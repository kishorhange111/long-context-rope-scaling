"""Train at 256 bytes, then test every RoPE extension at up to 16x the trained length.

    python -m rope.train --out results --device cuda

Phases. Each is resumable: finished records are kept in ``<out>/runs.jsonl``, and the base
checkpoint is kept in ``<out>/base.pt``.

1. ``base``: pre-train at ``train_len = 256`` on stdlib source plus 50% passkey examples.
2. ``zero_shot``: no further training. For each method and eval length L, set the scale to
   ``s = L / 256`` and measure val bits/byte, over the whole window and over the positions
   past 256 only (the *tail*), plus passkey accuracy at needle depths 10%, 50% and 90%.
3. ``finetune``: from the base checkpoint, a short run at length 1024 (s = 4) per method,
   then the same evaluation with s fixed at 4. This is the "extend with a few hundred steps"
   recipe from the PI and YaRN papers; ``none`` here means fine-tuning with plain RoPE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_corpus, passkey_example, text_window, train_batch
from .model import GPT, Config, n_params
from .scaling import METHODS, Rope

DEPTHS = (0.1, 0.5, 0.9)


def train(model: GPT, src, steps: int, length: int, bs: int, lr: float, device: str,
          rope: Rope | None = None, warmup: int = 100, seed: int = 0, cosine: bool = True,
          log=None) -> list[dict]:  # fmt: skip
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        f = min(1.0, step / warmup)
        if cosine:
            f *= 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.02)
        xy = torch.from_numpy(train_batch(src, bs, length, rng)).to(device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            logits = model(xy[:, :-1], rope)
        loss = F.cross_entropy(logits.float().flatten(0, 1), xy[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, steps // 10) == 0:
            hist.append({"step": step, "loss": loss.item()})
            if log:
                log(f"  step {step} loss {loss.item():.3f}")
    model.eval()
    return hist


@torch.no_grad()
def evaluate(model: GPT, val, length: int, rope: Rope, device: str, n_windows: int = 16,
             n_passkey: int = 32, bs: int = 8, seed: int = 1234) -> dict:  # fmt: skip
    """val bits/byte (whole window and tail past train_len) and passkey accuracy by depth."""
    amp = device == "cuda"
    rng = np.random.default_rng(seed)
    tl = model.c.train_len
    full, tail = [], []
    for i in range(0, n_windows, bs):
        rows = [text_window(val, length + 1, rng) for _ in range(min(bs, n_windows - i))]
        xy = torch.from_numpy(np.stack(rows)).to(device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            logits = model(xy[:, :-1], rope)
        ce = F.cross_entropy(logits.float().transpose(1, 2), xy[:, 1:], reduction="none")
        full.append(ce.mean(1))
        if length > tl:
            tail.append(ce[:, tl:].mean(1))
    rec = {"bpb": float(torch.cat(full).mean()) / math.log(2),
           "bpb_tail": float(torch.cat(tail).mean()) / math.log(2) if tail else None}  # fmt: skip
    acc = {}
    for d in DEPTHS:
        hits = 0
        for i in range(0, n_passkey, bs):
            exs = [passkey_example(val, length, d, rng) for _ in range(min(bs, n_passkey - i))]
            x = torch.from_numpy(np.stack([e[0] for e in exs])).to(device)
            a = exs[0][1]
            with torch.autocast("cuda", torch.float16, enabled=amp):
                pred = model(x, rope)[:, a - 1 : -1].argmax(-1)
            hits += int((pred == x[:, a:]).all(1).sum())
        acc[str(d)] = hits / n_passkey
    rec["passkey"] = acc
    rec["passkey_mean"] = float(np.mean(list(acc.values())))
    return rec


def zero_shot_rope(method: str, length: int, train_len: int) -> Rope:
    s = 1.0 if method == "none" else max(1.0, length / train_len)
    return Rope(method, scale=s, orig_len=train_len)


def _table(recs, phase, key, lens, pct=False):
    rows = ["| method | " + " | ".join(f"{L:,}" for L in lens) + " |",
            "|---|" + "---|" * len(lens)]  # fmt: skip
    for m in METHODS:
        by = {r["length"]: r for r in recs if r["phase"] == phase and r["method"] == m}
        if not by:
            continue
        cells = []
        for L in lens:
            v = by.get(L, {}).get(key)
            cells.append("–" if v is None else (f"{100 * v:.0f}%" if pct else f"{v:.3f}"))
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def report(recs, out: str, gpu: str):
    lens = sorted({r["length"] for r in recs if "length" in r})
    base = next((r for r in recs if r["phase"] == "base"), None)
    lines = [f"GPU **{gpu}**. Base model: {base['params']:,} params, trained at "
             f"{base['train_len']} bytes for {base['steps']} steps.\n" if base else ""]  # fmt: skip
    for phase, title in (("zero_shot", "Zero-shot (s = L / 256)"),
                         ("finetune", "After fine-tuning at 1,024 (s = 4)")):  # fmt: skip
        if not any(r["phase"] == phase for r in recs):
            continue
        lines += [f"### {title}\n", "Val bits/byte, positions past 256 only:\n",
                  _table(recs, phase, "bpb_tail", lens), "",
                  "Val bits/byte, whole window:\n", _table(recs, phase, "bpb", lens), "",
                  "Passkey accuracy (mean over depths 10/50/90%):\n",
                  _table(recs, phase, "passkey_mean", lens, pct=True), ""]  # fmt: skip
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--phases", nargs="+", default=["base", "zero_shot", "finetune"])
    ap.add_argument("--tiny", action="store_true", help="smoke test on CPU")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    path, ckpt = os.path.join(a.out, "runs.jsonl"), os.path.join(a.out, "base.pt")
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    done = {(r["phase"], r.get("method")) for r in recs}

    def emit(rs):
        with open(path, "a") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        recs.extend(rs)

    cfg = Config(d_model=64, n_layers=2, n_heads=2) if a.tiny else Config()
    bs = 4 if a.tiny else 32
    train_src, val = load_corpus()
    log(f"corpus: {len(train_src):,} train bytes, {len(val):,} val bytes")
    model = GPT(cfg).to(a.device)
    if ("base", None) in done and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=a.device))
        log("loaded base checkpoint")
    elif "base" in a.phases:
        t = time.time()
        hist = train(model, train_src, a.steps, cfg.train_len, bs, 1e-3, a.device, log=log)
        torch.save(model.state_dict(), ckpt)
        emit([{"phase": "base", "method": None, "params": n_params(model), "steps": a.steps,
               "train_len": cfg.train_len, "minutes": (time.time() - t) / 60,
               "history": hist}])  # fmt: skip
        log("base done")
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    if "zero_shot" in a.phases:
        for m in METHODS:
            if ("zero_shot", m) in done:
                continue
            rs = [{"phase": "zero_shot", "method": m, "length": L,
                   **evaluate(model, val, L, zero_shot_rope(m, L, cfg.train_len), a.device)}
                  for L in a.lens]  # fmt: skip
            emit(rs)
            log(f"zero_shot {m}: " + ", ".join(f"{r['length']}:{r['bpb']:.3f}" for r in rs))

    if "finetune" in a.phases:
        ft_len = 4 * cfg.train_len
        for m in ("none", "pi", "ntk", "yarn"):
            if ("finetune", m) in done:
                continue
            model.load_state_dict(base_state)
            rope = Rope(m, scale=1.0 if m == "none" else 4.0, orig_len=cfg.train_len)
            t = time.time()
            train(model, train_src, a.ft_steps, ft_len, max(1, bs // 4), 2e-4, a.device,
                  rope=rope, warmup=20, seed=1, cosine=False)  # fmt: skip
            mins = (time.time() - t) / 60
            rs = [{"phase": "finetune", "method": m, "length": L, "ft_steps": a.ft_steps,
                   "ft_minutes": mins, **evaluate(model, val, L, rope, a.device)}
                  for L in a.lens]  # fmt: skip
            emit(rs)
            log(f"finetune {m}: " + ", ".join(f"{r['length']}:{r['bpb']:.3f}" for r in rs))

    gpu = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
    print(report(recs, a.out, gpu))


if __name__ == "__main__":
    main()
