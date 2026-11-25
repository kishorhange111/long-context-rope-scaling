# long-context-rope-scaling

[![CI](https://github.com/kishorhange111/long-context-rope-scaling/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/long-context-rope-scaling/actions/workflows/ci.yml)

How do you run a RoPE model at 16× the length it was trained on? This repo implements the
four standard answers from scratch and compares them under controlled conditions:
- Position Interpolation;
- NTK-aware scaling;
- dynamic NTK;
- YaRN.

## Why RoPE breaks past its training length

RoPE rotates channel pair *i* of q and k by angle `m·θᵢ` at position *m*, with
`θᵢ = base^(−2i/d)`. Rotations compose, so `⟨R_m q, R_n k⟩` depends only on `m − n`.

A model trained on length *L* has only seen angles below `L·θᵢ`. High-frequency pairs wrap
around many times within *L*, so every angle is familiar. Low-frequency pairs, with
wavelength `2π/θᵢ > L`, never complete a turn. Past *L* they produce angles the model has
never seen.

| method ([`scaling.py`](src/rope/scaling.py)) | change, with `s = L_new / L` | trade-off |
|---|---|---|
| `pi`: Position Interpolation (Chen et al. 2023) | `θᵢ → θᵢ / s` | every pair is compressed, so neighbouring tokens become harder to tell apart |
| `ntk`: NTK-aware (bloc97 2023) | `base → base · s^(d/(d−2))` | highest frequency unchanged, lowest divided by exactly *s*, smooth in between |
| `yarn` (Peng et al. 2023) | per-pair ramp on `rᵢ = L / wavelengthᵢ`: unchanged if `r > β`, PI if `r < α`, blended between; q and k scaled by `0.1·ln s + 1` | targets only the pairs that are actually out of distribution, and corrects the attention entropy |
| `dynamic` | NTK-aware with `s = max(1, len / L)` | no change at all within the trained length |

The tests check:
- scores depend only on relative position;
- the rotation preserves norms;
- every method reduces to plain RoPE at `s = 1`;
- PI maps position `s·m` exactly onto `m`;
- NTK keeps the top frequency and divides the bottom one by *s*;
- YaRN leaves `r > β` pairs alone and interpolates `r < α` pairs exactly like PI;
- YaRN's mscale multiplies the logits by its square;
- dynamic NTK is a no-op within *L*.

## Experiment ([`train.py`](src/rope/train.py))

A 4.8M-parameter byte-level GPT is trained once at **256 bytes**, on Python standard-library
source mixed 50/50 with synthetic **passkey retrieval**. A passkey example hides
`The pass key is 48213…` at some depth in filler text and asks for it at the end. It is then
evaluated at 256 to 4,096 bytes under every method:

- **zero-shot**: no training, `s = L / 256`;
- **fine-tuned**: 300 steps at 1,024 (`s = 4`) per method, as in the PI and YaRN recipes.
  Here `none` means fine-tuning with unmodified RoPE.

The metrics are:
- val bits/byte over the whole window;
- val bits/byte over the **tail** (positions past 256, where extrapolation actually happens);
- passkey accuracy at needle depths of 10%, 50% and 90%, counted as correct only if all
  five digits are argmax-correct.

```
pytest -q                                          # 16 tests, CPU
python -m rope.train --out results --device cuda   # resumable: runs.jsonl + base.pt
```

## Results (Tesla T4, one seed)

The base model trains in 3.9 minutes, and each fine-tune takes 20 seconds (300 steps).
Passkey cells are 96 examples each (32 per depth). Raw records are in
[`results/runs.jsonl`](results/runs.jsonl).

GPU **Tesla T4**. Base model: 4,787,456 params, trained at 256 bytes for 4000 steps.

### Zero-shot (s = L / 256)

Val bits/byte, positions past 256 only:

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | – | 2.519 | 3.937 | 4.441 | 4.889 |
| pi | – | 3.001 | 4.127 | 4.630 | 4.892 |
| ntk | – | 1.341 | 2.161 | 3.027 | 3.688 |
| yarn | – | 1.400 | 1.542 | 1.721 | 1.965 |
| dynamic | – | 1.341 | 2.161 | 3.027 | 3.688 |

Val bits/byte, whole window:

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | 1.325 | 1.886 | 3.277 | 4.048 | 4.662 |
| pi | 1.325 | 2.948 | 4.103 | 4.616 | 4.875 |
| ntk | 1.325 | 1.304 | 1.967 | 2.831 | 3.552 |
| yarn | 1.325 | 1.417 | 1.573 | 1.735 | 1.959 |
| dynamic | 1.325 | 1.304 | 1.967 | 2.831 | 3.552 |

Passkey accuracy (mean over depths 10/50/90%):

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | 100% | 0% | 0% | 0% | 0% |
| pi | 100% | 0% | 0% | 0% | 0% |
| ntk | 100% | 26% | 0% | 0% | 0% |
| yarn | 100% | 72% | 2% | 0% | 0% |
| dynamic | 100% | 26% | 0% | 0% | 0% |

### After fine-tuning at 1,024 (s = 4)

Val bits/byte, positions past 256 only:

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | – | 1.229 | 1.173 | 1.280 | 1.758 |
| pi | – | 1.294 | 1.228 | 2.439 | 3.966 |
| ntk | – | 1.218 | 1.162 | 1.597 | 2.321 |
| yarn | – | 1.219 | 1.162 | 2.098 | 3.596 |

Val bits/byte, whole window:

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | 1.361 | 1.255 | 1.215 | 1.287 | 1.729 |
| pi | 1.441 | 1.334 | 1.274 | 2.310 | 3.803 |
| ntk | 1.358 | 1.249 | 1.206 | 1.564 | 2.257 |
| yarn | 1.358 | 1.251 | 1.205 | 2.003 | 3.452 |

Passkey accuracy (mean over depths 10/50/90%):

| method | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| none | 98% | 99% | 99% | 3% | 0% |
| pi | 46% | 85% | 99% | 0% | 0% |
| ntk | 26% | 100% | 100% | 0% | 0% |
| yarn | 1% | 49% | 100% | 0% | 0% |

### What it shows

**Zero-shot, YaRN is the only method that degrades gracefully.**
- At 16× the trained length it keeps tail bits/byte at 1.97, against 4.89 for plain RoPE,
  where a uniform guess over bytes is 8.
- It is also the only one that still retrieves the passkey at 2× (72%).
- NTK-aware scaling helps at 2× and then collapses.

**PI is *worse* than doing nothing without fine-tuning** (3.00 vs 2.52 at 2×). Compressing
every frequency, including the high-frequency pairs that encode local order, breaks the
model's short-range modelling. That is why the PI paper always fine-tunes.

**Dynamic NTK and NTK give identical numbers here, as they should.** Each window is scored
in a single forward pass of length L, so dynamic NTK's `s = L/256` equals the fixed NTK
scale. The two only differ in incremental decoding, where dynamic NTK leaves the early
tokens unscaled.

**After 300 steps at 1,024, every method fits 1,024 well** (tail bpb 1.16–1.23, passkey
≥ 99%). Two things are less expected:
- **Fine-tuning with plain RoPE extrapolates best beyond the fine-tuned length** (1.76 at
  4,096, against 2.32 for NTK and 3.60 for YaRN). With `s` fixed at 4, the scaled methods
  have only learned positions up to 4 × 256, while plain RoPE trained on 1,024 real
  positions and still extrapolates its high-frequency pairs.
- **Scaled fine-tuning hurts short-context retrieval.** Passkey accuracy at 256 drops to 46%
  (PI), 26% (NTK) and 1% (YaRN), while plain fine-tuning keeps 98%. Their perplexity at
  256 barely changes (1.36 vs 1.33). Retrieval at short range relies on fine positional
  resolution, which a fixed `s = 4` removes. Dynamic scaling, or mixing short sequences into
  the fine-tune, is the usual fix and the obvious next experiment.

**Caveats:**
- one seed;
- a 4.8M-parameter byte model;
- a 300-step fine-tune;
- passkey is a synthetic retrieval task.

The direction of each effect is what to take away, not the exact magnitudes.
