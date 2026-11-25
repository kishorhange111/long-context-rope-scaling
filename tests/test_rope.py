import math

import numpy as np
import pytest
import torch

from rope.data import N_DIGITS, QUESTION, needle, passkey_example, train_batch
from rope.model import GPT, Config
from rope.scaling import METHODS, Rope, apply_rope, base_inv_freq, cos_sin, inv_freq, yarn_ramp
from rope.train import evaluate, main, zero_shot_rope

D = 64


def _rot(x, pos):
    cos, sin = cos_sin(pos + 1, D, Rope(), dtype=torch.float64)
    return apply_rope(x, cos[pos], sin[pos])


def test_scores_depend_only_on_relative_position():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(D, generator=g, dtype=torch.float64)
    k = torch.randn(D, generator=g, dtype=torch.float64)
    for shift in (1, 37, 500):
        torch.testing.assert_close(_rot(q, 9) @ _rot(k, 4), _rot(q, 9 + shift) @ _rot(k, 4 + shift))


def test_rotation_preserves_norm_and_position_zero_is_identity():
    x = torch.randn(5, D, dtype=torch.float64)
    cos, sin = cos_sin(5, D, Rope(), dtype=torch.float64)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(y[0], x[0])


@pytest.mark.parametrize("method", METHODS)
def test_scale_one_is_plain_rope(method):
    f, m = inv_freq(D, Rope(method, scale=1.0), seq_len=100)
    torch.testing.assert_close(f, base_inv_freq(D))
    assert m == 1.0


def test_position_interpolation_maps_position_s_m_to_m():
    s = 4.0
    c0, s0 = cos_sin(10, D, Rope(), dtype=torch.float64)
    c1, s1 = cos_sin(40, D, Rope("pi", scale=s), dtype=torch.float64)
    torch.testing.assert_close(c1[::4], c0)
    torch.testing.assert_close(s1[::4], s0)


def test_ntk_keeps_highest_frequency_and_divides_lowest_by_s():
    s = 8.0
    f0, f = base_inv_freq(D), inv_freq(D, Rope("ntk", scale=s))[0]
    assert f[0] == f0[0] == 1.0
    torch.testing.assert_close(f[-1], f0[-1] / s)
    assert torch.all(f <= f0) and torch.all(f >= f0 / s - 1e-15)


def test_yarn_ramp_regimes():
    s, L = 4.0, 256
    rope = Rope("yarn", scale=s, orig_len=L)
    f0, (f, mscale) = base_inv_freq(D), inv_freq(D, rope)
    turns = L * f0 / (2 * math.pi)
    hi, lo = turns > rope.beta, turns < rope.alpha
    assert hi.any() and lo.any()
    torch.testing.assert_close(f[hi], f0[hi])  # many turns in L: untouched
    torch.testing.assert_close(f[lo], f0[lo] / s)  # never a full turn: interpolated like PI
    g = yarn_ramp(D, L, 10000.0, 1.0, 32.0)
    assert torch.all(g[1:] <= g[:-1])  # monotone from high to low frequency
    assert mscale == pytest.approx(0.1 * math.log(s) + 1)


def test_yarn_mscale_scales_logits_by_its_square():
    q, k = torch.randn(D, dtype=torch.float64), torch.randn(D, dtype=torch.float64)
    rope = Rope("yarn", scale=4.0)
    c, s = cos_sin(1, D, rope, dtype=torch.float64)  # position 0: pure scaling, no rotation
    m = inv_freq(D, rope)[1]
    torch.testing.assert_close(
        apply_rope(q, c[0], s[0]) @ apply_rope(k, c[0], s[0]), m * m * (q @ k)
    )


def test_dynamic_ntk_is_plain_rope_within_trained_length():
    r = Rope("dynamic", orig_len=256)
    torch.testing.assert_close(inv_freq(D, r, 200)[0], base_inv_freq(D))
    torch.testing.assert_close(inv_freq(D, r, 1024)[0], inv_freq(D, Rope("ntk", scale=4.0))[0])


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        Rope("alibi")


def test_passkey_example_layout():
    rng = np.random.default_rng(0)
    src = np.frombuffer(b"abcdefghij" * 500, np.uint8)
    for depth in (0.0, 0.5, 1.0):
        x, a = passkey_example(src, 300, depth, rng)
        assert len(x) == 300 and a == 300 - N_DIGITS
        b = x.astype(np.uint8).tobytes()
        key = b[a:]
        assert key.isdigit() and b.endswith(QUESTION + key) and needle(key) in b
    assert train_batch(src, 6, 300, rng).shape == (6, 301)


def test_model_is_causal_and_runs_past_training_length():
    torch.manual_seed(0)
    m = GPT(Config(d_model=32, n_layers=2, n_heads=2, train_len=16)).double().eval()
    x = torch.randint(0, 256, (2, 40))
    y = x.clone()
    y[:, 30:] = 0
    for rope in (None, zero_shot_rope("yarn", 40, 16)):
        torch.testing.assert_close(m(x, rope)[:, :30], m(y, rope)[:, :30])


def test_evaluate_and_tiny_pipeline(tmp_path):
    m = GPT(Config(d_model=32, n_layers=1, n_heads=2, train_len=128))
    val = np.frombuffer(bytes(range(32, 127)) * 200, np.uint8)
    r = evaluate(m, val, 256, Rope("pi", 2.0, 128), "cpu", n_windows=2, n_passkey=2, bs=2)
    assert r["bpb"] > 0 and r["bpb_tail"] > 0 and 0 <= r["passkey_mean"] <= 1
    main(["--tiny", "--device", "cpu", "--out", str(tmp_path), "--steps", "3", "--ft-steps", "2",
          "--lens", "256", "512"])  # fmt: skip
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 1 + 5 * 2 + 4 * 2
    assert "Zero-shot" in (tmp_path / "summary.md").read_text()
    main(["--tiny", "--device", "cpu", "--out", str(tmp_path), "--lens", "256", "512"])  # resumes
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == len(lines)
