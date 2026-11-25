"""Rotary position embeddings and the four standard ways to stretch them past the trained length.

RoPE rotates each pair of channels (i, i + d/2) of q and k by angle ``m · θ_i`` at position m,
with ``θ_i = base^(−2i/d)``. Since rotations compose, ``<R_m q, R_n k>`` depends only on m − n.

A model trained on length ``L`` has only seen angles ``m·θ_i < L·θ_i``. The low-frequency
channels, whose wavelength ``2π/θ_i`` is longer than L, have never completed a turn, so at
longer contexts they produce angles the model never saw. Each method fixes this differently.
With ``s = L_new / L``:

- ``pi`` (Position Interpolation, Chen et al. 2023): ``θ_i → θ_i / s``. Every channel is
  compressed, so the high-frequency channels that encode local order lose resolution.
- ``ntk`` (NTK-aware, bloc97 2023): ``base → base · s^(d/(d−2))``. The highest frequency is
  unchanged and the lowest is divided by exactly s, with a smooth interpolation in between.
- ``yarn`` (Peng et al. 2023): per-channel ramp on ``r_i = L / wavelength_i``. Channels with
  ``r > β`` (many turns in L) are left alone, channels with ``r < α`` are interpolated like PI,
  and those in between are blended. It also sharpens attention, scaling q and k by
  ``0.1·ln(s) + 1``.
- ``dynamic``: NTK-aware with ``s = max(1, current_length / L)``, so short inputs are untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

METHODS = ("none", "pi", "ntk", "yarn", "dynamic")


@dataclass(frozen=True)
class Rope:
    method: str = "none"
    scale: float = 1.0
    orig_len: int = 256
    base: float = 10000.0
    alpha: float = 1.0  # YaRN ramp: below this many turns per orig_len, interpolate fully
    beta: float = 32.0  # above this many turns, leave the channel alone

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown RoPE method {self.method!r}; choose from {METHODS}")


def base_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def yarn_ramp(dim: int, orig_len: int, base: float, alpha: float, beta: float) -> torch.Tensor:
    """γ_i ∈ [0, 1]: 1 keeps channel i as is, 0 interpolates it fully."""
    turns = orig_len * base_inv_freq(dim, base) / (2 * math.pi)  # r_i = L / wavelength_i
    return ((turns - alpha) / (beta - alpha)).clamp(0.0, 1.0)


def inv_freq(dim: int, rope: Rope, seq_len: int | None = None) -> tuple[torch.Tensor, float]:
    """Per-channel-pair frequencies (dim/2,) in float64, and the attention mscale."""
    s, m = rope.scale, rope.method
    if m == "dynamic":
        s = max(1.0, (seq_len or rope.orig_len) / rope.orig_len)
    if m == "none" or s == 1.0:
        return base_inv_freq(dim, rope.base), 1.0
    if m == "pi":
        return base_inv_freq(dim, rope.base) / s, 1.0
    if m in ("ntk", "dynamic"):
        return base_inv_freq(dim, rope.base * s ** (dim / (dim - 2))), 1.0
    theta = base_inv_freq(dim, rope.base)
    g = yarn_ramp(dim, rope.orig_len, rope.base, rope.alpha, rope.beta)
    return (1 - g) * theta / s + g * theta, 0.1 * math.log(s) + 1.0


def cos_sin(seq_len: int, dim: int, rope: Rope, device=None, dtype=torch.float32,
            offset: int = 0):  # fmt: skip
    """cos, sin of shape (seq_len, dim), already multiplied by YaRN's mscale."""
    f, mscale = inv_freq(dim, rope, seq_len + offset)
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float64)
    ang = torch.outer(pos, f)
    ang = torch.cat([ang, ang], -1)
    return ((ang.cos() * mscale).to(device=device, dtype=dtype),
            (ang.sin() * mscale).to(device=device, dtype=dtype))  # fmt: skip


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., L, dim); cos/sin: (L, dim). Pairs channel i with channel i + dim/2."""
    return x * cos + rotate_half(x) * sin
