"""Byte-level pre-norm GPT with RoPE. The RoPE variant is an argument of ``forward``, so a
single trained model can be evaluated under every context-extension method."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .scaling import Rope, apply_rope, cos_sin


@dataclass
class Config:
    vocab: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    train_len: int = 256


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.h = c.n_heads
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model, bias=False), nn.GELU(),
                                 nn.Linear(4 * c.d_model, c.d_model, bias=False))  # fmt: skip

    def forward(self, x, cos, sin):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, l, d))
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style residual scaling
            nn.init.normal_(b.proj.weight, std=0.02 / (2 * c.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, rope: Rope | None = None):
        rope = rope or Rope(orig_len=self.c.train_len)
        hd = self.c.d_model // self.c.n_heads
        cos, sin = cos_sin(idx.shape[1], hd, rope, idx.device, self.emb.weight.dtype)
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x) @ self.emb.weight.t()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
