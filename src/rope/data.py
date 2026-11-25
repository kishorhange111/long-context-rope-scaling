"""Data: Python standard-library source as a byte corpus, plus synthetic passkey retrieval.

A passkey example hides ``The pass key is 48213. Remember it. 48213 is the pass key.`` at a
chosen depth inside filler text and ends with ``What is the pass key? The pass key is 48213``.
The model has to copy the five digits from wherever the needle is, so accuracy at length L
tests whether attention still finds a token L positions back.
"""

from __future__ import annotations

import glob
import os
import sysconfig
import zlib

import numpy as np

QUESTION = b" What is the pass key? The pass key is "
N_DIGITS = 5


def needle(key: bytes) -> bytes:
    return b" The pass key is " + key + b". Remember it. " + key + b" is the pass key. "


def load_corpus() -> tuple[np.ndarray, np.ndarray]:
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .py files under {root}")
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (np.frombuffer(b"\n".join(tr), np.uint8).copy(),
            np.frombuffer(b"\n".join(va), np.uint8).copy())  # fmt: skip


def passkey_example(src: np.ndarray, length: int, depth: float, rng: np.random.Generator):
    """(tokens of exactly ``length`` bytes, index of the first answer digit)."""
    key = "".join(str(d) for d in rng.integers(0, 10, N_DIGITS)).encode()
    ndl, q = needle(key), QUESTION + key
    n_fill = length - len(ndl) - len(q)
    if n_fill < 0:
        raise ValueError(f"length {length} is too short for a passkey example")
    start = int(rng.integers(0, len(src) - n_fill))
    fill = src[start : start + n_fill].tobytes()
    cut = int(round(depth * n_fill))
    seq = fill[:cut] + ndl + fill[cut:] + q
    return np.frombuffer(seq, np.uint8).astype(np.int64), length - N_DIGITS


def text_window(src: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    i = int(rng.integers(0, len(src) - length))
    return src[i : i + length].astype(np.int64)


def train_batch(src: np.ndarray, bs: int, length: int, rng: np.random.Generator,
                p_passkey: float = 0.5) -> np.ndarray:  # fmt: skip
    """(bs, length + 1) token rows; a fraction are passkey examples at a random depth."""
    rows = []
    for _ in range(bs):
        if rng.random() < p_passkey:
            rows.append(passkey_example(src, length + 1, float(rng.random()), rng)[0])
        else:
            rows.append(text_window(src, length + 1, rng))
    return np.stack(rows)
