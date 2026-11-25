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

**Results are pending** from the Colab run.
