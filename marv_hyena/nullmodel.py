"""Round 4: the random-weights null model.

Round 3 enumerated all 4**9 inputs to block 0 and found detector channels for
every one of the 64 three-letter words, median 46 per word, mean 45.2 -- a
strikingly FLAT distribution. The reading was "block 0 is a generic
3-letter-word detector bank; start and stop codons aren't special".

That reading has no null model, and a flat distribution over all words is
exactly what an UNTRAINED gated short convolution might produce: with a
9-letter window and random projections, "channels that respond to word W"
could be an artifact of the architecture rather than anything learned. The
bio-foundation-model review (bioRxiv 2026.03.04.709491) names this as the
control the field keeps skipping.

`random_weights` swaps a block's parameters for random ones and puts the
originals back, so the exact same enumeration can be run on an untrained
block 0 and the two distributions compared.

Two modes, and the difference matters:

- mode="shuffle" (default) permutes the entries of each parameter tensor.
  The multiset of weights is preserved EXACTLY -- same mean, variance,
  min, max, sparsity, every moment -- and only the arrangement is destroyed.
  Anything that survives this is a property of the architecture and the
  weight distribution, not of learned structure. This is the strong null.
- mode="gaussian" draws fresh values matching each tensor's mean and std.
  Weaker (it also destroys the shape of the distribution), kept as a check
  that a result is not an artifact of the shuffle.

Restoration is exact: the originals are cloned before anything is touched and
written back in `finally`, so a failed run cannot leave a scrambled model
behind. `verify_restored` re-checks that afterwards, because every later
number in a notebook depends on it.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .arch import HyenaModel

MODES = ("shuffle", "gaussian")


def _randomize(t: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    """A tensor of the same shape/dtype/device with the learned structure gone."""
    if t.numel() <= 1:
        return t.clone()  # a scalar has no structure to destroy
    flat = t.detach().reshape(-1).float()
    if mode == "shuffle":
        perm = torch.randperm(flat.numel(), generator=gen, device="cpu").to(flat.device)
        out = flat[perm]
    elif mode == "gaussian":
        out = torch.randn(flat.shape, generator=gen, device="cpu").to(flat.device)
        out = out * flat.std() + flat.mean()
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return out.reshape(t.shape).to(t.dtype)


def _tensors(hm: HyenaModel, block: int, include_norms: bool):
    """(module, kind, name) for every parameter/buffer of `block` we may touch."""
    out = []
    for mod in hm.block(block).modules():
        is_norm = "norm" in type(mod).__name__.lower()
        if is_norm and not include_norms:
            continue
        for name, p in mod._parameters.items():
            if p is not None:
                out.append((mod, "param", name))
        for name, b in mod._buffers.items():
            # Buffers that are not learned state (a cached filter, a position
            # index, a mask) must be left alone: scrambling them changes the
            # operator rather than its weights.
            if b is not None and b.is_floating_point() and name in _LEARNED_BUFFERS:
                out.append((mod, "buffer", name))
    return out


# Vortex keeps some learned Hyena parameters as buffers rather than Parameters.
# Only these are randomized; everything else buffered is cached/derived state.
_LEARNED_BUFFERS = {"h", "D", "bias", "filter", "short_filter_weight", "log_poles", "residues"}


@contextmanager
def random_weights(hm: HyenaModel, blocks, mode: str = "shuffle", seed: int = 0,
                   include_norms: bool = False):
    """Temporarily replace the weights of `blocks` with random ones.

    include_norms=False leaves LayerNorm/RMSNorm scales alone by default:
    scrambling them changes the scale the block writes at, which shows up as
    "the model broke" rather than "the detectors were not learned". Set True
    for a fully untrained block.

    Yields the number of tensors replaced.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    blocks = [blocks] if isinstance(blocks, int) else list(blocks)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    saved = []
    try:
        for b in blocks:
            for mod, kind, name in _tensors(hm, b, include_norms):
                store = mod._parameters if kind == "param" else mod._buffers
                t = store[name]
                cur = t.data if kind == "param" else t
                saved.append((store, name, kind, cur.detach().clone()))
                new = _randomize(cur, mode, gen)
                if kind == "param":
                    t.data = new
                else:
                    store[name] = new
        yield len(saved)
    finally:
        for store, name, kind, original in reversed(saved):
            if kind == "param":
                store[name].data = original
            else:
                store[name] = original


@torch.no_grad()
def verify_restored(hm: HyenaModel, seq: str, reference_logits: torch.Tensor,
                    tol: float = 1e-4) -> dict:
    """After a `random_weights` block, check the model gives the same logits as
    before. Every number measured later in the session depends on this."""
    now = hm.logits(seq)
    err = float((now - reference_logits.to(now.device)).abs().max())
    return {"max_abs_logit_diff": err, "restored": err <= tol}


def word_count_summary(ranks: list[dict], key: str = "controlled") -> dict:
    """Shape of the 64-word detector-count distribution, for comparing a trained
    block 0 against its shuffled null. `ranks` is `motifs.rank_all_words` output,
    whose per-word count lives under "controlled" (the composition-controlled
    count; "raw" is the uncontrolled one)."""
    import numpy as np

    v = np.array([float(r[key]) for r in ranks], dtype=float)
    return {
        "n_words": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        # Spread relative to the mean: the single number that separates "every
        # word looks alike" (low) from "some words are special" (high).
        "cv": float(v.std() / v.mean()) if v.mean() else float("nan"),
        "max_over_median": float(v.max() / np.median(v)) if np.median(v) else float("nan"),
    }
