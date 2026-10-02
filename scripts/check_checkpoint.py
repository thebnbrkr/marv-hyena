#!/usr/bin/env python3
"""Inspect an Evo 2 checkpoint file's weights directly. No GPU, no model build.

Written 2026-10-01 to settle a question the scaling runs raised. The 20B and 40B
runs produced *bit-identical* numbers for every block they share (0-23),
including the embedding write, which depends on no block weights at all. Two
explanations fit:

  H1  the merged `evo2_40b.pt` does not hold the 40B weights, so a 50-block
      skeleton got another checkpoint's 24 blocks and the rest kept their
      initialisation.
  H2  the checkpoints genuinely share their first 24 blocks -- Evo 2's 20B and
      40B have the SAME width (8192) and roughly double the depth (24 -> 50),
      which is what depth-growth training produces.

H2 is plausible and would make the 40B run valid, with "26 dead blocks" as a
real finding. This script separates them in about a minute:

  * If late blocks are absent from the file, or their tensors are all zero,
    that is H1 and the 40B run must be redone.
  * If late blocks carry trained-looking weights, that is H2, and the shared
    prefix is a fact about the checkpoints rather than a mistake.

Usage
-----
    python check_checkpoint.py /work/hf_cache/evo2_40b.pt
    python check_checkpoint.py /work/hf_cache/evo2_20b.pt /work/hf_cache/evo2_40b.pt

With two files it also reports whether their shared tensors are identical,
which is the direct test of the shared-prefix hypothesis.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def load_state_dict(path: str):
    """Load an Evo 2 checkpoint's state dict, tensors only.

    Two real failures, found by the first person to run this on real files
    (2026-10-02), both fixed here:

    1. `weights_only=True` refuses the file. Evo 2 checkpoints carry Transformer
       Engine `*._extra_state` entries, which are pickled `io.BytesIO` blobs of
       FP8 scaling metadata. Allowlisting `io.BytesIO` through
       `safe_globals` keeps the code-execution protection that
       `weights_only=False` would throw away -- BytesIO holds bytes, it does not
       execute anything.
    2. Those same entries are not tensors, so `.is_floating_point()` blows up on
       them later. They are dropped here, at the boundary, and counted.
    """
    import io

    import torch

    def _load(**kw):
        try:  # mmap keeps 80 GB off the heap; available from torch 2.1
            return torch.load(path, map_location="cpu", mmap=True, **kw)
        except TypeError:
            return torch.load(path, map_location="cpu", **kw)

    try:
        with torch.serialization.safe_globals([io.BytesIO]):
            sd = _load(weights_only=True)
    except AttributeError:  # torch < 2.3 has no safe_globals context manager
        torch.serialization.add_safe_globals([io.BytesIO])
        sd = _load(weights_only=True)
    while isinstance(sd, dict) and len(sd) <= 3 and any(
            k in sd for k in ("state_dict", "model", "module")):
        sd = sd.get("state_dict") or sd.get("model") or sd.get("module")
    dropped = [k for k, v in sd.items() if not isinstance(v, torch.Tensor)]
    if dropped:
        print(f"dropped {len(dropped)} non-tensor entries "
              f"(Transformer Engine FP8 metadata), e.g. {dropped[:2]}")
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    return sd


def block_index(key: str) -> int | None:
    m = re.match(r"(?:.*\.)?blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def describe(path: str) -> dict:
    import torch

    size_gb = Path(path).stat().st_size / 1e9
    print(f"\n=== {path}")
    print(f"file size: {size_gb:.1f} GB   "
          f"(Evo 2 bf16: 7B ~14 GB, 20B ~40 GB, 40B ~80 GB)")
    sd = load_state_dict(path)
    print(f"tensors: {len(sd)}")

    by_block: dict[int, list[str]] = {}
    for k in sd:
        b = block_index(k)
        if b is not None:
            by_block.setdefault(b, []).append(k)
    if not by_block:
        print("no keys matching blocks.<n>. -- is this an Evo 2 checkpoint?")
        return {"path": path, "blocks": []}

    blocks = sorted(by_block)
    print(f"block indices present: {blocks[0]}..{blocks[-1]}  (count {len(blocks)})")
    gaps = [b for b in range(blocks[0], blocks[-1] + 1) if b not in by_block]
    if gaps:
        print(f"*** MISSING block indices inside the range: {gaps}")

    print(f"\n{'block':>6s} {'tensors':>8s} {'abs mean':>12s} {'zero frac':>10s}  sample key")
    probe = sorted({blocks[0], blocks[len(blocks) // 4], blocks[len(blocks) // 2],
                    blocks[3 * len(blocks) // 4], blocks[-1], 23, 24} & set(blocks))
    stats = {}
    for b in probe:
        keys = sorted(k for k in by_block[b] if sd[k].is_floating_point()
                      and sd[k].numel() > 1024)
        if not keys:
            keys = sorted(by_block[b])
        t = sd[keys[0]].float()
        am, zf = float(t.abs().mean()), float((t == 0).float().mean())
        stats[b] = (am, zf)
        flag = "   <-- ALL ZERO" if zf > 0.99 else ("   <-- near zero" if am < 1e-8 else "")
        print(f"{b:6d} {len(by_block[b]):8d} {am:12.4e} {zf:10.3f}  {keys[0][:46]}{flag}")

    # Per-block probing above picks ONE tensor per block, and which one depends on the
    # block's type: a Hyena block offers filter.D / filter.h, an attention block offers
    # inner_mha_cls.Wqkv.weight. Those have different natural scales, so comparing their
    # |mean| ACROSS blocks is apples to oranges -- it is what made the 2026-10-02 reading of
    # "blocks 24 and 49 look near-initialisation" ambiguous. The only meaningful comparison
    # is within one tensor name, across the blocks that have it.
    print(f"\n{'tensor (by name)':44s} {'blocks':>7s} {'median |mean|':>14s} {'min':>11s} "
          f"{'max':>11s} {'max/min':>10s}  blocks < max/30")
    suffixes: dict[str, dict[int, float]] = {}
    for k, v in sd.items():
        b = block_index(k)
        if b is None or not v.is_floating_point() or v.numel() <= 1024:
            continue
        suffixes.setdefault(re.sub(r"^(?:.*\.)?blocks\.\d+\.", "", k), {})[b] = float(v.float().abs().mean())
    odd = {}
    for name, per in sorted(suffixes.items()):
        if len(per) < 3:
            continue
        vals = sorted(per.values())
        med = vals[len(vals) // 2]
        top = vals[-1]
        # Compare against the MAXIMUM, not the median. The median assumes most blocks are
        # healthy, and in the 40B more than half are not: its mlp.l1 median is 7.7e-6 because
        # 26 of 50 blocks sit there, so a median rule declares the affected majority normal
        # and flags nobody. The runner caught this. The max is the one order statistic that
        # survives majority contamination, since at least one healthy block sets it.
        out = sorted(b for b, x in per.items() if top > 0 and x < top / 30)
        spread = top / vals[0] if vals[0] > 0 else float("inf")
        for b in out:
            odd.setdefault(b, []).append(name)
        warn = "  <-- majority affected" if len(out) > len(per) / 2 else ""
        print(f"{name[:44]:44s} {len(per):7d} {med:14.4e} {vals[0]:11.4e} {vals[-1]:11.4e}"
              f" {spread:10.3g}  {out if out else ''}{warn}")
    if odd:
        print(f"\n*** Blocks far below the median FOR THEIR OWN TENSOR: "
              f"{ {b: v for b, v in sorted(odd.items())} }")
        print("    Same tensor, same model, so this comparison is like-for-like. Blocks that")
        print("    appear here look under-trained relative to their siblings.")
        print("\n    CAVEAT: Evo 2 checkpoints also carry Transformer Engine `_extra_state` blobs")
        print("    (dropped above) that hold FP8 scaling metadata. If part of a model is stored")
        print("    FP8-scaled and part is not, raw |mean| is NOT comparable across those parts,")
        print("    and this table would show exactly the pattern above. Treat a whole-contiguous-")
        print("    half result as a question about storage format, not a verdict on training.")
    else:
        print("\nNo block is an outlier within its own tensor name: nothing looks under-trained.")

    dead = [b for b, (am, zf) in stats.items() if zf > 0.99 or am < 1e-8]
    print()
    if dead:
        print(f"*** Blocks {dead} carry zero / near-zero weights: the file is incomplete (H1).")
        print("    The 40B run must be redone after re-merging the shards.")
    else:
        print("All probed blocks carry trained-looking weights (H2 side):")
        print("    the file is not obviously truncated.")
    return {"path": path, "blocks": blocks, "sd": sd, "stats": stats}


def compare(a: dict, b: dict) -> None:
    import torch

    sa, sb = a["sd"], b["sd"]
    shared = sorted(set(sa) & set(sb))
    print(f"\n=== comparing the two files")
    print(f"keys: {len(sa)} vs {len(sb)}, shared {len(shared)}")
    if not shared:
        print("no shared keys -- different naming, nothing to compare")
        return
    same = diff = 0
    first_diff = None
    for k in shared:
        ta, tb = sa[k], sb[k]
        if ta.shape != tb.shape:
            diff += 1
            continue
        if torch.equal(ta, tb):
            same += 1
        else:
            diff += 1
            first_diff = first_diff or k
    print(f"shared tensors BIT-IDENTICAL: {same}/{len(shared)}   differing: {diff}")
    if first_diff:
        print(f"first differing key: {first_diff}")
    if same == len(shared):
        print("\n*** Every shared tensor is identical.")
        print("    Either the two files are the same weights (one is mislabelled), or the")
        print("    checkpoints genuinely share a prefix. The per-block table above decides:")
        print("    late blocks with real weights => shared prefix; zeros/missing => mislabelled.")
    elif same == 0:
        print("\n*** No shared tensor matches: these are genuinely different checkpoints.")
        print("    Then the identical RUN numbers came from somewhere else -- report that.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or two .pt checkpoint files")
    args = ap.parse_args()
    for p in args.paths:
        if not Path(p).exists():
            print(f"missing: {p}")
            return 1
    out = [describe(p) for p in args.paths]
    if len(out) == 2 and all("sd" in o for o in out):
        compare(*out)
    print("\nSend this whole output back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
