"""The exact motif dictionary of Evo 2's first layer (direction B).

Measures block 0's receptive field, runs every possible k-mer through it
(4**9 = 262,144 for k=9), and reports the channels with the sharpest motifs.

    python scripts/block0_motifs.py --model evo2_7b --out block0_motifs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena import motifs  # noqa: E402


def information_content(pwm: np.ndarray) -> float:
    p = np.clip(pwm, 1e-9, 1)
    return float((2 + (p * np.log2(p)).sum(1)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--n-top", type=int, default=200)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--out", default="block0_motifs.json")
    args = ap.parse_args()

    hm = mh.HyenaModel.load(args.model)
    k = motifs.receptive_field(hm)
    print(f"block 0 receptive field: {k} letters -> enumerating {4 ** k:,} k-mers")
    md = motifs.enumerate_block0(hm, k=k, n_top=args.n_top)

    ic = np.array([information_content(md.pwm(c)) for c in range(hm.hidden_size)])
    order = np.argsort(-ic)
    print(f"\n{'channel':>7} {'IC(bits)':>8} {'consensus':>12} {'top value':>10}  top 3 k-mers")
    for c in order[: args.show]:
        print(f"{c:>7} {ic[c]:>8.2f} {md.consensus(c):>12} {md.top_vals[c, 0]:>10.3f}  {', '.join(md.top_kmers[c][:3])}")

    Path(args.out).write_text(json.dumps({
        "k": k,
        "channels": [{"channel": int(c), "ic": float(ic[c]), "consensus": md.consensus(c),
                      "top": md.top_kmers[c][:50], "top_vals": md.top_vals[c, :50].tolist(),
                      "bottom": md.bottom_kmers[c][:50], "bottom_vals": md.bottom_vals[c, :50].tolist()}
                     for c in range(hm.hidden_size)],
    }))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
