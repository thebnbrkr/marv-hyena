"""Test 1 (PREDICTIONS.md P1): who copies across long distances?

A random insert appears twice, `gap` letters apart, inside real E. coli DNA.
For each gap we score the second copy with the full model and with every
mixer of one operator type mean-ablated.

    python scripts/run_copy_test.py --genome NC_000913.gb --gaps 100 1000 10000 50000

Reading the table: second_acc near first_acc means the model could NOT copy.
If P1 holds, "-attn" collapses second_acc at long gaps while "-li" does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena.experiments import copy_test, summarize_copy  # noqa: E402
from marv_hyena.probes import load_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--gaps", type=int, nargs="+", default=[100, 1000, 10000, 50000])
    ap.add_argument("--insert", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-block", action="store_true", help="also ablate attention/LI blocks one at a time")
    ap.add_argument("--out", default="copy_test.json")
    args = ap.parse_args()

    hm = mh.HyenaModel.load(args.model)
    rows = copy_test(hm, load_sequence(args.genome), gaps=args.gaps, insert_len=args.insert, seeds=args.seeds,
                     per_block_kinds=("attn", "li") if args.per_block else ())
    Path(args.out).write_text(json.dumps(rows, indent=1))

    gaps, conds, a2, a1 = summarize_copy(rows)
    print(f"\nsecond-copy accuracy (first-copy accuracy in brackets), mean over {args.seeds} seeds")
    print(f"{'gap':>7} " + " ".join(f"{c:>14}" for c in conds))
    for i, gap in enumerate(gaps):
        print(f"{gap:>7} " + " ".join(f"{f'{a2[i, j]:.2f} ({a1[i, j]:.2f})':>14}" for j in range(len(conds))))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
