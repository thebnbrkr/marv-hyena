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
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena.probes import load_sequence  # noqa: E402

KINDS = ("se", "mr", "li", "attn")


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

    genome = load_sequence(args.genome)
    hm = mh.HyenaModel.load(args.model)
    rows = []
    for gap in args.gaps:
        for seed in range(args.seeds):
            rng = random.Random(seed)
            need = 1000 + gap + 50
            off = rng.randrange(0, len(genome) - need)
            probe = mh.copy_probe(genome[off:off + need], insert_len=args.insert, gap=gap, lead=1000, seed=seed)
            ids = hm.ids(probe.seq)
            base = mh.score_copy(hm.logits(ids), ids, probe)
            rows.append({"gap": gap, "seed": seed, "condition": "none", **base})
            conditions = {f"-{k}": mh.mixers_of(hm, k) for k in KINDS}
            if args.per_block:
                conditions.update({f"-L{b}({hm.kind(b)})": [(b, "mixer")]
                                   for b in hm.blocks_of("attn") + hm.blocks_of("li")})
            for name, comps in conditions.items():
                means = mh.mean_writes(hm, ids, comps)
                with mh.mean_ablate(hm, means):
                    s = mh.score_copy(hm.logits(ids), ids, probe)
                rows.append({"gap": gap, "seed": seed, "condition": name, **s})
            print(f"gap={gap} seed={seed} done")

    Path(args.out).write_text(json.dumps(rows, indent=1))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    print(f"\nsecond-copy accuracy (first-copy accuracy in brackets), mean over {args.seeds} seeds")
    print(f"{'gap':>7} " + " ".join(f"{c:>14}" for c in conds))
    for gap in args.gaps:
        cells = []
        for c in conds:
            sel = [r for r in rows if r["gap"] == gap and r["condition"] == c]
            a2 = sum(r["second_acc"] for r in sel) / len(sel)
            a1 = sum(r["first_acc"] for r in sel) / len(sel)
            cells.append(f"{a2:>7.2f} ({a1:.2f})")
        print(f"{gap:>7} " + " ".join(f"{x:>14}" for x in cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
