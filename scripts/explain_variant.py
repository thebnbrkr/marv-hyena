"""WHY does Evo 2 score this mutation the way it does?

    python scripts/explain_variant.py --genome chr17.fa --pos 41276044 --ref A --alt G

Prints (1) the zero-shot delta log-likelihood, (2) which components carry the
mutation's effect on the downstream letters (activation patching, total
effect), grouped by operator type, and (3) for the letter right after the
variant, a direct-effect trace in the ref and alt runs plus how far back each
Hyena block was looking.

--pos is 0-based into the first record of --genome.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena.probes import load_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--span", type=int, default=200, help="downstream letters scored for patching")
    args = ap.parse_args()

    hm = mh.HyenaModel.load(args.model)
    v = mh.make_variant(load_sequence(args.genome), args.pos, args.ref, args.alt, args.window)

    print(f"delta log-likelihood (alt - ref, mean per letter): {mh.delta_logp(hm, v):+.5f}\n")

    print("== total effect: patch ALT components into the REF run ==")
    sweep = mh.explain_variant(hm, v, span=args.span)
    sweep.show(k=15)

    nxt = v.index + 1
    print(f"\n== direct effect on the letter after the variant ({v.ref_window[nxt]}) ==")
    for name, w in (("ref", v.ref_window), ("alt", v.alt_window)):
        d = mh.decompose_prediction(hm, w, nxt, target=v.ref_window[nxt])
        print(f"[{name}]", "  ".join(f"{k}={x:+.3f}" for k, x in sorted(d.by_kind().items())),
              f"  (actual {d.actual:+.3f})")

    print("\n== how far back each Hyena block looked (alt run, direct logit contribution per lag band) ==")
    direction = mh.normed_direction(hm, v.alt_window, nxt, target=v.ref_window[nxt])
    mh.show_profile(mh.distance_profile(hm, v.alt_window[: nxt], nxt - 1), direction=direction)


if __name__ == "__main__":
    main()
