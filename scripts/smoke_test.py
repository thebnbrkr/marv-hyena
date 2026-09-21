"""RUN THIS FIRST on the A100. Checks every assumption marv-hyena makes about
the installed Vortex against the real Evo 2, before any experiment is trusted.

    python scripts/smoke_test.py --model evo2_7b --genome /path/NC_000913.gb

Each check prints PASS/FAIL. Exit code 1 if anything fails. Tolerances are
loose (bf16); the tiny-model tests hold the same code to 1e-4 in float32.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena.checks import run_smoke_checks  # noqa: E402
from marv_hyena.probes import load_sequence  # noqa: E402

DEFAULT_GENOME = Path(__file__).resolve().parents[2] / "evo2-main/notebooks/sparse_autoencoder/NC_000913.gb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--genome", default=str(DEFAULT_GENOME))
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    if Path(args.genome).exists():
        genome = load_sequence(args.genome)
        seq = genome[100_000:100_000 + args.length]
        print(f"sequence: {args.genome} [100000:{100_000 + args.length}]")
    else:
        rng = random.Random(0)
        seq = "".join(rng.choice("ACGT") for _ in range(args.length))
        print("sequence: random (genome file not found)")

    t0 = time.time()
    hm = mh.HyenaModel.load(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s on {hm.device}\n{hm.describe()}\n")
    ok = run_smoke_checks(hm, seq, tol=args.tol)

    print()
    mh.show_reach(mh.reach_table(hm))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
