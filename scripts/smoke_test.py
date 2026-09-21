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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402
from marv_hyena import motifs  # noqa: E402
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
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    ids = hm.ids(seq)
    L = len(seq)

    # 1. every residual write is captured
    w = mh.capture_writes(hm, ids, [L // 2, L - 1])
    err = w.reconstruction_error()
    check("embed + sum(writes) == final residual", err < args.tol, f"rel_err={err:.2e}")

    # 2. trace adds up to the real logit difference
    d = mh.decompose_prediction(hm, seq, L - 1)
    rel = abs(d.total - d.actual) / max(1e-6, abs(d.actual))
    check("trace rows sum to real logit diff", rel < args.tol, f"sum={d.total:+.3f} actual={d.actual:+.3f}")
    d.show(k=8)

    # 3. distance bands reconstruct one block of each Hyena kind
    for kind in ("se", "mr", "li"):
        b = hm.blocks_of(kind)[len(hm.blocks_of(kind)) // 2]
        r = mh.hyena_distance(hm, ids, b, L - 1)
        check(f"distance bands reconstruct block {b} ({kind})", r.rel_error < args.tol, f"rel_err={r.rel_error:.2e}")

    # 4. block 0 receptive field (expected 3 + 7 - 1 = 9 for evo2 configs)
    rf = motifs.receptive_field(hm)
    check("block 0 receptive field is small enough to enumerate", rf <= 10, f"measured {rf} letters")

    # 5. hooks leave no trace
    before = hm.logits(ids)
    means = mh.mean_writes(hm, ids, mh.mixers_of(hm, "li"))
    with mh.mean_ablate(hm, means):
        during = hm.logits(ids)
    after = hm.logits(ids)
    check("hooks removed after intervention", torch.equal(before, after))
    check("mean-ablating LI mixers changes the output", not torch.allclose(before, during))

    # 6. weight-only reach
    print()
    mh.show_reach(mh.reach_table(hm))

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
