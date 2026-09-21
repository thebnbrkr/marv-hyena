"""Smoke checks: every assumption marv-hyena makes about the installed Vortex,
tested against the loaded model. Used by scripts/smoke_test.py and the Colab
notebook. Tolerances default to bf16-loose; the tiny-model tests hold the
same code to 1e-4 in float32."""
from __future__ import annotations

import torch

from . import motifs
from .arch import HyenaModel
from .distance import hyena_distance
from .intervene import mean_ablate, mean_writes, mixers_of
from .trace import capture_writes, decompose_prediction


def run_smoke_checks(hm: HyenaModel, seq: str, tol: float = 5e-2, verbose: bool = True) -> bool:
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    ids = hm.ids(seq)
    L = len(seq)

    w = capture_writes(hm, ids, [L // 2, L - 1])
    err = w.reconstruction_error()
    check("embed + sum(writes) == final residual", err < tol, f"rel_err={err:.2e}")

    d = decompose_prediction(hm, seq, L - 1)
    rel = abs(d.total - d.actual) / max(1e-6, abs(d.actual))
    check("trace rows sum to real logit diff", rel < tol, f"sum={d.total:+.3f} actual={d.actual:+.3f}")

    for kind in ("se", "mr", "li"):
        blocks = hm.blocks_of(kind)
        b = blocks[len(blocks) // 2]
        r = hyena_distance(hm, ids, b, L - 1)
        check(f"distance bands reconstruct block {b} ({kind})", r.rel_error < tol, f"rel_err={r.rel_error:.2e}")

    rf = motifs.receptive_field(hm)
    check("block 0 receptive field small enough to enumerate", rf <= 10, f"measured {rf} letters")

    before = hm.logits(ids)
    means = mean_writes(hm, ids, mixers_of(hm, "li"))
    with mean_ablate(hm, means):
        during = hm.logits(ids)
    after = hm.logits(ids)
    check("hooks removed after intervention", torch.equal(before, after))
    check("mean-ablating LI mixers changes the output", not torch.allclose(before, during))

    ok = all(results)
    if verbose:
        print(f"\n{sum(results)}/{len(results)} checks passed" + ("" if ok else "  -- DO NOT trust experiment numbers"))
    return ok
