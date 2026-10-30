"""Smoke checks: every assumption marv-hyena makes about the loaded model,
tested against it. `run_operator_checks` needs only token ids, so it works on
any StripedHyena (Evo 2, StripedHyena-7B text); `run_smoke_checks` adds the
DNA-specific ones and is what scripts/smoke_test.py and the Evo 2 notebooks
run. Tolerances default to bf16-loose; the tiny-model tests hold the same code
to 1e-4 in float32."""
from __future__ import annotations

import torch

from . import motifs
from .arch import HyenaModel
from .distance import hyena_distance
from .gates import band_ablate, capture_gates
from .hooks import capture_outputs
from .intervene import mean_ablate, mean_writes, mixers_of
from .trace import capture_writes, decompose_prediction, decompose_writes, token_direction


class _Checks:
    def __init__(self, verbose: bool):
        self.results, self.verbose = [], verbose

    def __call__(self, name, ok, detail=""):
        self.results.append(bool(ok))
        if self.verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    def done(self) -> bool:
        ok = all(self.results)
        if self.verbose:
            print(f"\n{sum(self.results)}/{len(self.results)} checks passed"
                  + ("" if ok else "  -- DO NOT trust experiment numbers"))
        return ok


@torch.no_grad()
def _operator_checks(hm: HyenaModel, ids: torch.Tensor, check: _Checks, tol: float) -> None:
    L = ids.shape[1]
    w = capture_writes(hm, ids, [L // 2, L - 1])
    err = w.reconstruction_error()
    check("embed + sum(writes) == final residual", err < tol, f"rel_err={err:.2e}")

    top2 = w.logits[1].topk(2).indices
    direction, tw = token_direction(hm, int(top2[0]), int(top2[1]))
    d = decompose_writes(hm, w, direction, 1, "top-1 minus top-2 token", token_weights=tw)
    rel = abs(d.total - d.actual) / max(1e-6, abs(d.actual))
    check("trace rows sum to real logit diff (top-1 vs top-2 token)", rel < tol,
          f"sum={d.total:+.3f} actual={d.actual:+.3f}")

    for kind in ("se", "mr", "li"):
        blocks = hm.blocks_of(kind)
        if not blocks:
            continue
        b = blocks[len(blocks) // 2]
        r = hyena_distance(hm, ids, b, L - 1)
        check(f"distance bands reconstruct block {b} ({kind})", r.rel_error < tol, f"rel_err={r.rel_error:.2e}")
        g = capture_gates(hm, ids, [L - 1], blocks=[b])[b]
        check(f"gates reconstruct block {b} ({kind}): x2 * inner == output", g.rel_error < tol,
              f"rel_err={g.rel_error:.2e}")
        with band_ablate(hm, b, r.bands[1], mode="zero", positions=[L - 1]):
            with capture_outputs(hm, [(b, "mixer")], positions=[L - 1]) as cap:
                hm.model(ids)
        expected = r.actual - r.band_writes[1]
        err = float((cap[(b, "mixer")][0] - expected).norm() / r.actual.norm().clamp_min(1e-12))
        check(f"band_ablate {r.bands[1]} on block {b} removes exactly distance's band write", err < tol,
              f"rel_err={err:.2e}")


@torch.no_grad()
def _hook_checks(hm: HyenaModel, ids: torch.Tensor, check: _Checks) -> None:
    before = hm.logits(ids)
    means = mean_writes(hm, ids, mixers_of(hm, "li"))
    with mean_ablate(hm, means):
        during = hm.logits(ids)
    after = hm.logits(ids)
    check("hooks removed after intervention", torch.equal(before, after))
    check("mean-ablating LI mixers changes the output", not torch.allclose(before, during))


def run_operator_checks(hm: HyenaModel, seq_or_ids, tol: float = 5e-2, verbose: bool = True) -> bool:
    """Model-agnostic checks (any vocabulary): exact write reconstruction, exact
    trace, and per Hyena kind present: lag bands, gates, band ablation; then
    hook removal. Pass a string (tokenised with hm.ids) or a (1, L) id tensor."""
    ids = hm.ids(seq_or_ids) if isinstance(seq_or_ids, str) else seq_or_ids
    check = _Checks(verbose)
    _operator_checks(hm, ids, check, tol)
    _hook_checks(hm, ids, check)
    return check.done()


def run_smoke_checks(hm: HyenaModel, seq: str, tol: float = 5e-2, verbose: bool = True) -> bool:
    """Evo 2 (DNA): the operator checks plus the DNA-specific ones."""
    check = _Checks(verbose)
    ids = hm.ids(seq)
    L = len(seq)
    _operator_checks(hm, ids, check, tol)

    d = decompose_prediction(hm, seq, L - 1)
    rel = abs(d.total - d.actual) / max(1e-6, abs(d.actual))
    check("trace rows sum to real logit diff (letter vs other bases)", rel < tol,
          f"sum={d.total:+.3f} actual={d.actual:+.3f}")

    rf = motifs.receptive_field(hm)
    check("block 0 receptive field small enough to enumerate", rf <= 10, f"measured {rf} letters")

    _hook_checks(hm, ids, check)
    return check.done()
