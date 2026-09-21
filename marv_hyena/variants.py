"""Single-letter variants: score them the way the Evo 2 BRCA1 notebook does,
then ask WHY with patching.

Two different numbers, on purpose:
- delta_logp: mean log-likelihood of the alt window minus the ref window
  (the published zero-shot variant-effect score).
- downstream_metric: log-prob of the letters AFTER the variant, which are
  identical in ref and alt. Patching needs a metric evaluated on the same
  tokens in both runs, so this is the one `explain_variant` uses. It asks:
  "how much does the mutation disturb the model's reading of what follows?"
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .arch import HyenaModel
from .intervene import PatchSweep, kind_groups, patch_sweep, span_logprob, token_logprobs


@dataclass
class Variant:
    ref_window: str
    alt_window: str
    index: int  # position of the variant inside the windows
    ref: str
    alt: str


def make_variant(seq: str, pos: int, ref: str, alt: str, window: int = 8192) -> Variant:
    """Centre a `window`-letter window on seq[pos] and substitute `alt`."""
    if seq[pos].upper() != ref.upper():
        raise ValueError(f"reference mismatch at {pos}: genome has {seq[pos]!r}, variant says {ref!r}")
    lo = max(0, pos - window // 2)
    hi = min(len(seq), lo + window)
    w = seq[lo:hi].upper()
    i = pos - lo
    return Variant(w, w[:i] + alt.upper() + w[i + 1:], i, ref.upper(), alt.upper())


@torch.no_grad()
def delta_logp(hm: HyenaModel, v: Variant) -> float:
    """mean log p(alt window) - mean log p(ref window). Negative = the model
    finds the mutated sequence less natural (predicted more damaging)."""
    out = []
    for w in (v.ref_window, v.alt_window):
        ids = hm.ids(w)
        out.append(float(token_logprobs(hm.logits(ids), ids).mean()))
    return out[1] - out[0]


def downstream_metric(start: int, span: int):
    """metric(logits, ids) = sum log p of letters in [start, start+span)."""
    def metric(logits, ids):
        end = min(start + span, ids.shape[1])
        return span_logprob(logits, ids, start, end)
    return metric


def explain_variant(hm: HyenaModel, v: Variant, span: int = 200, components=None,
                    at: str = "all") -> PatchSweep:
    """Patch each component (and each operator-type group) from the ALT run into
    the REF run and report how much of the downstream disturbance it carries.

    at="all":     patch every position. Round 1 lesson: any component on the
                  path into an output bottleneck (block 30 in Evo 2 7B) then
                  "restores 100%" trivially -- total effect, not localisation.
    at="variant": patch only the mutated position. Asks which components'
                  writes AT THE MUTATION SITE carry the change onward.
    """
    if at not in ("all", "variant"):
        raise ValueError("at must be 'all' or 'variant'")
    metric = downstream_metric(v.index + 1, span)
    positions = None if at == "all" else [v.index]
    return patch_sweep(hm, source_seq=v.alt_window, target_seq=v.ref_window, metric=metric,
                       components=components, groups=kind_groups(hm), positions=positions)


@torch.no_grad()
def residual_patch_by_depth(hm: HyenaModel, v: Variant, span: int = 200) -> list[dict]:
    """Round 3: at what depth does the mutation's information LEAVE the mutated
    position? For each depth d (-1 = the embedding, i.e. the letter itself;
    0..n-1 = after block d), copy the ALT run's entire residual at the mutated
    position into the REF run and measure the downstream metric.

    At d=-1 this reproduces the ALT run exactly (the only difference between
    the runs is that letter), so fraction == 1. As d grows, later positions
    have already read the REF residual at the site in blocks <= d, so
    whatever information moved away from the site by then is NOT transferred,
    and the fraction falls. Where it falls is where the model moved the signal.
    """
    metric = downstream_metric(v.index + 1, span)
    site = v.index
    alt_ids, ref_ids = hm.ids(v.alt_window), hm.ids(v.ref_window)

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def grab(d):
        def hook(_m, _i, out):
            x = out[0] if isinstance(out, tuple) else out
            captured[d] = x[0, site].detach().clone()
        return hook

    mods = [(-1, hm.model.embedding_layer)] + [(d, hm.block(d)) for d in range(hm.n_blocks)]
    try:
        for d, m in mods:
            handles.append(m.register_forward_hook(grab(d)))
        alt_logits, _ = hm.model(alt_ids)
    finally:
        for h in handles:
            h.remove()
    alt_m = metric(alt_logits[0].float(), alt_ids)
    ref_logits, _ = hm.model(ref_ids)
    ref_m = metric(ref_logits[0].float(), ref_ids)

    rows = []
    for d, m in mods:
        def patch(_m, _i, out, d=d):
            x = out[0] if isinstance(out, tuple) else out
            x = x.clone()
            x[0, site] = captured[d].to(x.device, x.dtype)
            return (x, *out[1:]) if isinstance(out, tuple) else x

        h = m.register_forward_hook(patch)
        try:
            logits, _ = hm.model(ref_ids)
        finally:
            h.remove()
        pm = metric(logits[0].float(), ref_ids)
        rows.append({"depth": d, "kind": "embed" if d < 0 else hm.kind(d), "metric": pm,
                     "fraction": (pm - ref_m) / (alt_m - ref_m) if abs(alt_m - ref_m) > 1e-9 else float("nan")})
    return rows


@torch.no_grad()
def downstream_effect(hm: HyenaModel, v: Variant, span: int = 200) -> float:
    """metric(alt) - metric(ref) on the letters after the variant. Patching
    fractions are only meaningful when this is well away from zero (round 1's
    FUNC variant had 0.25, and its +-200% 'fractions' were noise)."""
    metric = downstream_metric(v.index + 1, span)
    out = []
    for w in (v.ref_window, v.alt_window):
        ids = hm.ids(w)
        out.append(metric(hm.logits(ids), ids))
    return out[1] - out[0]
