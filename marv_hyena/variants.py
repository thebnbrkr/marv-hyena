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


def explain_variant(hm: HyenaModel, v: Variant, span: int = 200, components=None) -> PatchSweep:
    """Patch each component (and each operator-type group) from the ALT run into
    the REF run and report how much of the downstream disturbance it carries.
    A component restoring ~100% alone carries the whole effect of the mutation."""
    metric = downstream_metric(v.index + 1, span)
    return patch_sweep(hm, source_seq=v.alt_window, target_seq=v.ref_window, metric=metric,
                       components=components, groups=kind_groups(hm))
