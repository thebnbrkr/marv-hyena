"""Causal tools: switch components off (ablation) or swap them between two
runs (activation patching), and measure what changed.

Ablation replaces a component's residual write with its MEAN write (over
positions of a reference run) rather than zero: zeroing whole operator
families pushes the residual far off-distribution and the model just breaks,
which tells you nothing. Mean-ablation removes the position-specific
information the component carries while keeping the residual's scale.

Patching runs sequence A, records a component's write, then runs sequence B
with that one write swapped in. The fraction of the A-vs-B difference it
restores is that component's TOTAL effect (direct + indirect) -- the
complement to trace.py's direct-only numbers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch

from .arch import HyenaModel
from .hooks import Component, capture_outputs, replace_outputs


# ---------------------------------------------------------------- ablation
@torch.no_grad()
def mean_writes(hm: HyenaModel, ids: torch.Tensor, components: list[Component],
                skip_first: int = 0) -> dict[Component, torch.Tensor]:
    """Mean residual write of each component over positions >= skip_first.
    The mean is taken inside the hook, so memory stays O(hidden) per component
    whatever the sequence length."""
    store: dict[Component, torch.Tensor] = {}
    handles = []

    def make(c):
        def hook(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            store[c] = x[0, skip_first:].float().mean(0).detach()
        return hook

    try:
        for c in components:
            handles.append(hm.component(*c).register_forward_hook(make(c)))
        hm.model(ids)
    finally:
        for h in handles:
            h.remove()
    return store


@contextmanager
def mean_ablate(hm: HyenaModel, means: dict[Component, torch.Tensor],
                positions: slice | list[int] | None = None):
    """Replace each component's write with its mean (at `positions`, default all)."""
    def make(mu):
        def fn(x):
            out = x.clone()
            if positions is None:
                out[:] = mu.to(x.device, x.dtype)
            else:
                out[:, positions] = mu.to(x.device, x.dtype)
            return out
        return fn

    with replace_outputs(hm, {c: make(mu) for c, mu in means.items()}):
        yield


@contextmanager
def zero_ablate(hm: HyenaModel, components: list[Component]):
    with replace_outputs(hm, {c: torch.zeros_like for c in components}):
        yield


def mixers_of(hm: HyenaModel, kind: str) -> list[Component]:
    return [(b, "mixer") for b in hm.blocks_of(kind)]


# ---------------------------------------------------------------- metrics
def token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(L-1,) log p(ids[j] | ids[<j]) for j = 1..L-1, from (L, vocab) logits."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, 1:].to(lp.device)
    return lp[:-1].gather(-1, tgt[:, None])[:, 0]


def span_logprob(logits: torch.Tensor, ids: torch.Tensor, start: int, end: int) -> float:
    """Sum of log p for letters in [start, end) -- each letter scored from the one before it."""
    lp = token_logprobs(logits, ids)
    return float(lp[start - 1:end - 1].sum())


def span_accuracy(logits: torch.Tensor, ids: torch.Tensor, start: int, end: int) -> float:
    pred = logits[start - 1:end - 1].argmax(-1).cpu()
    return float((pred == ids[0, start:end].cpu()).float().mean())


# ---------------------------------------------------------------- patching
@dataclass
class PatchResult:
    component: Component | str
    kind: str
    metric: float  # metric of the patched run
    fraction: float  # (patched - base) / (source - base): share of the difference restored


@dataclass
class PatchSweep:
    base_metric: float  # target run, unpatched
    source_metric: float  # source run
    results: list[PatchResult]

    def top(self, k: int = 10) -> list[PatchResult]:
        return sorted(self.results, key=lambda r: -abs(r.fraction))[:k]

    def show(self, k: int = 12) -> None:
        print(f"target run={self.base_metric:+.3f}  source run={self.source_metric:+.3f}  "
              f"difference={self.source_metric - self.base_metric:+.3f}")
        for r in self.top(k):
            name = r.component if isinstance(r.component, str) else f"L{r.component[0]:<2} {r.kind:<4} {r.component[1]}"
            print(f"  {name:<22} restores {r.fraction:+7.1%}   (metric {r.metric:+.3f})")


@torch.no_grad()
def patch_sweep(hm: HyenaModel, source_seq: str, target_seq: str, metric: Callable[[torch.Tensor, torch.Tensor], float],
                components: list[Component] | None = None, groups: dict[str, list[Component]] | None = None,
                positions: slice | None = None, cache_device: str = "cpu") -> PatchSweep:
    """Patch each component (and each named group) from the source run into the
    target run. Sequences must have the same length. `metric(logits, target_ids)`.

    Memory: caches every requested component's full-length write on
    `cache_device` -- 64 components x L x 4096 x 4 bytes, ~8 GB at L=8k.
    Restrict `components` or the sequence length for longer inputs.
    """
    if len(source_seq) != len(target_seq):
        raise ValueError("source and target must have the same length")
    components = hm.all_components() if components is None else components
    groups = groups or {}
    need = sorted(set(components) | {c for g in groups.values() for c in g})
    src_ids, tgt_ids = hm.ids(source_seq), hm.ids(target_seq)

    with capture_outputs(hm, need, to_cpu=(cache_device == "cpu")) as cache:
        src_logits, _ = hm.model(src_ids)
    src_m = metric(src_logits[0].float(), src_ids)
    base_logits, _ = hm.model(tgt_ids)
    base_m = metric(base_logits[0].float(), tgt_ids)
    denom = src_m - base_m if abs(src_m - base_m) > 1e-9 else float("nan")

    def run(comps):
        def make(c):
            src = cache[c]
            def fn(x):
                out = x.clone()
                sl = slice(None) if positions is None else positions
                out[0, sl] = src[sl].to(x.device, x.dtype)
                return out
            return fn
        with replace_outputs(hm, {c: make(c) for c in comps}):
            logits, _ = hm.model(tgt_ids)
        return metric(logits[0].float(), tgt_ids)

    results = []
    for c in components:
        m = run([c])
        results.append(PatchResult(c, hm.kind(c[0]), m, (m - base_m) / denom))
    for name, comps in groups.items():
        m = run(comps)
        results.append(PatchResult(name, "group", m, (m - base_m) / denom))
    return PatchSweep(base_m, src_m, results)


def kind_groups(hm: HyenaModel) -> dict[str, list[Component]]:
    """Named groups for patch_sweep: all mixers of each operator type, and all MLPs."""
    g = {f"all {k} mixers": mixers_of(hm, k) for k in ("se", "mr", "li", "attn")}
    g["all mlps"] = [(b, "mlp") for b in range(hm.n_blocks)]
    return g
