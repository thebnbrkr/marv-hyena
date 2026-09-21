"""Hook plumbing shared by every analysis. All interventions go through
`replace_outputs`, all reads through `capture_outputs`, and both remove their
hooks on exit even if the forward pass raises."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch

from .arch import HyenaModel

Component = tuple[int, str]  # (block, "mixer" | "mlp")


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _rebuild(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


@contextmanager
def capture_outputs(
    hm: HyenaModel,
    components: Iterable[Component],
    positions: list[int] | slice | None = None,
    to_cpu: bool = True,
):
    """Record each component's residual write at `positions` (all if None).
    Yields a dict {(block, part): (P, hidden) float32}. Only the requested
    positions are kept, which is what makes long sequences affordable: a full
    32-block capture of a 100k-letter sequence would be ~50 GB."""
    store: dict[Component, torch.Tensor] = {}
    handles = []

    def make(comp):
        def hook(_module, _inputs, output):
            x = _first(output)[0]
            x = x if positions is None else x[positions]
            x = x.detach().float()
            store[comp] = x.cpu() if to_cpu else x

        return hook

    try:
        for comp in components:
            handles.append(hm.component(*comp).register_forward_hook(make(comp)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def replace_outputs(hm: HyenaModel, replacements: dict[Component, Callable[[torch.Tensor], torch.Tensor]]):
    """Replace each component's residual write during the `with` block.
    Each value maps the original output (1, L, hidden) to its replacement."""
    handles = []

    def make(fn):
        def hook(_module, _inputs, output):
            x = _first(output)
            new = fn(x).to(dtype=x.dtype, device=x.device)
            return _rebuild(output, new)

        return hook

    try:
        for comp, fn in replacements.items():
            handles.append(hm.component(*comp).register_forward_hook(make(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_module_io(module: torch.nn.Module, want_input: bool = True):
    """Record a module's first positional input (and output) on the next call."""
    store: dict[str, torch.Tensor] = {}

    def hook(_m, inputs, output):
        if want_input:
            store["input"] = inputs[0].detach()
        store["output"] = _first(output).detach()

    h = module.register_forward_hook(hook)
    try:
        yield store
    finally:
        h.remove()
