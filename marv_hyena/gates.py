"""GATES: open a Hyena operator and read or edit what is inside it.

Every Hyena block computes, per channel c and position i (see filters.py),

    y_c(i) = x2_c(i) * inner_c(i),   inner_c(i) = sum_lag k_c(lag) u_c(i - lag) + D_c u_c(i)
    u = x1 * v

x1 and x2 are the input-dependent GATES; k is fixed after training. The
mixer write is out_filter_dense(y), so y is the last point before the
operator's channels are mixed together.

Reading: `capture_gates` records x2, x1, v, inner and the real filter output
at the requested positions, in one forward pass.

Editing (context managers, hooks removed on exit):

- `band_ablate`   remove one lag band from the convolution (mean or zero).
                  The causal counterpart of distance.py's direct split.
- `freeze_gate`   replace a gate with its mean over positions, so that
                  channel's operator stops depending on the input there.
- `scale_channels` multiply operator channels by a factor. Because the long
                  convolution is per channel, scaling x1, x2 or y of a
                  channel by s all scale y by exactly s; this is the one
                  function for all three.

Each recomputes y from the captured input and compares it with the real
output (`rel_error`, invariant 1 in AGENTS.md). Edits raise rather than run
when that check fails, because then the lag conventions disagree with the
installed Vortex and any number would be wrong. Edits change only the
recomputed piece, added to the REAL output, so a small reconstruction error
cannot leak into the untouched part.

Memory: each hooked block needs a transient O(hidden x L) float32 buffer,
like the forward pass itself; the convolution runs `chunk` channels at a
time. Stored captures are (positions x hidden) only (invariant 3).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .arch import HyenaModel
from .distance import _split_gates
from .filters import lag_kernel
from .hooks import _first, _rebuild

CHUNK = 512  # channels per FFT convolution


def _causal_conv(k: torch.Tensor, u: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """out[c, t] = sum_{lag < k.shape[1]} k[c, lag] * u[c, t - lag], for all t.
    k (H, n_lags) lag-indexed, u (H, L); float32, FFT, `chunk` channels at a time."""
    H, L = u.shape
    n = L + k.shape[1]
    out = torch.empty_like(u)
    k = k.to(u.device)
    for c in range(0, H, chunk):
        kf = torch.fft.rfft(k[c:c + chunk], n=n)
        uf = torch.fft.rfft(u[c:c + chunk], n=n)
        out[c:c + chunk] = torch.fft.irfft(kf * uf, n=n)[:, :L]
    return out


@dataclass
class _Parts:
    x2: torch.Tensor  # (H, L)
    x1: torch.Tensor
    v: torch.Tensor
    k: torch.Tensor  # (H, n_lags) lag-indexed
    D: torch.Tensor | None  # (H,)

    @property
    def u(self) -> torch.Tensor:
        return self.x1 * self.v

    def inner(self, u: torch.Tensor | None = None, k: torch.Tensor | None = None,
              with_D: bool = True) -> torch.Tensor:
        u = self.u if u is None else u
        out = _causal_conv(self.k if k is None else k, u)
        if with_D and self.D is not None:
            out = out + self.D.to(u.device)[:, None] * u
        return out


def _parts(hm: HyenaModel, block: int, filt, z: torch.Tensor) -> _Parts:
    """z: the filter module's input, (1, L, 3H)."""
    x2, x1, v = _split_gates(hm, filt, z)
    k, D = lag_kernel(filt, hm.kind(block), z.shape[1], hm.hidden_size)
    return _Parts(x2, x1, v, k, D)


def _rel(recon: torch.Tensor, actual: torch.Tensor) -> float:
    return float(((recon - actual).norm() / actual.norm().clamp_min(1e-12)).detach())


def _check_block(hm: HyenaModel, block: int) -> None:
    if not hm.is_hyena(block):
        raise ValueError(f"block {block} is attention; gates and lag bands exist only in Hyena blocks")


def _index(positions) -> torch.Tensor | slice:
    if positions is None:
        return slice(None)
    if isinstance(positions, slice):
        return positions
    return torch.as_tensor(positions, dtype=torch.long)


# ---------------------------------------------------------------- reading
@dataclass
class GateCapture:
    block: int
    kind: str
    positions: list[int]
    x2: torch.Tensor  # (P, H) output gate
    x1: torch.Tensor  # (P, H) input gate
    v: torch.Tensor  # (P, H)
    inner: torch.Tensor  # (P, H) convolution + D term, before the output gate
    output: torch.Tensor  # (P, H) the real filter output y = x2 * inner
    rel_error: float  # || x2 * inner - output || / || output ||

    def closed_fraction(self, gate: str = "x2", tol: float = 1e-2) -> torch.Tensor:
        """(H,) share of captured positions where |gate| < tol * its channel's mean |gate|."""
        g = getattr(self, gate).abs()
        return (g < tol * g.mean(0, keepdim=True).clamp_min(1e-12)).float().mean(0)


@torch.no_grad()
def capture_gates(hm: HyenaModel, ids: torch.Tensor, positions: list[int], blocks=None,
                  to_cpu: bool = True) -> dict[int, GateCapture]:
    """One forward pass; for every Hyena block (or `blocks`), the gates, the
    convolution and the real output at `positions`. Check `rel_error` before
    using any of it."""
    blocks = [b for b in range(hm.n_blocks) if hm.is_hyena(b)] if blocks is None else list(blocks)
    for b in blocks:
        _check_block(hm, b)
    pos = list(positions)
    idx = torch.as_tensor(pos, dtype=torch.long)
    store: dict[int, GateCapture] = {}
    handles = []

    def make(b, filt):
        def hook(_m, inputs, output):
            p = _parts(hm, b, filt, inputs[0])
            inner = p.inner()[:, idx]
            x2 = p.x2[:, idx]
            y = _first(output)[0].float().T[:, idx]
            rel = _rel(x2 * inner, y)
            move = (lambda t: t.T.contiguous().cpu()) if to_cpu else (lambda t: t.T.contiguous())
            store[b] = GateCapture(b, hm.kind(b), pos, move(x2), move(p.x1[:, idx]), move(p.v[:, idx]),
                                   move(inner), move(y), rel)
        return hook

    try:
        for b in blocks:
            filt = hm.hyena_filter(b)
            handles.append(filt.register_forward_hook(make(b, filt)))
        hm.model(ids[:, : max(pos) + 1])  # causal: nothing after the last position matters
    finally:
        for h in handles:
            h.remove()
    return store


# ---------------------------------------------------------------- editing
@contextmanager
def _edit_filter(hm: HyenaModel, block: int, edit, tol: float):
    """Run `edit(parts, y_actual) -> y_new` ((H, L) float32) inside the filter's
    forward hook, after checking that the recomputed y matches the real one."""
    _check_block(hm, block)
    filt = hm.hyena_filter(block)
    report = {"rel_error": None}

    def hook(_m, inputs, output):
        y_out = _first(output)
        p = _parts(hm, block, filt, inputs[0])
        y = y_out[0].float().T
        rel = _rel(p.x2 * p.inner(), y)
        report["rel_error"] = rel
        if rel > tol:
            raise ValueError(
                f"block {block}: recomputed operator output differs from the real one by {rel:.2e} "
                f"(> tol {tol:.0e}); lag conventions disagree with the installed Vortex, edit refused")
        new = edit(p, y).T[None].to(dtype=y_out.dtype, device=y_out.device)
        return _rebuild(output, new)

    h = filt.register_forward_hook(hook)
    try:
        yield report
    finally:
        h.remove()


def _replacement(term: torch.Tensor, mode: str, skip_first: int) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(term)
    if mode == "mean":
        return term[:, skip_first:].mean(1, keepdim=True).expand_as(term)
    raise ValueError(f"mode must be 'mean' or 'zero', got {mode!r}")


@contextmanager
def band_ablate(hm: HyenaModel, block: int, band: tuple[int, int | None], mode: str = "mean",
                positions=None, skip_first: int = 0, tol: float = 5e-2):
    """Remove lags lo..hi (hi=None: to the end) from one Hyena block's
    convolution, at `positions` (default all). The D term counts as lag 0.

    mode='mean' (default, invariant 4) replaces the band's term with its mean
    over positions >= skip_first of this run; 'zero' drops it. Yields a report
    dict whose 'rel_error' is filled in by the forward pass."""
    lo, hi = band

    def edit(p: _Parts, y: torch.Tensor) -> torch.Tensor:
        n = p.k.shape[1]
        mask = torch.zeros(n, device=p.k.device)
        mask[lo:(n if hi is None else min(hi, n - 1) + 1)] = 1.0
        term = p.inner(k=p.k * mask, with_D=(lo == 0))
        delta = p.x2 * (_replacement(term, mode, skip_first) - term)
        out = y.clone()
        sl = _index(positions)
        out[:, sl] = y[:, sl] + delta[:, sl]
        return out

    with _edit_filter(hm, block, edit, tol) as report:
        yield report


@contextmanager
def freeze_gate(hm: HyenaModel, block: int, gate: str = "x2", channels=None, skip_first: int = 0,
                tol: float = 5e-2):
    """Replace gate x2 (output gate) or x1 (input gate) of `channels` (default
    all) with its mean over positions >= skip_first. The block then applies a
    fixed convolution on those channels: what changes measures how much the
    operator's DATA-DEPENDENT control matters there."""
    if gate not in ("x1", "x2"):
        raise ValueError(f"gate must be 'x1' or 'x2', got {gate!r}")

    def edit(p: _Parts, y: torch.Tensor) -> torch.Tensor:
        inner = p.inner()
        if gate == "x2":
            delta = (_replacement(p.x2, "mean", skip_first) - p.x2) * inner
        else:
            u_frozen = _replacement(p.x1, "mean", skip_first) * p.v
            delta = p.x2 * (p.inner(u=u_frozen) - inner)
        out = y.clone()
        ch = slice(None) if channels is None else torch.as_tensor(channels, dtype=torch.long)
        out[ch] = y[ch] + delta[ch]
        return out

    with _edit_filter(hm, block, edit, tol) as report:
        yield report


@contextmanager
def scale_channels(hm: HyenaModel, block: int, channels, factor: float, positions=None):
    """Multiply operator channels' output y by `factor` (0 switches them off),
    at `positions` (default all). Exact, so no reconstruction check is needed."""
    _check_block(hm, block)
    filt = hm.hyena_filter(block)
    ch = torch.as_tensor(channels, dtype=torch.long)

    def hook(_m, _inputs, output):
        y = _first(output)
        out = y.clone()
        sl = _index(positions)
        if isinstance(sl, slice):
            out[0, sl, ch] = y[0, sl, ch] * factor
        else:
            out[0, sl[:, None], ch[None, :]] = y[0, sl[:, None], ch[None, :]] * factor
        return _rebuild(output, out)

    h = filt.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
