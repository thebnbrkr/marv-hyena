"""DISTANCE: how much of a Hyena block's write at position i came from the
letter right there, from 1-8 letters back, ..., from >16k letters back.

Given the actual gates of this sequence, a Hyena block is linear in u = x1*v:

    y(i) = x2(i) * [ sum_lag k(lag) u(i-lag) + D u(i) ]

so splitting the lag axis into bands splits y(i) -- and, through the linear
out_filter_dense, the block's residual write -- into per-band pieces that
add up EXACTLY. Each result carries the relative reconstruction error
against the real module output captured in the same forward pass; if it is
not small, the lag conventions in filters.py disagree with the installed
Vortex and nothing here should be trusted.

Attention blocks are not decomposed here yet (flash attention does not
expose its weights); use context truncation or patching for them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .arch import HyenaModel
from .filters import lag_kernel
from .hooks import capture_module_io

DEFAULT_BANDS: tuple[tuple[int, int | None], ...] = (
    (0, 0), (1, 8), (9, 127), (128, 1023), (1024, 16383), (16384, None),
)


def band_label(band: tuple[int, int | None]) -> str:
    lo, hi = band
    return f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")


@dataclass
class DistanceResult:
    block: int
    kind: str
    position: int
    bands: tuple[tuple[int, int | None], ...]
    band_writes: torch.Tensor  # (n_bands, H) residual write from each lag band (D term folded into lag 0)
    bias: torch.Tensor  # (H,) out_filter_dense bias (position-independent)
    actual: torch.Tensor  # (H,) the real mixer write at this position
    rel_error: float

    def project(self, direction: torch.Tensor) -> list[float]:
        return [float(w @ direction) for w in self.band_writes]

    def norms(self) -> list[float]:
        return [float(w.norm()) for w in self.band_writes]


def _split_gates(hm: HyenaModel, filt, z: torch.Tensor):
    """Reproduce HyenaCascade.parallel_forward up to the inner filter:
    short featurizer FIR -> interleave -> split into (x2, x1, v), each (H, L)."""
    H = hm.hidden_size
    zf = z[0].float().T.unsqueeze(0)  # (1, 3H, L)
    L = zf.shape[-1]
    w = filt.short_filter_weight.float()
    ls = w.shape[-1]
    zp = F.conv1d(zf, w, bias=None, padding=ls - 1, groups=zf.shape[1])[..., :L]
    if getattr(filt, "short_filter_bias", None) is not None:
        zp = zp + filt.short_filter_bias.float()[None, :, None]
    if hm.config.get("interleave", False):
        zp = torch.cat([zp[:, 0::3], zp[:, 1::3], zp[:, 2::3]], dim=1)
    zp = zp[0]
    if filt.column_split_hyena:
        nh, hd = filt.num_attention_heads, filt.hidden_size_per_attention_head
        r = zp.reshape(nh, 3 * hd, L)
        x2, x1, v = (r[:, :hd].reshape(H, L), r[:, hd:2 * hd].reshape(H, L), r[:, 2 * hd:].reshape(H, L))
    else:
        x2, x1, v = zp.split([H, H, H], dim=0)
    if filt.hyena_flip_x1x2:
        x1, x2 = x2, x1
    return x2, x1, v


@torch.no_grad()
def hyena_distance(hm: HyenaModel, ids: torch.Tensor, block: int, position: int,
                   bands=DEFAULT_BANDS) -> DistanceResult:
    kind = hm.kind(block)
    if kind == "attn":
        raise NotImplementedError("distance decomposition of attention blocks is not implemented")
    blk, filt = hm.block(block), hm.hyena_filter(block)
    with capture_module_io(filt) as fio, capture_module_io(blk.out_filter_dense, want_input=False) as oio:
        hm.model(ids[:, : position + 1])  # causal: nothing after `position` matters
    z = fio["input"]
    actual = oio["output"][0, position].float()

    x2, x1, v = _split_gates(hm, filt, z)
    u = x1 * v  # (H, L)
    L = u.shape[1]
    k, D = lag_kernel(filt, kind, L, hm.hidden_size)
    n = min(k.shape[1], position + 1)
    u_back = u[:, : position + 1].flip(-1)[:, :n]  # u_back[:, lag] = u(position - lag)
    contrib = k[:, :n].to(u.device) * u_back  # (H, n) per-lag inner sum terms

    g = x2[:, position]
    W = blk.out_filter_dense.weight.float()
    per_band = []
    for lo, hi in bands:
        hi_ = n - 1 if hi is None else min(hi, n - 1)
        inner = contrib[:, lo:hi_ + 1].sum(1) if lo <= hi_ else torch.zeros_like(g)
        if lo == 0 and D is not None:
            inner = inner + D.to(u.device) * u[:, position]
        per_band.append(W @ (g * inner))
    band_writes = torch.stack(per_band)
    b = blk.out_filter_dense.bias
    bias = b.float() if b is not None else torch.zeros_like(actual)
    recon = band_writes.sum(0) + bias
    rel = float((recon - actual).norm() / actual.norm().clamp_min(1e-12))
    return DistanceResult(block, kind, position, tuple(bands), band_writes.cpu(), bias.cpu(), actual.cpu(), rel)


def distance_profile(hm: HyenaModel, seq_or_ids, position: int, blocks=None, bands=DEFAULT_BANDS,
                     direction: torch.Tensor | None = None) -> list[DistanceResult]:
    """Run hyena_distance for every Hyena block (or the given ones)."""
    ids = hm.ids(seq_or_ids) if isinstance(seq_or_ids, str) else seq_or_ids
    blocks = [b for b in range(hm.n_blocks) if hm.is_hyena(b)] if blocks is None else blocks
    return [hyena_distance(hm, ids, b, position, bands) for b in blocks]


def show_profile(results: list[DistanceResult], direction: torch.Tensor | None = None) -> None:
    """Rows = blocks, columns = lag bands. Values are the write's projection
    on `direction` if given, else each band-write's norm as a share of the total."""
    if not results:
        return
    bands = results[0].bands
    head = " ".join(f"{band_label(b):>11}" for b in bands)
    print(f"{'block':>5} {'kind':>4} {head}   recon_err")
    for r in results:
        vals = r.project(direction) if direction is not None else r.norms()
        if direction is None:
            tot = sum(vals) or 1.0
            cells = " ".join(f"{v / tot:>10.1%} " for v in vals)
        else:
            cells = " ".join(f"{v:>+11.3f}" for v in vals)
        print(f"{r.block:>5} {r.kind:>4} {cells}   {r.rel_error:.1e}")
