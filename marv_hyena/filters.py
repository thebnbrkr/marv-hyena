"""Read Hyena filters straight from the weights -- the MARV move, applied to
convolutions instead of FFN neurons. No forward pass.

Every Hyena block computes, per channel c and position i,

    y_c(i) = x2_c(i) * [ sum_{lag>=0} k_c(lag) * u_c(i - lag)  +  D_c * u_c(i) ]
    with u = x1 * v   (x2, x1, v are the three gated projections)

where k_c is the block's filter written LAG-INDEXED (k[lag] multiplies the
letter `lag` positions back). How k is stored differs per operator, and
getting this wrong silently flips a filter end-for-end:

  SE  `filter.h`  (groups, 1, len<128)  applied by F.conv1d, which is a
      CROSS-correlation with left padding: tap t multiplies lag (len-1-t).
      -> k = h.flip(-1).  No D term.
  MR  `filter.h`  (groups, 1, 128)      applied by an FFT convolution:
      tap t multiplies lag t.  -> k = h.  D is a gated bias (lag 0).
  (Vortex picks the branch by filter LENGTH -- conv1d below 128 taps, FFT at
  128 and above -- not by operator name, so `lag_kernel` does the same.)
  LI  implicit:   k_c(t) = sum_s residues[c,s] * exp(log_poles[c,s] * t)
      (vortex HyenaCascade.compute_filter), true convolution. D at lag 0.

Groups: SE/MR filters are shared by blocks of hidden_size/groups channels
(`repeat_interleave`), exactly as vortex expands them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .arch import HyenaModel


FFT_FIR_THRESHOLD = 128  # vortex engine.parallel_fir: `elif fir_length >= 128:` -> fftconv


def _fir_to_lag(h: torch.Tensor) -> torch.Tensor:
    """(C, len) stored FIR taps -> lag-indexed kernel, per vortex's branch rule."""
    return h if h.shape[-1] >= FFT_FIR_THRESHOLD else h.flip(-1)


def lag_kernel(filt, kind: str, L: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(k, D): k is (hidden, n_lags) float32 lag-indexed, n_lags <= L."""
    if kind == "li":
        h, *_ = filt.compute_filter(L, filt.log_poles.device)  # (1, D, L)
        k = h[0].float()
    elif kind in ("se", "mr"):
        h = filt.h.float()
        if h.shape[0] != hidden_size:
            h = h.repeat_interleave(hidden_size // h.shape[0], 0)
        h = h[:, 0]
        k = _fir_to_lag(h)[:, :L]
    else:
        raise ValueError(f"no lag kernel for kind {kind!r}")
    D = filt.D.float() if getattr(filt, "D", None) is not None else None
    return k, D


# ---------------------------------------------------------------- reach
@dataclass
class Reach:
    block: int
    kind: str
    reach50: float  # median over channels of the lag holding 50% of |k| mass
    reach90: float
    reach99: float
    max_reach99: float  # the longest-reaching channel
    per_channel99: np.ndarray


def _mass_quantile_lags(lags: np.ndarray, absk: np.ndarray, qs=(0.5, 0.9, 0.99)) -> list[np.ndarray]:
    """absk: (C, n) |k| sampled at `lags` (possibly non-uniform). Trapezoid mass."""
    widths = np.gradient(lags.astype(np.float64))
    mass = absk * widths[None, :]
    cum = np.cumsum(mass, axis=1)
    cum /= np.maximum(cum[:, -1:], 1e-30)
    return [lags[np.argmax(cum >= q, axis=1)] for q in qs]


def li_kernel_on_grid(log_poles: torch.Tensor, residues: torch.Tensor, t: np.ndarray) -> np.ndarray:
    """Evaluate the implicit LI filter at arbitrary lags without materialising
    all L of them (so reach out to 1M letters is cheap). -> (C, len(t))."""
    lp = log_poles.detach().float().cpu().reshape(log_poles.shape[0], -1)  # (C, S)
    r = residues.detach().float().cpu()  # (C, S)
    tt = torch.as_tensor(t, dtype=torch.float32)
    out = (r[:, :, None] * torch.exp(lp[:, :, None] * tt[None, None, :])).sum(1)
    return out.numpy()


def _reach_from_parts(block: int, kind: str, *, h=None, log_poles=None, residues=None,
                      hidden_size: int, max_len: int) -> Reach:
    if kind == "li":
        grid = np.unique(np.concatenate([np.arange(0, 512), np.geomspace(512, max_len, 512).astype(np.int64)]))
        absk = np.abs(li_kernel_on_grid(log_poles, residues, grid))
        lags = grid
        if (log_poles.detach().float() > 0).any():
            print(f"warning: block {block} has positive log_poles -> growing filter; reach is not meaningful")
    else:
        hh = h.detach().float().cpu()
        if hh.shape[0] != hidden_size:
            hh = hh.repeat_interleave(hidden_size // hh.shape[0], 0)
        absk = _fir_to_lag(hh[:, 0]).abs().numpy()
        lags = np.arange(absk.shape[1])
    r50, r90, r99 = _mass_quantile_lags(lags, absk)
    return Reach(block, kind, float(np.median(r50)), float(np.median(r90)), float(np.median(r99)),
                 float(r99.max()), r99)


def reach_table(hm: HyenaModel, max_len: int = 1_000_000) -> list[Reach]:
    """How far back each Hyena block's filters actually look, from weights alone."""
    rows = []
    for b in range(hm.n_blocks):
        kind = hm.kind(b)
        if kind == "attn":
            continue
        f = hm.hyena_filter(b)
        rows.append(_reach_from_parts(b, kind, h=getattr(f, "h", None), log_poles=getattr(f, "log_poles", None),
                                      residues=getattr(f, "residues", None), hidden_size=hm.hidden_size,
                                      max_len=max_len))
    return rows


def reach_table_from_checkpoint(state_dict: dict, config: dict, max_len: int = 1_000_000) -> list[Reach]:
    """Same table straight from a checkpoint's state dict + the evo2 yml config.
    Runs on a laptop CPU: no model instantiation, no CUDA.

        sd = torch.load("evo2_7b.pt", map_location="cpu", mmap=True, weights_only=False)
        cfg = yaml.safe_load(open("evo2/configs/evo2-7b-1m.yml"))
    """
    kinds = {}
    for key, kind in (("hcs_layer_idxs", "se"), ("hcm_layer_idxs", "mr"), ("hcl_layer_idxs", "li")):
        for i in config.get(key, []):
            kinds[i] = kind
    rows = []
    for b in sorted(kinds):
        p = f"blocks.{b}.filter."
        rows.append(_reach_from_parts(
            b, kinds[b], h=state_dict.get(p + "h"), log_poles=state_dict.get(p + "log_poles"),
            residues=state_dict.get(p + "residues"), hidden_size=config["hidden_size"], max_len=max_len))
    return rows


def show_reach(rows: list[Reach]) -> None:
    print(f"{'block':>5} {'kind':>4} {'reach50':>9} {'reach90':>9} {'reach99':>9} {'max99':>9}   (letters)")
    for r in rows:
        print(f"{r.block:>5} {r.kind:>4} {r.reach50:>9.0f} {r.reach90:>9.0f} {r.reach99:>9.0f} {r.max_reach99:>9.0f}")
