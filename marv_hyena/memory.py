"""Memory fix for long sequences on a 40 GB GPU.

Vortex's HyenaCascade.compute_filter builds the Hyena-LI filter as
    h[d, t] = sum_s residues[d, s] * exp(log_poles[d, s] * t)
by materialising the whole (channels, states, L) tensor first: 4096 x 16 x L
float32. At L ~ 51k (round 3's 50k-letter context test) that one temporary is
12.5 GB and runs a 40 GB A100 out of memory.

`chunked_compute_filter` computes exactly the same thing a block of channels at
a time (each element's sum over the 16 states is unchanged, so results are
bit-identical), keeping the temporary under ~1 GB. `install()` swaps it in for
Vortex's reference path; Vortex's own opt-in Triton kernel path is left alone.
"""
from __future__ import annotations

import torch

CHUNK = 256


def chunked_compute_filter(self, L, device, chunk: int = CHUNK):
    self.update_time(L, device)
    residues, log_poles = self.residues.float(), self.log_poles.float()  # (D, S), (D, S, 1)
    t = self.t.reshape(1, -1).float()  # (1, L)
    D = residues.shape[0]
    h = torch.empty((D, t.shape[-1]), device=residues.device, dtype=torch.float32)
    for c in range(0, D, chunk):
        h[c:c + chunk] = (residues[c:c + chunk, :, None] * (log_poles[c:c + chunk] * t).exp()).sum(1)
    return h[None], torch.float32, log_poles, residues


def install(cascade_cls=None) -> None:
    """Replace compute_filter on Vortex's HyenaCascade (or the given class)."""
    if cascade_cls is None:
        from vortex.model.model import HyenaCascade as cascade_cls
    if getattr(cascade_cls.compute_filter, "_marv_chunked", False):
        return
    original = cascade_cls.compute_filter

    def compute_filter(self, L, device):
        if getattr(getattr(self, "engine", None), "use_hcl_kernel", False):
            return original(self, L, device)  # Vortex's tiled kernel already avoids the big temporary
        return chunked_compute_filter(self, L, device)

    compute_filter._marv_chunked = True
    cascade_cls.compute_filter = compute_filter
