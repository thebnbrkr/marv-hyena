"""The MARV/LARQL vindex, for Evo 2's MLPs -- plus the part LARQL cannot do
for DNA: label neurons with real biology.

Evo 2 MLP (vortex ParallelGatedMLP):  y = l3( act(l1 x) * (l2 x) )
  act = GELU at block 0, IDENTITY at every later block (evo2_style_activations).
So from block 1 on, neuron n's activation is the bilinear (l1 x)_n (l2 x)_n:
there is no "gate" in the Llama sense, l1 and l2 play symmetric roles, and the
sign of an activation carries meaning. A MARV "feature" (one neuron at one
layer) is: read rows l1[n], l2[n]; write column l3[:, n].

Logit-lens `describe` does not transfer: the output vocabulary is ~4
letters. Neurons are described instead by WHERE in a genome they fire
(`label_units` against GenBank annotations) -- LARQL's feature-label
programme with biology as the answer key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .arch import HyenaModel
from .probes import Track


@dataclass
class MlpVindex:
    """Per block: l1 (I, H), l2 (I, H), l3 (H, I) as float16 numpy."""

    layers: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    model_name: str = ""

    def save(self, path: str | Path) -> None:
        arrays = {f"b{b}_{k}": v for b, d in self.layers.items() for k, v in d.items()}
        np.savez_compressed(path, model_name=np.array(self.model_name), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MlpVindex":
        z = np.load(path)
        v = cls(model_name=str(z["model_name"]))
        for key in z.files:
            if key == "model_name":
                continue
            b, name = key[1:].split("_")
            v.layers.setdefault(int(b), {})[name] = z[key]
        return v


def extract_mlp_vindex(hm: HyenaModel, blocks=None, model_name: str = "") -> MlpVindex:
    blocks = range(hm.n_blocks) if blocks is None else blocks
    v = MlpVindex(model_name=model_name)
    for b in blocks:
        mlp = hm.block(b).mlp
        v.layers[b] = {n: getattr(mlp, n).weight.detach().to("cpu", torch.float16).numpy().copy()
                       for n in ("l1", "l2", "l3")}
    return v


def extract_mlp_vindex_from_checkpoint(state_dict: dict, n_blocks: int, blocks=None, model_name: str = "") -> MlpVindex:
    """CPU-only path: read l1/l2/l3 from a (mmap'd) checkpoint state dict."""
    blocks = range(n_blocks) if blocks is None else blocks
    v = MlpVindex(model_name=model_name)
    for b in blocks:
        v.layers[b] = {n: state_dict[f"blocks.{b}.mlp.{n}.weight"].to(torch.float16).numpy().copy()
                       for n in ("l1", "l2", "l3")}
    return v


@dataclass
class NeuronDelta:
    block: int
    neuron: int
    l1_cos: float
    l2_cos: float
    l3_cos: float
    norm_ratio: float  # ||l3 col|| tuned / base


def _row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def diff(base: MlpVindex, tuned: MlpVindex) -> list[NeuronDelta]:
    """Per-neuron weight-space change, e.g. evo2_7b_base vs evo2_7b_microviridae."""
    out = []
    for b in sorted(set(base.layers) & set(tuned.layers)):
        A, B = base.layers[b], tuned.layers[b]
        c1, c2 = _row_cos(A["l1"], B["l1"]), _row_cos(A["l2"], B["l2"])
        c3 = _row_cos(A["l3"].T, B["l3"].T)
        nr = np.linalg.norm(B["l3"].astype(np.float32), axis=0) / (np.linalg.norm(A["l3"].astype(np.float32), axis=0) + 1e-12)
        out.extend(NeuronDelta(b, n, float(c1[n]), float(c2[n]), float(c3[n]), float(nr[n])) for n in range(len(c1)))
    return out


def most_changed(deltas: list[NeuronDelta], k: int = 20) -> list[NeuronDelta]:
    return sorted(deltas, key=lambda d: min(d.l1_cos, d.l2_cos, d.l3_cos))[:k]


@torch.no_grad()
def neuron_acts(hm: HyenaModel, seq: str, block: int) -> torch.Tensor:
    """(L, I) neuron activations act(l1 x) * (l2 x) -- the exact input to l3."""
    store = {}

    def hook(_m, inputs):
        store["a"] = inputs[0][0].detach().float().cpu()

    h = hm.block(block).mlp.l3.register_forward_pre_hook(hook)
    try:
        hm.model(hm.ids(seq))
    finally:
        h.remove()
    return store["a"]


@dataclass
class UnitLabel:
    unit: int
    label: str
    enrichment: float  # P(label | top activations) / P(label)
    frac_top: float
    frac_bg: float


def annotation_classes(track: Track) -> dict[str, np.ndarray]:
    """Boolean masks per annotation class, aligned to letters of track.seq."""
    n = len(track.seq)
    classes = {f"feature:{t}": track.feature == t for t in set(track.feature.tolist()) if t}
    classes["intergenic"] = track.feature == ""
    for p in (0, 1, 2):
        classes[f"codon_pos{p + 1}"] = track.codon_phase == p
    starts = np.zeros(n, dtype=bool)
    cds = (track.codon_phase >= 0).astype(np.int8)
    on = np.flatnonzero(np.diff(np.concatenate([[0], cds])) == 1)
    for s in on:
        starts[max(0, s - 30):s + 30] = True
    classes["cds_start_+-30"] = starts
    return classes


def label_units(acts: torch.Tensor, track: Track, top_frac: float = 0.002, min_enrichment: float = 3.0,
                use_abs: bool = True) -> list[UnitLabel]:
    """For each unit (neuron or SAE feature), compare where its strongest
    activations fall against the genome's background composition; keep the
    best-enriched annotation class if it clears `min_enrichment`. use_abs:
    Evo 2's bilinear neurons can fire strongly with either sign."""
    a = acts.abs() if use_abs else acts
    a = a.numpy()
    L, U = a.shape
    k = max(1, int(L * top_frac))
    top_idx = np.argpartition(-a, k - 1, axis=0)[:k]  # (k, U)
    classes = annotation_classes(track)
    labels = []
    for u in range(U):
        idx = top_idx[:, u]
        best = None
        for name, mask in classes.items():
            bg = mask.mean()
            if bg == 0:
                continue
            ft = mask[idx].mean()
            enr = ft / bg
            if best is None or enr > best.enrichment:
                best = UnitLabel(u, name, float(enr), float(ft), float(bg))
        if best is not None and best.enrichment >= min_enrichment:
            labels.append(best)
    return labels
