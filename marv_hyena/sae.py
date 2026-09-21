"""The public Evo 2 SAE (Goodfire, layer 26) as a first-class object:
encode, decompose a feature by operator type, and edit features in a live
forward pass.

The SAE class and loader reproduce the ones in evo2's
notebooks/sparse_autoencoder/sparse_autoencoder.ipynb (BatchTopK, tied
weights, k=64, expansion 8 -> 32768 features, reads the OUTPUT of
blocks[26] of evo2_7b_262k). Note BatchTopK picks the top k*L activations
across the whole sequence, not k per position.

Because the encoder pre-activation is linear in the residual,

    pre_f(i) = (embed + sum_{b<=26} mixer_b + mlp_b)(i) . W[:, f] + b_enc[f]

is an exact sum over blocks -- so "which operator types build feature f"
has an exact direct-effect answer (`decompose_feature`).
"""
from __future__ import annotations

from contextlib import contextmanager
from math import prod

import torch

from .arch import HyenaModel
from .trace import Decomposition, Row, capture_writes

SAE_REPO = "Goodfire/Evo-2-Layer-26-Mixed"
SAE_FILE = "sae-layer26-mixed-expansion_8-k_64.pt"
SAE_LAYER = 26
SAE_MODEL = "evo2_7b_262k"


class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.d_in, self.d_hidden, self.k = d_in, d_hidden, k
        self.W = torch.nn.Parameter(torch.zeros(d_in, d_hidden))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x):
        f = torch.relu(self.encoder_pre(x))
        *shape, _ = f.shape
        top = torch.topk(f.flatten(), self.k * prod(shape))
        return torch.zeros_like(f.flatten()).scatter(-1, top.indices, top.values).reshape(f.shape)

    def decode(self, f):
        return f @ self.W.T + self.b_dec


def load_goodfire_sae(device="cuda", dtype=torch.bfloat16, path: str | None = None) -> BatchTopKTiedSAE:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE, repo_type="model")
    sd = torch.load(path, weights_only=True, map_location="cpu")
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    d_in, d_hidden = sd["W"].shape
    sae = BatchTopKTiedSAE(d_in, d_hidden, k=64)
    sae.load_state_dict(sd)
    return sae.to(device=device, dtype=dtype).eval()


@contextmanager
def _block_output_hook(hm: HyenaModel, layer: int, fn):
    def hook(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        new = fn(x)
        return (new, *output[1:]) if isinstance(output, tuple) else new

    h = hm.block(layer).register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def feature_acts(hm: HyenaModel, sae: BatchTopKTiedSAE, seq: str, layer: int = SAE_LAYER) -> torch.Tensor:
    """(L, n_features) SAE activations over the sequence."""
    store = {}

    def grab(x):
        store["x"] = x[0].detach()
        return x

    with _block_output_hook(hm, layer, grab):
        hm.model(hm.ids(seq))
    x = store["x"].to(sae.W.device, sae.W.dtype)
    return sae.encode(x).float()


@torch.no_grad()
def decompose_feature(hm: HyenaModel, sae: BatchTopKTiedSAE, seq: str, position: int, feature: int,
                      layer: int = SAE_LAYER) -> Decomposition:
    """Split feature f's encoder pre-activation at `position` into the direct
    contribution of every block <= layer, grouped by operator type via .by_kind()."""
    ids = hm.ids(seq)
    writes = capture_writes(hm, ids, [position], blocks=range(layer + 1))
    w = sae.W[:, feature].detach().float().cpu()
    rows = [Row(None, "embed", "embed", float(writes.embed[0] @ w))]
    for (b, part), x in sorted(writes.parts.items()):
        rows.append(Row(b, hm.kind(b), part, float(x[0] @ w)))
    rows.append(Row(None, "bias", "b_enc", float(sae.b_enc[feature])))

    store = {}

    def grab(x):
        store["x"] = x[0, position].detach().float().cpu()
        return x

    with _block_output_hook(hm, layer, grab):
        hm.model(ids)
    # actual comes from the real block output, independent of the captured writes
    actual = float(store["x"] @ w + sae.b_enc[feature].float().cpu())
    return Decomposition(position=position, label=f"SAE feature {feature} pre-activation", rows=rows, actual=actual)


@contextmanager
def edit_features(hm: HyenaModel, sae: BatchTopKTiedSAE, scales: dict[int, float], layer: int = SAE_LAYER,
                  positions: slice | None = None):
    """Rescale SAE features in the live forward pass (0.0 = suppress,
    2.0 = double). Error-preserving: only the decoded difference is added,
    so the SAE's reconstruction error never enters the model.

        with edit_features(hm, sae, {15680: 0.0}):
            logits = hm.logits(seq)
    """
    idx = torch.tensor(list(scales.keys()), dtype=torch.long)
    fac = torch.tensor(list(scales.values()))

    def fn(x):
        xs = x[0].to(sae.W.device, sae.W.dtype)
        f = sae.encode(xs)
        sel = f[:, idx.to(f.device)]
        delta_f = sel * (fac.to(f.device, f.dtype) - 1.0)
        delta = delta_f @ sae.W[:, idx.to(f.device)].T  # (L, H)
        if positions is not None:
            mask = torch.zeros(delta.shape[0], 1, device=delta.device, dtype=delta.dtype)
            mask[positions] = 1
            delta = delta * mask
        return x + delta.to(x.device, x.dtype).unsqueeze(0)

    with _block_output_hook(hm, layer, fn):
        yield
