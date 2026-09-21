"""Architecture adapter: map a loaded Vortex `StripedHyena` (what `evo2.Evo2`
wraps as `.model`) onto the pieces marv-hyena operates on.

Written against Vortex (`vtx`) 1.0.8, commit 8b00afe. The facts this file
relies on, each read from `vortex/model/model.py`:

- `model.blocks[i]` is either an `AttentionBlock` or a `ParallelGatedConvBlock`.
  Which operator a conv block runs is decided by the config index lists:
  `hcs_layer_idxs` (Hyena-SE, short explicit FIR), `hcm_layer_idxs` (Hyena-MR,
  medium explicit FIR), `hcl_layer_idxs` (Hyena-LI, long implicit filter),
  `attn_layer_idxs` (multi-head attention).
- Conv block:      z = out_filter_dense(filter(projections(pre_norm(u)))) + u
                   y = mlp(post_norm(z)) + z
  Attention block: z = inner_mha_cls(pre_norm(u)) + u
                   y = mlp(post_norm(z)) + z
  So every block writes exactly two additive terms into the residual stream:
  the MIXER write (output of `out_filter_dense` or `inner_mha_cls`) and the MLP
  write (output of `mlp`). Hooking those two modules captures everything.
- final logits = unembed(norm(residual)); embeddings are tied by default.
- Tokens are raw ASCII bytes (`CharLevelTokenizer`): A=65, C=67, G=71, T=84.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

KINDS = ("se", "mr", "li", "attn")
PARTS = ("mixer", "mlp")
BASES = "ACGT"


def encode(seq: str) -> list[int]:
    """DNA string -> Evo 2 token ids (ASCII bytes, same as CharLevelTokenizer)."""
    return list(seq.encode("ascii"))


def decode(ids: Iterable[int]) -> str:
    return bytes(int(i) for i in ids).decode("ascii", errors="replace")


def base_id(base: str) -> int:
    if len(base) != 1:
        raise ValueError(f"expected one letter, got {base!r}")
    return ord(base)


class HyenaModel:
    """Thin handle over a Vortex StripedHyena. Holds no state besides the
    model; every analysis function takes one of these."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
        self._kinds = [self._kind_from_config(i) for i in range(len(model.blocks))]

    # ------------------------------------------------------------ loading
    @classmethod
    def from_evo2(cls, evo2_obj) -> "HyenaModel":
        return cls(evo2_obj.model)

    @classmethod
    def load(cls, model_name: str = "evo2_7b", use_flash_attn: bool | None = None, **kwargs) -> "HyenaModel":
        """Load through the official `evo2` package (downloads weights on first
        use). Needs Linux + CUDA; the 7B models run in bf16 without
        Transformer Engine, which is the A100 path.

        use_flash_attn=None: use the flash-attn package if it is installed,
        otherwise PyTorch's built-in attention kernel (see noflash.py)."""
        from . import noflash

        noflash.ignore_broken_transformer_engine()
        if use_flash_attn is None:
            use_flash_attn = noflash.flash_attn_available()
        if not use_flash_attn:
            noflash.prepare()
        from evo2 import Evo2

        return cls.from_evo2(Evo2(model_name, **kwargs))

    # ------------------------------------------------------------ structure
    def _kind_from_config(self, i: int) -> str:
        c = self.config
        if i in c.get("attn_layer_idxs", []):
            return "attn"
        if i in c.get("hcl_layer_idxs", []):
            return "li"
        if i in c.get("hcm_layer_idxs", []):
            return "mr"
        if i in c.get("hcs_layer_idxs", []):
            return "se"
        raise ValueError(f"block {i} is in none of the config's operator index lists")

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.model.embedding_layer.weight.device

    def kind(self, block: int) -> str:
        return self._kinds[block]

    def blocks_of(self, kind: str) -> list[int]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        return [i for i, k in enumerate(self._kinds) if k == kind]

    def block(self, i: int) -> nn.Module:
        return self.model.blocks[i]

    def is_hyena(self, i: int) -> bool:
        return self.kind(i) != "attn"

    def hyena_filter(self, i: int) -> nn.Module:
        """The `HyenaCascade` of a conv block (holds h / log_poles / residues / D)."""
        if not self.is_hyena(i):
            raise ValueError(f"block {i} is attention, it has no Hyena filter")
        return self.block(i).filter

    def component(self, block: int, part: str) -> nn.Module:
        """The module whose forward OUTPUT is exactly this component's residual
        write. part='mixer' -> out_filter_dense | inner_mha_cls; part='mlp' -> mlp."""
        b = self.block(block)
        if part == "mlp":
            return b.mlp
        if part == "mixer":
            return b.inner_mha_cls if self.kind(block) == "attn" else b.out_filter_dense
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")

    def all_components(self, kinds: Iterable[str] = KINDS, parts: Iterable[str] = PARTS):
        kinds, parts = set(kinds), list(parts)
        return [(i, p) for i in range(self.n_blocks) if self.kind(i) in kinds for p in parts]

    def describe(self) -> str:
        counts = {k: len(self.blocks_of(k)) for k in KINDS}
        layout = " ".join(f"{i}:{k}" for i, k in enumerate(self._kinds))
        return f"{self.n_blocks} blocks {counts}\n{layout}"

    # ------------------------------------------------------------ unembedding
    def unembed_weight(self) -> torch.Tensor:
        """(vocab, hidden). Tied to the input embedding unless the config unties it."""
        unembed = self.model.unembed
        if isinstance(unembed, nn.Embedding):
            return unembed.weight
        return self.model.embedding_layer.weight

    def final_norm_params(self) -> tuple[torch.Tensor, float]:
        norm = self.model.norm
        if norm.__dict__.get("use_flash_rmsnorm", False):
            raise NotImplementedError(
                "final norm uses flash rmsnorm, whose formula differs from the one "
                "decompose.py linearises; set use_flash_rmsnorm=False"
            )
        return norm.scale, float(norm.eps)

    # ------------------------------------------------------------ running
    def ids(self, seq: str) -> torch.Tensor:
        return torch.tensor(encode(seq), dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def logits(self, ids_or_seq) -> torch.Tensor:
        """(L, vocab) float32 logits. logits[j] is the prediction for letter j+1."""
        ids = self.ids(ids_or_seq) if isinstance(ids_or_seq, str) else ids_or_seq
        out, _ = self.model(ids)
        return out[0].float()
