"""StripedHyena-7B: a TEXT model built from Hyena blocks, read with the same tools.

`togethercomputer/StripedHyena-Hessian-7B` (Together AI, Apache-2.0) is the
English-language ancestor of Evo 2's architecture. Read from its Hub code
(model.py, engine.py, layers.py at commit 9ae6335):

- 32 blocks: Hyena at the even indices (`hyena_layer_idxs`), grouped-query
  attention at the odd ones. Every Hyena block is one operator type: a long
  implicit filter in MODAL form,
      k_c(t) = Re( sum_s residues[c,s] * poles[c,s]**t ),
  with complex poles and residues stored as (..., 2) real pairs. That is
  Evo 2's LI operator with complex instead of real poles, so marv-hyena gives
  these blocks kind "li". There is no SE / MR split.
- Same module names as Vortex (`blocks`, `filter`, `projections`,
  `out_filter_dense`, `mlp`, `inner_mha_cls`, `embedding_layer`, `norm`,
  `unembed`), the same operator y = x2 * (conv(k, x1 v) + D x1 v), a biased
  length-3 featurizer FIR, column-split projections and no interleave.
- Its attention is `flash_attn.modules.mha.MHA`, imported at module load. On
  Colab flash-attn usually has no prebuilt wheel. `install_attention_standin`
  provides a module with the same parameter names (`Wqkv`, `out_proj`,
  `rotary_emb.inv_freq`) and the same maths -- fused QKV with grouped KV heads,
  non-interleaved rotary embeddings recomputed in float32, causal softmax --
  through PyTorch's scaled_dot_product_attention. It supports full-sequence
  forward passes only (no generation cache), which is all analysis needs.
- Its embedding calls `self.forward` directly, which bypasses forward hooks;
  `trace.capture_writes` falls back to a lookup for that case.

    hm = HyenaModel.load_stripedhyena()          # ~15 GB download, bf16, one A100
    ids = hm.ids("The capital of France is")
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "togethercomputer/StripedHyena-Hessian-7B"
CODE_FILES = ("model.py", "layers.py", "engine.py", "cache.py", "utils.py")
_STANDIN_MARKER = "_marv_attention_standin"


# ---------------------------------------------------------------- attention stand-in
def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotary embedding on x (B, L, heads, d). Non-interleaved (flash_attn's
    default, what the Hub code asks for): dims i and i + r/2 form a pair.
    Interleaved (GPT-J style): dims 2i and 2i+1 form a pair."""
    r = 2 * cos.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    if interleaved:
        x1, x2 = x[..., 0:r:2].float(), x[..., 1:r:2].float()
        rot = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2).to(x.dtype)
    else:
        x1, x2 = x[..., : r // 2].float(), x[..., r // 2: r].float()
        rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)
    return torch.cat([rot, x[..., r:]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim, self.base = dim, float(base)
        self.register_buffer("inv_freq", self._inv_freq(None), persistent=False)

    def _inv_freq(self, device) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))

    def cos_sin(self, L: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        # like flash_attn with pos_idx_in_fp32: a stored non-fp32 inv_freq is recomputed in fp32
        inv = self.inv_freq if self.inv_freq.dtype == torch.float32 else self._inv_freq(device)
        freqs = torch.outer(torch.arange(L, device=device, dtype=torch.float32), inv.to(device))
        return freqs.cos(), freqs.sin()


class MHA(nn.Module):
    """Stand-in for flash_attn.modules.mha.MHA (self-attention, causal, GQA)."""

    def __init__(self, embed_dim, num_heads, num_heads_kv=None, rotary_emb_dim=0, qkv_proj_bias=True,
                 rotary_emb_base=10000.0, causal=True, layer_idx=None, out_proj_bias=True,
                 rotary_emb_interleaved=False, **_unused):
        super().__init__()
        if not causal:
            raise NotImplementedError("the stand-in implements causal self-attention only")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.num_heads_kv = num_heads_kv or num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        qkv_dim = self.head_dim * (num_heads + 2 * self.num_heads_kv)
        self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        self.rotary_emb = RotaryEmbedding(rotary_emb_dim, rotary_emb_base) if rotary_emb_dim > 0 else None
        self.rotary_interleaved = rotary_emb_interleaved

    def forward(self, x, inference_params=None, **_unused):
        if inference_params is not None:
            raise NotImplementedError("the attention stand-in has no generation cache; run full sequences")
        B, L, _ = x.shape
        nh, nkv, hd = self.num_heads, self.num_heads_kv, self.head_dim
        qkv = self.Wqkv(x)
        q = qkv[..., : nh * hd].view(B, L, nh, hd)
        kv = qkv[..., nh * hd:].view(B, L, 2, nkv, hd)  # flash_attn: "(two hkv d)"
        k, v = kv[:, :, 0], kv[:, :, 1]
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb.cos_sin(L, x.device)
            q, k = (_rotate(t, cos, sin, self.rotary_interleaved) for t in (q, k))
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, heads, L, hd)
        if nkv != nh:
            k, v = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, nh * hd))


def set_rotary_interleaved(hm, interleaved: bool) -> int:
    """Switch every attention stand-in in the model between the two rotary
    conventions (a diagnostic: the Hub code asks for non-interleaved). Returns
    how many attention modules were switched."""
    n = 0
    for m in hm.model.modules():
        if isinstance(m, MHA):
            m.rotary_interleaved = bool(interleaved)
            n += 1
    return n


def flash_attn_mha_available() -> bool:
    mod = sys.modules.get("flash_attn.modules.mha")
    if mod is not None:
        return not getattr(mod, _STANDIN_MARKER, False)
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ValueError, ImportError):
        return False


def install_attention_standin(force: bool = False) -> bool:
    """Make `from flash_attn.modules.mha import MHA` return the stand-in. Does
    nothing if the real flash-attn is installed, unless force=True. Returns
    True if the stand-in is (now) in place."""
    if flash_attn_mha_available() and not force:
        return False
    for name in ("flash_attn", "flash_attn.modules"):
        if name not in sys.modules or force:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
    mha = types.ModuleType("flash_attn.modules.mha")
    mha.MHA = MHA
    setattr(mha, _STANDIN_MARKER, True)
    sys.modules["flash_attn.modules.mha"] = mha
    sys.modules["flash_attn.modules"].mha = mha
    return True


# ---------------------------------------------------------------- loading
def import_code(code_dir: str | Path):
    """Import the Hub's model code as a package (its files use relative imports)."""
    install_attention_standin()
    pkg_root = Path(tempfile.mkdtemp(prefix="marv_sh_"))
    pkg = pkg_root / "stripedhyena_hub"
    pkg.mkdir()
    for f in CODE_FILES:
        shutil.copy(Path(code_dir) / f, pkg / f)
    (pkg / "__init__.py").write_text("")
    for k in [k for k in sys.modules if k == "stripedhyena_hub" or k.startswith("stripedhyena_hub.")]:
        del sys.modules[k]
    sys.path.insert(0, str(pkg_root))
    try:
        return importlib.import_module("stripedhyena_hub.model"), importlib.import_module("stripedhyena_hub.utils")
    finally:
        sys.path.remove(str(pkg_root))


def build(code_dir: str | Path, config: dict, state_dict: dict, dtype=torch.bfloat16):
    """Instantiate the Hub StripedHyena on the meta device and load `state_dict`
    into it (keys without the 'backbone.' prefix), so no random init is ever
    materialised. Poles and residues stay float32, as the Hub wrapper keeps them."""
    model_mod, utils_mod = import_code(code_dir)
    cfg = utils_mod.dotdict(dict(config))
    # analysis path: no fused kernels (the stand-in attention, the reference RMSNorm the trace linearises)
    cfg.update(use_flash_attn=False, use_flash_rmsnorm=False, use_flash_depthwise=False, use_flashfft=False,
               inference_mode=False)
    with torch.device("meta"):
        model = model_mod.StripedHyena(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if unexpected:
        print(f"marv_hyena: ignoring {len(unexpected)} unexpected checkpoint tensors, e.g. {unexpected[:3]}")
    if dtype == torch.bfloat16:
        model.to_bfloat16_except_poles_residues()
    else:
        for name, p in model.named_parameters():
            if "poles" not in name and "residues" not in name:
                p.data = p.data.to(dtype)
    # analysis only: without this every hooked forward pass outside no_grad builds an autograd graph
    return model.eval().requires_grad_(False)


def load(repo: str = REPO, path: str | None = None, device: str = "cuda", revision: str | None = None,
         tokenizer=True, dtype=torch.bfloat16):
    """Download (or read from `path`) and load StripedHyena-7B as a HyenaModel.
    tokenizer=True loads the repo's tokenizer; pass a tokenizer object, or None."""
    from safetensors.torch import load_file

    from .arch import HyenaModel

    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo, revision=revision,
                                 allow_patterns=["*.py", "*.json", "*.safetensors", "tokenizer*"])
    config = json.loads((Path(path) / "config.json").read_text())
    state = {}
    for shard in sorted(glob(str(Path(path) / "*.safetensors"))):
        for k, v in load_file(shard, device=device).items():
            state[k[len("backbone."):] if k.startswith("backbone.") else k] = v
    model = build(path, config, state, dtype=dtype)
    if tokenizer is True:
        # the class tokenizer_config.json names, loaded directly: AutoTokenizer would read the
        # config's auto_map and stop to ask whether to run the repo's custom code
        from transformers import LlamaTokenizerFast

        tokenizer = LlamaTokenizerFast.from_pretrained(path)
    return HyenaModel(model, tokenizer=tokenizer)
