"""A tiny CPU StripedHyena that copies Vortex's module NAMES and forward MATH
(vtx 1.0.8, vortex/model/model.py + engine.py), in float32, so every
marv-hyena analysis can be tested without a GPU or a checkpoint.

The math is transcribed from Vortex, not re-derived: in particular the SE
filter goes through F.conv1d (cross-correlation), the MR filter (>=128 taps)
through the same rfft/irfft recipe as engine.fftconv_func, and the LI filter
through compute_filter + the FFT path of engine.parallel_iir. marv-hyena's
own decompositions are written independently and must reproduce these.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cfg(dict):
    __getattr__ = dict.__getitem__


def tiny_config(**over) -> Cfg:
    c = Cfg(
        hidden_size=16, num_attention_heads=2, vocab_size=128, num_layers=8, eps=1e-6,
        attn_layer_idxs=[3, 7], hcl_layer_idxs=[2, 6], hcm_layer_idxs=[1, 5], hcs_layer_idxs=[0, 4],
        hcs_filter_length=7, hcs_filter_groups=4, hcm_filter_length=128, hcm_filter_groups=4,
        short_filter_length=3, state_size=4, interleave=True, column_split_hyena=False,
        hyena_flip_x1x2=False, inner_mlp_size=24, tie_embeddings=True, evo2_style_activations=True,
    )
    c.update(over)
    return c


class RMSNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.eps, self.hidden_size = c.eps, c.hidden_size
        self.scale = nn.Parameter(1.0 + 0.1 * torch.randn(c.hidden_size))
        self.use_flash_rmsnorm = False

    def forward(self, x):
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        return self.scale * y


class ParallelGatedMLP(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.act = nn.Identity() if (layer_idx > 0 and c.get("evo2_style_activations")) else F.gelu
        self.l1 = nn.Linear(c.hidden_size, c.inner_mlp_size, bias=False)
        self.l2 = nn.Linear(c.hidden_size, c.inner_mlp_size, bias=False)
        self.l3 = nn.Linear(c.inner_mlp_size, c.hidden_size, bias=False)

    def forward(self, z):
        return self.l3(self.act(self.l1(z)) * self.l2(z))


def interleave(z_pre):
    return torch.cat([z_pre[:, 0::3, :], z_pre[:, 1::3, :], z_pre[:, 2::3, :]], dim=1)


def fftconv_func(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = k_f.squeeze()
    if u.dim() > k_f.dim():
        k_f = k_f.unsqueeze(0)
    u_f = torch.fft.rfft(u, n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return y + u * D.unsqueeze(-1)


class HyenaCascade(nn.Module):
    def __init__(self, c, layer_idx, hyena_filter_groups, fir_inner_filter_length=None):
        super().__init__()
        H = c.hidden_size
        self.hidden_size, self.num_attention_heads = H, c.num_attention_heads
        self.hidden_size_per_attention_head = H // c.num_attention_heads
        self.column_split_hyena, self.hyena_flip_x1x2 = c.column_split_hyena, c.hyena_flip_x1x2
        self.interleave = c.interleave
        self.hyena_filter_groups = hyena_filter_groups
        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = c.short_filter_length
        self.short_filter_weight = nn.Parameter(0.5 * torch.randn(3 * H, 1, c.short_filter_length))
        self.short_filter_bias = None
        if fir_inner_filter_length:
            self.h = nn.Parameter(0.3 * torch.randn(hyena_filter_groups, 1, fir_inner_filter_length))
            self.D = nn.Parameter(0.2 * torch.randn(H)) if fir_inner_filter_length >= 128 else None
        else:
            self.log_poles = nn.Parameter(-torch.rand(H, c.state_size, 1) * 0.5 - 0.01)
            self.residues = nn.Parameter(0.3 * torch.randn(H, c.state_size))
            self.D = nn.Parameter(0.2 * torch.randn(H))
            self.h = None
        self.t = None

    def update_time(self, L, device):
        if self.t is None or self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        h = (self.residues[..., None] * (self.log_poles * self.t).exp()).sum(1)[None]
        return h, torch.float32, self.log_poles, self.residues

    def forward(self, u, inference_params=None, padding_mask=None):
        H = self.hidden_size
        L = u.shape[1]
        z_pre = F.conv1d(u.permute(0, 2, 1), self.short_filter_weight, bias=None,
                         padding=self.short_filter_length - 1, groups=3 * H)[..., :L]
        if self.interleave:
            z_pre = interleave(z_pre)
        x2, x1, v = z_pre.split([H, H, H], dim=1)
        if self.h is None:  # LI
            h, *_ = self.compute_filter(L, u.device)
            x1v = x1 * v
            fft_size = 2 * L
            Hf = torch.fft.rfft(h, n=fft_size) / fft_size
            X = torch.fft.fft(x1v, n=fft_size)[..., : Hf.shape[-1]]
            y = torch.fft.irfft(X * Hf, n=fft_size, norm="forward")[..., :L]
            y = (y + x1v * self.D.unsqueeze(-1)) * x2
            return y.permute(0, 2, 1), None
        h = self.h
        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(H // self.hyena_filter_groups, 0)
        uu = x1 * v
        if self.fir_inner_filter_length >= 128:
            z = fftconv_func(uu, h[:, :, :L], self.D)
        else:
            z = F.conv1d(uu, h, bias=None, padding=self.fir_inner_filter_length - 1, groups=H)[..., :L]
        return (x2 * z).permute(0, 2, 1), None


class ParallelGatedConvBlock(nn.Module):
    def __init__(self, c, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None):
        super().__init__()
        self.pre_norm, self.post_norm = RMSNorm(c), RMSNorm(c)
        self.filter = HyenaCascade(c, layer_idx, hyena_filter_groups or c.hidden_size, fir_inner_filter_length)
        self.projections = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=False)
        self.out_filter_dense = nn.Linear(c.hidden_size, c.hidden_size, bias=True)
        self.mlp = ParallelGatedMLP(c, layer_idx)

    def forward(self, u, inference_params=None, padding_mask=None):
        z = self.projections(self.pre_norm(u))
        z, _ = self.filter(z)
        z_in = self.out_filter_dense(z) + u
        return self.mlp(self.post_norm(z_in)) + z_in, None


class MHA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.hd = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.Wqkv = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=False)
        self.out_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=True)

    def forward(self, x, inference_params=None):
        B, L, _ = x.shape
        q, k, v = self.Wqkv(x).reshape(B, L, 3, self.n, self.hd).unbind(2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / self.hd ** 0.5
        att = att.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        return self.out_proj(torch.einsum("bhts,bshd->bthd", att, v).reshape(B, L, -1))


class AttentionBlock(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.pre_norm, self.post_norm = RMSNorm(c), RMSNorm(c)
        self.inner_mha_cls = MHA(c)
        self.mlp = ParallelGatedMLP(c, layer_idx)

    def forward(self, u, inference_params=None, padding_mask=None):
        u = self.inner_mha_cls(self.pre_norm(u), inference_params=inference_params) + u
        return self.mlp(self.post_norm(u)) + u, None


class Unembed(nn.Module):
    """Stand-in for vortex's Lambda(embedding_layer.unembed): not an nn.Embedding."""

    def __init__(self, emb):
        super().__init__()
        self._emb = [emb]

    def forward(self, x):
        return x @ self._emb[0].weight.T


class StripedHyena(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding_layer = nn.Embedding(c.vocab_size, c.hidden_size)
        blocks = []
        for i in range(c.num_layers):
            if i in c.attn_layer_idxs:
                blocks.append(AttentionBlock(c, i))
            elif i in c.hcl_layer_idxs:
                blocks.append(ParallelGatedConvBlock(c, i))
            elif i in c.hcm_layer_idxs:
                blocks.append(ParallelGatedConvBlock(c, i, c.hcm_filter_groups, c.hcm_filter_length))
            else:
                blocks.append(ParallelGatedConvBlock(c, i, c.hcs_filter_groups, c.hcs_filter_length))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(c)
        self.unembed = Unembed(self.embedding_layer)

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        x = self.embedding_layer(x)
        for block in self.blocks:
            x, _ = block(x)
        return self.unembed(self.norm(x)), None


def make_tiny(seed: int = 0, **over) -> StripedHyena:
    torch.manual_seed(seed)
    return StripedHyena(tiny_config(**over)).eval()


# ---------------------------------------------------------------- StripedHyena-7B (text)
# Transcribed from the Hub code of togethercomputer/StripedHyena-Hessian-7B
# (Apache-2.0, commit 9ae6335): model.py ParallelHyenaFilter / ParallelGatedConvBlock,
# engine.py parallel_fir / parallel_iir, layers.py. Attention is marv_hyena's
# stand-in for flash_attn's MHA (stripedhyena.py), which the real model also runs
# under on Colab.

def tiny_sh_config(**over) -> Cfg:
    c = Cfg(
        hidden_size=16, num_attention_heads=4, proj_groups=2, vocab_size=64, num_layers=6, eps=1e-5,
        attn_layer_idxs=[1, 3, 5], hyena_layer_idxs=[0, 2, 4, 6],  # the Hub config also lists one index too many
        state_size=2, short_filter_length=3, short_filter_bias=True, column_split_hyena=True,
        hyena_filter_groups=1, inner_mlp_size=24, rotary_emb_base=500000, qkv_proj_bias=False,
        mha_out_proj_bias=False, tie_embeddings=False, final_norm=True,
    )
    c.update(over)
    return c


class SHFilter(nn.Module):
    def __init__(self, c):
        super().__init__()
        H = c.hidden_size
        self.hidden_size, self.state_size = H, c.state_size
        self.num_attention_heads = c.num_attention_heads
        self.hidden_size_per_attention_head = H // c.num_attention_heads
        self.column_split_hyena = c.column_split_hyena
        self.hyena_filter_groups = c.hyena_filter_groups
        self.short_filter_length = c.short_filter_length
        self.D = nn.Parameter(0.2 * torch.randn(H))
        self.short_filter_weight = nn.Parameter(0.5 * torch.randn(3 * H, 1, c.short_filter_length))
        self.short_filter_bias = nn.Parameter(0.1 * torch.randn(3 * H))
        # stable complex poles (|p| < 1) stored as (..., 2) real pairs, as in the checkpoint
        mag = 0.5 + 0.45 * torch.rand(H, c.state_size, 1)
        ang = (torch.rand(H, c.state_size, 1) - 0.5) * 2.0
        self.poles = nn.Parameter(torch.view_as_real(torch.polar(mag, ang)))
        self.residues = nn.Parameter(0.3 * torch.randn(H, c.state_size, 1, 2))

    def update_time(self, L, device):
        if not hasattr(self, "t") or self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        residues = torch.view_as_complex(self.residues.to(torch.float32))
        log_poles = torch.view_as_complex(self.poles.to(torch.float32)).log()
        h = (residues * (log_poles * self.t).exp()).real.sum(1)[None]
        return h, torch.float32, log_poles, residues

    def forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        z_pre = F.conv1d(u.permute(0, 2, 1), self.short_filter_weight, self.short_filter_bias, stride=1,
                         padding=self.short_filter_length - 1, groups=3 * self.hidden_size)[..., :L]
        h, *_ = self.compute_filter(L, u.device)
        nh, hd = self.num_attention_heads, self.hidden_size_per_attention_head
        z = z_pre.reshape(z_pre.shape[0], nh, 3 * hd, L)
        x2, x1, v = (z[:, :, :hd].reshape(1, -1, L), z[:, :, hd:2 * hd].reshape(1, -1, L),
                     z[:, :, 2 * hd:].reshape(1, -1, L))
        x1v = x1 * v
        fft_size = 2 * L
        Hf = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
        X = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)[..., : Hf.shape[-1]]
        y = torch.fft.irfft(X * Hf, n=fft_size, norm="forward")[..., :L]
        y = (y.to(x1v.dtype) + x1v * self.D.unsqueeze(-1)) * x2
        return y.permute(0, 2, 1), inference_params


class SHMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.l1 = nn.Linear(c.hidden_size, c.inner_mlp_size, bias=False)
        self.l2 = nn.Linear(c.hidden_size, c.inner_mlp_size, bias=False)
        self.l3 = nn.Linear(c.inner_mlp_size, c.hidden_size, bias=False)

    def forward(self, z):
        return self.l3(F.silu(self.l1(z)) * self.l2(z))


class SHConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pre_norm, self.post_norm = RMSNorm(c), RMSNorm(c)
        self.filter = SHFilter(c)
        self.projections = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.out_filter_dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.mlp = SHMLP(c)

    def forward(self, u, inference_params=None, padding_mask=None):
        z = self.projections(self.pre_norm(u))
        z, _ = self.filter(z)
        u = self.out_filter_dense(z) + u
        return self.mlp(self.post_norm(u)) + u, None


class SHAttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        from marv_hyena.stripedhyena import MHA

        self.pre_norm, self.post_norm = RMSNorm(c), RMSNorm(c)
        self.inner_mha_cls = MHA(c.hidden_size, c.num_attention_heads,
                                 num_heads_kv=c.num_attention_heads // c.proj_groups,
                                 rotary_emb_dim=c.hidden_size // c.num_attention_heads,
                                 qkv_proj_bias=c.qkv_proj_bias, rotary_emb_base=c.rotary_emb_base,
                                 out_proj_bias=c.mha_out_proj_bias)
        self.mlp = SHMLP(c)

    def forward(self, u, inference_params=None, padding_mask=None):
        u = self.inner_mha_cls(self.pre_norm(u), inference_params=inference_params) + u
        return self.mlp(self.post_norm(u)) + u, None


class SHEmbedding(nn.Embedding):
    """layers.VocabParallelEmbedding: embed() calls self.forward directly, so
    forward hooks on this module never fire (marv-hyena must not rely on them)."""

    def embed(self, x):
        return self.forward(x)

    def unembed(self, u):
        return u @ self.weight.T


class StripedHyena7BLike(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding_layer = SHEmbedding(c.vocab_size, c.hidden_size)
        self.norm = RMSNorm(c)
        self.unembed = SHEmbedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList(SHAttnBlock(c) if i in c.attn_layer_idxs else SHConvBlock(c)
                                    for i in range(c.num_layers))

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        x = self.embedding_layer.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        return self.unembed.unembed(self.norm(x)), None


def make_tiny_sh(seed: int = 0, **over) -> StripedHyena7BLike:
    torch.manual_seed(seed)
    return StripedHyena7BLike(tiny_sh_config(**over)).eval()


class CharTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer: one id per character, mod vocab."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) % (self.vocab_size - 2) + 2 for ch in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "".join(chr(int(i) + 30) for i in ids)
