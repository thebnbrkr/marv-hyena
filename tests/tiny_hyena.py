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
