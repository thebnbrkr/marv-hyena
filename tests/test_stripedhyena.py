"""The text-model path: a tiny copy of StripedHyena-7B's maths (complex modal
poles, biased featurizer, column split, hook-bypassing embedding, grouped-query
attention through marv-hyena's flash-attn stand-in)."""
import numpy as np
import pytest
import torch

import marv_hyena as mh
from marv_hyena import stripedhyena
from marv_hyena.checks import run_operator_checks
from marv_hyena.filters import li_kernel_on_grid, _modal
from marv_hyena.hooks import capture_outputs
from tiny_hyena import CharTokenizer, make_tiny_sh

TOL = 1e-4
TEXT = ("The quick brown fox jumps over the lazy dog. " * 6).strip()


@pytest.fixture(scope="module")
def sh():
    return mh.HyenaModel(make_tiny_sh(), tokenizer=CharTokenizer(64))


def test_kinds_and_tokenizer(sh):
    assert [sh.kind(i) for i in range(sh.n_blocks)] == ["li", "attn"] * 3
    ids = sh.ids("abc")
    assert ids.shape == (1, 4) and int(ids[0, 0]) == 1  # BOS-like first token from the tokenizer


def test_operator_checks_pass(sh):
    assert run_operator_checks(sh, TEXT, tol=TOL, verbose=False)


def test_capture_writes_without_embedding_hook(sh):
    """The embedding bypasses hooks; the lookup fallback must still reconstruct exactly."""
    ids = sh.ids(TEXT)
    w = mh.capture_writes(sh, ids, [10, ids.shape[1] - 1])
    assert w.reconstruction_error() < TOL


@pytest.mark.parametrize("block", [0, 2, 4])
def test_gates_and_band_ablation(sh, block):
    ids = sh.ids(TEXT)
    p = ids.shape[1] - 1
    g = mh.capture_gates(sh, ids, [5, p], blocks=[block])[block]
    assert g.rel_error < TOL
    d = mh.hyena_distance(sh, ids, block, p)
    assert d.rel_error < TOL
    for i, band in enumerate(d.bands[:4]):
        with mh.band_ablate(sh, block, band, mode="zero", positions=[p]):
            with capture_outputs(sh, [(block, "mixer")], positions=[p]) as cap:
                sh.model(ids)
        assert torch.allclose(cap[(block, "mixer")][0], d.actual - d.band_writes[i], atol=1e-4)


def test_reach_table_reads_complex_poles(sh):
    rows = mh.reach_table(sh, max_len=10_000)
    assert [r.block for r in rows] == [0, 2, 4] and all(np.isfinite(r.reach99) for r in rows)
    f = sh.hyena_filter(2)
    lp, res = _modal(f)
    k_grid = li_kernel_on_grid(lp, res, np.arange(50))
    k_model = f.compute_filter(50, f.D.device)[0][0].detach().numpy()
    assert np.allclose(k_grid, k_model, atol=1e-5)


def _reference_attention(mha, x, interleaved=False):
    """Independent re-derivation: rotary as complex rotation of (x_i, x_{i+d/2})
    pairs (or (x_2i, x_2i+1) when interleaved), explicit causal softmax,
    kv head = q head // group."""
    B, L, _ = x.shape
    nh, nkv, hd = mha.num_heads, mha.num_heads_kv, mha.head_dim
    W = mha.Wqkv.weight
    q = (x @ W[: nh * hd].T).view(B, L, nh, hd)
    k = (x @ W[nh * hd: (nh + nkv) * hd].T).view(B, L, nkv, hd)
    v = (x @ W[(nh + nkv) * hd:].T).view(B, L, nkv, hd)
    inv = 1.0 / (mha.rotary_emb.base ** (torch.arange(0, hd, 2).float() / hd))
    angle = torch.arange(L).float()[:, None] * inv[None, :]  # (L, hd/2)
    rot = torch.polar(torch.ones_like(angle), angle)[None, :, None, :]

    def rotate(t):
        if interleaved:
            z = torch.complex(t[..., 0::2], t[..., 1::2]) * rot
            return torch.stack([z.real, z.imag], dim=-1).flatten(-2)
        z = torch.complex(t[..., : hd // 2], t[..., hd // 2:]) * rot
        return torch.cat([z.real, z.imag], dim=-1)

    q, k = rotate(q), rotate(k)
    out = torch.zeros(B, L, nh, hd)
    for h in range(nh):
        kv_h = h // (nh // nkv)
        s = q[:, :, h] @ k[:, :, kv_h].transpose(-1, -2) / hd ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
        out[:, :, h] = s.softmax(-1) @ v[:, :, kv_h]
    return mha.out_proj(out.reshape(B, L, nh * hd))


@pytest.mark.parametrize("interleaved", [False, True])
def test_attention_standin_matches_reference(interleaved):
    torch.manual_seed(0)
    mha = stripedhyena.MHA(32, num_heads=4, num_heads_kv=2, rotary_emb_dim=8, qkv_proj_bias=False,
                           rotary_emb_base=10000, out_proj_bias=False, rotary_emb_interleaved=interleaved)
    x = torch.randn(1, 40, 32)
    with torch.no_grad():
        assert torch.allclose(mha(x), _reference_attention(mha, x, interleaved), atol=1e-5)
        assert not torch.allclose(mha(x), _reference_attention(mha, x, not interleaved), atol=1e-3)


def test_switching_rotary_convention(sh):
    ids = sh.ids(TEXT)
    before = sh.logits(ids)
    assert stripedhyena.set_rotary_interleaved(sh, True) == 3
    switched = sh.logits(ids)
    assert stripedhyena.set_rotary_interleaved(sh, False) == 3
    assert not torch.allclose(before, switched)
    assert torch.equal(before, sh.logits(ids))


def test_standin_answers_the_flash_attn_import():
    assert stripedhyena.install_attention_standin() or stripedhyena.flash_attn_mha_available()
    from flash_attn.modules.mha import MHA

    if not stripedhyena.flash_attn_mha_available():
        assert MHA is stripedhyena.MHA
