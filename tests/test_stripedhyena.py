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


def _reference_attention(mha, x, interleaved=False, inv=None, window=None):
    """Independent re-derivation: rotary as complex rotation of (x_i, x_{i+d/2})
    pairs (or (x_2i, x_2i+1) when interleaved), explicit causal softmax,
    kv head = q head // group. `inv` overrides the rotary frequencies; `window`
    limits attention to that many positions back, plus position 0."""
    B, L, _ = x.shape
    nh, nkv, hd = mha.num_heads, mha.num_heads_kv, mha.head_dim
    W = mha.Wqkv.weight
    q = (x @ W[: nh * hd].T).view(B, L, nh, hd)
    k = (x @ W[nh * hd: (nh + nkv) * hd].T).view(B, L, nkv, hd)
    v = (x @ W[(nh + nkv) * hd:].T).view(B, L, nkv, hd)
    if inv is None:
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
        if window is not None:
            far = torch.tril(torch.ones(L, L, dtype=torch.bool), -window)
            far[:, 0] = False
            s = s.masked_fill(far, float("-inf"))
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


def _mha(**kw):
    torch.manual_seed(0)
    return stripedhyena.MHA(32, num_heads=4, num_heads_kv=2, rotary_emb_dim=8, qkv_proj_bias=False,
                            rotary_emb_base=10000, out_proj_bias=False, **kw)


def test_rotary_precisions():
    """A bf16 inv_freq (as in the checkpoint) is recomputed in fp32 by default;
    'stored' uses it as loaded; 'bf16' rounds positions too."""
    mha, x = _mha(), torch.randn(1, 300, 32)
    exact = mha.rotary_emb.inv_freq.clone()
    mha.rotary_emb.inv_freq = exact.to(torch.bfloat16)
    with torch.no_grad():
        assert torch.allclose(mha(x), _reference_attention(mha, x), atol=1e-5)
        mha.rotary_emb.precision = "stored"
        stored = _reference_attention(mha, x, inv=exact.to(torch.bfloat16).float())
        assert torch.allclose(mha(x), stored, atol=1e-5)
        assert not torch.allclose(stored, _reference_attention(mha, x), atol=1e-4)
        mha.rotary_emb.precision = "bf16"
        out_bf16 = mha(x)
        assert not torch.allclose(out_bf16, stored, atol=1e-4)
        cos, _ = mha.rotary_emb.cos_sin(300, "cpu")
        pos = torch.arange(300).to(torch.bfloat16)
        assert torch.equal(cos, (pos[:, None] * exact.to(torch.bfloat16)[None]).cos().float())


@pytest.mark.parametrize("window", [1, 7, 39])
def test_attention_window_matches_reference(window):
    mha, x = _mha(), torch.randn(1, 40, 32)
    mha.window = window
    with torch.no_grad():
        assert torch.allclose(mha(x), _reference_attention(mha, x, window=window), atol=1e-5)


def test_switching_precision_and_window_restores(sh):
    ids = sh.ids(TEXT)
    before = sh.logits(ids)
    assert stripedhyena.set_attention_window(sh, 8) == 3
    assert not torch.allclose(before, sh.logits(ids))
    stripedhyena.set_attention_window(sh, None)
    assert stripedhyena.set_rotary_precision(sh, "bf16") == 3
    assert not torch.allclose(before, sh.logits(ids))
    stripedhyena.set_rotary_precision(sh, "fp32")
    assert torch.equal(before, sh.logits(ids))
    with pytest.raises(ValueError):
        stripedhyena.set_rotary_precision(sh, "fp16")


def test_standin_answers_the_flash_attn_import():
    assert stripedhyena.install_attention_standin() or stripedhyena.flash_attn_mha_available()
    from flash_attn.modules.mha import MHA

    if not stripedhyena.flash_attn_mha_available():
        assert MHA is stripedhyena.MHA


def _bf16_sh():
    """The tiny model with bf16 weights and float32 poles/residues, like the checkpoint."""
    m = make_tiny_sh()
    for n, p in m.named_parameters():
        if "poles" not in n and "residues" not in n:
            p.data = p.data.to(torch.bfloat16)
    return mh.HyenaModel(m, tokenizer=CharTokenizer(64))


def test_compute_precision_whole_model():
    sh16 = _bf16_sh()
    ids = sh16.ids(TEXT)
    dtypes = {n: p.dtype for n, p in sh16.model.named_parameters()}
    before = sh16.logits(ids)
    # reference: the same bf16-valued weights, built in float32 from the start
    ref = make_tiny_sh()
    ref.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sh16.model.state_dict().items()})
    expected = mh.HyenaModel(ref, tokenizer=CharTokenizer(64)).logits(ids)
    with stripedhyena.compute_precision(sh16, torch.float32):
        assert all(p.dtype == torch.float32 for p in sh16.model.parameters())
        during = sh16.logits(ids)
    assert torch.allclose(during.float(), expected.float(), atol=1e-4)
    assert not torch.equal(during.float(), before.float())
    assert {n: p.dtype for n, p in sh16.model.named_parameters()} == dtypes
    assert torch.equal(before, sh16.logits(ids))


def test_compute_precision_some_blocks():
    sh16 = _bf16_sh()
    ids = sh16.ids(TEXT)
    before = sh16.logits(ids)
    with stripedhyena.compute_precision(sh16, torch.float32, blocks=sh16.blocks_of("li")):
        assert sh16.block(0).projections.weight.dtype == torch.float32
        assert sh16.block(1).inner_mha_cls.Wqkv.weight.dtype == torch.bfloat16
        during = sh16.logits(ids)
    assert during.dtype == before.dtype and not torch.equal(during, before)
    assert float((during.float() - before.float()).norm() / before.float().norm()) < 0.05  # bf16 rounding only
    assert torch.equal(before, sh16.logits(ids))
