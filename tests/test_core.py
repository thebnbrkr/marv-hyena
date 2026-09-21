"""Exactness tests: every decomposition must add back up to the real model."""
import torch

import marv_hyena as mh
from marv_hyena.filters import reach_table_from_checkpoint


def test_layout(hm):
    assert [hm.kind(i) for i in range(hm.n_blocks)] == ["se", "mr", "li", "attn"] * 2
    assert hm.blocks_of("li") == [2, 6]
    assert hm.component(3, "mixer") is hm.block(3).inner_mha_cls
    assert hm.component(0, "mixer") is hm.block(0).out_filter_dense


def test_writes_reconstruct_final_residual(hm, seq):
    w = mh.capture_writes(hm, hm.ids(seq), [10, 150, 299])
    assert w.reconstruction_error() < 1e-5
    assert len(w.parts) == 2 * hm.n_blocks


def test_decomposition_sums_to_real_logit_difference(hm, seq):
    for idx in (5, 120, 299):
        d = mh.decompose_prediction(hm, seq, idx)
        assert abs(d.total - d.actual) < 1e-4 * max(1.0, abs(d.actual))
        assert set(d.by_kind()) <= {"se", "mr", "li", "attn", "mlp", "embed"}
    d = mh.decompose_prediction(hm, seq, 50, target="G", baseline="T")
    assert abs(d.total - d.actual) < 1e-4 * max(1.0, abs(d.actual))


def test_distance_bands_reconstruct_every_hyena_block(hm, seq):
    ids = hm.ids(seq)
    for b in (0, 1, 2, 4, 5, 6):
        for pos in (0, 3, 64, 200, 299):
            r = mh.hyena_distance(hm, ids, b, pos)
            assert r.rel_error < 1e-4, (b, pos, r.rel_error)


def test_distance_respects_filter_length(hm, seq):
    """SE (7 taps) can only reach 6 letters back: bands beyond must be exactly zero
    before the gate; MR (128 taps) nothing beyond lag 127."""
    ids = hm.ids(seq)
    r = mh.hyena_distance(hm, ids, 0, 250, bands=((0, 0), (1, 6), (7, None)))
    assert r.band_writes[2].abs().max() == 0
    r = mh.hyena_distance(hm, ids, 1, 250, bands=((0, 127), (128, None)))
    assert r.band_writes[1].abs().max() == 0


def test_wrong_lag_orientation_is_caught(hm, seq, monkeypatch):
    """The reconstruction check must be able to fail: if the analysis reads
    the SE filter the wrong way round, rel_error has to blow up."""
    from marv_hyena import filters as F

    ids = hm.ids(seq)
    k, _ = F.lag_kernel(hm.hyena_filter(0), "se", 10, hm.hidden_size)
    assert torch.allclose(k[:, 0], hm.hyena_filter(0).h.data.repeat_interleave(4, 0)[:, 0, -1])  # lag 0 = LAST tap
    monkeypatch.setattr(F, "_fir_to_lag", lambda h: h)  # wrong: treat conv1d taps as lags
    assert mh.hyena_distance(hm, ids, 0, 100).rel_error > 1e-2


def test_normed_direction_matches_decomposition(hm, seq):
    idx = 180
    d = mh.normed_direction(hm, seq, idx)
    w = mh.capture_writes(hm, hm.ids(seq[:idx]), [idx - 1])
    dec = mh.decompose_writes(hm, w, mh.logit_direction(hm, seq[idx])[0], 0, "x")
    assert abs(float((w.embed[0] + sum(w.parts.values())[0]) @ d) - dec.total) < 1e-3


def test_reach_tables_agree(hm):
    live = mh.reach_table(hm, max_len=4096)
    sd = {k: v.detach() for k, v in hm.model.state_dict().items()}
    off = reach_table_from_checkpoint(sd, dict(hm.config), max_len=4096)
    assert [(r.block, r.kind) for r in live] == [(r.block, r.kind) for r in off]
    for a, b in zip(live, off):
        assert abs(a.reach90 - b.reach90) < 1e-6
    se = [r for r in live if r.kind == "se"]
    mr = [r for r in live if r.kind == "mr"]
    assert all(r.max_reach99 <= 6 for r in se)
    assert all(r.max_reach99 <= 127 for r in mr)
