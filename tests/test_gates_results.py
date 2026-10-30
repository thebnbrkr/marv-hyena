"""Gate capture, band / gate / channel edits, and the results table, on the tiny model.

The edits are checked against code written separately: band_ablate against
distance.py's per-band split, freeze_gate against capture_gates' gates."""
import json
import math

import pytest
import torch

import marv_hyena as mh
from marv_hyena.hooks import capture_module_io, capture_outputs

TOL = 1e-4


def _hyena_blocks(hm):
    return [b for b in range(hm.n_blocks) if hm.is_hyena(b)]


def test_capture_gates_reconstructs_every_hyena_block(hm, seq):
    ids = hm.ids(seq)
    pos = [5, 150, 299]
    caps = mh.capture_gates(hm, ids, pos)
    assert sorted(caps) == _hyena_blocks(hm)
    for b, g in caps.items():
        assert g.rel_error < TOL, (b, g.kind, g.rel_error)
        assert g.x2.shape == g.inner.shape == g.output.shape == (len(pos), hm.hidden_size)
        assert torch.allclose(g.x2 * g.inner, g.output, atol=1e-5)


def test_capture_gates_agrees_with_distance(hm, seq):
    """W @ y + b from the gate capture == the real mixer write distance.py reads."""
    ids = hm.ids(seq)
    for b in (0, 1, 2):  # se, mr, li
        g = mh.capture_gates(hm, ids, [250], blocks=[b])[b]
        d = mh.hyena_distance(hm, ids, b, 250)
        lin = hm.block(b).out_filter_dense
        write = lin.weight.float() @ g.output[0] + lin.bias.float()
        assert torch.allclose(write, d.actual, atol=1e-4)


def test_gates_refuse_attention_blocks(hm, seq):
    with pytest.raises(ValueError):
        mh.capture_gates(hm, hm.ids(seq), [10], blocks=[3])
    with pytest.raises(ValueError):
        with mh.band_ablate(hm, 3, (0, 0)):
            pass


@pytest.mark.parametrize("block", [1, 2])  # mr, li
def test_zero_band_ablation_matches_distance_split(hm, seq, block):
    """Zeroing a band at one position removes exactly that band's write."""
    ids, p = hm.ids(seq), 280
    d = mh.hyena_distance(hm, ids, block, p)
    for i, band in enumerate(d.bands):
        with mh.band_ablate(hm, block, band, mode="zero", positions=[p]) as rep:
            with capture_outputs(hm, [(block, "mixer")], positions=[p]) as cap:
                hm.model(ids[:, : p + 1])
        assert rep["rel_error"] < TOL
        expected = d.actual - d.band_writes[i]
        assert torch.allclose(cap[(block, "mixer")][0], expected, atol=1e-4), (block, band)


def test_band_beyond_reach_changes_nothing_and_hooks_are_removed(hm, seq):
    ids = hm.ids(seq)
    before = hm.logits(ids)
    with mh.band_ablate(hm, 0, (9, None), mode="zero"):  # SE looks 6 lags back at most
        during = hm.logits(ids)
    assert torch.equal(before, during)
    with mh.band_ablate(hm, 2, (1, None)):  # LI beyond lag 0, mean mode
        changed = hm.logits(ids)
    assert not torch.allclose(before, changed)
    assert torch.equal(before, hm.logits(ids))


def test_band_ablation_at_one_position_is_causal(hm, seq):
    ids, p = hm.ids(seq), 200
    before = hm.logits(ids)
    with mh.band_ablate(hm, 2, (1, None), positions=[p]):
        during = hm.logits(ids)
    # later FFT-convolution blocks mix float rounding across the whole sequence,
    # so "unchanged" means rounding-level here, not bit-identical
    assert (during[:p] - before[:p]).abs().max() < 1e-5
    assert (during[p:] - before[p:]).abs().max() > 1e-2


def test_freeze_output_gate_matches_capture(hm, seq):
    ids, b, ch = hm.ids(seq), 2, [0, 5]
    all_pos = list(range(len(seq)))
    g = mh.capture_gates(hm, ids, all_pos, blocks=[b])[b]
    expected = g.output.clone()
    expected[:, ch] = g.x2[:, ch].mean(0) * g.inner[:, ch]
    with mh.freeze_gate(hm, b, "x2", channels=ch) as rep:
        with capture_module_io(hm.hyena_filter(b), want_input=False) as io:
            hm.model(ids)
    assert rep["rel_error"] < TOL
    assert torch.allclose(io["output"][0].float(), expected, atol=1e-5)


def test_freeze_input_gate_touches_only_its_channels(hm, seq):
    ids, b = hm.ids(seq), 1
    with capture_module_io(hm.hyena_filter(b), want_input=False) as io:
        hm.model(ids)
    base = io["output"][0]
    with mh.freeze_gate(hm, b, "x1", channels=[3]):
        with capture_module_io(hm.hyena_filter(b), want_input=False) as io2:
            hm.model(ids)
    out = io2["output"][0]
    others = [c for c in range(hm.hidden_size) if c != 3]
    assert torch.allclose(out[:, others], base[:, others], atol=1e-6)
    assert not torch.allclose(out[:, 3], base[:, 3])


def test_scale_channels(hm, seq):
    ids, b = hm.ids(seq), 1
    before = hm.logits(ids)
    with mh.scale_channels(hm, b, list(range(hm.hidden_size)), 1.0):
        assert torch.equal(before, hm.logits(ids))
    with mh.scale_channels(hm, b, list(range(hm.hidden_size)), 0.0):
        with capture_outputs(hm, [(b, "mixer")]) as cap:
            hm.model(ids)
    bias = hm.block(b).out_filter_dense.bias.float()
    assert torch.allclose(cap[(b, "mixer")], bias.expand_as(cap[(b, "mixer")]), atol=1e-6)
    with mh.scale_channels(hm, b, [2], 2.0, positions=[10, 20]):
        with capture_module_io(hm.hyena_filter(b), want_input=False) as io:
            hm.model(ids)
    with capture_module_io(hm.hyena_filter(b), want_input=False) as io0:
        hm.model(ids)
    y, y0 = io["output"][0], io0["output"][0]
    assert torch.allclose(y[[10, 20], 2], 2 * y0[[10, 20], 2])
    y[[10, 20], 2] = y0[[10, 20], 2]
    assert torch.equal(y, y0)


def test_failed_check_refuses_the_edit(hm, seq):
    ids = hm.ids(seq)
    before = hm.logits(ids)
    with pytest.raises(ValueError, match="edit refused"):
        with mh.band_ablate(hm, 2, (1, None), tol=-1.0):  # forces the check to fail
            hm.logits(ids)
    assert torch.equal(before, hm.logits(ids))


# ---------------------------------------------------------------- results table
def test_results_table_appends_and_filters(tmp_path):
    t = mh.ResultsTable(tmp_path / "sub" / "table.jsonl")
    assert t.rows() == []
    rows = [{"effect_a": 0.1 * i, "effect_b": 0.1 * i - 0.3} for i in range(20)]
    r1 = t.add(model="tiny", component=(2, "mixer"), intervention="band_ablate:1+", task="copy@100",
               dataset="random", stats=mh.paired_effect_size(rows, n_boot=200), unit="nats", seed=0,
               commit="abc123", depth=3)
    t.add(model="tiny", component="all li mixers", intervention="mean_ablate", task="copy@100",
          dataset="random", n=5, effect=float("nan"), commit=None)
    lines = (tmp_path / "sub" / "table.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == r1
    assert r1["component"] == "L2 mixer" and r1["n"] == 20 and r1["extra"] == {"depth": 3}
    assert r1["ci_low"] <= r1["effect"] <= r1["ci_high"] and math.isclose(r1["effect"], -0.3, abs_tol=1e-9)
    assert t.rows()[1]["effect"] is None  # NaN is stored as null
    assert [r["intervention"] for r in t.where(component=(2, "mixer"))] == ["band_ablate:1+"]
    assert len(t.where(task="copy@100", model="tiny")) == 2


def test_results_table_needs_n_and_effect(tmp_path):
    t = mh.ResultsTable(tmp_path / "t.jsonl")
    with pytest.raises(ValueError):
        t.add(model="m", component="c", intervention="i", task="t", dataset="d", effect=1.0)
    assert not (tmp_path / "t.jsonl").exists()
