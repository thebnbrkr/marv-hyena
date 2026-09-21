"""Interventions, probes, SAE, vindex, motifs on the tiny model."""
import random

import numpy as np
import torch

import marv_hyena as mh
from marv_hyena import motifs, sae as sae_mod, vindex
from marv_hyena.intervene import token_logprobs


def test_hooks_are_removed_and_ablation_changes_output(hm, seq):
    ids = hm.ids(seq)
    before = hm.logits(ids)
    means = mh.mean_writes(hm, ids, mh.mixers_of(hm, "li"))
    with mh.mean_ablate(hm, means):
        during = hm.logits(ids)
    after = hm.logits(ids)
    assert torch.equal(before, after)
    assert not torch.allclose(before, during)


def test_patching_is_causal_and_bounded(hm, seq):
    alt = seq[:150] + ("A" if seq[150] != "A" else "C") + seq[151:]
    metric = lambda logits, ids: float(token_logprobs(logits, ids)[100:140].sum())  # letters before the edit
    sweep = mh.patch_sweep(hm, alt, seq, metric=lambda lg, ids: float(token_logprobs(lg, ids)[151:200].sum()),
                           groups=mh.kind_groups(hm))
    assert len(sweep.results) == 2 * hm.n_blocks + 5
    assert sweep.base_metric != sweep.source_metric
    # the model is causal: letters before the variant score the same in both runs
    v = mh.make_variant(seq, 150, seq[150], alt[150], window=300)
    early = mh.patch_sweep(hm, v.alt_window, v.ref_window, metric=metric, components=[(0, "mixer"), (6, "mlp")])
    assert early.base_metric == early.source_metric


def test_variant_pipeline(hm, seq):
    v = mh.make_variant(seq, 120, seq[120], "T" if seq[120] != "T" else "G", window=200)
    assert v.ref_window[v.index] == seq[120] and v.alt_window[v.index] != seq[120]
    assert isinstance(mh.delta_logp(hm, v), float)
    sweep = mh.explain_variant(hm, v, span=50)
    assert any(r.component == "all li mixers" for r in sweep.results)


def test_copy_probe_and_truncation(hm):
    rng = random.Random(0)
    bg = "".join(rng.choice("ACGT") for _ in range(3000))
    p = mh.copy_probe(bg, insert_len=40, gap=500, lead=200)
    assert p.seq[slice(*p.first)] == p.seq[slice(*p.second)]
    ids = hm.ids(p.seq)
    s = mh.score_copy(hm.logits(ids), ids, p, skip=5)
    assert set(s) == {"first_acc", "second_acc", "first_lp", "second_lp"}
    curve = mh.truncation_curve(hm, p.seq, p.second, [10, 100, 1000])
    # only p.second[0] letters exist before the span, so the largest request is capped
    assert [c["context"] for c in curve] == [10, 100, p.second[0]]


def test_codon_phase_and_labels(hm, seq):
    n = len(seq)
    phase = np.full(n, -1, dtype=np.int8)
    phase[30:240] = np.arange(210) % 3
    feat = np.full(n, "", dtype=object)
    feat[30:240] = "CDS"
    track = mh.Track(0, seq, phase, feat, np.zeros(n, dtype=np.int8))
    out = mh.codon_phase_accuracy(hm, track)
    assert out["codon_pos1"]["n"] == 70
    acts = vindex.neuron_acts(hm, seq, 5)
    assert acts.shape == (n, hm.config.inner_mlp_size)
    labels = vindex.label_units(acts, track, top_frac=0.05, min_enrichment=1.0)
    assert all(l.enrichment >= 1.0 for l in labels)


def test_vindex_roundtrip_and_diff(hm, tmp_path):
    v = vindex.extract_mlp_vindex(hm, blocks=[0, 5], model_name="tiny")
    v.save(tmp_path / "v.npz")
    w = vindex.MlpVindex.load(tmp_path / "v.npz")
    assert w.model_name == "tiny" and np.array_equal(v.layers[5]["l3"], w.layers[5]["l3"])
    d = vindex.diff(v, w)
    assert all(abs(x.l1_cos - 1) < 1e-3 and abs(x.norm_ratio - 1) < 1e-3 for x in d)


def test_sae_decomposition_and_edit(hm, seq):
    torch.manual_seed(0)
    sae = sae_mod.BatchTopKTiedSAE(hm.hidden_size, 64, k=4)
    with torch.no_grad():
        sae.W.normal_(0, 0.3)
        sae.b_enc.normal_(0, 0.1)
    layer = 5
    acts = sae_mod.feature_acts(hm, sae, seq, layer=layer)
    pos, feat = divmod(int(acts.argmax()), acts.shape[1])
    d = sae_mod.decompose_feature(hm, sae, seq, pos, feat, layer=layer)
    assert abs(d.total - d.actual) < 1e-4 * max(1.0, abs(d.actual))
    base = hm.logits(seq)
    with sae_mod.edit_features(hm, sae, {feat: 1.0}, layer=layer):
        same = hm.logits(seq)
    with sae_mod.edit_features(hm, sae, {feat: 0.0}, layer=layer):
        edited = hm.logits(seq)
    assert torch.allclose(base, same, atol=1e-5)
    assert not torch.allclose(base, edited)


def test_block0_receptive_field_and_enumeration(hm):
    assert motifs.receptive_field(hm) == 3 + 7 - 1
    md = motifs.enumerate_block0(hm, k=5, n_top=10, batch=256)
    kmer = md.top_kmers[0][0]
    direct = motifs._block0_filter_out(hm, hm.ids(kmer))[0, -1, 0]
    assert abs(float(direct) - md.top_vals[0, 0]) < 1e-5
    assert md.pwm(0).shape == (5, 4)
