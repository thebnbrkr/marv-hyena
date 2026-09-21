"""Round-2 additions: diagnostics, conditions, position importance, motif control."""
import itertools
import random

import numpy as np
import torch

import marv_hyena as mh
from marv_hyena import diagnostics, motifs
from marv_hyena.experiments import codon_test, copy_test, make_conditions, periodicity


def test_write_norm_shares_and_bottlenecks(hm, seq):
    rows = diagnostics.write_norms(hm, seq, [100, 200])
    assert len(rows) == 1 + 2 * hm.n_blocks and all(r.share >= 0 for r in rows)
    # make block 6's mixer write enormous -> it must be found as the bottleneck
    with mh.intervene.replace_outputs(hm, {(6, "mixer"): lambda x: x * 1e4}):
        assert diagnostics.find_bottlenecks(hm, seq, [100, 200]) == [6]


def test_health_flags_a_broken_model(hm, seq):
    h = diagnostics.health(hm, seq)
    assert set(h) == {"health_acc", "health_lp", "broken"}
    # swamp the residual with one constant vector: every position predicts the same
    # thing, which can't beat uniform guessing on a mixed sequence -> broken
    const = torch.randn(hm.hidden_size) * 1e4
    with mh.intervene.replace_outputs(hm, {(7, "mlp"): lambda x: x * 0 + const}):
        assert diagnostics.health(hm, seq)["broken"]


def test_make_conditions_excludes_and_singles(hm):
    c = make_conditions(hm, exclude_blocks=[6], single_blocks=[3, 6])
    assert c["none"] == []
    assert ("-li (keep L6)") in c and c["-li (keep L6)"] == [(2, "mixer")]
    assert c["-se"] == [(0, "mixer"), (4, "mixer")]  # SE family untouched by the exclusion
    assert c["-L3(attn)"] == [(3, "mixer")] and c["-L6(li)"] == [(6, "mixer")]


def test_experiments_carry_health(hm, seq):
    rng = random.Random(0)
    genome = "".join(rng.choice("ACGT") for _ in range(4000))
    conds = make_conditions(hm, families=("attn",), single_blocks=[2])
    rows = copy_test(hm, genome, gaps=(100,), insert_len=30, seeds=1, lead=100, conditions=conds,
                     health_seq=genome[:500], verbose=False)
    assert {r["condition"] for r in rows} == {"none", "-attn", "-L2(li)"} and all("broken" in r for r in rows)
    n = len(seq)
    phase = np.full(n, -1, dtype=np.int8)
    phase[30:240] = np.arange(210) % 3
    feat = np.full(n, "", dtype=object)
    feat[30:240] = "CDS"
    crow = codon_test(hm, mh.Track(0, seq, phase, feat, np.zeros(n, dtype=np.int8)), conditions=conds)
    assert set(periodicity(crow)) == {"none", "-attn", "-L2(li)"}


def test_position_importance_matches_brute_force(hm):
    k = 4
    md = motifs.enumerate_block0(hm, k=k, n_top=5, batch=64)
    kmers = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    ids = torch.tensor([[ord(ch) for ch in s] for s in kmers])
    out = motifs._block0_filter_out(hm, ids)[:, -1].double().numpy()  # (4^k, H)
    digits = np.array([[ "ACGT".index(ch) for ch in s] for s in kmers])
    imp = np.stack([np.var([out[digits[:, p] == b].mean(0) for b in range(4)], axis=0) for p in range(k)])
    imp = imp / imp.sum(0, keepdims=True)
    assert np.allclose(md.position_importance, imp, atol=1e-5)


def test_motif_control_rejects_composition_only_channels():
    rng = random.Random(0)
    at_rich = ["".join(rng.choice("AT") for _ in range(9)) for _ in range(50)]  # contains TAA by chance, often
    taa_specific = ["".join(rng.choice("ACGT") for _ in range(3)) + "TAA" + "".join(rng.choice("ACGT") for _ in range(3))
                    for _ in range(50)]
    md = motifs.MotifDictionary(9, [at_rich, taa_specific], np.zeros((2, 50)), [[], []], np.zeros((2, 50)))
    raw, ctrl = motifs.motif_channels(md, "TAA", min_frac=0.5)
    assert 1 in {h.channel for h in ctrl}
    assert 0 not in {h.channel for h in ctrl}


def test_variant_position_patching(hm, seq):
    v = mh.make_variant(seq, 150, seq[150], "A" if seq[150] != "A" else "C", window=300)
    eff = mh.variants.downstream_effect(hm, v, span=50)
    sweep = mh.explain_variant(hm, v, span=50, at="variant")
    assert abs((sweep.source_metric - sweep.base_metric) - eff) < 1e-4
