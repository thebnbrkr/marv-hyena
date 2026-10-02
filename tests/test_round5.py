"""Round 5: substitution-matched designs and the controls behind them.

Following the tiny_hyena rule, the biology these tests check (which letter
changes are transitions, which codon swaps are silent) is written out by hand
here rather than derived from the module under test.
"""
import random

import numpy as np
import pytest
import torch

from marv_hyena import codons, controls, genome, motifs, nullmodel, probes

# Transitions stay inside a chemical family: purines A,G or pyrimidines C,T.
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


def _rand_dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _cds_track(seq: str) -> probes.Track:
    n = len(seq)
    return probes.Track(0, seq, np.arange(n, dtype=np.int8) % 3,
                        np.full(n, "CDS", dtype=object), np.ones(n, dtype=np.int8))


# ---------------------------------------------------------------- letter types
def test_transition_table():
    for a in "ACGT":
        for b in "ACGT":
            if a == b:
                continue
            assert codons.is_transition(a, b) == ((a, b) in TRANSITIONS)
    assert codons.substitution_type("A", "G") == "transition"
    assert codons.substitution_type("T", "G") == "transversion"
    with pytest.raises(ValueError):
        codons.substitution_type("A", "A")


def test_gc_change():
    assert codons.gc_change("A", "G") == 1
    assert codons.gc_change("C", "T") == -1
    assert codons.gc_change("A", "T") == 0
    assert codons.gc_change("G", "C") == 0


# ---------------------------------------------------------------- designs
def _check_site(seq, s):
    assert seq[s.pos] == s.ref
    start = s.pos - s.codon_pos
    assert seq[start:start + 3] == s.codon
    for arm in (s.a, s.b):
        assert arm.alt != s.ref
        assert arm.alt_codon == s.codon[:s.codon_pos] + arm.alt + s.codon[s.codon_pos + 1:]
        assert arm.subst == ("transition" if (s.ref, arm.alt) in TRANSITIONS else "transversion")
    assert s.a.alt != s.b.alt


def test_matched_design_holds_letter_type_fixed():
    seq = _rand_dna(30000, 1)
    sites = codons.paired_sites(_cds_track(seq), ("silent", "transversion"), ("missense", "transversion"),
                                "matched", min_spacing=3)
    assert sites
    for s in sites:
        _check_site(seq, s)
        assert s.a.subst == s.b.subst == "transversion"
        assert codons.GENETIC_CODE[s.a.alt_codon] == s.aa
        assert codons.GENETIC_CODE[s.b.alt_codon] not in (s.aa, "*")
    # the only families where this exists: Ile wobble and Arg first position
    assert {s.aa for s in sites} <= {"I", "R"}


def test_flipped_design_turns_the_confound_against_the_hypothesis():
    seq = _rand_dna(30000, 2)
    sites = codons.paired_sites(_cds_track(seq), ("silent", "transversion"), ("missense", "transition"),
                                "flipped", min_spacing=3)
    assert sites
    for s in sites:
        _check_site(seq, s)
        assert s.a.subst == "transversion" and s.b.subst == "transition"
    # Two places in the code allow it: arginine's first letter (AGG->CGG silent,
    # AGG->GGG glycine) and isoleucine's rare codon ATA (ATA->ATT silent,
    # ATA->ATG methionine).
    for s in sites:
        assert (s.aa, s.codon_pos) == ("R", 0) or (s.codon, s.codon_pos) == ("ATA", 2)


def test_fourfold_design_changes_no_amino_acid():
    seq = _rand_dna(6000, 3)
    sites = codons.paired_sites(_cds_track(seq), ("silent", "transition"), ("silent", "transversion"),
                                "fourfold", min_spacing=3)
    assert sites
    for s in sites:
        _check_site(seq, s)
        assert codons.GENETIC_CODE[s.a.alt_codon] == codons.GENETIC_CODE[s.b.alt_codon] == s.aa


def test_stop_designs():
    seq = _rand_dna(30000, 4)
    t = _cds_track(seq)
    matched = codons.paired_sites(t, ("nonsense", "transversion"), ("missense", "transversion"), "m", min_spacing=3)
    flipped = codons.paired_sites(t, ("nonsense", "transition"), ("missense", "transversion"), "f", min_spacing=3)
    assert matched and flipped
    for s in matched + flipped:
        _check_site(seq, s)
        assert codons.GENETIC_CODE[s.a.alt_codon] == "*"
        assert codons.GENETIC_CODE[s.b.alt_codon] not in ("*", s.aa)
    assert all(s.a.subst == "transition" for s in flipped)


def test_real_stops_are_never_mutated():
    seq = _rand_dna(20000, 5)
    for design in codons.design_sites(_cds_track(seq), max_sites=None, min_spacing=3).values():
        assert all(s.aa != "*" for s in design if s.codon)


def test_sites_are_spread_and_seeded():
    seq = _rand_dna(30000, 6)
    t = _cds_track(seq)
    a = codons.paired_sites(t, ("silent", "transition"), ("silent", "transversion"), "x", min_spacing=60, seed=1)
    b = codons.paired_sites(t, ("silent", "transition"), ("silent", "transversion"), "x", min_spacing=60, seed=1)
    assert [s.b.alt for s in a] == [s.b.alt for s in b]
    pos = [s.pos for s in a]
    assert all(y - x >= 60 for x, y in zip(pos, pos[1:]))
    # the transversion letter is drawn, not always the first in ACGT order
    assert len({s.b.alt for s in a}) > 1


def test_noncoding_sites_avoid_annotations():
    seq = _rand_dna(3000, 7)
    n = len(seq)
    feat = np.full(n, "", dtype=object)
    feat[1000:2000] = "CDS"
    t = probes.Track(0, seq, np.full(n, -1, dtype=np.int8), feat, np.zeros(n, dtype=np.int8))
    sites = codons.noncoding_sites(t, min_spacing=10, flank=20)
    assert sites
    for s in sites:
        assert not (1000 - 20 <= s.pos <= 1999 + 20)
        assert s.a.subst == "transition" and s.b.subst == "transversion"
        assert s.a.alt != s.ref and s.b.alt != s.ref


def test_from_wobble_reproduces_round4_sites():
    seq = _rand_dna(3000, 8)
    ws = codons.wobble_sites(_cds_track(seq), min_spacing=3)
    ps = codons.from_wobble(ws)
    assert len(ps) == len(ws)
    for w, p in zip(ws, ps):
        assert (p.pos, p.a.alt, p.b.alt) == (w.pos, w.syn_alt, w.nonsyn_alt)
        assert p.a.kind == "silent" and p.b.kind == "missense"


def test_codon_usage_counts_in_frame_codons_only():
    seq = "ATGATGAAA" + "CCC"
    t = _cds_track(seq)
    u = codons.codon_usage(t)
    assert u["ATG"] == pytest.approx(2 / 4) and u["AAA"] == pytest.approx(1 / 4)
    assert u["TGA"] == 0.0  # out-of-frame ATG|ATG does not create TGA


# ---------------------------------------------------------------- measurement
def test_paired_test_keeps_every_block(hm, seq):
    t = _cds_track(seq)
    sites = codons.paired_sites(t, ("silent", None), ("missense", None), "any", min_spacing=40, margin=40)[:2]
    if not sites:
        pytest.skip("no sites in fixture")
    rows = codons.paired_test(hm, seq, sites, window=256, span=8, downstream_span=20,
                              usage=codons.codon_usage(t))
    assert rows
    for r in rows:
        assert len(r["div_a"]) == len(r["div_b"]) == hm.n_blocks == len(r["kinds"])
        assert r["d_missense"] == 1 and r["d_nonsense"] == 0
        assert np.isfinite(r["effect_a"]) and np.isfinite(r["effect_b"])


def test_paired_test_identical_arms_give_zero_difference(hm, seq):
    """If both arms make the same substitution, every paired statistic must be
    exactly neutral -- the exactness check for this design."""
    t = _cds_track(seq)
    sites = codons.paired_sites(t, ("silent", None), ("missense", None), "any", min_spacing=40, margin=40)[:1]
    if not sites:
        pytest.skip("no sites in fixture")
    s = sites[0]
    same = codons.PairedSite("same", s.pos, s.ref, s.a, s.a, s.codon_pos, s.codon, s.aa)
    r = codons.paired_test(hm, seq, [same], window=256, span=8, downstream_span=20)[0]
    assert r["effect_a"] == pytest.approx(r["effect_b"], abs=1e-6)
    assert np.allclose(r["div_a"], r["div_b"], atol=1e-6)


# ---------------------------------------------------------------- controls: stats
def test_sign_test_and_interval():
    assert controls.sign_test(5, 10) == pytest.approx(1.0)
    assert controls.sign_test(0, 10) == pytest.approx(2 / 1024)
    assert controls.sign_test(10, 10) == pytest.approx(2 / 1024)
    lo, hi = controls.wilson_interval(8, 17)
    assert lo < 0.47 < hi and lo > 0.2 and hi < 0.75


def test_safe_ratio_refuses_tiny_denominators():
    assert controls.safe_ratio(1.0, 0.028, 0.1) != controls.safe_ratio(1.0, 0.028, 0.1)  # NaN
    assert controls.safe_ratio(1.0, 2.0, 0.1) == 0.5


def _row(ea, eb, **kw):
    r = {"effect_a": ea, "effect_b": eb, "design": "t", "a_label": "a", "b_label": "b",
         "d_missense": 0, "d_nonsense": 0, "d_transversion": 0, "d_gc": 0, "d_usage": 0.0}
    r.update(kw)
    return r


def test_summarize_paired():
    rows = [_row(-1, -2), _row(-1, -3), _row(-2, -1)]
    s = controls.summarize_paired(rows)
    assert s["n"] == 3 and s["b_more_disruptive_frac"] == pytest.approx(2 / 3)
    assert s["median_paired_diff"] == pytest.approx(-1.0)


def test_paired_regression_separates_letter_type_from_amino_acid():
    """Synthetic truth: transversions cost 2 nats, amino-acid change costs 0.
    Round 4's design (missense always paired with transversion) cannot tell
    these apart; adding matched and fourfold rows lets the regression do it."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):  # round-4 style: silent/ts vs missense/tv
        rows.append(_row(0.0, -2.0 + rng.normal(0, .3), d_missense=1, d_transversion=1))
    for _ in range(200):  # matched: silent/tv vs missense/tv
        rows.append(_row(0.0, rng.normal(0, .3), d_missense=1, d_transversion=0))
    for _ in range(200):  # fourfold: silent/ts vs silent/tv
        rows.append(_row(0.0, -2.0 + rng.normal(0, .3), d_missense=0, d_transversion=1))
    fit = controls.paired_regression(rows, n_boot=200)
    assert fit["coef"]["d_transversion"] == pytest.approx(-2.0, abs=0.1)
    lo, hi = fit["ci95"]["d_missense"]
    assert lo < 0 < hi  # no amino-acid effect, correctly
    assert "d_gc" in fit["dropped"]  # never varied, not estimated


def test_peak_blocks_ratio_bias_and_difference():
    """Block 2 has a tiny silent divergence, so the raw ratio picks it even
    though block 5 is where the arms really separate."""
    row = {"div_a": [1.0, 1.0, 1e-4, 1.0, 1.0, 1.0],
           "div_b": [1.1, 1.1, 1e-3, 1.1, 1.1, 3.0],
           "kinds": ["se", "mr", "li", "attn", "se", "mr"]}
    p = controls.peak_blocks(row)
    assert p["ratio"] == 2
    assert p["ratio_floor"] == 5 and p["diff"] == 5
    shares = controls.peak_kind_shares([row], method="diff")
    assert shares["share"]["mr"] == 1.0 and shares["base_rate"]["se"] == pytest.approx(2 / 6)


# ---------------------------------------------------------------- controls: block-0 null
def test_best_word_share_and_cutoff_sweep(hm):
    md = motifs.enumerate_block0(hm, k=4, n_top=20)
    live = controls.live_channels(md)
    shares = controls.best_word_share(md, length=2, n_top=20, live=live)
    assert shares.shape == (int(live.sum()),)
    assert ((shares > 0) & (shares <= 1)).all()
    sweep = controls.cutoff_sweep(shares, (0.2, 0.5, 0.9))
    assert sweep[0.2] >= sweep[0.5] >= sweep[0.9]
    with nullmodel.random_weights(hm, 0, seed=0):
        null = controls.best_word_share(motifs.enumerate_block0(hm, k=4, n_top=20), length=2, n_top=20)
    assert null.shape[0] == hm.hidden_size


def test_repeat_gain_kept_is_nan_on_a_tiny_baseline():
    rows = [
        {"condition": "none", "arm": "real", "gap": 100, "first_lp": -0.03, "second_lp": -0.002,
         "retrieval_gain": 0.028},
        {"condition": "-attn", "arm": "real", "gap": 100, "first_lp": -0.05, "second_lp": -0.14,
         "retrieval_gain": -0.0877},
    ]
    out = {r["condition"]: r for r in genome.summarize_repeat_test(rows)}
    assert out["-attn"]["gain_kept"] != out["-attn"]["gain_kept"]  # NaN, not -313%


# ---------------------------------------------------------------- round-5 scoring
def _summ(frac, n=150, p=0.001):
    return {"b_more_disruptive_frac": frac, "n": n, "sign_test_p": p}


def _score_inputs(**over):
    summary = {"fourfold": _summ(0.65), "noncoding": _summ(0.58), "matched": _summ(0.52, p=0.7),
               "flipped": _summ(0.40), "stop_matched": _summ(0.05), "stop_flipped": _summ(0.10)}
    summary.update({k: v for k, v in over.items() if k in summary})
    fit = over.get("fit", {"coef": {"d_missense": 0.1, "d_transversion": -1.0},
                           "ci95": {"d_missense": (-0.5, 0.6), "d_transversion": (-1.5, -0.5)}})
    se = over.get("se", {"round4": 0.55, "fourfold": 0.50, "round4_shuffled": 0.60})
    peaks = [{"design": d, "method": "diff", "se share": v, "se base": 9 / 32} for d, v in se.items()]
    peaks += [{"design": d, "method": "ratio", "se share": 0.99, "se base": 9 / 32} for d in se]
    sw = lambda *v: dict(zip((0.5, 0.6, 0.7, 0.8, 0.9), v))
    block0 = over.get("block0", {"trained": {"median": 0.9, "sweep": sw(100, 90, 80, 70, 60)},
                                 "shuffle0": {"median": 0.5, "sweep": sw(50, 40, 30, 20, 10)},
                                 "gaussian0": {"median": 0.4, "sweep": sw(40, 30, 20, 10, 5)}})
    return summary, fit, peaks, block0


def test_score_round5_letter_type_reading():
    """The review's expected world: letter type explains round 4."""
    s = controls.score_round5(*_score_inputs())
    assert s["P23"]["verdict"] == "CONFIRMED"
    assert s["P24"]["verdict"] == "CONFIRMED"
    assert s["P25"]["verdict"] == "REFUTED"
    assert s["P26"]["verdict"] == "CONFIRMED"      # only the diff rows count, not ratio's 0.99
    assert s["P27"]["verdict"] == "CONFIRMED"      # stop is arm a: 1 - 0.05 = 95%, 1 - 0.10 = 90%
    assert s["P28"]["verdict"] == "CONFIRMED"


def test_score_round5_protein_aware_reading():
    """The other world: round 4's reading survives every control."""
    fit = {"coef": {"d_missense": -2.0, "d_transversion": -0.5},
           "ci95": {"d_missense": (-3.0, -1.0), "d_transversion": (-1.0, 0.1)}}
    s = controls.score_round5(*_score_inputs(
        fourfold=_summ(0.50), noncoding=_summ(0.49), matched=_summ(0.72, p=1e-4), flipped=_summ(0.61),
        fit=fit, se={"round4": 0.85, "fourfold": 0.30, "round4_shuffled": 0.35}))
    assert s["P23"]["verdict"] == "REFUTED"
    assert s["P24"]["verdict"] == "REFUTED"
    assert s["P25"]["verdict"] == "CONFIRMED"
    assert s["P26"]["verdict"] == "REFUTED"


def test_score_round5_edges():
    s = controls.score_round5(*_score_inputs(stop_flipped=_summ(0.2, n=12)))
    assert s["P27"]["verdict"].startswith("UNTESTABLE")
    s = controls.score_round5(*_score_inputs(stop_flipped=_summ(0.35)))   # stop wins only 65%
    assert s["P27"]["verdict"] == "REFUTED"
    s = controls.score_round5(*_score_inputs(matched=_summ(0.58, p=0.2)))
    assert s["P25"]["verdict"].startswith("PARTLY")
    sw = lambda *v: dict(zip(("0.5", "0.6", "0.7", "0.8", "0.9"), v))   # JSON round-trip turns keys into str
    b0 = {"trained": {"median": 0.9, "sweep": sw(100, 90, 80, 70, 60)},
          "shuffle0": {"median": 0.87, "sweep": sw(100, 95, 30, 20, 10)}}
    assert controls.score_round5(*_score_inputs(block0=b0))["P28"]["verdict"] == "REFUTED"


# ---------------------------------------------------------------- P28's artifact
def test_best_word_share_refuses_inputs_shorter_than_the_word(hm):
    """Round 5's null came out at exactly 0 because block 0 was enumerated over
    1-2-letter inputs; a 3-letter word can never appear in those."""
    md = motifs.enumerate_block0(hm, k=2, n_top=5)
    with pytest.raises(ValueError, match="2-letter inputs"):
        controls.best_word_share(md, length=3, n_top=5)


def test_only_scrambles_one_named_tensor(hm, seq):
    names = nullmodel.tensor_names(hm, 0)
    assert len(names) == len(set(names)) > 1
    before = {n: p.detach().clone() for n, p in hm.block(0).named_parameters()}
    with nullmodel.random_weights(hm, 0, seed=0, only=names[0]) as n_changed:
        now = dict(hm.block(0).named_parameters())
        changed = [n for n in before if not torch.equal(before[n], now[n])]
    assert n_changed == 1 and changed == [names[0]]
    assert all(torch.equal(before[n], p) for n, p in hm.block(0).named_parameters())


def test_block0_sensitivity_sees_the_receptive_field(hm):
    """The trained tiny block 0 reaches back k-1 letters and no further, so
    reach[d] is clearly positive inside the field and float rounding (~1e-7)
    outside it."""
    k = motifs.receptive_field(hm)
    s = nullmodel.block0_sensitivity(hm, k=k, n=64)
    assert s["finite"] and s["rel_spread"] > 0
    assert all(r > 0.05 for r in s["reach"][:k])
    assert all(r < 1e-5 for r in s["reach"][k:])
    assert all(u == 0.0 for u in s["unchanged"][:k])


def test_rarity_split_separates_the_confound():
    """A codon-frequency confound lives in d_usage's sign: negative means the
    protein-changing arm is the rarer codon, which would favour the hypothesis
    for free."""
    def row(d_usage, b_wins):
        return {"effect_a": 0.0, "effect_b": -1.0 if b_wins else 1.0, "d_usage": d_usage,
                "design": "x", "a_label": "a", "b_label": "b"}
    rows = ([row(-1.0, True)] * 3 + [row(-1.0, False)] * 1          # favours: 3/4
            + [row(+1.0, True)] * 8 + [row(+1.0, False)] * 2        # opposes: 8/10
            + [row(0.0, True)])                                     # neutral: 1/1
    out = controls.rarity_split(rows)
    assert out["favours"]["n"] == 4 and out["favours"]["b_more_disruptive_frac"] == 0.75
    assert out["opposes"]["n"] == 10 and out["opposes"]["b_more_disruptive_frac"] == 0.8
    assert out["neutral"]["n"] == 1
    assert controls.rarity_split([])== {}


# ---------------------------------------------------------------- replicate-aware statistics
def test_holm_adjusts_and_stays_monotone():
    p = {"a": 0.001, "b": 0.04, "c": 0.5}
    adj = controls.holm(p)
    assert adj["a"] == pytest.approx(0.003)      # 3 * 0.001
    assert adj["b"] == pytest.approx(0.08)       # 2 * 0.04
    assert adj["c"] == pytest.approx(0.5)
    assert adj["a"] <= adj["b"] <= adj["c"]      # monotone by construction
    assert controls.holm({"only": 0.02})["only"] == pytest.approx(0.02)


def test_cluster_bootstrap_widens_when_sites_share_a_cluster():
    """The replicate unit matters: 40 sites in 2 genes carry far less
    information than 40 sites in 40 genes, and the interval must say so."""
    rows = [{"effect_a": 0.0, "effect_b": -1.0, "gene": f"g{i}"} for i in range(30)]
    rows += [{"effect_a": 0.0, "effect_b": 1.0, "gene": f"g{i}"} for i in range(30, 40)]
    spread = controls.cluster_bootstrap(rows, lambda r: r["gene"], n_boot=2000)
    clumped = controls.cluster_bootstrap(rows, lambda r: r["gene"][:2], n_boot=2000)
    assert spread["n_clusters"] == 40 and clumped["n_clusters"] < 40
    assert spread["estimate"] == clumped["estimate"] == pytest.approx(0.75)
    width = lambda d: d["ci95"][1] - d["ci95"][0]
    assert width(clumped) > width(spread)


def test_paired_effect_size_reports_magnitude_not_just_sign():
    rows = [{"effect_a": 0.0, "effect_b": -2.0} for _ in range(20)]
    es = controls.paired_effect_size(rows, n_boot=1000)
    assert es["median_diff_nats"] == pytest.approx(-2.0)
    assert es["ci95"][0] <= -2.0 <= es["ci95"][1]
    assert controls.paired_effect_size([]) == {"n": 0}


def test_copy_test_sites_crosses_inserts_and_offsets(hm):
    """Round 6's replicate unit: inserts and insertion sites must vary
    independently, so a bootstrap can resample either."""
    from marv_hyena import experiments
    g = _rand_dna(4000, 7)
    rows = experiments.copy_test(hm, g, gaps=(50,), insert_len=40, seeds=3, sites=2,
                                 lead=100, conditions={"none": []}, verbose=False)
    assert len(rows) == 3 * 2
    assert {r["insert_seed"] for r in rows} == {0, 1, 2}
    assert len({r["site_offset"] for r in rows}) == 2
    # each insert is seen at each site exactly once
    assert len({(r["insert_seed"], r["site_seed"]) for r in rows}) == 6
    legacy = experiments.copy_test(hm, g, gaps=(50,), insert_len=40, seeds=3,
                                   lead=100, conditions={"none": []}, verbose=False)
    assert len(legacy) == 3 and {r["site_seed"] for r in legacy} == {0, 1, 2}


def test_score_copy_refuses_an_insert_shorter_than_skip():
    """A 20-letter insert with the default skip=20 scored nothing and divided by
    zero; it now says so."""
    import torch
    from marv_hyena import probes
    probe = probes.CopyProbe("A" * 200, (0, 20), (100, 120), 80)
    with pytest.raises(ValueError, match="not longer than skip"):
        probes.score_copy(torch.zeros(1, 200, 8), torch.zeros(1, 200, dtype=torch.long), probe)


# ---------------------------------------------------------------- organelle guard (round 7 prep)
def _fake_gb(tmp_path, source_quals=None, transl_table=None, desc="Synthetic test record"):
    """A minimal GenBank file: biopython writes it, genbank_track reads it."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from Bio import SeqIO
    rec = SeqRecord(Seq("ATG" + "AAACCCGGGTTT" * 20 + "TAA"), id="TEST", name="TEST",
                    description=desc, annotations={"molecule_type": "DNA"})
    rec.features.append(SeqFeature(FeatureLocation(0, len(rec.seq)), type="source",
                                   qualifiers=source_quals or {"organism": ["Testus fakus"]}))
    q = {"gene": ["x"]}
    if transl_table is not None:
        q["transl_table"] = [str(transl_table)]
    rec.features.append(SeqFeature(FeatureLocation(0, 60, strand=1), type="CDS", qualifiers=q))
    path = tmp_path / "t.gb"
    SeqIO.write(rec, str(path), "genbank")
    return str(path)


def test_genbank_track_accepts_a_nuclear_record(tmp_path):
    path = _fake_gb(tmp_path)
    tr = probes.genbank_track(path, 0, 60)
    assert len(tr.seq) == 60


def test_genbank_track_refuses_an_organelle_record(tmp_path):
    """Yeast and vertebrate mitochondria use a different code; codons.py assumes the
    standard one, so the loader must refuse rather than mislabel substitutions."""
    path = _fake_gb(tmp_path, source_quals={"organism": ["Testus fakus"],
                                            "organelle": ["mitochondrion"]})
    with pytest.raises(ValueError, match="organelle"):
        probes.genbank_track(path, 0, 60)


def test_genbank_track_refuses_a_nonstandard_transl_table(tmp_path):
    path = _fake_gb(tmp_path, transl_table=2)        # 2 = vertebrate mitochondrial
    with pytest.raises(ValueError, match="transl_table"):
        probes.genbank_track(path, 0, 60)
    ok = _fake_gb(tmp_path, transl_table=1)          # 1 = the standard code
    assert len(probes.genbank_track(ok, 0, 60).seq) == 60


def test_genbank_track_accepts_table_11_like_e_coli(tmp_path):
    """Table 11 (bacterial) differs from the standard code only in which codons may
    initiate translation, so every codon still encodes the same amino acid. E. coli's
    own record declares 11, and refusing it would break every notebook in this repo."""
    assert len(probes.genbank_track(_fake_gb(tmp_path, transl_table=11), 0, 60).seq) == 60
    for bad in (2, 3, 4, 5):          # 2 vertebrate mito, 3 yeast mito, 4, 5 -- all reassign codons
        with pytest.raises(ValueError, match="transl_table"):
            probes.genbank_track(_fake_gb(tmp_path, transl_table=bad), 0, 60)


# ---------------------------------------------------------------- sharded-model prep (20B / 40B)
def test_device_map_reports_a_single_device_model(hm):
    """On the tiny CPU model everything is on one device, so sharded is False and
    every block is accounted for. The 20B/40B runs use this to read a device
    mismatch, so its shape matters more than its values here."""
    from marv_hyena import diagnostics
    dm = diagnostics.device_map(hm)
    assert dm["sharded"] is False and dm["n_devices"] == 1
    assert sorted(dm["blocks"]) == list(range(hm.n_blocks))
    assert sum(len(v) for v in dm["by_device"].values()) == hm.n_blocks
    assert dm["embedding"] == dm["unembed"] == dm["devices"][0]


# ---------------------------------------------------------------- model identity (20B/40B postmortem)
def test_model_fingerprint_separates_different_weights(hm):
    """The check that the 20B/40B scaling runs needed: two different
    checkpoints must not fingerprint the same."""
    from marv_hyena import diagnostics
    a = diagnostics.model_fingerprint(hm)
    again = diagnostics.model_fingerprint(hm)
    assert a["fingerprint"] == again["fingerprint"]        # deterministic
    # perturbing one weight must change it
    with torch.no_grad():
        p = next(hm.block(0).parameters())
        p.add_(1.0)
    try:
        assert diagnostics.model_fingerprint(hm)["fingerprint"] != a["fingerprint"]
    finally:
        with torch.no_grad():
            p.sub_(1.0)
    assert diagnostics.model_fingerprint(hm)["fingerprint"] == a["fingerprint"]


def test_find_dead_tail_is_quiet_on_a_working_model(hm, seq):
    from marv_hyena import diagnostics
    d = diagnostics.find_dead_tail(hm, seq)
    assert d["largest_write"] > 0
    assert d["suspicious"] is False, d
    assert d["dead_tail"] <= max(2, hm.n_blocks // 10)
