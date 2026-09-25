"""Round 5: substitution-matched designs and the controls behind them.

Following the tiny_hyena rule, the biology these tests check (which letter
changes are transitions, which codon swaps are silent) is written out by hand
here rather than derived from the module under test.
"""
import random

import numpy as np
import pytest

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
