"""Round 4: null model, real-repeat probes, translation test.

The genetic-code test follows the tiny_hyena rule: the reference table is
written out independently (NCBI table 1 in its usual TCAG order) rather than
re-derived from the same expression the module uses. That is what caught the
TGT/TGG swap in the first draft.
"""
import random

import pytest
import torch

from marv_hyena import codons, genome, motifs, nullmodel, probes

# NCBI translation table 1, written in the conventional TCAG codon order.
_B1 = "TTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGG"
_B2 = "TTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGGTTTTCCCCAAAAGGGG"
_B3 = "TCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
REFERENCE_CODE = {_B1[i] + _B2[i] + _B3[i]: _AA[i] for i in range(64)}


def _rand_dna(n: int, seed: int) -> str:
    """One rng for the whole string. (Calling random.Random(seed) inside the
    comprehension re-seeds every letter and yields a constant sequence.)"""
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# ---------------------------------------------------------------- genetic code
def test_genetic_code_matches_ncbi_table_1():
    assert codons.GENETIC_CODE == REFERENCE_CODE


def test_stops_and_degeneracy():
    assert set(codons.STOPS) == {"TAA", "TAG", "TGA"}
    assert len(codons.AA_TO_CODONS["L"]) == 6
    assert len(codons.AA_TO_CODONS["M"]) == 1
    assert codons.translate("ATGAAATAG") == "MK*"
    assert "TTG" in codons.synonyms("TTA") and "TTA" not in codons.synonyms("TTA")


def _cds_track(seq: str) -> probes.Track:
    """A Track whose whole length is a forward-strand CDS in frame 0."""
    import numpy as np
    n = len(seq)
    return probes.Track(0, seq, np.arange(n, dtype=np.int8) % 3,
                        np.full(n, "CDS", dtype=object), np.ones(n, dtype=np.int8))


def test_wobble_sites_are_valid():
    seq = _rand_dna(3000, 7)
    sites = codons.wobble_sites(_cds_track(seq), min_spacing=3)
    assert sites, "no wobble sites found in 1000 random codons"
    for s in sites:
        assert seq[s.codon_start:s.pos + 1] == s.codon
        assert s.codon[2] == s.ref
        # both alternatives are single-letter changes at the third position
        assert s.syn_codon == s.codon[:2] + s.syn_alt
        assert s.nonsyn_codon == s.codon[:2] + s.nonsyn_alt
        assert s.syn_alt != s.ref and s.nonsyn_alt != s.ref
        # and they do what their names say
        assert codons.GENETIC_CODE[s.syn_codon] == s.aa
        assert codons.GENETIC_CODE[s.nonsyn_codon] != s.aa
        assert s.nonsyn_aa != "*", "allow_stop=False must exclude nonsense"
        assert s.aa != "*"


def test_wobble_sites_allow_stop_opt_in():
    seq = _rand_dna(6000, 11)
    with_stop = codons.wobble_sites(_cds_track(seq), allow_stop=True, min_spacing=3)
    without = codons.wobble_sites(_cds_track(seq), allow_stop=False, min_spacing=3)
    assert len(with_stop) >= len(without)
    assert any(s.is_nonsense for s in with_stop)
    assert not any(s.is_nonsense for s in without)


def test_wobble_sites_respect_spacing():
    seq = _rand_dna(4000, 3)
    sites = codons.wobble_sites(_cds_track(seq), min_spacing=60)
    pos = [s.pos for s in sites]
    assert all(b - a >= 60 for a, b in zip(pos, pos[1:]))


# ---------------------------------------------------------------- null model
def test_random_weights_restores_exactly(hm, seq):
    before = hm.logits(seq).clone()
    with nullmodel.random_weights(hm, 0, seed=0) as n:
        assert n > 0
    after = hm.logits(seq)
    assert torch.equal(before, after), "weights were not restored bit-for-bit"
    assert nullmodel.verify_restored(hm, seq, before)["restored"]


def test_random_weights_restores_after_an_exception(hm, seq):
    before = hm.logits(seq).clone()
    with pytest.raises(RuntimeError):
        with nullmodel.random_weights(hm, 0, seed=0):
            raise RuntimeError("boom")
    assert torch.equal(before, hm.logits(seq))


def test_random_weights_changes_the_model_while_active(hm, seq):
    before = hm.logits(seq).clone()
    with nullmodel.random_weights(hm, 0, seed=0):
        during = hm.logits(seq).clone()
    assert not torch.allclose(before, during), "scrambling block 0 changed nothing"


def test_shuffle_preserves_the_weight_multiset(hm):
    w = hm.block(0).out_filter_dense.weight
    before = w.detach().flatten().sort().values.clone()
    with nullmodel.random_weights(hm, 0, mode="shuffle", seed=1):
        during = w.detach().flatten().sort().values.clone()
    assert torch.equal(before, during), "shuffle must permute, not resample"


def test_gaussian_mode_does_not_preserve_the_multiset(hm):
    w = hm.block(0).out_filter_dense.weight
    before = w.detach().flatten().sort().values.clone()
    with nullmodel.random_weights(hm, 0, mode="gaussian", seed=1):
        during = w.detach().flatten().sort().values.clone()
    assert not torch.allclose(before, during)


def test_random_weights_rejects_unknown_mode(hm):
    with pytest.raises(ValueError):
        with nullmodel.random_weights(hm, 0, mode="nonsense"):
            pass


def test_word_count_summary_flags_a_flat_distribution():
    flat = nullmodel.word_count_summary([{"controlled": 46} for _ in range(64)])
    peaked = nullmodel.word_count_summary([{"controlled": 46} for _ in range(63)] + [{"controlled": 400}])
    assert flat["cv"] == pytest.approx(0.0)
    assert peaked["cv"] > flat["cv"]
    assert peaked["max_over_median"] > flat["max_over_median"]


# ---------------------------------------------------------------- real repeats
def test_copy_probe_accepts_a_given_insert():
    bg = _rand_dna(3000, 2)
    ins = "ACGTACGTAC" * 5
    p = probes.copy_probe(bg, gap=200, lead=300, insert=ins)
    assert p.seq[p.first[0]:p.first[1]] == ins
    assert p.seq[p.second[0]:p.second[1]] == ins
    assert p.second[0] - p.first[1] == 200


def test_copy_probe_without_insert_is_unchanged():
    bg = _rand_dna(3000, 2)
    a = probes.copy_probe(bg, insert_len=50, gap=200, lead=300, seed=5)
    b = probes.copy_probe(bg, insert_len=50, gap=200, lead=300, seed=5, insert=None)
    assert a.seq == b.seq and a.first == b.first


def test_sequence_helpers():
    assert genome.reverse_complement("ACGTT") == "AACGT"
    assert genome.gc_content("GGCC") == 1.0 and genome.gc_content("ATAT") == 0.0
    s = "AACCGGTT"
    sh = genome.shuffle_letters(s, random.Random(0))
    assert sorted(sh) == sorted(s) and len(sh) == len(s)


def test_repeat_probes_share_everything_but_the_insert():
    bg = _rand_dna(6000, 4)
    fam = genome.RepeatFamily(
        "test", "rRNA",
        [genome.RepeatCopy(0, 300, 1, _rand_dna(300, 8))],
        kmer_similarity=1.0)
    rps = genome.repeat_probes(bg, fam, gaps=(500,), insert_len=100, lead=400, seed=0)
    arms = {r.arm: r for r in rps}
    assert set(arms) == {"real", "shuffled", "random"}
    real, shuf = arms["real"].probe, arms["shuffled"].probe
    r_ins = real.seq[real.first[0]:real.first[1]]
    s_ins = shuf.seq[shuf.first[0]:shuf.first[1]]
    assert r_ins == fam.copies[0].seq[:100]
    assert sorted(r_ins) == sorted(s_ins), "shuffled arm must match composition"
    assert r_ins != s_ins
    # same geometry in every arm
    assert {r.probe.first for r in rps} == {real.first}
    assert {r.probe.second for r in rps} == {real.second}


def test_summarize_repeat_test_reports_gain_kept():
    rows = [
        {"condition": "none", "arm": "real", "gap": 100, "first_lp": -1.0, "second_lp": 0.0,
         "retrieval_gain": 1.0},
        {"condition": "-attn", "arm": "real", "gap": 100, "first_lp": -1.0, "second_lp": -0.75,
         "retrieval_gain": 0.25},
    ]
    out = {r["condition"]: r for r in genome.summarize_repeat_test(rows)}
    assert out["none"]["gain_kept"] == pytest.approx(1.0)
    assert out["-attn"]["gain_kept"] == pytest.approx(0.25)


# ---------------------------------------------------------------- translation
def test_divergence_by_block_is_zero_for_an_identical_arm(hm, seq):
    rows = codons.divergence_by_block(hm, seq, 100, {"same": seq}, span=8)
    assert rows
    assert all(r["abs_divergence"] == pytest.approx(0.0, abs=1e-5) for r in rows)


def test_divergence_by_block_covers_every_component(hm, seq):
    alt = seq[:100] + ("A" if seq[100] != "A" else "C") + seq[101:]
    rows = codons.divergence_by_block(hm, seq, 100, {"alt": alt}, span=8)
    assert {(r["block"], r["part"]) for r in rows} == set(hm.all_components())
    assert any(r["abs_divergence"] > 0 for r in rows)


def test_translation_test_runs_end_to_end(hm, seq):
    sites = codons.wobble_sites(_cds_track(seq), min_spacing=40)[:3]
    if not sites:
        pytest.skip("no wobble sites in this fixture sequence")
    rows = codons.translation_test(hm, seq, sites, window=256, span=8, downstream_span=20)
    assert rows
    for r in rows:
        assert r["codon"] in codons.GENETIC_CODE
        assert codons.GENETIC_CODE[r["nonsyn_codon"]] != r["aa"]
        assert codons.GENETIC_CODE[r["syn_codon"]] == r["aa"]
    s = codons.summarize_translation(rows)
    assert s["n"] == len(rows)
    assert 0.0 <= s["nonsyn_more_disruptive_frac"] <= 1.0


def test_enumerate_block0_still_works_under_a_null(hm):
    """The null-model comparison only means anything if the same enumeration
    runs unchanged on scrambled weights."""
    k = 4
    trained = motifs.enumerate_block0(hm, k=k, n_top=5)
    with nullmodel.random_weights(hm, 0, seed=0):
        null = motifs.enumerate_block0(hm, k=k, n_top=5)
    assert trained.top_vals.shape == null.top_vals.shape
    assert trained.top_kmers != null.top_kmers


def test_translation_ratio_is_gated_on_a_usable_silent_effect():
    """A ratio of two signed quantities that straddle zero is not a statistic;
    round 1's FUNC variant taught this. Only clearly-disruptive silent arms
    may anchor one."""
    rows = [
        {"syn_effect": -2.0, "nonsyn_effect": -4.0, "usable": True, "effect_ratio": 2.0,
         "peak_divergence_block": 9},
        # silent arm barely moved: ratio must not count
        {"syn_effect": -0.01, "nonsyn_effect": -3.0, "usable": False, "effect_ratio": float("nan"),
         "peak_divergence_block": 9},
    ]
    s = codons.summarize_translation(rows)
    assert s["n"] == 2 and s["n_usable"] == 1
    assert s["median_effect_ratio"] == pytest.approx(2.0)
    # the paired sign test still uses both sites
    assert s["nonsyn_more_disruptive_frac"] == pytest.approx(1.0)


def test_translation_test_marks_unusable_rows(hm, seq):
    sites = codons.wobble_sites(_cds_track(seq), min_spacing=40)[:3]
    if not sites:
        pytest.skip("no wobble sites in this fixture sequence")
    rows = codons.translation_test(hm, seq, sites, window=256, span=8, downstream_span=20)
    for r in rows:
        assert r["usable"] == (r["syn_effect"] < -codons.MIN_EFFECT)
        if not r["usable"]:
            assert r["effect_ratio"] != r["effect_ratio"]  # NaN


# ---------------------------------------------------------------- repeat finding
def test_family_key_strips_per_copy_suffixes():
    """E. coli names every IS copy separately. Grouping on the raw name put each
    copy in a group of one and found zero mobile-element families in the real
    genome."""
    assert genome._family_key("insertion sequence:IS1A") == "IS1"
    assert genome._family_key("insertion sequence:IS1I") == "IS1"
    assert genome._family_key("insertion sequence:IS186A") == "IS186"
    assert genome._family_key("insertion sequence:IS911A-1") == "IS911"
    assert genome._family_key("insertion sequence:IS30B") == "IS30"
    # a name whose trailing letter is not a copy suffix must survive intact:
    # stripping ISX's X would leave a bare "IS" and merge unrelated elements
    assert genome._family_key("insertion sequence:ISX") == "ISX"
    # names without the "type:" prefix still work
    assert genome._family_key("16S ribosomal RNA") == "16S ribosomal RNA"


def test_kmer_similarity_survives_an_indel():
    """The seven real 23S rRNA copies differ in length by one base. A
    prefix-by-prefix comparison scored them 0.87 and rejected the best repeat
    family in the genome."""
    a = _rand_dna(600, 21)
    b = a[:300] + "G" + a[300:]          # one insertion, mid-sequence
    assert len(b) == len(a) + 1
    prefix_identity = sum(x == y for x, y in zip(a, b)) / len(a)
    # The first half still lines up and the shifted half agrees ~1/4 of the time
    # by chance, so prefix comparison lands near (300 + 0.25*300)/600 = 0.625 --
    # under the 0.90 threshold find_repeat_families applies, which is exactly how
    # the real 23S family got thrown away.
    assert 0.5 < prefix_identity < 0.7, "the fixture must actually break prefix alignment"
    assert genome.kmer_similarity(a, b, k=16) > 0.90


def test_kmer_similarity_bounds():
    a = _rand_dna(500, 22)
    assert genome.kmer_similarity(a, a) == pytest.approx(1.0)
    assert genome.kmer_similarity(a, _rand_dna(500, 23)) < 0.1
    # shorter than k falls back to containment
    assert genome.kmer_similarity("ACGT", "TTACGTTT", k=16) == 1.0
    assert genome.kmer_similarity("AAAA", "CCCCCCCC", k=16) == 0.0
