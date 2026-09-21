"""The shared experiment functions (used by scripts + Colab) run end to end."""
import random

import numpy as np

import marv_hyena as mh
from marv_hyena.checks import run_smoke_checks
from marv_hyena.experiments import codon_test, context_test, copy_test, summarize_copy


def _genome(n=6000, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_smoke_checks_pass_on_tiny(hm, seq):
    assert run_smoke_checks(hm, seq, tol=1e-4, verbose=False)


def test_copy_test_rows_and_summary(hm):
    rows = copy_test(hm, _genome(), gaps=(50, 500), insert_len=30, seeds=2, per_block_kinds=("attn",),
                     lead=100, verbose=False)
    gaps, conds, a2, a1 = summarize_copy(rows)
    assert gaps == [50, 500]
    assert conds[:5] == ["none", "-se", "-mr", "-li", "-attn"] and "-L3(attn)" in conds
    assert a2.shape == (2, len(conds)) and np.all((0 <= a2) & (a2 <= 1))


def test_codon_and_context_tests(hm, seq):
    n = len(seq)
    phase = np.full(n, -1, dtype=np.int8)
    phase[30:240] = np.arange(210) % 3
    feat = np.full(n, "", dtype=object)
    feat[30:240] = "CDS"
    rows = codon_test(hm, mh.Track(0, seq, phase, feat, np.zeros(n, dtype=np.int8)))
    assert {r["condition"] for r in rows} == {"none", "-se", "-mr", "-li", "-attn"}
    ctx = context_test(hm, _genome(3000), target=(2500, 2700), short=100)
    assert len(ctx) == 5 and all(r["full_context_letters"] == 2500 for r in ctx)
