"""Checks that can explain a finding away. Round 5.

A review of round 4 found three claims that looked solid and were not, and in
each case the fix was not new machinery but a control that should have been
there from the start:

- P21 ("the model knows amino acids") was mostly a letter-type effect: the
  silent arm was nearly always a transition and the missense arm a
  transversion. -> compare arms matched on substitution type (`codons.paired_sites`)
  and fit the paired difference against every candidate explanation at once
  (`paired_regression`).
- "90.8% of sites peak in SE blocks" picked, per site, the block with the
  largest missense/silent RATIO. Ratios explode where the denominator is small,
  and the maximum of 32 noisy ratios finds exactly those blocks.
  -> rank blocks by DIFFERENCE, or by ratio only where the denominator is not
  tiny (`peak_blocks`), and compare against a design with no amino-acid change.
- "Block 0 is learned: trained 46 detectors per word, shuffled 0" used one hard
  cutoff (>= 80% of a channel's top-50 inputs share the word). A cutoff can
  turn "less selective" into "zero". -> look at the whole distribution and
  sweep the cutoff (`best_word_share`, `cutoff_sweep`).

Nothing here is specific to Hyena or to Evo 2: the statistics take plain rows
and arrays. Only the inputs (per-block divergences, block-0 enumerations) come
from the model.
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ small stats
def sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes in n fair coin flips."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion that behaves at small n (unlike p +- 2se)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def safe_ratio(num: float, den: float, min_den: float) -> float:
    """num/den, or NaN when |den| is too small for the ratio to mean anything.
    Round 1 (FUNC variant, +-200%) and round 4 (P20: -attn 'keeps -313%' of a
    0.028-nat gain) both reported ratios over near-zero denominators."""
    return num / den if abs(den) >= min_den else float("nan")


# ------------------------------------------------------------------ paired designs
def summarize_paired(rows: list[dict]) -> dict:
    """Per design: how often arm b disturbs the model more than arm a at the
    same site (effects are negative when disruptive, so 'more' = lower).
    The sign test is paired and threshold-free; it is the primary statistic."""
    if not rows:
        return {"n": 0}
    a = np.array([r["effect_a"] for r in rows], float)
    b = np.array([r["effect_b"] for r in rows], float)
    k, n = int((b < a).sum()), len(rows)
    lo, hi = wilson_interval(k, n)
    return {
        "design": rows[0].get("design"),
        "a": rows[0].get("a_label"), "b": rows[0].get("b_label"),
        "n": n,
        "b_more_disruptive_frac": k / n,
        "ci95": (lo, hi),
        "sign_test_p": sign_test(k, n),
        "median_effect_a": float(np.median(a)),
        "median_effect_b": float(np.median(b)),
        "median_paired_diff": float(np.median(b - a)),
    }


def stratify(rows: list[dict], key) -> dict:
    """summarize_paired within each group of `key(row)`."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {g: summarize_paired(v) for g, v in sorted(groups.items(), key=lambda t: str(t[0]))}


COVARIATES = ("d_missense", "d_nonsense", "d_transversion", "d_gc", "d_usage")


def paired_regression(rows: list[dict], covariates=COVARIATES, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Regress the within-site difference (effect_b - effect_a) on the
    differences in each candidate explanation, pooled over designs.

    Because both arms share the site, everything about the site cancels. What
    is left is: did the amino acid change, did the protein end, was it a
    transversion, did G/C content change, did the codon get rarer. A negative
    coefficient means 'this makes the change more disruptive'. Bootstrap over
    sites gives the interval. Covariates that never vary in `rows` are dropped
    (they cannot be estimated).
    """
    if not rows:
        return {"n": 0}
    y = np.array([r["effect_b"] - r["effect_a"] for r in rows], float)
    cols = [c for c in covariates if len({r[c] for r in rows}) > 1]
    X = np.column_stack([np.ones(len(rows))] + [np.array([r[c] for r in rows], float) for c in cols])
    names = ["intercept"] + cols
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        bb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots.append(bb)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {
        "n": len(rows),
        "coef": {n: float(v) for n, v in zip(names, beta)},
        "ci95": {n: (float(a), float(b)) for n, a, b in zip(names, lo, hi)},
        "dropped": [c for c in covariates if c not in cols],
    }


# ------------------------------------------------------------------ where arms diverge
def peak_blocks(row: dict, floor: float = 0.05) -> dict:
    """Which block separates arm b from arm a most, three ways.

    - ratio:       argmax b/a over all blocks (round 4's statistic, biased toward
                   blocks where a is tiny)
    - ratio_floor: argmax b/a over blocks where a is at least `floor` x its max
    - diff:        argmax (b - a), no denominator at all
    """
    a = np.asarray(row["div_a"], float)
    b = np.asarray(row["div_b"], float)
    ok = a > 1e-9
    out = {}
    out["ratio"] = int(np.argmax(np.where(ok, b / np.where(ok, a, 1), -np.inf))) if ok.any() else None
    big = a >= floor * a.max() if a.max() > 0 else ok
    out["ratio_floor"] = int(np.argmax(np.where(big, b / np.where(big, a, 1), -np.inf))) if big.any() else None
    out["diff"] = int(np.argmax(b - a))
    return out


def peak_kind_shares(rows: list[dict], method: str = "diff", floor: float = 0.05) -> dict:
    """Share of sites whose peak block (by `method`) is each operator kind,
    next to that kind's share of blocks. A kind that wins far above its base
    rate under 'diff' AND under a no-amino-acid control is about letters, not
    the genetic code."""
    if not rows:
        return {}
    kinds = rows[0]["kinds"]
    base = {k: kinds.count(k) / len(kinds) for k in set(kinds)}
    peaks = [peak_blocks(r, floor)[method] for r in rows]
    peaks = [p for p in peaks if p is not None]
    share = {k: sum(kinds[p] == k for p in peaks) / len(peaks) for k in base} if peaks else {}
    return {"method": method, "n": len(peaks), "share": share, "base_rate": base,
            "modal_block": max(set(peaks), key=peaks.count) if peaks else None}


# ------------------------------------------------------------------ block-0 null, no cutoff
def best_word_share(md, length: int = 3, n_top: int = 50, live: np.ndarray | None = None) -> np.ndarray:
    """For each channel: the largest fraction of its top-`n_top` inputs that
    contain any single `length`-letter word. 1.0 = every top input shares one
    word; ~0.3 = no word stands out. `live` (bool per channel) drops dead ones.

    This is the quantity the >= 80% cutoff thresholds, shown whole, so a
    trained-vs-null comparison does not depend on where the cutoff sits.
    """
    import itertools

    words = ["".join(p) for p in itertools.product("ACGT", repeat=length)]
    out = []
    for c, kmers in enumerate(md.top_kmers):
        if live is not None and not live[c]:
            continue
        top = kmers[:n_top]
        out.append(max(sum(w in s for s in top) for w in words) / len(top))
    return np.array(out)


def live_channels(md, tol: float = 1e-6) -> np.ndarray:
    """Channels whose output varies at all over the full enumeration. Round 3
    found 724 dead ones whose 'top inputs' are arbitrary orderings of ties."""
    return md.position_importance.sum(0) >= tol


def cutoff_sweep(shares: np.ndarray, cutoffs=(0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    """How many channels clear each cutoff. If the trained/null gap only exists
    at 0.8 and vanishes at 0.6, the 'zero detectors' result was the cutoff."""
    return {float(c): int((shares >= c).sum()) for c in cutoffs}
