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

    if md.k < length:
        # Round 5's null: enumerate_block0 re-measured the receptive field on
        # scrambled weights, got 1-2 letters, and every share came out 0.
        raise ValueError(f"enumeration used {md.k}-letter inputs; cannot count {length}-letter words. "
                         "Pass the trained model's k to enumerate_block0.")
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


# ------------------------------------------------------------------ round-5 scoring
def _verdict(confirmed: bool, refuted: bool, between: str = "IN BETWEEN") -> str:
    return "CONFIRMED" if confirmed else "REFUTED" if refuted else between


def score_round5(summary: dict, fit: dict, peaks: list[dict], block0: dict) -> dict:
    """Apply the thresholds registered for P23-P28 in PREDICTIONS.md, mechanically.

    Inputs are the notebook's SUMMARY (summarize_paired per design), the
    paired_regression result, the peak-share rows ({design, method, 'se share'})
    and the block-0 dict ({name: {'median', 'sweep'}}). The output is a reading
    aid, not the record: outcomes are written into PREDICTIONS.md by hand,
    against the prediction text.
    """
    out = {}
    frac = lambda d: summary[d]["b_more_disruptive_frac"]

    ff, nc = frac("fourfold"), frac("noncoding")
    out["P23"] = {"verdict": _verdict(ff >= 0.60, ff <= 0.55),
                  "observed": f"fourfold transversion more disruptive {ff:.1%} (n={summary['fourfold']['n']})",
                  "rule": "confirmed >= 60%; refuted <= 55%"}
    out["P24"] = {"verdict": _verdict(nc >= 0.55 and nc < ff, nc <= 0.50 or nc > ff + 0.10),
                  "observed": f"noncoding {nc:.1%} (n={summary['noncoding']['n']}) vs fourfold {ff:.1%}",
                  "rule": "confirmed >= 55% and below fourfold; refuted <= 50% or > fourfold + 10 pts"}

    m, fl = summary["matched"], summary["flipped"]
    lo, hi = fit["ci95"].get("d_missense", (float("nan"), float("nan")))
    has_tv = "d_transversion" in fit["coef"]
    out["P25"] = {
        "verdict": _verdict(m["b_more_disruptive_frac"] >= 0.60 and m["sign_test_p"] < 0.05
                            and fl["b_more_disruptive_frac"] >= 0.50 and hi < 0 and has_tv,
                            m["b_more_disruptive_frac"] <= 0.55 and lo <= 0 <= hi,
                            "PARTLY (letter type explains part)"),
        "observed": (f"matched {m['b_more_disruptive_frac']:.1%} (n={m['n']}, p={m['sign_test_p']:.3g}); "
                     f"flipped {fl['b_more_disruptive_frac']:.1%} (n={fl['n']}); "
                     f"d_missense 95% CI {lo:+.3f}..{hi:+.3f}; d_transversion in model: {has_tv}"),
        "rule": "confirmed: matched >= 60% & p < .05, flipped >= 50%, d_missense CI < 0; "
                "refuted: matched <= 55% and d_missense CI includes 0"}

    se = {r["design"]: r["se share"] for r in peaks if r["method"] == "diff"}
    r4, f4, sh = se["round4"], se["fourfold"], se["round4_shuffled"]
    base = next(r.get("se base", 9 / 32) for r in peaks if r["design"] == "round4")
    out["P26"] = {"verdict": _verdict(r4 < 0.60 and abs(f4 - r4) <= 0.20,
                                      r4 >= 0.80 and f4 <= 0.50 and sh <= base + 0.15),
                  "observed": f"SE share of diff-peaks: round4 {r4:.1%}, fourfold {f4:.1%}, shuffled {sh:.1%} "
                              f"(SE base rate {base:.1%})",
                  "rule": "confirmed: round4 < 60% and fourfold within 20 pts; "
                          "refuted (finding 19 survives): round4 >= 80%, fourfold <= 50%, shuffled <= base + 15"}

    sm, sf = summary["stop_matched"], summary["stop_flipped"]
    # arm a is the stop, so "stop more disruptive" is 1 - b_more_disruptive_frac
    s1, s2 = 1 - sm["b_more_disruptive_frac"], 1 - sf["b_more_disruptive_frac"]
    untestable = min(sm["n"], sf["n"]) < 30
    out["P27"] = {"verdict": "UNTESTABLE (< 30 sites)" if untestable else
                  _verdict(s1 >= 0.90 and s2 >= 0.85, s1 < 0.70 or s2 < 0.70),
                  "observed": f"stop more disruptive: matched {s1:.1%} (n={sm['n']}), flipped {s2:.1%} (n={sf['n']})",
                  "rule": "confirmed: matched >= 90%, flipped >= 85%; refuted: either < 70%; untestable: n < 30"}

    sweep = {k: {float(c): v for c, v in d["sweep"].items()} for k, d in block0.items()}
    nulls = [k for k in block0 if k != "trained"]
    shuf = [k for k in nulls if k.startswith("shuffle")]
    gap = block0["trained"]["median"] - float(np.mean([block0[k]["median"] for k in shuf]))
    beats_all = all(sweep["trained"][c] > sweep[k][c] for k in nulls for c in sweep["trained"])
    tie_at_06 = any(sweep[k][0.6] >= sweep["trained"][0.6] for k in nulls)
    out["P28"] = {"verdict": _verdict(gap >= 0.20 and beats_all, gap < 0.05 or tie_at_06),
                  "observed": f"median gap {gap:+.3f}; trained beats every null at every cutoff: {beats_all}; "
                              f"a null matches trained at 0.6: {tie_at_06}",
                  "rule": "confirmed: gap >= 0.20 and beats every null everywhere; "
                          "refuted: gap < 0.05 or any null >= trained at 0.6"}
    return out


def rarity_split(rows: list[dict]) -> dict:
    """Split paired rows by whether codon rarity pushes with or against the
    hypothesis that arm b (the one that changes the protein) disturbs more.

    `d_usage` is log(usage[b_codon] / usage[a_codon]), so:
      d_usage < 0  arm b lands on the rarer codon -> rarity FAVOURS the hypothesis
      d_usage > 0  arm a lands on the rarer codon -> rarity OPPOSES it
      d_usage == 0 neither (no usage table, or equal frequencies)

    A result that holds on the 'opposes' half is not a codon-frequency artifact.
    Round 5's `flipped` design at isoleucine ATA is the one place where rarity
    favours the hypothesis, because ATA is rare and the silent arm leaves it.
    """
    groups = {"favours": [], "opposes": [], "neutral": []}
    for r in rows:
        d = r.get("d_usage", 0.0)
        groups["favours" if d < 0 else "opposes" if d > 0 else "neutral"].append(r)
    return {k: summarize_paired(v) for k, v in groups.items() if v}


# ------------------------------------------------------------------ replicate-aware statistics
def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, family-wise. Round 5 reported seven
    designs' sign tests uncorrected; with seven tests at 0.05 the chance of one
    false positive is about 30%."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cluster_bootstrap(rows: list[dict], cluster_key, stat=None, n_boot: int = 10000,
                      seed: int = 0) -> dict:
    """Bootstrap a paired statistic by resampling CLUSTERS, not rows.

    The replicate unit is what you resample. Resampling sites treats two sites
    in one gene as independent; resampling genes does not. `cluster_key(row)`
    returns the cluster id (a gene, a sequence family, an insert). `stat` maps a
    list of rows to a number, and defaults to the share of sites where arm b is
    more disruptive.

    Returns the point estimate, the percentile interval, and both counts, so a
    reader can see how much clustering there was.
    """
    if not rows:
        return {"n": 0}
    if stat is None:
        def stat(rs):
            return float(np.mean([r["effect_b"] < r["effect_a"] for r in rs]))
    groups: dict = {}
    for r in rows:
        groups.setdefault(cluster_key(r), []).append(r)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for i in pick for r in groups[keys[i]]]
        draws.append(stat(sample))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_rows": len(rows), "n_clusters": len(keys),
            "estimate": stat(rows), "ci95": (float(lo), float(hi)),
            "boot_sd": float(np.std(draws, ddof=1))}


def paired_effect_size(rows: list[dict], n_boot: int = 10000, seed: int = 0,
                       cluster_key=None) -> dict:
    """Median within-site difference (effect_b - effect_a) in nats, with a
    bootstrap interval. A sign test says whether b wins more often; this says by
    how much, which is what a reader needs beside it. Negative = arm b is more
    disruptive. `cluster_key` resamples clusters instead of rows."""
    if not rows:
        return {"n": 0}
    d = np.array([r["effect_b"] - r["effect_a"] for r in rows], float)
    rng = np.random.default_rng(seed)
    if cluster_key is None:
        draws = [float(np.median(rng.choice(d, len(d), replace=True))) for _ in range(n_boot)]
    else:
        groups: dict = {}
        for r, x in zip(rows, d):
            groups.setdefault(cluster_key(r), []).append(x)
        keys = list(groups)
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            draws.append(float(np.median([x for i in pick for x in groups[keys[i]]])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n": len(d), "median_diff_nats": float(np.median(d)),
            "ci95": (float(lo), float(hi)),
            "n_clusters": None if cluster_key is None else len(keys)}
