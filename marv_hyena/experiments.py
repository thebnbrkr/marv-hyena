"""The PREDICTIONS.md experiments as functions over a loaded model, so the
scripts and the Colab notebooks share one implementation. Each returns plain
rows (list of dicts) ready for pandas / JSON.

Round 2 changes (RESEARCH_LOG.md): every experiment takes explicit ablation
`conditions` (build them with make_conditions, which can exclude bottleneck
blocks and add single-layer ablations), and every row can carry a `health`
check so a broken model is never read as a result.
"""
from __future__ import annotations

import random

import numpy as np

from .arch import KINDS, HyenaModel
from .diagnostics import health
from .intervene import mean_ablate, mean_writes, mixers_of
from .probes import Track, codon_phase_accuracy, copy_probe, score_copy, truncation_curve


def make_conditions(hm: HyenaModel, families=KINDS, exclude_blocks=(), single_blocks=(),
                    include_none: bool = True) -> dict[str, list]:
    """name -> list of (block, 'mixer') to mean-ablate.
    families:       '-li' = every LI mixer (minus exclude_blocks; the name notes the exclusion)
    single_blocks:  '-L7(se)' = just that block's mixer
    """
    excl = set(exclude_blocks)
    suffix = f" (keep L{','.join(map(str, sorted(excl)))})" if excl else ""
    conds: dict[str, list] = {"none": []} if include_none else {}
    for k in families:
        comps = [c for c in mixers_of(hm, k) if c[0] not in excl]
        if comps:
            conds[f"-{k}" + (suffix if any(b in excl for b in hm.blocks_of(k)) else "")] = comps
    for b in single_blocks:
        conds[f"-L{b}({hm.kind(b)})"] = [(b, "mixer")]
    return conds


def _default_conditions(hm, conditions, per_block_kinds=()):
    if conditions is not None:
        return conditions
    return make_conditions(hm, single_blocks=[b for k in per_block_kinds for b in hm.blocks_of(k)])


def _run(hm, ids, comps, fn):
    if comps:
        with mean_ablate(hm, mean_writes(hm, ids, comps)):
            return fn()
    return fn()


def _health(hm, health_seq, comps):
    if health_seq is None:
        return {}
    return _run(hm, hm.ids(health_seq), comps, lambda: health(hm, health_seq))


def copy_test(hm: HyenaModel, genome: str, gaps=(100, 1000, 10000), insert_len: int = 200, seeds: int = 3,
              per_block_kinds=(), lead: int = 1000, conditions: dict | None = None,
              health_seq: str | None = None, verbose: bool = True) -> list[dict]:
    """P1. Score both copies of a repeated random insert under each condition.
    Means for mean-ablation come from the probe sequence itself."""
    conds = _default_conditions(hm, conditions, per_block_kinds)
    hrows = {name: _health(hm, health_seq, comps) for name, comps in conds.items()}
    rows = []
    for gap in gaps:
        for seed in range(seeds):
            rng = random.Random(seed)
            need = lead + gap + 50
            off = rng.randrange(0, len(genome) - need)
            probe = copy_probe(genome[off:off + need], insert_len=insert_len, gap=gap, lead=lead, seed=seed)
            ids = hm.ids(probe.seq)
            for name, comps in conds.items():
                s = _run(hm, ids, comps, lambda: score_copy(hm.logits(ids), ids, probe))
                rows.append({"gap": gap, "seed": seed, "condition": name, **s, **hrows[name]})
            if verbose:
                print(f"copy test: gap={gap} seed={seed} done")
    return rows


def codon_test(hm: HyenaModel, track: Track, conditions: dict | None = None) -> list[dict]:
    """P3. Accuracy by codon position (and intergenic) per condition. The
    genome window itself is the health check: 'all' = accuracy over every letter."""
    conds = _default_conditions(hm, conditions)
    ids = hm.ids(track.seq)
    rows = []
    for name, comps in conds.items():
        out = _run(hm, ids, comps, lambda: codon_phase_accuracy(hm, track))
        h = _run(hm, ids, comps, lambda: health(hm, track.seq))
        for region, v in out.items():
            rows.append({"condition": name, "region": region, **v, **h})
    return rows


def periodicity(rows: list[dict]) -> dict[str, float]:
    """Per condition: mean accuracy at codon positions 1-2 minus position 3
    (the wobble position). Round 1 full model: ~0.25."""
    out = {}
    for name in dict.fromkeys(r["condition"] for r in rows):
        acc = {r["region"]: r["acc"] for r in rows if r["condition"] == name}
        out[name] = (acc["codon_pos1"] + acc["codon_pos2"]) / 2 - acc["codon_pos3"]
    return out


def context_test(hm: HyenaModel, seq: str, target: tuple[int, int], short: int = 500,
                 conditions: dict | None = None) -> list[dict]:
    """P2. Context benefit = mean log-p(target | all upstream) - mean
    log-p(target | `short` letters upstream), per condition. The ablation
    means come from the full-length sequence."""
    conds = _default_conditions(hm, conditions)
    full_ids = hm.ids(seq[: target[1]])
    rows = []
    for name, comps in conds.items():
        c = _run(hm, full_ids, comps, lambda: truncation_curve(hm, seq, target, [short, target[0]]))
        short_lp, full_lp = c[0]["mean_logprob"], c[-1]["mean_logprob"]
        rows.append({"condition": name, "short_ctx_lp": short_lp, "full_ctx_lp": full_lp,
                     "context_benefit": full_lp - short_lp, "full_context_letters": c[-1]["context"],
                     "broken": short_lp < -1.2863})
    return rows


def summarize_copy(rows: list[dict]):
    """(gaps, conditions, second_acc matrix, first_acc matrix) averaged over seeds."""
    gaps = sorted({r["gap"] for r in rows})
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    a2 = np.zeros((len(gaps), len(conds)))
    a1 = np.zeros_like(a2)
    for i, g in enumerate(gaps):
        for j, c in enumerate(conds):
            sel = [r for r in rows if r["gap"] == g and r["condition"] == c]
            a2[i, j] = np.mean([r["second_acc"] for r in sel])
            a1[i, j] = np.mean([r["first_acc"] for r in sel])
    return gaps, conds, a2, a1
