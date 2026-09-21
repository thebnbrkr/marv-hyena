"""The PREDICTIONS.md experiments as functions over a loaded model, so the
scripts and the Colab notebook share one implementation. Each returns plain
rows (list of dicts) ready for pandas / JSON."""
from __future__ import annotations

import random

import numpy as np

from .arch import KINDS, HyenaModel
from .intervene import mean_ablate, mean_writes, mixers_of
from .probes import Track, codon_phase_accuracy, copy_probe, score_copy, truncation_curve


def _conditions(hm: HyenaModel, kinds=KINDS, per_block_kinds=()):
    conds = {"none": []}
    conds.update({f"-{k}": mixers_of(hm, k) for k in kinds})
    for k in per_block_kinds:
        conds.update({f"-L{b}({k})": [(b, "mixer")] for b in hm.blocks_of(k)})
    return conds


def copy_test(hm: HyenaModel, genome: str, gaps=(100, 1000, 10000), insert_len: int = 200, seeds: int = 3,
              per_block_kinds=(), lead: int = 1000, verbose: bool = True) -> list[dict]:
    """P1. For each gap/seed: score both copies with the full model and with
    each operator type's mixers mean-ablated (means from the same sequence)."""
    rows = []
    for gap in gaps:
        for seed in range(seeds):
            rng = random.Random(seed)
            need = lead + gap + 50
            off = rng.randrange(0, len(genome) - need)
            probe = copy_probe(genome[off:off + need], insert_len=insert_len, gap=gap, lead=lead, seed=seed)
            ids = hm.ids(probe.seq)
            for name, comps in _conditions(hm, per_block_kinds=per_block_kinds).items():
                if comps:
                    with mean_ablate(hm, mean_writes(hm, ids, comps)):
                        s = score_copy(hm.logits(ids), ids, probe)
                else:
                    s = score_copy(hm.logits(ids), ids, probe)
                rows.append({"gap": gap, "seed": seed, "condition": name, **s})
            if verbose:
                print(f"copy test: gap={gap} seed={seed} done")
    return rows


def codon_test(hm: HyenaModel, track: Track) -> list[dict]:
    """P3. Accuracy by codon position (and intergenic) for the full model and
    each operator type ablated."""
    rows = []
    ids = hm.ids(track.seq)
    for name, comps in _conditions(hm).items():
        if comps:
            with mean_ablate(hm, mean_writes(hm, ids, comps)):
                out = codon_phase_accuracy(hm, track)
        else:
            out = codon_phase_accuracy(hm, track)
        for region, v in out.items():
            rows.append({"condition": name, "region": region, **v})
    return rows


def context_test(hm: HyenaModel, seq: str, target: tuple[int, int], short: int = 500) -> list[dict]:
    """P2. Context benefit = mean log-p(target | all upstream) - mean
    log-p(target | `short` letters upstream), per ablation condition. The
    ablation means come from the full-length sequence."""
    rows = []
    full_ids = hm.ids(seq[: target[1]])
    for name, comps in _conditions(hm).items():
        def curve():
            return truncation_curve(hm, seq, target, [short, target[0]])
        if comps:
            with mean_ablate(hm, mean_writes(hm, full_ids, comps)):
                c = curve()
        else:
            c = curve()
        rows.append({"condition": name, "short_ctx_lp": c[0]["mean_logprob"], "full_ctx_lp": c[-1]["mean_logprob"],
                     "context_benefit": c[-1]["mean_logprob"] - c[0]["mean_logprob"],
                     "full_context_letters": c[-1]["context"]})
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
