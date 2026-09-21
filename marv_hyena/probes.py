"""DNA test inputs with a known right answer -- the DNA analogue of MARV's
probe batteries. Each builder returns the sequence plus the spans to score.

- copy_probe:     a random segment appears twice, `gap` letters apart. The
                  first copy is unpredictable; the second is predictable only
                  if the model can find and copy the first. (Test 1)
- truncation:     score the same target span with shorter and shorter
                  upstream context. (Test 2)
- codon phase:    accuracy at codon positions 1/2/3 inside annotated genes. (Test 3)

LARQL's rule applies to every probe here: a sequence shorter than an
operator's reach cannot distinguish that operator. MR reaches 128 letters
per block, so any "long-range" claim needs gaps far beyond that.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .arch import HyenaModel
from .intervene import span_accuracy, span_logprob, token_logprobs


def load_sequence(path: str) -> str:
    """First record of a GenBank (.gb/.gbk/.genbank) or FASTA file, upper-case."""
    from Bio import SeqIO

    fmt = "genbank" if path.lower().endswith((".gb", ".gbk", ".genbank")) else "fasta"
    return str(next(SeqIO.parse(path, fmt)).seq).upper()


def random_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


@dataclass
class CopyProbe:
    seq: str
    first: tuple[int, int]  # [start, end) of the first copy
    second: tuple[int, int]
    gap: int


def copy_probe(background: str, insert_len: int = 200, gap: int = 1000, lead: int = 1000,
               seed: int = 0) -> CopyProbe:
    """lead letters of background, the random insert, `gap` letters of
    background, the insert again, 50 letters of background. Background must
    be at least lead + gap + 50 long (use real genome, not random DNA, so the
    model is in-distribution everywhere except the inserts)."""
    need = lead + gap + 50
    if len(background) < need:
        raise ValueError(f"background too short: need {need}, have {len(background)}")
    rng = random.Random(seed)
    ins = random_dna(insert_len, rng)
    a = background[:lead]
    g = background[lead:lead + gap]
    tail = background[lead + gap:lead + gap + 50]
    seq = a + ins + g + ins + tail
    s1 = len(a)
    s2 = s1 + insert_len + gap
    return CopyProbe(seq, (s1, s1 + insert_len), (s2, s2 + insert_len), gap)


def score_copy(logits: torch.Tensor, ids: torch.Tensor, probe: CopyProbe, skip: int = 20) -> dict:
    """Accuracy + mean log-prob on each copy. `skip` letters at the start of the
    second copy are excluded: the model needs a few letters to recognise the repeat."""
    a0, a1 = probe.first
    b0, b1 = probe.second
    return {
        "first_acc": span_accuracy(logits, ids, a0 + skip, a1),
        "second_acc": span_accuracy(logits, ids, b0 + skip, b1),
        "first_lp": span_logprob(logits, ids, a0 + skip, a1) / (a1 - a0 - skip),
        "second_lp": span_logprob(logits, ids, b0 + skip, b1) / (b1 - b0 - skip),
    }


@torch.no_grad()
def truncation_curve(hm: HyenaModel, seq: str, target: tuple[int, int], contexts: list[int]) -> list[dict]:
    """Mean log-prob of seq[target] when the model sees only `c` letters before
    the target span, for each c in contexts. The drop from full context to
    short context is how much the far DNA is worth for this span."""
    t0, t1 = target
    out = []
    for c in sorted(contexts):
        start = max(0, t0 - c)
        sub = seq[start:t1]
        ids = hm.ids(sub)
        logits = hm.logits(ids)
        lp = span_logprob(logits, ids, t0 - start, t1 - start) / (t1 - t0)
        out.append({"context": t0 - start, "mean_logprob": lp})
    return out


# ---------------------------------------------------------------- annotations
@dataclass
class Track:
    """Per-letter annotation of a genome window."""

    start: int
    seq: str
    codon_phase: np.ndarray  # 0/1/2 inside a CDS on the forward strand, -1 elsewhere
    feature: np.ndarray  # object array: 'CDS', 'tRNA', 'rRNA', ..., '' for none
    strand: np.ndarray  # +1 / -1 / 0


def genbank_track(path: str, start: int, end: int, types=("CDS", "tRNA", "rRNA", "ncRNA", "tmRNA",
                                                           "mobile_element", "regulatory")) -> Track:
    """Annotate genome[start:end] from a GenBank file (needs biopython)."""
    from Bio import SeqIO

    rec = next(SeqIO.parse(path, "genbank"))
    seq = str(rec.seq[start:end]).upper()
    n = end - start
    phase = np.full(n, -1, dtype=np.int8)
    feat = np.full(n, "", dtype=object)
    strand = np.zeros(n, dtype=np.int8)
    for f in rec.features:
        if f.type not in types:
            continue
        s, e = int(f.location.start), int(f.location.end)
        if e <= start or s >= end:
            continue
        lo, hi = max(s, start) - start, min(e, end) - start
        feat[lo:hi] = f.type
        strand[lo:hi] = f.location.strand or 0
        if f.type == "CDS" and f.location.strand == 1 and len(f.location.parts) == 1:
            idx = np.arange(lo, hi)
            phase[lo:hi] = (idx + start - s) % 3
    return Track(start, seq, phase, feat, strand)


@torch.no_grad()
def codon_phase_accuracy(hm: HyenaModel, track: Track) -> dict:
    """Next-letter accuracy and mean log-prob at codon positions 1/2/3 of
    forward-strand genes, plus outside genes. A strong 3-periodicity means the
    model tracks the reading frame."""
    ids = hm.ids(track.seq)
    logits = hm.logits(ids)
    lp = token_logprobs(logits, ids).cpu().numpy()
    correct = (logits[:-1].argmax(-1).cpu() == ids[0, 1:].cpu()).numpy()
    ph = track.codon_phase[1:]
    out = {}
    for p in (0, 1, 2):
        m = ph == p
        out[f"codon_pos{p + 1}"] = {"acc": float(correct[m].mean()) if m.any() else float("nan"),
                                    "lp": float(lp[m].mean()) if m.any() else float("nan"), "n": int(m.sum())}
    m = track.feature[1:] == ""
    out["intergenic"] = {"acc": float(correct[m].mean()) if m.any() else float("nan"),
                         "lp": float(lp[m].mean()) if m.any() else float("nan"), "n": int(m.sum())}
    return out
