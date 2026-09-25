"""Round 4: does the copying circuit fire on REAL repeats, or only on ours?

Rounds 1-3 established "attention does exact copying" with `probes.copy_probe`:
a RANDOM 200-letter stretch inserted into E. coli DNA and repeated later. The
first copy is unpredictable, the second is predictable only by retrieval, and
ablating attention takes the second copy to chance.

The objection (from a biologist reading the round-3 log): that is a
string-matching task with no biology in it. It shows the model CAN retrieve; it
does not show the circuit is used when reading a genome. The random insert may
recruit machinery that real DNA rarely touches.

This module runs the same experiment on real repeat families that E. coli
actually contains -- the seven rRNA operons, the IS elements -- so the only
thing that changes between arms is the CONTENT of the repeated stretch:

  random    a random 200-letter insert            (rounds 1-3)
  shuffled  a real repeat, letters shuffled       (same composition, no biology)
  real      an actual genomic repeat copy         (biology intact)

The measurement that matters is the RETRIEVAL GAIN, second_lp - first_lp, not
the raw score on the second copy. A real rRNA copy scores well on its FIRST
appearance too, because the model knows rRNA; only the gain isolates what
retrieval added on top of prior knowledge. Ablating attention should flatten
the gain while leaving the first copy alone. If it does not -- if the gain on
real repeats survives attention ablation -- then the round-1-3 copying result
really was about the synthetic probe.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from .arch import HyenaModel
from .intervene import mean_ablate, mean_writes
from .probes import CopyProbe, copy_probe, score_copy

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(s: str) -> str:
    return s.translate(COMPLEMENT)[::-1]


def gc_content(s: str) -> float:
    return (s.count("G") + s.count("C")) / max(1, len(s))


def shuffle_letters(s: str, rng: random.Random) -> str:
    """Composition control: same base counts, no sequence structure."""
    letters = list(s)
    rng.shuffle(letters)
    return "".join(letters)


# ------------------------------------------------------------------ finding repeats
@dataclass
class RepeatCopy:
    start: int
    end: int
    strand: int
    seq: str  # already oriented to the + strand of the family


@dataclass
class RepeatFamily:
    name: str
    feature_type: str
    copies: list[RepeatCopy]
    # Mean pairwise k-mer similarity, NOT an alignment identity: see
    # kmer_similarity for why the distinction matters here.
    kmer_similarity: float

    @property
    def n_copies(self) -> int:
        return len(self.copies)

    @property
    def length(self) -> int:
        return min(len(c.seq) for c in self.copies)

    def __repr__(self) -> str:
        return (f"RepeatFamily({self.name!r}, {self.feature_type}, {self.n_copies} copies, "
                f"{self.length} bp, k-mer similarity {self.kmer_similarity:.3f})")


def _family_key(name: str) -> str:
    """Collapse a per-copy feature name to its family.

    E. coli annotates each insertion-sequence copy individually
    ("insertion sequence:IS1A", "IS1B", "IS1C", "IS911A-1"), so grouping on the
    raw name puts every copy in a group of one and finds no repeats at all.
    The family is the name with its copy suffix removed.

    A trailing "-<digits>" goes first, then ONE trailing uppercase letter, but
    only when what remains still ends in a digit. That last guard keeps IS1A ->
    IS1 while leaving a genuine name like ISX alone (stripping its X would
    leave a bare "IS").
    """
    base = name.split(":")[-1].strip()
    m = re.match(r"^(.*?)-\d+$", base)
    if m:
        base = m.group(1)
    if len(base) > 1 and base[-1].isupper() and base[-2].isdigit():
        base = base[:-1]
    return base


def kmer_similarity(a: str, b: str, k: int = 16) -> float:
    """Share of the shorter sequence's k-mers that also occur in the longer one.

    Deliberately NOT an alignment identity. E. coli's seven 23S rRNA copies
    differ in length by one base (2904 vs 2905); a prefix-by-prefix comparison
    puts everything after that indel out of register and scores them 0.87,
    which rejected the single best repeat family in the genome. Counting shared
    k-mers is indel-robust: one indel only disturbs the k k-mers spanning it.

    Called "similarity", not "identity", because %identity has a specific
    meaning in biology and this is not it.
    """
    if len(a) > len(b):
        a, b = b, a
    if len(a) < k:
        return 1.0 if a in b else 0.0
    big = {b[i:i + k] for i in range(len(b) - k + 1)}
    small = [a[i:i + k] for i in range(len(a) - k + 1)]
    return sum(s in big for s in small) / len(small)


def _feature_name(f) -> str | None:
    q = f.qualifiers
    for key in ("mobile_element_type", "product", "gene", "note"):
        if key in q and q[key]:
            return str(q[key][0]).strip()
    return None


def _mean_similarity(seqs: list[str], k: int = 16) -> float:
    """Mean pairwise k-mer similarity over the copies."""
    tot, pairs = 0.0, 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            tot += kmer_similarity(seqs[i], seqs[j], k)
            pairs += 1
    return tot / max(1, pairs)


def find_repeat_families(path: str, types=("rRNA", "mobile_element"), min_copies: int = 2,
                         min_len: int = 150, min_similarity: float = 0.90,
                         k: int = 16) -> list[RepeatFamily]:
    """Repeat families from a GenBank file's annotations.

    Groups features of the same type by FAMILY (see `_family_key`: E. coli names
    every IS copy separately, so the raw name gives groups of one), keeps groups
    with at least `min_copies` members that really are near-identical, judged by
    indel-robust k-mer similarity. Copies on the minus strand are
    reverse-complemented so every copy in a family shares an orientation.
    """
    from Bio import SeqIO

    rec = next(SeqIO.parse(path, "genbank"))
    genome = str(rec.seq).upper()
    groups: dict[tuple[str, str], list[RepeatCopy]] = {}
    for f in rec.features:
        if f.type not in types or len(f.location.parts) != 1:
            continue
        name = _feature_name(f)
        if name is None:
            continue
        s, e = int(f.location.start), int(f.location.end)
        if e - s < min_len:
            continue
        strand = f.location.strand or 1
        sub = genome[s:e]
        if set(sub) - set("ACGT"):
            continue
        groups.setdefault((f.type, _family_key(name)), []).append(
            RepeatCopy(s, e, strand, sub if strand >= 0 else reverse_complement(sub)))

    out = []
    for (ftype, name), copies in groups.items():
        if len(copies) < min_copies:
            continue
        sim = _mean_similarity([c.seq for c in copies], k)
        if sim < min_similarity:
            continue
        out.append(RepeatFamily(name, ftype, sorted(copies, key=lambda c: c.start), sim))
    return sorted(out, key=lambda f: (-f.n_copies, -f.length))


# ------------------------------------------------------------------ probes
@dataclass
class RepeatProbe:
    probe: CopyProbe
    arm: str  # "real" | "shuffled" | "random"
    family: str
    gap: int
    insert_gc: float


def repeat_probes(background: str, family: RepeatFamily, gaps=(1000, 10000), insert_len: int = 200,
                  lead: int = 1000, seed: int = 0, copy_index: int = 0) -> list[RepeatProbe]:
    """The three arms at each gap, from one repeat family.

    The insert is the first `insert_len` letters of one real copy. All three
    arms share the same background, gap and length, so the ONLY difference is
    what is being repeated.
    """
    rng = random.Random(seed)
    real = family.copies[copy_index].seq[:insert_len]
    if len(real) < insert_len:
        raise ValueError(f"family {family.name!r} copies are shorter than insert_len={insert_len}")
    arms = {"real": real, "shuffled": shuffle_letters(real, rng)}
    out = []
    for gap in gaps:
        for arm, ins in arms.items():
            p = copy_probe(background, insert_len=insert_len, gap=gap, lead=lead, seed=seed, insert=ins)
            out.append(RepeatProbe(p, arm, family.name, gap, gc_content(ins)))
        p = copy_probe(background, insert_len=insert_len, gap=gap, lead=lead, seed=seed)
        out.append(RepeatProbe(p, "random", family.name, gap,
                               gc_content(p.seq[p.first[0]:p.first[1]])))
    return out


@torch.no_grad()
def run_repeat_test(hm: HyenaModel, probes: list[RepeatProbe], conditions: dict | None = None,
                    skip: int = 20) -> list[dict]:
    """Score every probe under every ablation condition.

    `conditions` maps a label to a list of (block, part) components to
    mean-ablate, exactly like `experiments.make_conditions`. The reported
    `retrieval_gain` is second_lp - first_lp: what retrieval added on top of
    whatever the model already knew about the sequence.
    """
    conditions = {"none": []} if conditions is None else conditions
    rows = []
    for label, comps in conditions.items():
        for rp in probes:
            ids = hm.ids(rp.probe.seq)
            if comps:
                with mean_ablate(hm, mean_writes(hm, ids, comps)):
                    logits = hm.logits(ids)
            else:
                logits = hm.logits(ids)
            s = score_copy(logits, ids, rp.probe, skip=skip)
            rows.append({
                "condition": label, "arm": rp.arm, "family": rp.family, "gap": rp.gap,
                "insert_gc": rp.insert_gc, **s,
                "retrieval_gain": s["second_lp"] - s["first_lp"],
                "acc_gain": s["second_acc"] - s["first_acc"],
            })
    return rows


def summarize_repeat_test(rows: list[dict], min_gain: float = 0.1) -> list[dict]:
    """Mean retrieval gain per (condition, arm, gap), and how much of the
    unablated gain each condition leaves standing.

    `gain_kept` is NaN when the unablated gain is below `min_gain` nats: round 4
    reported "-attn keeps -313%" of a 0.028-nat gain, a ratio over nothing."""
    import numpy as np

    key = lambda r: (r["condition"], r["arm"], r["gap"])  # noqa: E731
    agg: dict[tuple, list[dict]] = {}
    for r in rows:
        agg.setdefault(key(r), []).append(r)
    base = {k[1:]: float(np.mean([x["retrieval_gain"] for x in v]))
            for k, v in agg.items() if k[0] == "none"}
    out = []
    for k, v in sorted(agg.items()):
        gain = float(np.mean([x["retrieval_gain"] for x in v]))
        b = base.get(k[1:])
        out.append({
            "condition": k[0], "arm": k[1], "gap": k[2], "n": len(v),
            "first_lp": float(np.mean([x["first_lp"] for x in v])),
            "second_lp": float(np.mean([x["second_lp"] for x in v])),
            "retrieval_gain": gain,
            "gain_kept": gain / b if b is not None and abs(b) >= min_gain else float("nan"),
        })
    return out
