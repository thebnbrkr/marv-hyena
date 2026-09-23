"""Round 4: the translation test. Does Evo 2 represent AMINO ACIDS, or only letters?

Round 3 showed the reading frame lives in the MR layers: remove them and
next-letter accuracy goes flat across the three codon positions. That says the
model tracks WHERE the codon boundaries are. It does not say the model knows
what a codon MEANS.

The genetic code is redundant: six codons encode leucine, and the third
("wobble") position is usually free. So there is a clean test. Take one codon
site and make two single-letter substitutions AT THE SAME POSITION:

    synonymous      ATT -> ATC   (both isoleucine)
    non-synonymous  ATT -> ATG   (isoleucine -> methionine)

Same site, same codon position, same edit distance, same local sequence. The
only difference is whether the encoded amino acid changed. If the model merely
models nucleotide statistics, the two disturbances should be comparable. If it
represents the protein, the non-synonymous one should disturb it more -- and
the depth at which the two diverge is where the genetic code gets applied.

Eight codon families admit both kinds of change at position 3, which is what
`wobble_sites` looks for:

    AT[TCA]/ATG  Ile/Met     TT[TC]/TT[AG]  Phe/Leu    TA[TC]/TA[AG]  Tyr/stop
    CA[TC]/CA[AG]  His/Gln   AA[TC]/AA[AG]  Asn/Lys    GA[TC]/GA[AG]  Asp/Glu
    TG[TC]/TGA/TGG  Cys/stop/Trp           AG[TC]/AG[AG]  Ser/Arg

Premature stop codons are excluded by default (`allow_stop=False`). A stop is a
much larger biological event than an amino-acid swap -- Evo 2's own SAE feature
f/24278 fires on them -- so mixing them into the non-synonymous arm would
confound "the protein changed" with "the protein ended". They are worth a
separate arm, which is what `allow_stop=True` is for.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .arch import HyenaModel
from .probes import Track
from .trace import capture_writes

# Standard genetic code (NCBI table 1). Bacterial table 11 differs only in which
# codons may act as STARTs, which does not affect any comparison here.
GENETIC_CODE = {}
for _i, _a in enumerate("KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"):
    GENETIC_CODE["ACGT"[_i >> 4] + "ACGT"[(_i >> 2) & 3] + "ACGT"[_i & 3]] = _a

AA_TO_CODONS: dict[str, list[str]] = {}
for _c, _aa in GENETIC_CODE.items():
    AA_TO_CODONS.setdefault(_aa, []).append(_c)

STOPS = tuple(AA_TO_CODONS["*"])


def translate(seq: str) -> str:
    """Whole codons only; a trailing partial codon is dropped."""
    s = seq.upper()
    return "".join(GENETIC_CODE.get(s[i:i + 3], "X") for i in range(0, len(s) - len(s) % 3, 3))


def synonyms(codon: str) -> list[str]:
    """Other codons for the same amino acid."""
    c = codon.upper()
    return [x for x in AA_TO_CODONS[GENETIC_CODE[c]] if x != c]


@dataclass
class WobbleSite:
    """One codon where position 3 admits both a silent and a missense change."""
    pos: int  # genome/window index of the THIRD codon letter (the one substituted)
    codon_start: int
    codon: str
    aa: str
    ref: str  # the third letter
    syn_alt: str  # substitution that keeps the amino acid
    syn_codon: str
    nonsyn_alt: str  # substitution that changes it
    nonsyn_codon: str
    nonsyn_aa: str

    @property
    def is_nonsense(self) -> bool:
        return self.nonsyn_aa == "*"


def wobble_sites(track: Track, allow_stop: bool = False, max_sites: int | None = None,
                 min_spacing: int = 60) -> list[WobbleSite]:
    """Find codon sites in `track` where substituting the third letter can be
    either synonymous or non-synonymous.

    Only forward-strand CDS positions with a complete in-window codon are used
    (`genbank_track` sets `phase` for exactly those). `min_spacing` keeps the
    sites far enough apart that their downstream measurement windows do not
    overlap.
    """
    seq, phase = track.seq, track.codon_phase
    out: list[WobbleSite] = []
    last = -10 ** 9
    for i in range(len(seq)):
        if phase[i] != 2 or i < 2:  # third position of a codon
            continue
        if i - last < min_spacing:
            continue
        start = i - 2
        codon = seq[start:i + 1]
        if len(codon) != 3 or set(codon) - set("ACGT"):
            continue
        aa = GENETIC_CODE[codon]
        if aa == "*":
            continue  # do not mutate a real stop codon
        syn, non = [], []
        for b in "ACGT":
            if b == codon[2]:
                continue
            alt = codon[:2] + b
            alt_aa = GENETIC_CODE[alt]
            (syn if alt_aa == aa else non).append((b, alt, alt_aa))
        if not allow_stop:
            non = [x for x in non if x[2] != "*"]
        if not syn or not non:
            continue
        sb, sc, _ = syn[0]
        nb, nc, naa = non[0]
        out.append(WobbleSite(pos=i, codon_start=start, codon=codon, aa=aa, ref=codon[2],
                              syn_alt=sb, syn_codon=sc, nonsyn_alt=nb, nonsyn_codon=nc,
                              nonsyn_aa=naa))
        last = i
        if max_sites is not None and len(out) >= max_sites:
            break
    return out


# ------------------------------------------------------------------ measurement
@torch.no_grad()
def divergence_by_block(hm: HyenaModel, ref: str, index: int, alts: dict[str, str],
                        span: int = 24) -> list[dict]:
    """How far each block's write moves when the letter at `index` changes.

    Captured at the `span` positions AFTER the substitution -- the positions
    whose reading of the protein would change -- not at the site itself, where
    a letter swap always registers. Reported relative to the reference write's
    own size, so blocks of wildly different scale (block 30 writes ~1e12) stay
    comparable.
    """
    positions = list(range(index + 1, min(index + 1 + span, len(ref))))
    if not positions:
        raise ValueError("span leaves no positions after the substitution")
    ref_w = capture_writes(hm, hm.ids(ref), positions)
    rows = []
    for label, alt in alts.items():
        if len(alt) != len(ref):
            raise ValueError(f"alt {label!r} has a different length from ref")
        alt_w = capture_writes(hm, hm.ids(alt), positions)
        for (b, part), x in sorted(ref_w.parts.items()):
            d = float((alt_w.parts[(b, part)] - x).norm())
            n = float(x.norm())
            rows.append({"arm": label, "block": b, "kind": hm.kind(b), "part": part,
                         "abs_divergence": d, "rel_divergence": d / n if n > 1e-12 else float("nan")})
    return rows


def translation_test(hm: HyenaModel, seq: str, sites: list[WobbleSite], window: int = 4096,
                     span: int = 24, downstream_span: int = 200) -> list[dict]:
    """The full comparison at every site: how much does a silent change disturb
    the model, versus a missense change at the same position?

    Returns one row per site with the downstream effect of each arm, plus the
    block at which the two arms diverge most.
    """
    from .variants import downstream_effect, make_variant

    rows = []
    for s in sites:
        lo = max(0, s.pos - window // 2)
        hi = min(len(seq), lo + window)
        if hi - lo < 64 or s.pos - lo < 8:
            continue
        ref_w = seq[lo:hi]
        i = s.pos - lo
        arms = {"syn": ref_w[:i] + s.syn_alt + ref_w[i + 1:],
                "nonsyn": ref_w[:i] + s.nonsyn_alt + ref_w[i + 1:]}
        eff = {}
        for arm, alt in arms.items():
            v = make_variant(seq, s.pos, s.ref, s.syn_alt if arm == "syn" else s.nonsyn_alt,
                             window=window)
            eff[arm] = downstream_effect(hm, v, span=downstream_span)
        div = divergence_by_block(hm, ref_w, i, arms, span=span)
        by_block: dict[int, dict[str, float]] = {}
        for r in div:
            by_block.setdefault(r["block"], {})[r["arm"]] = (
                by_block.get(r["block"], {}).get(r["arm"], 0.0) + r["rel_divergence"])
        ratios = {b: (v.get("nonsyn", 0.0) / v["syn"]) for b, v in by_block.items()
                  if v.get("syn", 0.0) > 1e-9}
        peak = max(ratios, key=ratios.get) if ratios else None
        rows.append({
            "pos": s.pos, "codon": s.codon, "aa": s.aa,
            "syn_codon": s.syn_codon, "nonsyn_codon": s.nonsyn_codon, "nonsyn_aa": s.nonsyn_aa,
            "is_nonsense": s.is_nonsense,
            "syn_effect": eff["syn"], "nonsyn_effect": eff["nonsyn"],
            # A ratio of two signed quantities that can straddle zero is not a
            # statistic. Round 1 learned this the hard way: its FUNC variant had
            # a downstream effect of 0.25 and produced "fractions" of +-200% that
            # were pure noise (see variants.downstream_effect). So the ratio is
            # only defined where the SILENT arm is clearly disruptive, and
            # `usable` says whether this row may enter a ratio average.
            "usable": bool(eff["syn"] < -MIN_EFFECT),
            "effect_ratio": (eff["nonsyn"] / eff["syn"]) if eff["syn"] < -MIN_EFFECT else float("nan"),
            "peak_divergence_block": peak,
            "peak_divergence_ratio": ratios.get(peak) if peak is not None else float("nan"),
        })
    return rows


# Below this magnitude (nats, summed over `downstream_span` letters) a
# downstream effect is indistinguishable from noise and cannot anchor a ratio.
MIN_EFFECT = 0.25


def summarize_translation(rows: list[dict]) -> dict:
    """The headline numbers: does a missense change disturb the model more than
    a silent one at the same site, and how often?

    Two statistics, on purpose. `nonsyn_more_disruptive_frac` is a paired sign
    test over EVERY site -- it needs no ratio and no threshold, so it is the one
    to trust. `median_effect_ratio` gives the size of the gap but is only
    averaged over `usable` sites, and is meaningless if `n_usable` is small.
    """
    import numpy as np

    if not rows:
        return {"n": 0, "n_usable": 0}
    syn = np.array([r["syn_effect"] for r in rows], dtype=float)
    non = np.array([r["nonsyn_effect"] for r in rows], dtype=float)
    blocks = [r["peak_divergence_block"] for r in rows if r["peak_divergence_block"] is not None]
    ratios = [r["effect_ratio"] for r in rows if r.get("usable") and np.isfinite(r["effect_ratio"])]
    return {
        "n": len(rows),
        "n_usable": len(ratios),
        "mean_syn_effect": float(syn.mean()),
        "mean_nonsyn_effect": float(non.mean()),
        # downstream_effect is negative when the mutation hurts, so "more
        # disturbing" means MORE NEGATIVE. This is the paired, threshold-free
        # statistic and the one P21 is decided on.
        "nonsyn_more_disruptive_frac": float((non < syn).mean()),
        "median_effect_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "modal_peak_block": max(set(blocks), key=blocks.count) if blocks else None,
    }
