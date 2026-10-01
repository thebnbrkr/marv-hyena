# Open items

Everything owed before this work is published, in one place. Written 2026-10-01, revised the same day
to put the reviewer's points first. Sources: an external review of the research log, the list in
`RELATED_WORK.md` §8, and a full read of Mathur & Sachidanandam.

Cost key: **re-analysis** = no GPU, the data is already in `results/`; **1 run** = one Colab A100
session; **code** = new analysis code plus a run.

**Order of work.** Group 1 is the reviewer's points, and they come first — they are what stands
between "interesting" and "defensible", and three of them cost nothing. Group 2 is engaging with the
closest published paper. Everything after that can wait.

---

# GROUP 1 — The reviewer's points, in his order

> *"The main thing is probably to do a few repeats and have stats."*
> *"Premature conclusion on mutation propagation, as it is only based on a few variants."*
> *"Some results based on only one biological sequence family. Maybe do it for at least n = 3."*
> *"Need to show biological replication for a different dataset."*
> *"Could do with better uncertainty quantification, completion of the matched biological control, and
> better health-matched control."*
> *"Maybe need to reframe some claims."*

## 1.1 Repeats and statistics — his first and strongest point

What is actually in the stored data, checked 2026-10-01:

| result | replicates stored | reported with uncertainty? |
|---|---|---|
| Copying | **2 seeds** × 3 gaps × 5 conditions | no — recoverable; spread is tiny (±0.004–0.008 at a 10k gap) |
| Write magnitudes / funnel | **3 genome regions** | partly (ranges given in the log) |
| Precision check | **3 genome regions** | no — recoverable |
| Load-bearing map | 1 per block per run, but **found in 4 separate runs** | no |
| Codon rhythm (MR) | **none.** The four rows per condition are codon positions 1/2/3 plus intergenic — the signal, not replicates | no, and nothing to recover |
| Mutation propagation | **5 variants**, all BRCA1 | no |
| Repeat / copying on real DNA | **1 family** (16S rRNA) | no |
| Round-5 biology | 150 sites per design | **yes** — exact sign tests, Wilson and bootstrap intervals |
| Block-0 nulls | 3 shuffle + 3 Gaussian seeds | yes |

**Actions.**

1. **Report intervals for copying and the precision check.** *re-analysis.* The seeds and regions are
   already stored; only the reporting is missing. **Do this first — it is free.**
2. **Replicate the codon-rhythm result.** *1 run.* It is a single measurement on one stretch of DNA
   and it is a headline claim. Needs several disjoint genome regions, reported with spread.
3. **Add seeds to the single-measurement wiring results.** *1 run.* The load-bearing map turning up in
   four runs is reassuring, but no single run carries an interval.

## 1.2 Mutation propagation rests on five variants

4. *1 run.* "A mutation's effect leaves the position within blocks 0–7" comes from five BRCA1 variants
   (4 LOF, 1 benign). Marked "measured once" in the log, but he is right that it should not be stated
   as a finding at this n. Target: 50+ variants across several genes, reported as a distribution of
   hand-off depth rather than a range.

## 1.3 One sequence family → n ≥ 3

5. *1 run.* Round 4 ran 16S rRNA only. The repeat finder was fixed afterwards and now returns **six**
   families in E. coli (IS2 ×7, IS3 ×5 among them), so the tooling is ready and the run is cheap.
   This is the single cheapest way to answer him.

## 1.4 Biological replication on a different dataset

6. **A second genome.** *1 run.* All biology is E. coli. A second bacterium with GenBank annotations
   is the cheapest version of what he is asking for.
7. **A second checkpoint.** *1 run.* Everything is `evo2_7b`. Rerunning the notebooks with
   `MODEL_NAME = 'evo2_7b_base'` needs no new code. Both external reviews asked for this.

## 1.5 The controls he names

8. **Better health-matched control.** *code.* Specified in `RESEARCH_LOG.md` (round 3b) and never
   built: degrade unrelated components until health matches attention's 0.571, then re-measure.
   Without it the far-context question stays unanswerable, which is why that claim is withdrawn.
9. **A damage-curve control for the MR claim.** *code.* Removing MR is the most damaging family
   ablation, so the rhythm drop could be damage rather than MR's specific role. Same machinery as 8.
10. **"Completion of the matched biological control."** Largely done — matched, flipped, fourfold,
    noncoding and two stop designs all ran, with the codon-rarity control and the per-amino-acid
    split. What is *not* done: it covers **two amino acids**, because the genetic code allows a
    same-site substitution-type-matched pair nowhere else. Going wider means giving up same-site
    matching and controlling across sites — a different design, not a bigger run.
11. **Close the reopened scrambled-null leg behind the SE claim.** *code.* Round 5 dismissed it by
    calling the fully scrambled model near-dead; round 5b showed that argument failing for block 0.
12. **Make the SE claim causal.** *code.* `peak_blocks` says where a difference shows up, not that SE
    computes it. Ablate or patch SE writes on the same paired sites. Until then the verb is **"peaks
    in"**, never "carries".

## 1.6 Reframing

13. His proposed wording for round 4 — *initial synonymous-versus-missense experiments were confounded
    by substitution class and therefore do not establish amino-acid-level representation* — **is
    already the repo's position** (finding 23). But it is now out of date: round 5 ran the matched
    controls and the effect survived. Tell him that. The current defensible claim is narrower *and*
    stronger than either version:

    > At isoleucine and arginine sites in E. coli, in one checkpoint, a substitution that changes the
    > amino acid disturbs Evo 2 more than a silent one at the same position — with substitution type
    > held equal (72.0%, n = 150) or reversed (80.0%, n = 150), and with codon frequency working
    > against the result at 266 of 300 sites.

14. **Never write "Evo 2 knows amino acids."** Agreed with him, and already removed from the README
    and the notebook page.

---

# GROUP 2 — Mathur & Sachidanandam

[*Benchmarking DNA Foundation Models: Biological Blind Spots in Evo2 Variant-Effect
Prediction*](https://doi.org/10.64898/2026.03.10.710786) (bioRxiv, March 2026, not peer reviewed).
Read in full; see `RELATED_WORK.md` §2 for what it claims. It is the closest published work to round
5, it touches no internals, and four of its results need an action from us.

## 2.1 Their transversion result versus our matched design — the same-data demonstration

They report transversions ~1.9× more disruptive than transitions (mean ΔL −0.0113 vs −0.0060), from
**46 transversions against 727 transitions**, unmatched, unpaired, pooled across D-loop, RNA genes,
synonymous and missense variants. Our `fourfold` design asks the same question with 150 paired
same-site comparisons holding the protein constant: **54.7%, p = 0.29**.

15. **Run their statistic and ours on the same data, and report both.** *re-analysis.* A first pass
    pooling every arm of every round-5 design by substitution type gives transition mean −7.34 against
    transversion −5.82 — *transitions* looking worse, the opposite of their direction, because the pool
    is dominated by the stop designs' huge effects. Medians go the other way (−0.881 vs −1.006). **That
    is the point, and it needs doing properly**: an unmatched group comparison measures whatever the
    sampling happens to contain. Restricted to comparable designs and reported beside the paired test,
    this is one figure showing that the same sequences give different answers depending only on whether
    you match. It is the strongest argument for the whole matched-design approach, and it costs nothing.
16. **State the relationship as a refinement, never a contradiction.** Their measurement stands; ours
    shows what survives when position and coding consequence are held fixed. Our own round-4 numbers
    are the cautionary half: unmatched, our version of this comparison gave 68.9% and read as biology.

## 2.2 Their codon-usage finding, in a prokaryote

They find Evo 2 does **not** internalise codon usage bias: the preferred wobble base is chosen at
24.4% of 966 positions, against 25% for a coin flip (mean JSD 0.254), on human TTN exon 305.

17. **Repeat their wobble test on E. coli.** *1 run.* `codons.codon_usage` already builds the
    empirical table from the genome, so the comparison is a short notebook step. Two reasons to do it:
    it tests whether their conclusion holds in a prokaryote, where Evo 2 has far more training data
    per genome; and it would make our own rarity control airtight, since that control assumes the
    codon-frequency confound is weak and their result is currently the only evidence for that
    assumption.
18. **Cite them for the codon half of the dissociation.** The joint claim — *Evo 2 tracks what a codon
    means while not tracking which synonymous codon is preferred* — is half theirs. Already in the
    README and the page; keep it that way in any abstract.

## 2.3 Their genetic-code point exposes a bug waiting to happen in our code

They show Evo 2 ignores the mitochondrial code reassignments (AGA/AGG are stops in vertebrate mtDNA,
ATA is methionine, TGA is tryptophan), calling 26 of 26 valid mito start-preserving variants
pathogenic.

19. **Make the genetic code table swappable.** *code, small.* `marv_hyena/codons.py` hardcodes the
    standard code (`GENETIC_CODE`, built once at import). Anyone pointing our tools at mitochondrial
    DNA would get silently wrong synonymous/missense calls — exactly the error their paper is about.
    Add a code-table argument with the standard code as default, a vertebrate-mito table, and a test
    that the two disagree where they should.

## 2.4 Two things to borrow from them

20. **Their permutation control.** *code.* Moving a sequence while leaving it intact is a clean null
    manipulation, and it is close to what our far-context test needs (item 8). Their version collapsed
    tRNA sensitivity 65.8% → 5.1%, which is how a failing control should look. Consider adapting it
    rather than inventing one.
21. **Their stratified reporting.** Their central methodological complaint is that aggregate metrics on
    imbalanced variant sets overstate performance, and they name the BRCA1 benchmark specifically. Our
    round-2 BRCA1 replication is an aggregate AUROC (0.88) on **40 variants**. Either stratify it or
    stop giving it prominence; it is a replication of someone else's number, not a finding of ours.

## 2.5 Strategic consequence

22. Their paper is a benchmarking framework built on null manipulations, so the "controls toolkit"
    idea in `RESEARCH_LOG.md` is **no longer an empty niche**. Our differentiator has to be the
    internals — operator ablation, per-block divergence, weight nulls — not the controls framing alone.

---

# GROUP 2b — Round 7: replication across organisms and models

Added 2026-10-01 after a review asked for organism- and model-level replication. This is the
reviewer's "different dataset" point taken seriously, and it is the condition he set before drafting a
paper. **Nothing here needs new analysis code** — the designs, statistics and loaders already exist.

## Why yeast specifically, and not just "another organism"

| reason | detail |
|---|---|
| It is a **eukaryote** | every result so far is one bacterium |
| **Arginine codon usage flips** | in E. coli AGA and AGG are among the rarest codons; in yeast AGA is the *most used* arginine codon. The matched designs lean on arginine sites, so this is a natural experiment for the codon-rarity control: if the effect holds in both, codon frequency cannot be driving it |
| **Low GC** | ~38% against E. coli's ~51%, so it also tests base composition |

Third organism: a **high-GC non-pathogenic** bacterium, *Caulobacter crescentus* (~67% GC). E. coli,
yeast and Caulobacter then span 38–67% GC and both domains, which also satisfies the n ≥ 3 instinct.

## Three yeast-specific checks, all now enforced in code

27. **Nuclear genes only.** Yeast mitochondria use translation table 3, which really does reassign
    codons. `probes.genbank_track` now **refuses** any organelle record or any `transl_table` outside
    {1, 11} — table 11 (bacterial) is allowed because it differs from the standard code only in which
    codons may *initiate*, not in what any codon encodes. Skip the mitochondrial record (NC_001224).
    *This is item 2.3, built; it is what makes yeast safe to run.*
28. **Exclude sites near splice junctions.** *code.* Few yeast genes have introns, but a "silent"
    change near an exon boundary can disrupt splicing. `genbank_track` already skips CDS features with
    more than one location part when assigning codon phase, so spliced genes contribute no phased
    sites — confirm that is enough, and otherwise add an explicit margin.
29. **Same strand and overlap rules as E. coli.** Reverse-complement minus-strand genes, drop
    overlapping ones. Round 6's `pick_genes` already does both; reuse it rather than rewriting it.

## Getting the data (Colab, two commands)

Request GenBank format so the existing loader works unchanged:

```bash
!curl -sO https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets && chmod +x datasets
!./datasets download genome accession GCF_000146045.2 --include genome,gbff --filename yeast_S288C.zip
!unzip -q yeast_S288C.zip -d yeast
```

`GCF_000146045.2` is the S288C reference assembly (R64).

30. **Per organism:** all round-5 designs at 150 sites each, the copying test, and the reading-frame
    test on ≥ 20 genes. *1 run per organism.* Pre-register the pass condition **before running**: the
    matched design's interval excludes 50% in each genome.

## A second checkpoint — and what our hardware allows

| checkpoint | context | FP8 / Hopper needed? | runs on our A100? |
|---|---|---|---|
| **`evo2_7b_base`** | 8K | no | **yes — do this one** |
| `evo2_7b` (what every result so far used) | 1M | no | yes |
| `evo2_7b_262k` | 262K | no | yes, low value (same lineage) |
| **`evo2_7b_microviridae`** | 8K | no | **yes — the most divergent weights we can reach** |
| `evo2_1b_base` | 8K | **FP8 required** | **no** |
| `evo2_20b` | 1M | **FP8 required** | no |
| `evo2_40b` / `evo2_40b_base` | 1M / 8K | **FP8, and multiple H100s** | no |

*Verified against the Arc Institute `evo2` README on 2026-10-01, not from memory. Every 7B variant is
BF16-compatible and needs no Transformer Engine; the 1B, 20B and 40B all require FP8 on Hopper.*

**So the only independently-trained second model (`evo2_1b_base`) is the one we cannot run**, and the
40B needs several H100s. That is a hardware fact, not a choice, and the paper should say so plainly
rather than leave the single-checkpoint limitation unexplained.

31. **Run `evo2_7b_base`.** *1 run.* Copying, round 5 and the SE peak. **State the caveat in the
    paper:** `evo2_7b` was produced by context-extending `evo2_7b_base`, so this is the same training
    run at an earlier stage — a **stability check, not an independent model.**
32. **`evo2_1b_base` is out of reach on our hardware.** We have A100s; it needs an H100. Two honest
    options: rent H100 hours for a few hours, or name it as future work. If it does get run, its layer
    layout differs, so **test for the pattern** — a single dominant late block, attention doing the
    copying, the SE peak — **never for specific indices like "block 30".**
33. **The BioNeMo 1B checkpoint** is fine-tuned to support BF16 as well as FP8, which may avoid the
    H100 — but it is modified weights in a different framework, so the Vortex hooks would need
    porting. Use it as a fallback for the **behavioural** tests (round 5) only, never the internal
    ones.
34. **`evo2_7b_microviridae` is the best A100-reachable divergence test.** It is `evo2_7b_base`
    fine-tuned on a bacteriophage family, so unlike `evo2_7b` and `evo2_7b_262k` it differs from the
    parent by *training* rather than by context extension. Running the operator map and the funnel on
    it answers a question no other accessible checkpoint can: does fine-tuning move the wiring, or is
    the map a property of the pre-trained model? Read-only analysis of a published checkpoint, so it
    sits inside this repo's scope note (understanding and measuring, never inserting capabilities).
35. **20B and 40B are future work**, named as such, with the hardware reason given.

## His condition for drafting the paper

A concrete "done": (a) repeats and statistics — intervals, gene-level bootstrap, the six-family repeat
run (round 6); (b) round 5 replicated in yeast, plus Caulobacter if possible; (c) `evo2_7b_base`
replication of copying, round 5 and the SE peak.

**Writing need not wait.** The introduction, methods and the controls section — with its five cases
where a control reversed a finding — do not depend on any of these results. Draft those while the runs
go.

---

# GROUP 3 — Citation hygiene

23. **Read every `SUMMARY` and `SECONDHAND` entry in `RELATED_WORK.md` before citing it.** Michalak &
    Abreu, the ICML 2026 load-bearing paper, the pruning study, Bick, Arora, Lu.
24. **Rename or explicitly distinguish "load-bearing layers",** which an ICML 2026 paper already uses
    on Evo 2 with a different method. Check its layer numbering (0- or 1-based) before citing any
    layer number.
25. **Do not claim** that our funnel explains that paper's deep load-bearing layer.
26. **Credit BioRiskEval for the magnitude growth** wherever the funnel is discussed.

---

## Cheapest path to a defensible paper

| step | items | cost | answers |
|---|---|---|---|
| 1 | Intervals from stored data; the matched-vs-unmatched figure | re-analysis, hours | his 1.1; their 2.1 |
| 2 | Repeat families n ≥ 3; codon-rhythm replication | 1 run | his 1.1, 1.3 |
| 3 | Second checkpoint on the existing notebooks | 1 run | his 1.4 |
| 4 | SE causal test + the scrambled-null leg | 1 run + code | our strongest claim |
| 5 | Mutation propagation at n ≥ 50 | 1 run | his 1.2 |
| 6 | Health-matched controls (far context, MR damage curve) | code + 1 run | his 1.5 |
| 7 | Their wobble test on E. coli | 1 run | their 2.2 |
| 8 | `evo2_7b_base` replication (copying, round 5, the SE peak) | 1 run | his 1.4 |
| 9 | Round 5 + copying + reading frame in **yeast**, then Caulobacter | 1 run each | his 1.4, and the arginine-rarity natural experiment |

**Step 1 is free and answers part of both groups — start there.** Steps 1–4 turn the strongest claims
from suggestive into solid. Steps 5–6 close the two results marked "measured once" and "withdrawn".

**What is already done** and should not be re-litigated: round 5's matched substitution designs, the
codon-rarity control (266 of 300 sites), the per-amino-acid split, the size-matched SE comparison,
the block-0 null with six seeds, and determinism replication of the whole round-3 result set in
round 3b.
