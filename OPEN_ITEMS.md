# Open items

Everything owed before this work is published, in one place. Written 2026-10-01, merging the list in
`RELATED_WORK.md` §8 with an external review of the research log (2026-10-01, recorded in
`RESEARCH_LOG.md`). Ordered by what each item unblocks, not by effort.

Cost key: **re-analysis** = no GPU, the data is already in `results/`; **1 run** = one Colab A100
session; **code** = new analysis code plus a run.

---

## A. Replication and uncertainty

The reviewer's first and strongest point: *"do a few repeats and have stats."* Round 5's biology has
this (exact sign tests, Wilson intervals, bootstrap intervals, a two-proportion test). **The wiring
results from rounds 1–3 mostly do not.** What is actually in the stored data:

| result | replicates in the data | reported with uncertainty? |
|---|---|---|
| Copying (`copy_families_keep_lb`) | **2 seeds** × 3 gaps × 5 conditions | no — but recoverable by re-analysis; spread is tiny (±0.004–0.008 at a 10k gap) |
| Write magnitudes / funnel | **3 genome regions** | partly (the log gives ranges) |
| Precision check | **3 genome regions** | no — recoverable |
| Load-bearing map | 1 measurement per block per run, but **found in 4 separate runs** | no |
| Codon rhythm (MR) | **none.** The four rows per condition are codon positions 1/2/3 plus intergenic — the signal, not replicates | no, and nothing to recover |
| Mutation propagation | **5 variants** (all BRCA1) | no |
| Repeat / copying-on-real-DNA | **1 family** (16S rRNA) | no |
| Round-5 biology | 150 sites per design | **yes** |
| Block-0 nulls | 3 shuffle seeds + 3 Gaussian seeds | yes (medians + cutoff sweep) |

1. **Report uncertainty on the copying and precision results.** *re-analysis.* The seeds and regions
   are already stored; only the reporting is missing.
2. **Replicate the codon-rhythm result.** *1 run.* It is a single measurement on one stretch of DNA,
   and it is one of the project's headline claims (MR carries the reading frame). Needs several
   disjoint genome regions, reported with spread.
3. **Add seeds to the single-measurement wiring results.** *1 run.* The load-bearing map has been
   recovered four times across runs, which is reassuring, but no single run carries an interval.

## B. Sample size on specific claims

4. **Mutation propagation rests on 5 variants.** *1 run.* "A mutation's effect leaves the mutated
   position within blocks 0–7" comes from five BRCA1 variants (4 LOF, 1 benign). The claim is marked
   "measured once" in the log and on the page, but the reviewer is right that it should not be stated
   as a finding at this n. Target: 50+ variants, spread across genes, reported as a distribution of
   hand-off depth.
5. **The repeat test used one sequence family.** *1 run.* Round 4 ran 16S rRNA only. The repeat
   finder was fixed afterwards and now returns **six** families in E. coli (including IS2 ×7 and IS3
   ×5), so the tooling is ready and the run is cheap. Target: n ≥ 3 families, as the reviewer asks.
6. **The amino-acid designs cover two amino acids.** *design work.* Isoleucine and arginine are the
   only ones the genetic code allows for a same-site, substitution-type-matched pair. Going beyond
   them means giving up same-site matching and controlling across sites instead — a different design,
   not a bigger run.

## C. Controls that are designed but not built

7. **A health-matched control for the far-context test.** *code.* Specified in
   `RESEARCH_LOG.md` (round 3b) and never built: degrade unrelated components until health matches
   attention's 0.571, then re-measure. Without it the far-context question stays unanswerable, which
   is why that claim is withdrawn rather than reported.
8. **A damage-curve control for the MR / reading-frame claim.** *code.* Removing MR is the most
   damaging family ablation, so the rhythm drop could be damage rather than MR's specific role. Needs
   the same health-matching logic as item 7.
9. **Close the reopened scrambled-null leg behind the SE result.** *code.* Round 5 dismissed it by
   calling the fully scrambled model near-dead; round 5b showed that style of argument failing for
   block 0. Needs the per-block equivalent of `nullmodel.block0_sensitivity`.
10. **Make the SE claim causal.** *code.* `peak_blocks` says where a difference shows up, not that SE
    computes it. Ablate or patch SE writes on the same paired sites. Until then the verb is **"peaks
    in"**, never "carries".

## D. Generalisation

11. **A second checkpoint.** *1 run.* Everything is `evo2_7b`. Rerunning the notebooks with
    `MODEL_NAME = 'evo2_7b_base'` needs no new code. Every reviewer will ask, and both external
    reviews did.
12. **A second genome.** *1 run.* All biology is E. coli. The reviewer asks for biological
    replication on a different dataset; a second bacterium with GenBank annotations is the cheapest
    version.

## E. Framing and citation hygiene

13. **Do not write "Evo 2 knows amino acids".** The reviewer's proposed wording for round 4 is
    already the repo's position: *initial synonymous-versus-missense experiments were confounded by
    substitution class and therefore do not establish amino-acid-level representation.* Round 5 then
    ran the matched controls and the effect survived, so the current defensible claim is narrower and
    more specific than either: *at isoleucine and arginine sites in E. coli, in one checkpoint, a
    substitution that changes the amino acid disturbs Evo 2 more than a silent one at the same
    position — with substitution type held equal (72.0%, n = 150) or reversed (80.0%, n = 150), and
    with codon frequency working against the result at 266 of 300 sites.*
14. **Read every `SUMMARY` and `SECONDHAND` entry in `RELATED_WORK.md` before citing it.** Michalak &
    Abreu, the ICML 2026 load-bearing paper, the pruning study, Bick, Arora, Lu.
15. **Rename or explicitly distinguish "load-bearing layers",** which an ICML 2026 paper already uses
    on Evo 2 with a different method. Check its layer numbering (0- or 1-based) before citing any
    layer number.
16. **Do not claim** that our funnel explains that paper's deep load-bearing layer.
17. **Credit BioRiskEval for the magnitude growth** wherever the funnel is discussed, and
    Mathur & Sachidanandam for the codon-usage half of the "means but not preferred" dissociation.

---

## Cheapest path to a defensible paper

| step | items | cost |
|---|---|---|
| 1 | Report uncertainty from data already stored (1) | re-analysis, hours |
| 2 | Repeat families n ≥ 3, and the codon-rhythm replication (5, 2) | 1 run |
| 3 | Second checkpoint on the existing notebooks (11) | 1 run |
| 4 | SE causal test + the scrambled-null leg (9, 10) | 1 run + code |
| 5 | Mutation propagation at n ≥ 50 (4) | 1 run |
| 6 | Health-matched controls (7, 8) | code + 1 run |

Steps 1–4 are what turn the strongest claims from suggestive into solid. Steps 5–6 close the two
results currently marked "measured once" and "withdrawn".

**What is already done** and should not be re-litigated: round 5's matched substitution designs, the
codon-rarity control (266 of 300 sites), the per-amino-acid split, the size-matched SE comparison,
the block-0 null with six seeds, and determinism replication of the whole round-3 result set in
round 3b.
