# Pre-registered predictions

Borrowed from LARQL's `META_MODEL.md`. Each prediction is written down
**before** the experiment that tests it, together with what would refute
it. After a run, **append** an `Outcome (date):` block under the prediction.
Never edit a prediction in place. If it was wrong, say so and write the
corrected model underneath.

All predictions below were registered on 2026-09-21. They concern
`evo2_7b` unless stated otherwise.

A standing rule, also from LARQL: a probe shorter than an operator's reach
cannot distinguish that operator. MR reaches 128 letters per block, so a
"long-range" claim needs gaps of at least 1,000 letters.

---

### P1: Exact copying across long distances is done by attention, not LI

**Test:** `scripts/run_copy_test.py`, gaps 100 / 1,000 / 10,000 / 50,000, a
200-letter random insert, 3 seeds.

**Prediction:**
- The full model's second-copy accuracy is far above its first-copy accuracy
  at every gap.
- Mean-ablating all attention mixers (`-attn`) brings second-copy accuracy
  to within 0.10 of first-copy accuracy at gaps ≥ 1,000.
- Mean-ablating all LI mixers (`-li`) lowers second-copy accuracy by less
  than 0.10 at those gaps.
- At gap 100, `-attn` hurts less than at gap ≥ 1,000, because MR can bridge
  short gaps.

**Refuted if:** `-attn` leaves second-copy accuracy within 0.10 of the full
model at gap ≥ 10,000 (something other than attention copies), or `-li`
destroys copying as badly as `-attn`.

**Outcome (2026-09-21, round 1, `evo2_7b`, gaps 100 / 1,000 / 10,000, 2 seeds):** PARTLY CONFIRMED.
- Full model: second-copy accuracy 1.000 / 1.000 / 0.997 vs. first copy ~0.23. Confirmed.
- `-attn`: second copy 0.253 / 0.244 / 0.247 (chance) while the model still worked on genes (codon accuracy
  0.70–0.76). Confirmed: attention does the copying.
- "At gap 100, `-attn` hurts less": REFUTED. It hurt just as much; even 100-letter repeats need attention.
- "`-li` lowers copying by < 0.10": UNTESTABLE. `-li` (like `-se` and `-mr`) broke the whole model (uniform
  guessing), because it also removed block 30, which appears to dominate the output (RESEARCH_LOG round 1, finding 2).
  Round 2 repeats this without the bottleneck block and with a health check.


---

### P2: Regional context (far upstream DNA) travels through LI

**Test:** `truncation_curve` on gene spans, with upstream context of 500
and 50,000 letters. The context benefit is mean log-p(full) − mean
log-p(500). Measure that benefit under `-li` and under `-attn`.

**Prediction:** the full model shows a positive context benefit. `-li`
shrinks it by ≥ 50%. `-attn` shrinks it by < 50%.

**Refuted if:** `-attn` removes more of the benefit than `-li`, or neither
alone removes ≥ 50% (redundancy; then ablate both together and record that
as the finding).

**Outcome (2026-09-21, round 1, `evo2_7b`, 1 gene: ilvI):** INCONCLUSIVE. The full-model context benefit was
tiny: +0.0167 nats per letter (500 → 50,000 letters of upstream context). `-attn` removed it (→ −0.0027) while the model
still worked (log-prob −0.64). `-li` and `-mr` pushed the model to uniform guessing (log-prob −1.38), so their "zero
benefit" is uninterpretable. That hints against P2, but it's one gene with a tiny effect. Round 2: 5 genes, families
ablated without the bottleneck block, health-flagged.


---

### P3: The reading frame (codon structure) is carried by SE

**Test:** `codon_phase_accuracy` on forward-strand E. coli genes, full model
vs `-se`, `-mr`, `-li`, `-attn`.

**Prediction:** the full model's accuracy differs by codon position, with
position 3 (the wobble position) the lowest. `-se` shrinks the gap between
the best and worst codon position by ≥ 50%, while changing intergenic
accuracy proportionally less than coding accuracy.

**Refuted if:** another operator type removes more of the periodicity than
`-se`.

**Outcome (2026-09-21, round 1):** PARTLY CONFIRMED / UNTESTABLE. The full model is least accurate at codon
position 3 (the wobble position): 0.936 / 0.964 / 0.700, intergenic 0.618. Confirmed. But `-se` broke the model
(accuracy 0.000–0.003, below chance), and so did `-mr` and `-li` (≈ chance). Only `-attn` left it working
(0.701 / 0.759 / 0.585). A broken model loses every rhythm, so "SE carries the rhythm" can't be tested by
family ablation. Round 2: single-layer ablations with a health check.


---

### P4: How far LI looks, read from the weights, predicts its measured context use

**Test:** `scripts/filter_reach.py` (weights), then per-block LI ablation on
`truncation_curve`.

**Prediction:** the LI blocks' median reach99 spans at least two orders of
magnitude, so some LI blocks are effectively local and some are global.
Across LI blocks, weight-read reach and the context benefit lost when that
single block is ablated have Spearman correlation > 0.5.

**Refuted if:** all LI blocks have similar reach (within 10×), or the
correlation is ≤ 0.

**Outcome (2026-09-21, round 1, weights of `evo2_7b`):** FIRST PART REFUTED. The median reach99 of the LI blocks
is 4–7 letters for all nine LI blocks (block 2: 5, 6: 6, 9: 6, 13: 6, 16: 7, 20: 7, 23: 6, 27: 4, 30: 4), not
spread over two orders of magnitude. Long reach exists only in a minority of channels (longest channel per block, up
to 4,528 letters in block 2 and 1,303 in block 27). Corrected model: LI blocks are mostly local, with a few
long-range channels. The correlation part was not run.


---

### P5: Coding SNVs are judged locally

**Test:** `scripts/explain_variant.py` on BRCA1 coding SNVs from the evo2
BRCA1 notebook data, with an 8,192-letter window and a 200-letter
downstream span.

**Prediction:** the operator-type groups "all se mixers" plus "all mr
mixers" restore more of the downstream disturbance than "all li mixers" plus
"all attn mixers". In the `distance` profile of the letter after the
variant, ≥ 80% of the Hyena blocks' absolute direct contribution comes from
lags < 128.

**Refuted if:** LI or attention groups dominate the patching restoration, or
most of the Hyena contribution comes from lags ≥ 128.

**Outcome (2026-09-21, round 1):** NOT TESTED (design flaw). Patching every position of a late component
"restored ~100%" for almost any component, because (hypothesis) block 30's write dominates the output, so everything
on the path into it looks responsible. The FUNC variant's downstream effect was only +0.25, so its fractions (±200%)
were noise. Zero-shot scores went the right way for the one pair tested (LOF −0.00134 vs FUNC +0.00025). Round 2:
patch at the mutation site only, only variants with |effect| ≥ 1, plus a 40-variant AUROC check.


---

### P6: The first layer is a motif detector bank with recognisable biology

**Test:** `scripts/block0_motifs.py`.

**Prediction:** ≥ 10% of block-0 channels have a top-100 PWM with
information content ≥ 8 bits (sharp motifs, not diffuse composition).
Among the sharpest channels there are recognisable signals: start codon
ATG, stop codons TAA/TAG/TGA, and Shine-Dalgarno-like AGGAGG.

**Refuted if:** fewer than 2% of channels exceed 8 bits (block 0 encodes
composition, not motifs), or none of the named signals appears among the
top 100 channels by information content.

**Outcome (2026-09-21, round 1):** CONFIRMED AS WRITTEN, BUT THE TEST WAS TOO WEAK. 67.8% of channels have
information content ≥ 8 bits, and channels whose top-50 inputs are ≥ 80% start/stop/Shine-Dalgarno exist
(ATG 46, TAA 145, TAG 50, TGA 41, AGGAG 1). However, the 8-bit threshold passes almost any channel (the top 100 of
262,144 inputs look alike by construction), and the motif counts had no control for letter composition (AT-rich
channels contain TAA by chance). Round 2: composition-controlled counts and an exact per-position importance measure.


---

## Round 2 predictions (registered 2026-09-21, before running `notebooks/marv_hyena_round2_colab.ipynb`)

### P7: Block 30 is an output bottleneck

**Test:** R2.1, write norms at 3 positions in 3 genome regions.

**Prediction:** block 30's mixer write is ≥ 90% of the final residual's norm in all three regions, and no other
single write exceeds 10%.

**Refuted if:** block 30's share is < 50% anywhere, or another block's write is comparable in size.

---

### P8: With block 30 kept on, LI does not copy

**Test:** R2.3, `-li (keep L30)` on the copy test, with the health check.

**Prediction:** the model stays healthy (health accuracy ≥ 0.5, not flagged broken), and second-copy accuracy stays
≥ 0.9 at gaps 100 / 1,000 / 10,000.

**Refuted if:** the model stays healthy but copying falls below 0.5, meaning LI takes part in copying.
If the model is flagged broken, the result is UNTESTABLE again, not refuted.

---

### P9: Some single SE layer carries the codon rhythm

**Test:** R2.2, single-mixer ablations.

**Prediction:** at least one single SE layer reduces codon rhythm (mean accuracy at positions 1–2 minus position 3) by
≥ 50% while health accuracy stays ≥ 0.5. No single attention layer does.

**Refuted if:** no healthy single-layer ablation of any kind halves the rhythm (the rhythm is distributed), or the
layers that do are not SE.

---

### P10: Start/stop detectors survive a composition control

**Test:** R2.4, `motif_channels` with composition control (≥ 80% of top-50 inputs contain the motif, and ≥ 3× the
composition-matched chance rate).

**Prediction:** ≥ 10 controlled channels each for ATG and for at least two of the three stop codons. The control
motifs CCC and GCG have fewer controlled channels than ATG. Mean position importance is highest at the most recent
3 positions.

**Refuted if:** fewer than 5 controlled channels survive for ATG and for every stop codon, meaning the round-1 counts were
composition artifacts.

---

### P11: Zero-shot scores separate harmful from harmless BRCA1 variants

**Test:** R2.5, 20 LOF + 20 FUNC variants, AUROC of −delta_logp.

**Prediction:** AUROC ≥ 0.65. This is a small-sample sanity check; the Evo 2 paper reports strong separation on
the full set.

**Refuted if:** AUROC < 0.55.
