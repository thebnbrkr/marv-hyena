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
