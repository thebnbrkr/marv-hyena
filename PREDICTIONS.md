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

**Outcome (2026-09-21, round 3, the LI part, now testable):** REFUTED. With the load-bearing layers kept on, `-li` is
healthy and copying at the 10,000 gap drops from 0.997 to 0.556 (a 0.44 drop, far more than the predicted < 0.10).
Short gaps are barely affected (0.986 at 100). Corrected model: attention is essential for copying at every distance,
and LI contributes substantially at long range.

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

**Outcome (2026-09-21, round 3, fair test with load-bearing layers kept on):** REFUTED. `-se` shrinks the codon
rhythm from 0.251 to 0.109 (−57%, which meets the "≥ 50%" bar), but `-mr` removes more: 0.251 → −0.016 (−106%, the
rhythm is gone). That matches the refutation condition "another operator type removes more of the periodicity than
`-se`". Health under `-mr` on the gene window is 0.461 vs. 0.571 under `-se`, so both models are degraded but working.
Corrected model: the reading frame is carried mainly by the MR (128-letter) layers.

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

**Outcome (2026-09-21, round 2):** CONFIRMED, more strongly than predicted. Block 30's mixer write is 100.0% of the
final residual's norm in all three regions (7.1e11, 4.4e12, 1.3e12). The next-largest write is block 29's MLP at
~6e6, about 123,000× smaller. Block 30's MLP (~4e-15) and block 31 (mixer 0.38, MLP ~7e-14) write almost nothing.
Ablating block 31 gives bit-identical outputs. In bf16, every write before block 30 is below rounding resolution
once block 30 writes, so the output is a function of block 30's output alone.

---

### P8: With block 30 kept on, LI does not copy

**Test:** R2.3, `-li (keep L30)` on the copy test, with the health check.

**Prediction:** the model stays healthy (health accuracy ≥ 0.5, not flagged broken), and second-copy accuracy stays
≥ 0.9 at gaps 100 / 1,000 / 10,000.

**Refuted if:** the model stays healthy but copying falls below 0.5, meaning LI takes part in copying.
If the model is flagged broken, the result is UNTESTABLE again, not refuted.

**Outcome (2026-09-21, round 2):** UNTESTABLE AGAIN. `-li (keep L30)` still broke the model (health 0.214), because
LI contains a second load-bearing layer: L9 alone breaks the model (health 0.254). Side evidence from single-layer
ablations: removing L2 (the LI block with the longest-reaching filters) lowers copying at the 10,000 gap from 0.994 to
0.822 with the model healthy (0.845). So LI takes a modest part in long-range copying. That's not below the 0.5
refutation bar, but it goes against "LI does not copy". Round 3: family ablations that keep L0, L1, L9, L29, L30 on.

**Outcome (2026-09-21, round 3):** REFUTED once testable. Keeping all load-bearing layers on (L0, L1, L4, L9, L29,
L30), `-li` is healthy (0.528) and copying at the 10,000 gap falls to 0.556. That's not below the 0.5 refutation bar, but
LI clearly takes part in long-range copying, which goes against "LI does not copy".

---

### P9: Some single SE layer carries the codon rhythm

**Test:** R2.2, single-mixer ablations.

**Prediction:** at least one single SE layer reduces codon rhythm (mean accuracy at positions 1–2 minus position 3) by
≥ 50% while health accuracy stays ≥ 0.5. No single attention layer does.

**Refuted if:** no healthy single-layer ablation of any kind halves the rhythm (the rhythm is distributed), or the
layers that do are not SE.

**Outcome (2026-09-21, round 2):** REFUTED. No healthy single-layer ablation halves the codon rhythm (normal
0.251). The largest healthy drops are L10 (attention) → 0.166 (−34%) and L5 (MR) → 0.191 (−24%). No SE layer comes
close (L0 kills the rhythm but breaks the whole model, health 0.288). Corrected model: the reading-frame signal is
distributed across layers, with attention and MR contributing at least as much as SE.


---

### P10: Start/stop detectors survive a composition control

**Test:** R2.4, `motif_channels` with composition control (≥ 80% of top-50 inputs contain the motif, and ≥ 3× the
composition-matched chance rate).

**Prediction:** ≥ 10 controlled channels each for ATG and for at least two of the three stop codons. The control
motifs CCC and GCG have fewer controlled channels than ATG. Mean position importance is highest at the most recent
3 positions.

**Refuted if:** fewer than 5 controlled channels survive for ATG and for every stop codon, meaning the round-1 counts were
composition artifacts.

**Outcome (2026-09-21, round 2):** PARTLY CONFIRMED.
- Controlled channels: ATG 46, TAA 65 (down from 145; 80 were composition artifacts), TAG 50, TGA 41,
  AGGAG 1. "≥ 10 for ATG and two stops" is confirmed.
- Position importance is highest at the current letter (0.204), then −2 (0.142), −1 (0.125). "Most recent 3
  positions highest" is confirmed.
- "Control motifs have fewer channels than ATG": REFUTED for GCG (53 > 46); holds for CCC (8).

So 3-letter-word detectors are common in block 0, and ATG/stops are not shown to be special without a comparison
across all 64 three-letter words. Also found: the main-effect importance measure scores purely combinatorial channels
as ~0 (e.g. channel 263, top inputs ending in ATGC).


---

### P11: Zero-shot scores separate harmful from harmless BRCA1 variants

**Test:** R2.5, 20 LOF + 20 FUNC variants, AUROC of −delta_logp.

**Prediction:** AUROC ≥ 0.65. This is a small-sample sanity check; the Evo 2 paper reports strong separation on
the full set.

**Refuted if:** AUROC < 0.55.

**Outcome (2026-09-21, round 2):** CONFIRMED. AUROC 0.880 on 20 LOF + 20 FUNC variants. Mean delta log-likelihood:
LOF −0.00458 vs. FUNC −0.00100. Mean downstream effect: −22.3 vs. −4.4 nats.

---

## Round 3 predictions (registered 2026-09-21, before running `notebooks/marv_hyena_round3_colab.ipynb`)

### P12: The block-30 blow-up is in the weights, not the arithmetic

**Test:** R3.1, block 30's mixer write recomputed in float32; logits recomputed in float32 with and without every
earlier write.

**Prediction:** the float32 norm is within 1% of the bf16 norm in all 3 regions. At float32, adding back every earlier
write changes the logits by < 1% of their scale.

**Refuted if:** the float32 norm differs by > 10% (arithmetic artifact), or earlier writes shift the float32 logits by
> 10% (they would matter at full precision).

**Outcome (2026-09-21, round 3):** CONFIRMED. float32 vs. bf16 norm: 6.394e11 vs. 6.418e11, 2.2104e12 vs. 2.2116e12,
1.7612e12 vs. 1.7525e12 (all within 0.5%). Earlier writes change the float32 logits by ≤ 2.2e-5 on a scale of 12–24.

---

### P13: With the load-bearing layers kept on, LI takes a minor part in copying and carries some far context

**Test:** R3.3, families ablated except the load-bearing layers found in R3.2.

**Prediction:** `-li` (keeping the load-bearing layers) is healthy (not flagged broken). Its copying at the 10,000
gap falls to between 0.5 and 0.95 (L2 alone gave 0.82 in round 2), and it removes < 50% of the far-context benefit.
`-attn` still kills copying (≤ 0.3) and removes ≥ 80% of the context benefit.

**Refuted if:** healthy `-li` leaves 10,000-gap copying ≥ 0.95 (LI plays no part), or kills it ≤ 0.3 (LI is a
main copier); or `-li` removes more of the context benefit than `-attn`. If `-li` is still broken, UNTESTABLE.

**Outcome (2026-09-21, round 3):** COPYING PART CONFIRMED; CONTEXT PART NOT RUN (out of memory on a 40 GB GPU).
Load-bearing on this run: L0, L1, L4, L9, L29, L30. `-li` (keeping those on) is healthy (0.528, not broken) with copying
0.986 / 0.944 / 0.556 at gaps 100 / 1,000 / 10,000, inside the predicted 0.5–0.95 at 10,000. `-attn`: 0.253 / 0.244 /
0.247 (≤ 0.3, confirmed).

---

### P14: Block 30 reads locally, mostly from the latest writes

**Test:** R3.4, integrated-gradients attribution at block 30's input, 3 gene + 3 intergenic positions.

**Prediction:** completeness error < 10% at every position. ≥ 80% of the absolute attribution comes from ≤ 8 letters
back (block 30's filters reach ~4 letters). Blocks 28–29 (the largest writes entering block 30) get ≥ 50% of the
absolute attribution.

**Refuted if:** completeness error > 25% (the method fails here), or most attribution comes from > 16 letters back.

**Outcome (2026-09-21, round 3):** NOT RUN. It crashed on a bug: Vortex loads weights inside torch.inference_mode(),
and autograd rejects those tensors. Fixed (`interface._autograd_safe`, with a reproduction test).

---

### P15: A mutation's signal leaves the mutated position within the first ~10 blocks

**Test:** R3.5, residual patching at the mutation site after each block, 5 large-effect BRCA1 variants.

**Prediction:** the fraction of the effect transferred stays ≥ 0.9 through block 2, then falls below 0.5 by block 10
for at least 4 of 5 variants.

**Refuted if:** the fraction stays ≥ 0.5 past block 20 for most variants (the signal stays at the site until late),
or it falls below 0.5 already at block 0.

**Outcome (2026-09-21, round 3):** PARTLY REFUTED. The signal leaves the site even earlier than predicted.
- "< 0.5 by block 10": held for 5/5 variants.
- "≥ 0.9 through block 2": held for 0/5.
- The refutation condition "below 0.5 already at block 0" was met for 2/5 (41256881: 0.09; 41256880: 0.07), and the
  FUNC variant was at 0.50.

Corrected model: mutation effects are handed to neighbouring positions very early, either in block 0 directly or by
blocks 4–7.

---

### P16: ATG and the stop codons rank high among all 64 three-letter words

**Test:** R3.6, controlled detector counts for all 64 words.

**Prediction:** ATG and at least two of TAA/TAG/TGA rank in the top 16 of 64.

**Refuted if:** ATG and all three stops rank below the median (then round 2's "codon detectors" were generic
3-letter-word detectors).

**Outcome (2026-09-21, round 3):** NOT CONFIRMED. ATG ranks 31/64 (46 channels, exactly the median of 46). Stops: TAA
#1 (65), TAG #26, TGA #48. Only one stop is in the top 16. The refutation condition is not met (TAA is #1), but the
reading is that block 0 is a generic 3-letter-word detector bank, and start/stop codons are not special.
Caveat: 724 dead channels were included in the counts (P17).

---

### P17: Total-effect importance catches the channels the main effect missed

**Test:** R3.6, total-effect index from the full enumeration.

**Prediction:** channel 263 (zero main effect in round 2) has ≥ 80% of its total effect in its last 4 positions,
matching its `…ATGC` top inputs. Mean total effect peaks within the last 3 positions.

**Refuted if:** channel 263 has near-zero total effect everywhere (then the channel is dead, not combinatorial).

**Outcome (2026-09-21, round 3):** REFUTED FOR CHANNEL 263 / PEAK CONFIRMED.
- Channel 263 has zero total effect at all 9 positions: it's a dead channel, and 724 channels (18%) are.
- Mean total effect peaks at the current letter (0.309) and 2 back (0.301), which is within the last 3 positions.
- The indices sum to 1.75 on average, meaning strong letter-combination effects.

---

## Round 3b outcomes (rerun of the two failed steps, 2026-09-23)

Round 3's R3.3 far-context step ran out of memory and R3.4 crashed. Both were fixed (`memory.py`,
`interface._autograd_safe`) and the whole notebook was rerun on an 80 GB A100 / CUDA 13.0, versus round 3's 40 GB /
CUDA 12.8. Raw outputs: `results/round3b/`. The ten result keys shared with round 3 came back **bit-identical**,
including baseline health to 16 digits (0.6959707140922546), so findings 12 and 14-17 replicated exactly across two
A100 SKUs and two CUDA versions. These outcomes are appended, not substituted; the round-3 entries above stand.

### P13, context half (was: NOT RUN, out of memory)

**Outcome (2026-09-23, round 3b):** ATTENTION CLAUSE CONFIRMED; LI CLAUSE UNTESTABLE.

Baseline far-context benefit is 0.0160 nats/letter over 5 genes (creD .0163, gspE .0236, ilvI .0167, uup .0078,
yehQ .0158), consistent with round 2's finding 9.

- `-attn` removes **109%** of it (mean −0.0014), on 5/5 genes, none flagged broken, model still well clear of
  guessing (−0.59 to −1.25 vs −1.386). The predicted ≥80% is met. **Far context goes through attention.**
- `-li` (keeping the load-bearing layers on) came out at mean −0.1394, apparently removing 970%. That number is
  **an artifact of a single broken run**: yehQ under `-li` scored −2.18 nats/letter, worse than uniform guessing, and
  is flagged `broken`. Excluding it, the remaining four genes average **+0.0268** — LI removes none of the benefit.
  P13's own clause applies: *"If `-li` is still broken, UNTESTABLE."* So the LI half is untestable, not confirmed,
  and the refutation condition ("`-li` removes more than `-attn`") is **not** met on the healthy runs.

**Caveats to carry forward.** Every ablated condition sits at −0.96 to −1.37 nats against baseline's −0.30 to −1.02,
so "benefit removed" is partly confounded with "model degraded" — a model near guessing has little headroom to show a
0.016-nat gain. `-mr` going negative on 4/5 genes is most likely this. The surviving `-li` genes swing +.0715 /
+.0565 / −.0019 / −.0188, a spread four times the signal. yehQ is the weakest gene at baseline and breaks in two
conditions; it is a poor probe and should be replaced. Round 4 should raise the gene count and match conditions on
baseline headroom before this is reported as a number.

### P14 (was: NOT RUN, crashed)

**Outcome (2026-09-23, round 3b):** UNTESTABLE — the method fails its own gate.

The autograd fix works and all 6 positions ran. Every one fails the completeness self-check at ~100%:

| region | pos | f(u)−f(base) | sum(attr) | completeness err |
|---|---|---|---|---|
| gene | 204001 | +3.173 | −0.0005 | 100.0% |
| gene | 305848 | +0.087 | +0.0008 | 99.1% |
| gene | 399989 | +5.886 | −0.0022 | 100.0% |
| intergenic | 209630 | +0.834 | +0.0000 | 100.0% |
| intergenic | 303820 | +4.315 | −0.0002 | 100.0% |
| intergenic | 399582 | −0.156 | +0.0005 | 100.4% |

P14 pre-registered "completeness error > 25% (the method fails here)", so this is UNTESTABLE, and under invariant 1
nothing underneath is reportable — including the striking near-exact cancellation of L29's mixer (+0.206) against its
own MLP (−0.207) at every position. Integrated gradients attributes zero where the real change is up to 5.9.

The failure has a consistent shape: the gradient reaches back **exactly one block**. Share of absolute attribution is
L29 97.7–100.0%, L28 0.0–2.3%, L27 and earlier 0.0% — the funnel running backwards, the gradient dying as fast as the
activations grow. The likely cause is that `f` along the straight path from the embedding-only baseline to the real
residual is near-discontinuous (block 30 amplifies to ~10¹², the final RMSNorm divides the scale back out, so `f`
depends on a direction that can flip sharply at one point on the path), and 32 midpoint samples step over it.
Diagnosis and replacement are R4.0 and P18 below.

---

## Round 4 predictions (registered 2026-09-23, before running `notebooks/marv_hyena_round4_colab.ipynb`)

Round 4 answers three challenges to the round-3 results, two of them raised by a biologist reading the lab notebook,
one owed to the literature review since the round-3 novelty check. Ordered so that the test which can *invalidate
existing findings* runs first.

### P18: the integrated-gradients failure is a sharp path, not a wrong gradient

**Test:** R4.0, `f(e + a(u−e))` evaluated on a dense grid of 512 values of `a` at the same 6 positions, forward passes
only.

**Prediction:** `f` is not smooth along the path. At least one adjacent pair of grid points differs by ≥ 25% of the
total `f(u) − f(e)` range, and ≥ 80% of the total variation is concentrated in < 10% of the path.

**Refuted if:** `f` varies smoothly (no single step above 10% of the range), in which case integrated gradients should
have worked and the ~100% completeness error is a bug in `interface.block_input_attribution`, not a property of the
model.

**Either way:** attribution at block 30's input moves to causal mean-ablation of each incoming write (R4.0b), which
needs no autograd, no completeness assumption, and yields a total effect rather than a direct one.

---

### P19: block 0's word-detector bank is architectural, not learned

**Test:** R4.1, the full 4⁹ enumeration rerun with block 0's weights shuffled (`nullmodel.random_weights`,
mode="shuffle", 3 seeds), compared against the trained block 0 on the same composition-controlled 64-word counts.

This is the null model the bio-foundation-model review (bioRxiv 2026.03.04.709491) asks for and the round-3 literature
review recorded as owed. Shuffling preserves the weight multiset exactly, so anything that survives is a property of
the architecture and the weight distribution, not of training.

**Prediction (the honest expectation, which is that our own finding is mostly a null result):** the shuffled block 0
also produces detector channels for all 64 three-letter words with a flat distribution — its coefficient of variation
across the 64 words is within 50% of the trained model's, and its median count is within a factor of 2 of 46.

**Refuted if:** the shuffled null gives a visibly different distribution — CV differing by more than 2×, or median
count below 10 or above 150.

**What each outcome costs us.** If confirmed, findings 4, 10, 15 and 16 describe the *architecture* of a gated
9-letter convolution, not anything Evo 2 learned, and every block-0 claim in `RESEARCH_LOG.md` and the README must be
reworded to say so. If refuted, block 0 learned something, and the difference from the null is the first honest
description of *what*.

**Secondary:** the 724 dead channels (18%). Prediction: the shuffled null has fewer than half as many dead channels
(< 9%), because deadness is a learned outcome rather than an architectural one.

---

### P20: the copying circuit fires on real genomic repeats, not just our synthetic one

**Test:** R4.2, `genome.repeat_probes` on E. coli's real repeat families (rRNA operons, IS elements), three arms at
matched geometry — real repeat / letter-shuffled repeat / random insert — scored under the round-3 family ablations
with the load-bearing layers kept on. The statistic is the **retrieval gain**, `second_lp − first_lp`, not the raw
second-copy score: a real rRNA copy scores well on its first appearance because the model knows rRNA, and only the
gain isolates what retrieval added.

**Prediction:** the real arm shows a substantial retrieval gain (≥ 0.5 nats at a 1,000-letter gap), and `-attn`
removes ≥ 70% of it while leaving `first_lp` within 0.15 nats of baseline. The gain on the real arm is within a
factor of 2 of the gain on the random arm.

**Refuted if:** the real arm's retrieval gain survives `-attn` (< 30% removed), which would mean the round-1-3 copying
result was specific to the synthetic probe and attention is not what reads real repeats; or the real arm shows
essentially no gain (< 0.1 nats), which would mean the model predicts real repeats from prior knowledge alone and
never retrieves.

---

### P21: Evo 2 represents amino acids, not just letters

**Test:** R4.3, `codons.wobble_sites` + `translation_test` on ≥ 100 forward-strand CDS codon sites in E. coli where
substituting the third letter can be either silent or missense. Same site, same codon position, same edit distance;
the only difference is whether the encoded amino acid changed. Premature stops excluded (`allow_stop=False`).

**Prediction (decided on the paired sign test):** the missense substitution disturbs the model more than the silent
one **at the same site** in ≥ 65% of sites (`nonsyn_more_disruptive_frac`). This statistic is paired, threshold-free
and sign-aware, which is why it is the one that decides P21.

**Secondary, reported with its sample size:** the median ratio of downstream effects is ≥ 1.2. The ratio is only
defined at sites where the silent arm's own effect is clearly disruptive (< −0.25 nats over the 200-letter
downstream span); a ratio of two signed quantities that straddle zero is not a statistic, which is the mistake round
1 made with its FUNC variant's ±200% "fractions". If `n_usable` < 20 this number is not reported at all.

**Refuted if:** missense and silent are indistinguishable — between 45% and 55% of sites on the sign test — which
would say Evo 2 models nucleotide statistics and the codon rhythm of finding 13 is periodicity without meaning.

**Localisation (reported either way):** the block at which the missense and silent arms diverge most,
`peak_divergence_block` from `codons.divergence_by_block`. Prediction: the modal peak block is > 7, i.e. later than
the early hand-off stage of finding 14, because reading an amino acid needs the whole codon assembled.

---

### P22: nonsense beats missense

**Test:** R4.3 rerun with `allow_stop=True`, comparing sites whose non-synonymous alternative is a premature stop
against those whose is an ordinary amino-acid change.

**Prediction:** premature stops disturb the model more than missense changes, by a ratio of median downstream effects
≥ 1.5 (comparing the nonsense arm's median `nonsyn_effect` against the missense arm's, both over `usable` sites).
This is the behavioural counterpart of Evo 2's SAE feature f/24278, which the Evo 2 paper reports firing on
frameshifts and premature stops.

**Refuted if:** stops and ordinary missense changes are indistinguishable (ratio within 1.0 ± 0.1), which would put
our measurement at odds with the published SAE feature and mean one of the two is not measuring what it claims.

**Untestable if** fewer than 20 nonsense sites clear the `usable` threshold in the window, in which case widen the
track rather than reporting a number.

---

## Round 4 outcomes (2026-09-23)

Run on an 80 GB A100, smoke checks 8/8, 60/60 tests, all 19 cells clean, load-bearing `[0,1,4,9,29,30]` for the third
consecutive run. Raw outputs: `results/round4/`.

### P18 — PARTLY CONFIRMED

A **single step out of 512** carries 97.4–99.9% of `f`'s range at all six positions (predicted ≥25%). `f` is a step
function, so round 3b's ~100% completeness failure was not a bug in the attribution code. The refutation condition
(smooth, no step above 10%) is nowhere near met.

The second clause failed: 80% of the *total variation* needs 25–46% of the path, not <10%. Not a contradiction —
`f` is a near-discontinuous jump **plus** heavy high-frequency noise. Both defeat integrated gradients: midpoint
sampling steps over the jump, bf16 jitter swamps the gradient elsewhere.

Causal replacement (R4.0b) inverts the answer. IG said L29 97.7–100%, L28 ≤2.3%, earlier 0.0%. Mean-ablation says
L29 12.0%, L1 10.3%, L28 8.3%, L2 7.9%, L9 7.3%, L0 7.2%, L4 6.4% — reaching **block 0**, and recovering the
load-bearing set by an independent route. Block 30 reads from the whole network.

### P19 — REFUTED (the prediction was that our own finding was an artifact; it is not)

Weight-shuffled block 0 produces **zero** detector channels: all 64 words, all 3 seeds, no NaNs, every value exactly
0. Trained: median 46, mean 45.2, cv 0.280, range 6–65. Predicted the null would match the trained model within 50%
on cv and a factor of 2 on median; it matches on neither.

**Block 0's motif bank is learned.** Findings 4, 10, 15 and 16 survive and now carry the null model the bio-FM
review asks for. Secondary clause (dead channels) is not separately reportable: the null has no live detectors to
compare against.

### P20 — REFUTED, via its own fallback clause

Real 16S arm: first_lp −0.032 at **98.9% accuracy on first appearance**, retrieval gain **0.028** (predicted ≥0.5).
Random arm: first_lp −1.431, gain 1.421. The clause that fires is *"the real arm shows essentially no gain (<0.1
nats), which would mean the model predicts real repeats from prior knowledge alone and never retrieves."*

`-attn` collapses the random/shuffled gain 1.42 → 0.01, and `-li` keeps 69% at a 1k gap but 29% at 10k, replicating
findings 1, 7 and 13 on a new probe. So the circuit is real and attention-dependent; it is simply redundant on a
conserved repeat.

Two limits, both fixable: the run used **one family** (it cloned before the repeat-finder fix — six families,
including IS2 ×7 and IS3 ×5, are available now), and `retrieval_gain` cannot measure anything when `first_lp ≈ 0`.

### P21 — CONFIRMED on all three clauses

| clause | result | predicted |
|---|---|---|
| missense more disruptive than silent, same site | **68.9%** (119 sites) | ≥ 65% |
| median effect ratio | **1.319** (n_usable 48) | ≥ 1.20 |
| modal divergence block | **11** | > 7 |

Robust to the one outlier (68.6% without it). **Evo 2 represents amino acids, not just letters.**

Unpredicted and more interesting: **90.8% of sites peak in SE blocks**, 87.4% in blocks 7/11/14 alone. With finding
13 (MR carries the frame) that gives a division of labour — MR (128 letters ≈ 40 codons) tracks *where* the frame is,
SE (7 letters ≈ 2 codons) reads *what* the codon says.

### P22 — UNTESTABLE, by its own pre-registered gate

Only 2 of 11 nonsense sites cleared `usable`, against the ≥20 required. Direction is unambiguous — **11/11** sign
test, mean effect −38.1 nats vs −0.7 for missense — but not reportable.

Our own design flaw contributed: `usable` gates on the **silent** arm, which is correct for P21 and wrong for P22.
At these sites the silent arm averaged +0.23, disqualifying rows whose nonsense arm was enormous. Round 5: widen the
track, and gate P22 on its own arm.

---

## Round 4 review (2026-09-25): outcomes that need a caveat

Appended, not edited: the round-4 outcomes above stand as recorded. This section records why four of them are
weaker than they read, found by re-analysing `results/round4/results_round4.json` and `results/round3b/`.

- **P21 (CONFIRMED above) is confounded by letter type.** At the third codon letter the silent change is almost
  always a *transition* (A↔G, C↔T) and the missense change a *transversion*. Of the 119 sites, 102 were
  silent-transition vs missense-transversion; missense won 72.5% of those. On the 17 sites where both arms were
  transversions (all isoleucine, ATT/ATC → ATA vs ATG), missense won **8/17 = 47%** (95% CI 26–69%). Those 17
  are themselves confounded the other way (the silent arm lands on ATA, a rare codon; Ile→Met is a mild swap), so
  they cannot show amino-acid blindness either. P21's clauses were met; what they measured is not yet separable from
  "transversions surprise the model more". Tested properly in P25.
- **P21's localisation clause ("modal peak block 11", "90.8% SE") used the argmax of a missense/silent ratio**
  across 32 blocks. At the winning blocks the ratio was 5–7×, against 1.32× end-to-end — the signature of small
  denominators. Per-block numbers were not saved, so it cannot be rechecked from the JSON. Tested in P26.
- **P19 (REFUTED, "block 0 is learned") rests on one hard cutoff.** A channel counts as a word detector only if ≥80%
  of its top-50 inputs share the word. An exact 0 across all 64 words and 3 seeds is what a cutoff cliff produces
  when shuffled channels are *less* selective, not necessarily *un*selective. Tested without a cutoff in P28.
- **P20's "-attn keeps −313%"** is a ratio over a 0.028-nat baseline gain. `genome.summarize_repeat_test` now
  returns NaN below 0.1 nats.
- **P13's context half ("ATTENTION CLAUSE CONFIRMED", round 3b) does not survive its own negative control.** In the
  same run, removing SE — which reaches 7 letters and cannot carry 50,000 letters of context — cut the benefit from
  +0.0160 to +0.0030 (−81%, 4 healthy genes), and MR erased it (−0.0056). Every ablation drops health from 0.696 to
  0.43–0.57 against a 0.016-nat effect: the test measures damage, not a pathway. The attention reading is withdrawn;
  which operator carries far context is unknown. (This reading was written on 2026-09-23 in an uncommitted working
  copy and is restored here.)
- **P22 context:** the Evo 2 paper (Brixi et al., Nature 2026, Fig. 2) already shows, across 36 species, that
  missense and premature-stop changes lower likelihood more than silent ones. The *behaviour* in P21/P22 is not new;
  only its localisation would be. Its main text does not report a transition/transversion or codon-usage control.

---

## Round 5 predictions (registered 2026-09-25, before running `notebooks/marv_hyena_round5_colab.ipynb`)

Every design compares two single-letter substitutions **at the same position**, picked by (kind, letter type) with
`codons.paired_sites`, from sites spread across the whole E. coli genome. "More disruptive" means a more negative
downstream effect (log-prob of the next 200 letters, alt minus ref). The primary statistic for P23–P25 and P27 is the
paired sign test: the share of sites where arm b disturbs the model more than arm a, with a 95% Wilson interval.

### P23: letter type matters even when the protein does not change

**Test:** R5.1 `fourfold` — silent transition vs silent transversion at four-fold degenerate third positions (e.g.
GCT → GCC vs GCA, all alanine).

**Prediction:** the transversion is more disruptive at ≥ 60% of sites.

**Refuted if:** ≤ 55%, in which case letter type is not a live explanation for P21 and round 4's reading stands
stronger.

### P24: …and outside genes, but less

**Test:** R5.1 `noncoding` — transition vs transversion at the same position in unannotated DNA (no feature within
20 letters).

**Prediction:** the transversion is more disruptive at ≥ 55% of sites, and at a lower rate than in P23 (coding
context amplifies letter-type preferences, because the model knows the third codon letter is where variation
normally happens).

**Refuted if:** ≤ 50%, or more than 10 points above P23's rate.

### P25: Evo 2 represents amino acids beyond letter type

**Test:** R5.2. `matched` — silent transversion vs missense transversion (Ile ATx → ATA vs ATG; Arg CGx → AGx vs
GGx). `flipped` — silent transversion vs missense *transition* (Arg AGG → CGG vs GGG; Ile ATA → ATT vs ATG), so
letter type pushes against the hypothesis. Then `controls.paired_regression` over round4 + fourfold + noncoding +
matched + flipped: the within-site difference against Δmissense, Δtransversion, ΔGC and Δcodon-usage together.

**Prediction:** matched ≥ 60% (sign test p < 0.05); flipped ≥ 50%; and the regression's `d_missense` coefficient is
negative with its 95% bootstrap interval entirely below 0 while `d_transversion` is in the model.

**Refuted if:** matched ≤ 55% **and** the `d_missense` interval includes 0 — then round 4's 68.9% was letter type,
and Evo 2 is gene-aware (knows where the wobble position is) without evidence of being protein-aware. Anything in
between is reported as "letter type explains part of it", with the regression coefficients as the split.

### P26: the SE-peak claim does not survive a fair ranking

**Test:** R5.3, round 4's own 120 sites rerun with every block kept, peaks picked three ways (`ratio` as round 4,
`ratio_floor`, `diff`), compared with the same statistic on `fourfold` (no amino-acid change) and on round 4's sites
through a fully weight-shuffled Evo 2 (`round4_shuffled`, 60 sites). Base rate: SE is 9 of 32 blocks (28%).

**Prediction (the review's expectation, against round 4's finding 19):** under `diff`, SE's share of peaks on the
round-4 sites falls below 60%; and SE's share on `fourfold` is within 20 points of its share on `round4`.

**Refuted (finding 19's localisation survives) if:** under `diff` SE's share on `round4` stays ≥ 80%, **and** it is
≤ 50% on `fourfold`, **and** ≤ 43% (base rate + 15) on `round4_shuffled`.

### P27: premature stops are far more disruptive than missense, whatever the letter type

**Test:** R5.4. `stop_matched` — stop transversion vs missense transversion (e.g. Cys TGT → TGA vs TGG).
`stop_flipped` — stop *transition* vs missense transversion (e.g. Trp TGG → TGA vs TGT; Gln CAA → TAA vs AAA). No
`usable` gate: every site enters the sign test.

**Prediction:** the stop is more disruptive at ≥ 90% of `stop_matched` sites and ≥ 85% of `stop_flipped` sites.

**Refuted if:** either is < 70%. **Untestable if** either design has fewer than 30 sites.

### P28: block 0's selectivity is learned, measured without a cutoff

**Test:** R5.5, `controls.best_word_share` for every live channel (dead channels removed, the round-3 open item):
trained vs weight-shuffled (3 seeds) vs Gaussian (3 seeds), plus `cutoff_sweep` at 0.5–0.9.

**Prediction:** the trained median best-word share exceeds the mean shuffled median by ≥ 0.20, and the trained model
has more channels than every null at every cutoff from 0.5 to 0.9.

**Refuted if:** the median gap is < 0.05, or any null has at least as many channels as the trained model at cutoff
0.6 — then "block 0 is learned" is downgraded to "training sharpens a selectivity the architecture already has",
and the 46-vs-0 contrast is reported as a cutoff effect.

---

## Round 5 outcomes (run 2026-09-25, recorded 2026-09-30)

Run on an 80 GB A100 from commit `1f2b29d` (`main`), about 1.5 hours after P23–P28 were committed. Smoke checks 8/8,
83/83 tests, every cell clean, load-bearing `[0,1,4,9,29,30]` and bottleneck `[30]` for the fourth consecutive run.
150 sites per design from the whole genome, round 4's own 120 sites, and 60 sites through the shuffled model. Raw
outputs: `results/round5/`. Verdicts below apply each prediction's registered thresholds as written (also computed
by `controls.score_round5`); the reading after each verdict is not part of the prediction.

### P23 — REFUTED

Four-fold sites, silent transition vs silent transversion: the transversion was more disruptive at **54.7%** (82/150,
95% CI 47–62%, sign test p = 0.29). Predicted ≥ 60%; refuted at ≤ 55%. By the registered clause, **letter type alone
is not a live explanation for P21**. When the protein does not change, swapping a transition for a transversion
barely moves the model (median paired difference −0.10 nats).

### P24 — NEITHER (one clause met, one missed)

Outside genes the transversion was more disruptive at **60.7%** (91/150, CI 53–68%, p = 0.011): ≥ 55% is met. But
the rate is 6 points **above** four-fold sites rather than below, so the "less than in genes" clause fails. It is not
refuted (that needs ≤ 50%, or more than 10 points above P23). Letter type matters a little outside genes and less
inside them, the reverse of the predicted ordering.

### P25 — IN BETWEEN by the registered rule; the sign tests are met strongly, the regression clause is not identifiable

| clause | result | predicted |
|---|---|---|
| `matched`: missense more disruptive, both arms transversions | **72.0%** (108/150, CI 64–79%, p = 6.9e-8) | ≥ 60%, p < 0.05 |
| `flipped`: missense more disruptive, letter type pushing against it | **80.0%** (120/150, CI 73–86%, p = 6e-14) | ≥ 50% |
| pooled regression `d_missense` | **+0.924** (95% CI +0.222 .. +1.689) | negative, CI entirely below 0 |

The regression clause fails in the opposite direction, so the rule gives "in between" and that is the recorded
verdict. A post-hoc check of the regression (not a re-scoring) shows it could not have decided P25 either way.
`d_missense` and `d_transversion` are constant within each design, so the pooled fit is a straight line through
four design means: (1, +1) round4, (1, 0) matched, (1, −1) flipped, (0, +1) fourfold and noncoding. Its intercept
extrapolates to (0, 0), which no design contains. The fit also gives `d_transversion` = +1.77 ("transversions are
*less* disruptive"), which the direct four-fold and noncoding comparisons contradict. Refit on the sign of each
paired difference instead of its size, `d_missense` is −0.17 (CI −0.36 .. +0.02). The design flaw was ours, at
registration.

Scope: these designs exist for only two amino acids. Within them, missense wins at Ile ATC/ATT → ATA vs **ATG**
(65%, p = 0.02; 69%, p = 0.004) and at Arg first-letter sites (CGA, CGG: 93–100%). In `flipped`, Arg CGG → AGG vs
TGG (Trp) wins 96% of 67 sites, while Ile ATA → ATC vs ATG wins only 43% of 23 (p = 0.68). The Ile missense arm is
always ATG, which is also the start codon. Codon rarity works *against* the hypothesis at Ile, because the silent
arm lands on ATA, a rare codon, yet missense still wins.

### P26 — NOT CONFIRMED; the refutation clause holds on two of its three legs

The review predicted that SE's share of peaks would fall below 60% once blocks are ranked by difference. It did not.

| design (ranked by `diff`) | SE share of peaks | modal block |
|---|---|---|
| round4 (amino acid changes) | **85.0%** | 7 |
| matched (amino acid changes, letter type fixed) | 86.0% | 7 |
| fourfold (no amino-acid change) | **11.3%** | 1 |
| noncoding (no gene) | 30.7% | 1 |
| round4 sites, all weights shuffled | 61.7% | 14 |

SE base rate 9/32 = 28.1%. Refutation (finding 19's localisation survives) needed round4 ≥ 80% ✓, fourfold ≤ 50% ✓
and shuffled ≤ 43% ✗. The leg that fails rests on a near-dead model: through the shuffled network, the median
per-block divergence is about 0.01 (trained: 0.1–0.8), and block 0 shows no divergence at all. Recorded as not
confirmed and not refuted. In the trained model, SE peaks appear when the amino acid changes and not when it does
not, which is the comparison the shuffled leg was meant to back up.

### P27 — CONFIRMED

The stop was more disruptive than a missense change at the same site at **93.3%** of `stop_matched` sites (140/150,
p = 2e-30) and **92.0%** of `stop_flipped` sites (138/150, p = 3e-28). Predicted ≥ 90% and ≥ 85%. Median effect:
stop −21.9 vs missense −2.5 nats (matched), −16.9 vs −2.7 (flipped). Letter type set against the stop changes
nothing. The behaviour is in the Evo 2 paper. New here is that it survives a same-site, letter-type-controlled
comparison at 150 sites each.

### P28 — NOT SCORABLE: the null enumeration is degenerate (measurement artifact)

As run, every threshold is met: trained median best-word share 0.98 against **0.000** for all six nulls, and 0
channels at every cutoff. But 0.000 is impossible for a real 9-letter enumeration. Every 9-letter input contains a
3-letter word, so a channel's best-word share is at least 1/64. A share of exactly 0 means the null's top inputs
were **shorter than 3 letters**.

Cause: `motifs.enumerate_block0` re-measures block 0's receptive field on whatever weights are loaded. On scrambled
weights, `receptive_field` (tolerance 1e-3) declares 1–2 letters, so the null was enumerated over 1–2-letter inputs.
The round-4 null (P19) went through the same code path and reported the same exact zeros. The rehearsal on
2026-09-30 reproduces it (shuffled median 0.000). Corroboration from the shuffled-model rows: the scrambled block 0's
output does not move at all when a letter changes. The scrambled block is close to input-insensitive at bf16, which
is a broken block rather than an untrained one.

Consequence: **P19's outcome (round 4: "block 0's motif bank is learned") is unsupported** until the null is rerun
with the receptive field pinned to the trained 9 letters and a null block that still responds to its input. This
does not make the claim false; it makes it untested.

---

## Round 5b predictions (registered 2026-09-30, before running `notebooks/marv_hyena_round5b_colab.ipynb`)

Round 5b redoes only the block-0 null that P28 could not score. Two fixes: the enumeration's input length is pinned
to the trained receptive field (9 letters) instead of being re-measured on scrambled weights, and
`nullmodel.block0_sensitivity` checks first whether each null block responds to its input at all. Already known when
these were written: the trained median best-word share is 0.98 (round 5), and the trained tiny test model's block 0
has a reach of exactly 9 letters.

### P29: the P28 zeros are the receptive-field artifact

**Test:** R5b.1 and R5b.3. `motifs.receptive_field` on each null (3 shuffle and 3 Gaussian seeds), then
`best_word_share` with `k` pinned at 9.

**Prediction:** `receptive_field` reports ≤ 2 letters on at least 4 of the 6 nulls, and with `k` pinned at 9 no
null's median best-word share is 0 (each is ≥ 0.10).

**Refuted if:** `receptive_field` reports ≥ 9 letters on every null, or a pinned-k null still has a median share of
0. Either would mean the zeros come from somewhere else.

### P28, rerun under a gate fixed now

P28's text and thresholds are unchanged. The gate is new: a null block counts toward P28 only if it responds to its
input. That means its median `unchanged` share, over offsets 0–8, is below 0.5. A null that fails the gate is
reported and not scored, because comparing a working block with one that ignores its input says nothing about
learning. If no null passes, P28 stays **NOT SCORABLE**, and the round-6 design has to build a null that responds
to its input.

---

## Round 5b outcomes (2026-09-30)

Run on an 80 GB A100, 89/89 tests, smoke checks 8/8, block 0 restored after every scramble (asserted). Raw outputs:
`results/round5b/`.

### P29 — CONFIRMED on both clauses

`motifs.receptive_field` reports **1 letter on 6 of 6 nulls** (predicted ≤ 2 letters on ≥ 4 of 6). With `k` pinned at
the trained value of 9, **no null has a median best-word share of 0**: the six nulls sit at 0.90–0.94 (predicted
≥ 0.10). The P28 zeros were the receptive-field artifact, as diagnosed.

### P28 (rerun under the round-5b gate) — REFUTED

Every null passed the responsiveness gate, so all six were scored.

| | live channels | median best-word share | channels ≥ 0.6 | ≥ 0.8 |
|---|---|---|---|---|
| trained | 3,372 | **0.980** | 3,269 | 2,735 |
| shuffle 0/1/2 | 4,096 | 0.940 | 3,958 / 3,939 / 3,938 | 3,148 / 3,072 / 3,072 |
| gaussian 0/1/2 | 4,096 | 0.920 / 0.900 / 0.900 | 3,934 / 3,918 / 3,934 | 2,972 / 2,871 / 2,944 |

Median gap trained − shuffled = **+0.040**, below the registered 0.05 refutation threshold, and **every null has more
channels than the trained model at every cutoff from 0.5 to 0.9**. Both refutation conditions fire.

By the registered wording, "block 0 is learned" is downgraded to **"training sharpens a selectivity the architecture
already has"**, and round 4's 46-vs-0 contrast is an artifact. A weight-shuffled block 0 produces word-selective
channels about as strongly as the trained one. **Round 4's finding 18 is refuted, not merely withdrawn.** Round 3's
descriptive results about block 0 (all 64 words have detectors, start and stop codons are not special, 724 dead
channels) are unaffected — they never depended on the null. The dead channels are now the one block-0 property that
clearly separates trained from untrained: 724 of 4,096 in the trained model, 0 in every null.

### Mechanism, and a correction to round 5's reading of it

`receptive_field` compares outputs with `torch.allclose(atol=1e-3, rtol=1e-3)`. The trained block-0 cascade output
has median magnitude **3.7e-4**; the nulls' is **5.6e-9**, about 65,000× smaller, so every comparison falls inside the
absolute tolerance and the search stops at k=1.

**The nulls are not input-insensitive.** Their relative sensitivity profile is nearly identical to the trained
block's: median movement per changed letter 0.41–0.44 against the trained 0.47, non-zero for offsets 0–8 and exactly
0 from offset 9, with under 1% of outputs unmoved. Round 5's entry claimed the scrambled block was "close to
input-insensitive at bf16, a broken block rather than an untrained one". **That reading was wrong**; the block works,
its outputs are merely tiny. Scrambling one tensor at a time (R5b.2) localises the magnitude collapse to
`projections.weight`: scrambling it alone gives rf=1 and abs_mean 9.5e-9, while every other tensor leaves rf=9.

Limitation of R5b.2 worth recording: the diagnostic measures the Hyena cascade's output, so tensors applied after the
filter (`out_filter_dense`, the MLP) cannot change it, and their rows are identical to trained by construction rather
than by finding.

**Knock-on for P26.** Round 5 dismissed P26's shuffled-model leg (SE still took 62% of peaks) on the grounds that the
fully scrambled model was "near-dead", citing small per-block divergences. That is the same style of argument that
just proved wrong here. The P26 divergences are relative rather than absolute, so the argument is not refuted by this
result, but it is no longer trustworthy without its own check. **P26's shuffled leg is reopened**: it needs the
`block0_sensitivity` treatment applied across blocks before the leg can be dismissed or accepted.

---

## Round 5 re-analysis (2026-09-30): three checks on the amino-acid result

No new GPU run. All three are re-analyses of `results/round5/results_round5.json`, prompted by an outside review that
asked whether the amino-acid result could be an amino-acid-composition effect, a codon-frequency effect, or a
comparison between different populations of sites. Outcomes are recorded here because they change how P25 should be
stated, not what it scored.

### 1. Split by amino acid: the effect is graded by how drastic the swap is

| design | amino acid | sites | missense more disruptive | exact sign test |
|---|---|---|---|---|
| `matched` | **Ile** (ATT/ATC → ATA vs ATG; Ile→Met) | 124 | **66.9%** | p = 2.0e-4 |
| `matched` | **Arg** (CGA/CGG/AGG → …; Arg→Gly or Arg→Trp) | 26 | **96.2%** | p = 8.1e-7 |
| `flipped` | **Ile** (ATA → ATT/ATC vs ATG) | 57 | 63.2% | p = 0.063 |
| `flipped` | **Arg** | 93 | **90.3%** | p = 2.2e-16 |

Isoleucine is **not** at chance: 66.9% with p = 2.0e-4 in `matched`, and 63.2% (p = 0.063, not significant) in
`flipped`. Round 4's 8/17 at Ile was 17 sites; this is 124. Arginine is far stronger. Ile→Met is one of the mildest
substitutions (both hydrophobic, similar size) and Arg→Gly / Arg→Trp are drastic, so the effect is **graded by
severity** — the ordering a protein-aware model predicts and letter statistics do not.

This also resolves the review's warning that `flipped` (80%) beats `matched` (72%) although letter type works against
`flipped`. It is composition, not letter type: `flipped` is 62% arginine sites (93/150), `matched` only 17% (26/150).
Within each amino acid, `matched` ≥ `flipped` as expected (Ile 66.9% vs 63.2%; Arg 96.2% vs 90.3%).

### 2. Codon frequency works against the result at 266 of 300 sites

`controls.rarity_split` splits paired rows by the sign of `d_usage` = log(usage[b] / usage[a]).

| group | rarity favours the hypothesis | rarity opposes it |
|---|---|---|
| `matched` Ile | 0 sites | 124 — 66.9%, p = 2.0e-4 |
| `matched` Arg | 0 | 26 — 96.2%, p = 8.1e-7 |
| `flipped` Arg | 0 | 93 — 90.3%, p = 2.2e-16 |
| `flipped` Ile, ATA → ATT | **34 — 76.5%, p = 2.9e-3** | 0 |
| `flipped` Ile, ATA → ATC | 0 | 23 — 43.5%, p = 0.68 |

In **266 of 300** paired sites the *silent* arm lands on the rarer codon, so codon rarity pushes against "the
protein-changing arm disturbs more". Restricted to those 266 sites the result is **75.9%, p = 7.7e-18**. The confound
exists in exactly one subgroup of 34 sites (`flipped` Ile ATA → ATT), and that subgroup is the one the review
identified. Removing it does not weaken the result; it strengthens it.

### 3. The letter-type baseline is the same population-independent

The review asked whether `fourfold`'s 54.7% came from a different set of amino acids than the Ile/Arg designs. It is
sampled across 9 amino acids (A, G, I, L, P, R, S, T, V). Restricted to the two the matched designs use:

| | sites | transversion more disruptive | p |
|---|---|---|---|
| all `fourfold` | 150 | 54.7% | 0.29 |
| Ile + Arg only | 27 | 55.6% | 0.70 |
| Arg only | 18 | 61.1% | 0.48 |
| Ile only | 9 | 44.4% | 1.00 |

The restricted baseline (55.6%) matches the full one (54.7%), so the comparison was not across populations.

### 4. P26's base rate and null, verified

The peak is taken over **32 blocks**, one per block rather than one per write, so SE's chance level is 9/32 = 28.1%
(attention 5/32 = 15.6%). Confirmed from the stored `kinds` vector, which has 32 entries. The weight-shuffled null
**was** run, on 60 sites, and is the leg reopened by round 5b.

### How P25 should now be stated

Unchanged as a score (in between, by its own rule). Stated for use: *at isoleucine and arginine sites in E. coli, in
one checkpoint, a substitution that changes the amino acid disturbs Evo 2 more than a silent substitution at the same
position — with substitution type held equal (72.0%, n = 150) or set against it (80.0%, n = 150), with codon frequency
working against the result at 266 of 300 sites (75.9% there, p = 7.7e-18), and graded by how drastic the substitution
is (Ile→Met 66.9%, Arg→Gly/Trp 96.2%).* `fourfold` sites, where no amino acid changes, give 54.7% (p = 0.29).

---

## Round 5 re-analysis, second pass (2026-10-01): the SE localisation is partly a size effect

No new GPU run; all from `results/round5/results_round5.json`. A second review asked whether the 85%-vs-11% contrast
compares *signal against noise* rather than *meaning against no meaning*: the four-fold design has nothing real
separating its two arms, so its peaks may land wherever noise peaks. The check is to match designs on how big the
per-block difference is before comparing where it peaks. The concern is partly right.

### Within a design, the SE preference does not depend on difference size

| design | all sites | top third by peak size | bottom third |
|---|---|---|---|
| `matched` | 86.0% | 90.0% | 82.0% |
| `round4` | 85.0% | 82.5% | 77.5% |
| `fourfold` | 11.3% | 16.0% | 16.0% |
| `noncoding` | 30.7% | **50.0%** | 16.0% |

`matched` holds 82–90% across its own range and `fourfold` holds 11–16% across its, so neither is driven by
magnitude internally. But `noncoding` rises from 16% to 50% with difference size, which is the effect the review
predicted.

### Matched on difference size, the gap shrinks but survives

Restricting every design to sites whose peak difference is at least `matched`'s median (0.313):

| | sites | peak is in an SE block | 95% CI |
|---|---|---|---|
| `matched` (amino acid changes) | 75 | **89.3%** | 0.80–0.94 |
| `noncoding` (no gene) | 51 | **51.0%** | 0.38–0.64 |
| `fourfold` (no amino-acid change) | 6 | — | too few to report |
| SE chance level (9 of 32 blocks) | — | 28.1% | — |

Two-proportion test: z = 4.81, p = 1.5e-6.

**So the honest claim is 89% versus 51%, not 85% versus 11%.** Both are above the 28% chance level, so large
differences of any kind are somewhat more likely to peak in an SE block — part of the original contrast was size.
What survives is that an amino-acid change is still far more SE-localised than a size-matched difference with no
gene present. The four-fold design cannot contribute to this comparison at all: only 6 of its 150 sites reach the
matched threshold, which is itself the point — silent-vs-silent differences are small.

### The peak statistic is not defined for the stop designs

`peak_blocks(..., "diff")` takes `argmax(div_b − div_a)`. In `stop_matched` that quantity is **≤ 0 at 111 of 150
sites** (median exactly 0.0000), and in `stop_flipped` at 25 of 150, because the stop arm diverges more than the
missense arm at *every* block. The argmax then returns the least-negative block, which is not a peak. Any statement
about "where stops and missense separate most" computed this way, including the line the round-5 notebook prints, is
**not interpretable** and is withdrawn. A signed version of the statistic is needed, or the stop designs must use
`div_a − div_b`.

### Consequence for P26

P26's score is unchanged. Its reading is narrowed: "the missense-vs-silent difference peaks in SE blocks, and does so
far more than a size-matched difference outside genes (89% vs 51%, chance 28%)". The word *carry* should not be used
until SE writes are ablated or patched on these pairs; `peak_blocks` is descriptive, not causal. Still open from
round 5b: the weight-shuffled leg.

---

## Round 5 statistics, added 2026-10-01 (re-analysis, no new run)

Prompted by a review asking for replicate-aware statistics. Three things were missing from round 5's
reporting: the replicate unit was never named, the seven designs' p-values were uncorrected, and effect
sizes were not reported beside the sign tests. All three are recoverable from
`results/round5/results_round5.json`. Sites were mapped to genes by parsing the GenBank CDS features.

**Clustering is not a problem for round 5.** The designs draw sites with `min_spacing=300`, which put
almost one site per gene: 142–147 distinct genes per 150 sites. A gene-level (cluster) bootstrap is
therefore almost identical to the site-level interval:

| design | sites | distinct genes | site-level 95% CI | **gene-level 95% CI** | median paired difference (nats), gene-bootstrap CI |
|---|---|---|---|---|---|
| `round4` | 120 | **15** | 0.60–0.77 | 0.602–0.757 | −0.488 (−0.807, −0.219) |
| `matched` | 150 | 142 | 0.64–0.79 | 0.649–0.788 | −0.640 (−0.817, −0.398) |
| `flipped` | 150 | 143 | 0.73–0.86 | 0.735–0.861 | −2.722 (−3.689, −1.578) |
| `fourfold` | 150 | 147 | 0.47–0.62 | 0.468–0.624 | **−0.102 (−0.203, +0.058)** |
| `noncoding` | 150 | 133 (5 kb bins) | 0.53–0.68 | 0.524–0.683 | −0.372 (−0.646, −0.050) |
| `stop_matched` | 150 | 146 | 0.04–0.12 | 0.032–0.108 | +17.135 (+14.008, +20.396) |
| `stop_flipped` | 150 | 142 | 0.05–0.13 | 0.040–0.126 | +11.611 (+9.630, +14.544) |

**Round 4's sites are the clustered ones**, in only 15 genes, because they came from a single 60 kb
window. Round 5's whole-genome sampling fixed that without anyone noticing it was a fix.

**Holm correction across the seven designs** changes no conclusion:

| design | raw p | Holm-adjusted p |
|---|---|---|
| `flipped` | 6.0e-14 | 3.0e-13 |
| `stop_matched` | 1.8e-30 | 1.2e-29 |
| `stop_flipped` | 2.6e-28 | 1.6e-27 |
| `matched` | 6.9e-8 | **2.8e-7** |
| `round4` | 3.2e-5 | 9.7e-5 |
| `noncoding` | 0.0111 | **0.0222** |
| `fourfold` | 0.288 | 0.288 (not significant) |

**The best statement of the four-fold result is its effect size, not its p-value.** The median paired
difference is −0.102 nats with a gene-level interval of (−0.203, **+0.058**) — the interval **includes
zero**. "Substitution type alone moves the model by an amount indistinguishable from zero" is both
stronger and more honest than "p = 0.29".

New in `controls`: `holm`, `cluster_bootstrap`, `paired_effect_size`. All tested.

---

## Round 6 predictions (registered 2026-10-01, before running `notebooks/marv_hyena_round6_colab.ipynb`)

Round 6 is replication, not new ground: it re-runs the claims that rest on a single measurement, with
the **replicate unit, sample size and statistic fixed here, before the run**. That ordering is the
point — round 5's P25 registered a statistic that turned out not to be computable from its design.

Conventions for every prediction below. A **replicate** is a different sequence, never a rerun of the
same input. Intervals are 95% percentile bootstrap over the named replicate unit, 10,000 draws.
Where several designs are tested at once, p-values are **Holm-corrected across them** and the
corrected value is what the threshold applies to. Each prediction names what would refute it.

### P30: copying needs attention at every gap, with inserts and sites as separate replicates

**Test:** R6.1. `copy_test(..., seeds=10, sites=5)` — 10 random inserts crossed with 5 independent
genomic insertion sites, 50 replicates per gap, at gaps 100, 1,000 and 10,000, under the five family
conditions keeping the load-bearing layers on. Replicate unit: the (insert, site) pair. Statistic:
mean second-copy accuracy with a bootstrap interval resampling inserts and sites separately, so the
larger of the two intervals is reported.

**Prediction:** with attention ablated, mean second-copy accuracy is below 0.35 at all three gaps and
its interval excludes 0.90; unablated it is above 0.95 at all three gaps.

**Refuted if:** the attention-ablated interval overlaps 0.90 at any gap, or the unablated mean falls
below 0.95.

### P31: LI's long-range contribution survives replication

**Test:** R6.1, the `-li` condition.

**Prediction:** at a 10,000-letter gap, mean second-copy accuracy without LI is below 0.80 and its
interval excludes the unablated mean; at a 100-letter gap its interval includes the unablated mean.

**Refuted if:** the 10,000-gap interval includes the unablated mean — the single-measurement 55.6%
would then be a sampling artifact.

### P32: six load-bearing layers, defined by a threshold fixed in advance

**Test:** R6.2. Health measured on **5 independent DNA sets**, each 4,096 letters, drawn from disjoint
genome regions and each mixing coding and intergenic sequence. A layer counts as load-bearing only if
removing its mixer leaves health below 0.5 on **all 5** sets. The 0.5 cutoff and the all-5 rule are
fixed here.

**Prediction:** exactly the six layers found in rounds 2–5 (L0, L1, L4, L9, L29, L30) meet the rule,
and no others do.

**Refuted if:** any of the six fails on any set, or a seventh layer meets the rule. Either outcome
means the map is DNA-dependent, which is itself reportable.

### P33: the reading frame goes with MR, across 20 genes

**Test:** R6.3. Codon-position accuracy on **20 randomly chosen genes** (minus-strand genes
reverse-complemented, overlapping genes excluded), under the family conditions. Replicate unit: the
gene. Statistic: the periodicity measure per gene, with a gene-level bootstrap interval, Holm-corrected
across the four family conditions.

**Prediction:** removing MR reduces periodicity and its interval excludes the unablated interval, while
removing SE does not.

**Refuted if:** MR's interval overlaps the unablated one, or SE's interval also excludes it. This claim
currently rests on one region, so a refutation is a real possibility and is the reason to run it.

### P34: retrieval gain on real repeat families, with a headroom rule fixed in advance

**Test:** R6.4. At least **3** of the six repeat families now found in E. coli. A family is
**excluded before scoring** if the model predicts its first copy above 95% accuracy, because that
leaves no headroom for retrieval to add anything — the flaw that made 16S rRNA uninformative in round
4. Replicate unit: the family. Statistic: retrieval gain per family with an interval over probes.

**Prediction:** at least 3 families pass the headroom rule; among those, mean retrieval gain exceeds
0.5 nats unablated, and removing attention cuts it by at least 70%.

**Refuted if:** fewer than 3 families pass (the test is then **untestable**, not refuted), or attention
ablation leaves more than half the gain.

### P35: a mutation's effect leaves its position early, at n ≥ 100

**Test:** R6.5. At least **100** variants: ClinVar BRCA1 plus in-silico E. coli variants spanning
silent, missense, nonsense and non-coding. The **hand-off block** is defined here as the first block
after which under 10% of the downstream effect still transfers when the mutated position's residual is
patched.

**Prediction:** the median hand-off block is at or below 7, the 90th percentile at or below 12, and the
distribution differs between variant classes (nonsense later than silent).

**Refuted if:** the median exceeds 10, or the interquartile range spans more than 15 blocks — the
"blocks 0–7" claim from five variants would then not generalise.

### What round 6 does not do

No new biology. The matched substitution designs are not rerun; extending them past isoleucine and
arginine needs a cross-site saturation design, and replicating them on other genomes and a second
checkpoint is round 7. Both are listed in `OPEN_ITEMS.md`.

---

## Round 6 rehearsal (2026-10-01) — NOT the registered run

Run on an 80 GB A100, 95/95 tests, smoke checks 8/8, every cell clean, `mode: quick`. Raw files in
`results/round6/` with `rehearsal` in the names so they are never mistaken for the record. **No outcome
below is recorded against P30–P35**, with one exception noted at the top. Reduced n: 3 inserts × 2 sites
(not 10 × 5), 4 genes (not 20), 2 families (not 6), 12 variants (not 120).

### P32 — CONFIRMED, and this one is not reduced

`find_load_bearing` runs on all 5 DNA sets regardless of `QUICK`, so **this is the full registered
test**. Health per set, against baselines of 0.633–0.858:

| block | kind | set 0 | 1 | 2 | 3 | 4 | below 0.5 in |
|---|---|---|---|---|---|---|---|
| 0 | se | 0.291 | 0.264 | 0.286 | 0.284 | 0.284 | **5/5** |
| 1 | mr | 0.434 | 0.354 | 0.370 | 0.386 | 0.383 | **5/5** |
| 4 | se | 0.397 | 0.325 | 0.359 | 0.390 | 0.322 | **5/5** |
| 9 | li | 0.244 | 0.252 | 0.236 | 0.241 | 0.235 | **5/5** |
| 29 | mr | 0.282 | 0.350 | 0.260 | 0.268 | 0.271 | **5/5** |
| 30 | li | 0.282 | 0.246 | 0.260 | 0.268 | 0.271 | **5/5** |

Exactly the six layers predicted, no borderline cases, and no seventh layer meets the rule. The
load-bearing map is now replicated on 5 independent DNA sets against a threshold fixed in advance.

Also newly visible: **baseline health varies 0.633–0.858 across the five sets.** Rounds 2–5 measured it
on one sequence and reported 0.696 — mid-range, but the spread is wide enough that any future
health-based threshold should be stated relative to the set it was measured on. A descriptive companion
rule (health below 60% of each set's own baseline) drops block 1 and keeps the other five.

### What the reduced runs point at, with no outcome recorded

- **P30 looks likely to confirm.** Attention ablated gives 0.264 / 0.259 / 0.258 at gaps 100 / 1,000 /
  10,000, every interval excluding 0.90, unablated ≥ 0.95 throughout. All clauses met at n = 6.
- **P31's second clause looks likely to fail.** Without LI the 10k mean is 0.660 (interval
  0.467–0.854, excluding the unablated 1.000) — the first clause. But at a **100-letter** gap, −LI gives
  0.949 with an interval of 0.902–0.996, which **excludes** the unablated 1.000, where P31 predicted it
  would include it. If that holds at full n, LI contributes a little even at short range.
- **P33 looks likely to be REFUTED, by its own clause.** Removing MR drops periodicity from 0.276 to
  0.035 and its interval excludes the unablated interval — but **removing SE also excludes it** (0.108,
  interval 0.060–0.151), which P33 named as a refutation condition. Every ablation lowers periodicity
  (−li 0.148, −attn 0.173), and health falls with it (none 0.650; −mr 0.459, −se 0.484, −li 0.518,
  −attn 0.497). **This is the far-context problem again:** periodicity tracks damage, and MR is both the
  lowest periodicity and close to the lowest health. The health-matched control (`OPEN_ITEMS.md` items
  8–9) is no longer optional for this claim — it is the only way to separate "MR carries the frame" from
  "removing MR hurts most".
- **P34's headroom rule bites hard, as designed.** Of the two families the rehearsal reached, 23S rRNA
  scored 0.956 on first copy and was **excluded**; IS5 scored 0.656 and was kept. With 16S at 0.989 in
  round 4, both rRNAs are out. The full run tests 6 families, so whether 3 pass rests on the IS
  elements. P34 may still come back **UNTESTABLE** by its own gate.
- **P35 points away from the registered prediction.** Median hand-off block **24** (IQR 4–28, 90th
  percentile 28) against a predicted median ≤ 7. The per-class split is the interesting part: non-coding
  3, nonsense 14.5, missense 27, silent 28 — in-gene variants handing off *late* and non-coding *early*,
  which is the opposite ordering round 3's five BRCA1 variants suggested.

### A deviation in P35, corrected before the full run

P35 registered "**ClinVar** BRCA1 plus in-silico E. coli variants". Two problems, both now fixed:

1. The rehearsal's sampler drew **E. coli variants only** — the BRCA1 arm was never implemented, so the
   numbers above are not the registered test even setting n aside. Round 3's claim came from human
   BRCA1 sequence, so comparing against E. coli variants alone cannot bear on it.
2. The BRCA1 variant set this project has access to is **Findlay et al.'s saturation-genome-editing
   data** from the Evo 2 repository, labelled LOF / FUNC — the same source round 2 used — not ClinVar.
   The registered wording is wrong, and is corrected here rather than quietly substituted.

The notebook now runs both arms and reports the hand-off distribution by `source`, so the full run can
say whether the "blocks 0–7" claim holds on the human variants it came from, on E. coli, or neither.

---

## Scaling predictions (registered 2026-10-01, before `notebooks/marv_hyena_scaling_{20b,40b}_h100.ipynb` run)

Evo 2's 20B and 40B checkpoints need FP8 on Hopper GPUs, which this project does not have; the
notebooks are written to be run by someone else on borrowed hardware (`notebooks/SCALING_RUN_README.md`
is what gets sent with them). They have **never run on real H100 hardware** — only dry-run against the
tiny CPU model — so Stage 0 is a gate that stops before anything expensive if the self-checks fail.

**Every prediction below is about a pattern, never an index.** 20B and 40B have different depths and
attention layouts, so "block 30" is meaningless for them. Where a prediction names a position it names
it relative to the end of the network.

### P36: the funnel is architectural

**Test:** Stage 2. Write magnitude of every (block, part) at the last position, in 1 region (quick) or
3 (full), plus each write's share of the final residual.

**Prediction:** in both 20B and 40B, one block's write exceeds the second largest by a factor of at
least 10³, that block is a **Hyena** block in the **last quarter** of the network, and its share of the
final residual is above 0.99.

**Refuted if:** the largest-to-second ratio is under 10² in either model, or the dominant block is an
attention block, or it sits in the first half. Then the 7B funnel is a property of that checkpoint and
the whole bottleneck story narrows to it.

### P37: the last block is inert

**Test:** Stage 2b. Mean-ablate the final block's mixer and MLP; measure the largest logit change.

**Prediction:** the maximum logit change from ablating the final block is below 10⁻² in both models.

**Refuted if:** it exceeds 10⁻¹ in either. (7B gives exactly 0 — bit-identical — but that is a bf16
rounding consequence of the funnel's size, so an exact zero is not required.)

### P38: attention is required for copying, at every scale

**Test:** Stage 4. `copy_test` with the load-bearing layers kept on, 10 inserts × 5 sites per gap in
the full run.

**Prediction:** with attention ablated, mean second-copy accuracy is below 0.40 at every gap tested,
while the unablated model is above 0.90.

**Refuted if:** attention-ablated accuracy exceeds 0.60 at any gap — the copying circuit would then be
routed differently at scale.

### P39: load-bearing layers in every Hyena family, none in attention

**Test:** Stage 3. One mixer at a time, health below 0.5 on all 5 DNA sets (2 in the quick run).

**Prediction:** at least one load-bearing layer in each of SE, MR and LI, and **zero** in attention.

**Refuted if:** any attention block is load-bearing, or any Hyena family has none. Note this test is
the one most likely to be affected by the absolute 0.5 cutoff: baseline health varies by DNA set
(0.633–0.858 in 7B), and a larger model may sit elsewhere. The notebook prints baselines so the cutoff
can be judged, and a borderline outcome is reported as borderline rather than forced.

### P40: the amino-acid effect and its SE localisation hold at scale

**Test:** Stage 6, only if `RUN_BIOLOGY = True`. `matched`, `fourfold` and `stop_matched` designs at 60
sites each in the full run.

**Prediction:** on `matched`, the protein-changing substitution is more disruptive at ≥ 60% of sites;
on `fourfold` the rate is within 10 points of 50%; on `stop_matched` the stop wins at ≥ 85%; and SE's
share of peak blocks is at least 20 points higher on `matched` than on `fourfold`.

**Refuted if:** `matched` falls below 55%, or SE's share on `matched` is within 10 points of its share
on `fourfold`. **Untestable** if fewer than 30 sites are scored per design.

### What these runs cannot settle

20B and 40B share Evo 2's training corpus and the StripedHyena 2 architecture, so agreement shows the
pattern is **scale-stable within one model family** — not that it is a property of hybrid architectures
in general. That would need a differently-trained model, and the only independently-trained Evo 2
checkpoint (`evo2_1b_base`) also requires Hopper. Both limits belong in the paper.
