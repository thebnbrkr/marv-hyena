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
