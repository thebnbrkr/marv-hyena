# Research log

A running diary of what we tried, what happened, what went wrong, and what we changed. Entries are appended, never
rewritten. `PREDICTIONS.md` holds the formal predictions and their outcomes; this file is the story around them.

Written for a software engineer first. There's a glossary at the bottom.

---

## The mental model (read this once)

Think of Evo 2 as a **pipeline of 32 services that all write to one shared log**, the "residual stream".

- The input DNA letter is turned into a vector and written to the log (the "embedding").
- Each of the 32 blocks **reads the log, computes something, and appends its result** (the "write"). Every block
  appends two writes: one from its **mixer** (the part that looks at other positions in the DNA) and one from its
  **MLP** (the part that only processes the current position).
- At the end, the log is summed up and turned into a guess for the next letter (A, C, G or T).

There are four kinds of mixer, which differ in **how far back along the DNA they can look**:

| mixer | how far back it looks | analogy |
|---|---|---|
| **SE** (Hyena short) | the last 7 letters | reading through a keyhole |
| **MR** (Hyena medium) | the last 128 letters | a sliding window |
| **LI** (Hyena long) | everything so far, fading with distance | a running summary |
| **attention** | anything so far, exactly | search-and-copy |

Our question is **which kind of mixer does which job**. The architecture paper assumes they specialize, but nobody has
checked in the trained model.

Our main tool, **ablation**, is like replacing one service with a **stub that always returns that service's average
response**. If something downstream breaks, that service was doing it. One trap matters in everything below: if you stub
a service that *everything* depends on, *everything* breaks, and you learn nothing specific.

## Current picture: how Evo 2 7B works (living summary)

*This section gets rewritten as we learn. The dated entries below are the permanent record it's built from.*

Follow one position of DNA through the 32 blocks:

1. **Reading the letters (block 0, SE, load-bearing).** About 3,400 live detector channels each look at the last
   9 letters, weighted toward the nearest ~6. They respond to letter **combinations**, not single letters, and they
   cover all 64 three-letter words roughly evenly. They're generic "word" detectors: start and stop codons aren't
   special. 18% of the channels (724) are dead. *(Findings 4, 10, 15, 16)*
2. **Early hand-off (blocks 0–7).** A mutation's effect is passed to the neighbouring positions here: sometimes in
   block 0 itself, otherwise by blocks 4–7. After block 7 nothing is left to transfer at the mutated position. Three
   of the six load-bearing layers sit here (L0, L1, L4), which suggests an essential early encoding stage.
   *(Findings 6, 13, 14)*
3. **Reading frame (MR layers).** The medium 128-letter layers carry the 3-letter codon rhythm inside genes. Remove
   them and accuracy becomes flat across codon positions. *(Finding 13; supersedes finding 8, which saw only
   single layers)*
4. **Look-up and copy (attention, mainly L3; LI helps at long range).** Exact repeats are recognised at any distance,
   from 100 to 10,000 letters. Attention is essential; the LI layers (especially L2) help only at long range.
   Copying is its own circuit: it survives even when ordinary reading is broken (e.g. with L1 removed).
   *(Findings 1, 7, 13)*
5. **Far context (tentative: attention).** 50,000 letters of upstream DNA help only slightly (~0.01–0.02 nats per
   letter), and removing attention erases that. The fair LI test is still pending. *(Finding 9)*
6. **What "long" LI layers really do.** Most LI filter channels reach only 4–7 letters; a few reach thousands. That's
   consistent with LI helping long-range copying without being the main long-range channel. L9 is load-bearing.
   *(Findings 3, 13)*
7. **The funnel (blocks 28–30).** Output sizes escalate: ~10⁴ (block 28) → ~10⁶ (block 29) → ~10¹² (block 30).
   Block 30 (LI, looking ~4 letters back) produces the representation the output reads, and **only** that. This is
   built into the weights, not a rounding effect. Block 31 and block 30's MLP have no effect. *(Findings 2, 5, 12)*
8. **The middle (roughly blocks 10–28).** Individually expendable: removing any one costs little, so the work is
   spread out or redundant. *(Finding 6)*

**Confidence.**
- Solid (causal, replicated across rounds, or exact from the weights): 4 (copying), 6 (LI reach), 7 (the funnel).
- Measured once, needs replication: 1 (the enumeration is exact, but the "generic words" reading is one analysis), 2
  (5 variants), 3 (one gene window), 8.
- Tentative: 5 (far context via attention).

**Not yet known:**
- what block 30 actually reads (R3.4, crashed; rerun pending);
- what the redundant middle layers compute;
- whether LI carries far context;
- whether this holds for other checkpoints (evo2_7b_262k, evo2_7b_base);
- how it lines up with the Goodfire SAE features at layer 26.

**Superseded along the way (kept in the record):**
- finding 8 ("codon rhythm is spread out") was only true one layer at a time; as a family, MR carries it;
- finding 10's "combinatorial channel 263" was a dead channel (finding 16).


---

## 2026-09-21: Round 0, getting it to run on Colab

The code was written against Vortex's source (the library Evo 2 runs on) and tested on a tiny fake model on a Mac.
Getting it running on a Colab A100 took four fixes:

1. **flash-attention compiled for 45+ minutes.** `pip install flash-attn` found no prebuilt binary for Colab's torch,
   so it compiled from source. *Fix:* don't use flash-attn at all. Vortex has a code path that uses PyTorch's built-in
   attention (also fast on an A100). Vortex still *imports* flash-attn at startup, so we register an empty placeholder
   module to satisfy the import (`noflash.py`). Verified that this attention is mathematically identical to standard
   attention.
2. **The wrong Evo 2 got installed.** Colab moved to Python 3.13, and every current Evo 2 release says "Python < 3.13".
   pip silently fell back to the old 0.3.0 instead of failing. *Fix:* install `evo2==0.6.0 --ignore-requires-python`
   (it's plain Python, so the limit is conservative) and assert the version.
3. **Colab ships a broken `transformer-engine` package** whose import raises `RuntimeError`. Evo 2 only handles
   `ImportError` ("not installed"). *Fix:* uninstall it, and convert that failure into an `ImportError` in code.
4. **My own bug:** the "is flash-attn installed?" check crashed on the placeholder module (Python's `find_spec` raises
   on modules without import metadata). *Fix:* mark the placeholder and check for the marker first.

Lesson: a fresh cloud image is a moving target. Pin versions, assert them, and fail loudly.

---

## 2026-09-21: Round 1, first real results (Evo 2 7B, A100)

### The smoke checks all passed (8/8)

Every "exact" tool checks itself against the real model. For example, the per-block credit must add up to the model's
real output. All checks passed within bf16 rounding (0.3–0.8%). **So the tools measure what they claim to.** What
went wrong below was experiment *design*, not the tools.

### Finding 1: attention does the copying (solid)

Test: insert a random 200-letter stretch into real E. coli DNA, then repeat it `gap` letters later. The first copy is
unpredictable. The second is predictable *only* by finding and copying the first.

| gap | full model | attention stubbed |
|---|---|---|
| 100 | 100% correct | 25% (chance) |
| 1,000 | 100% | 24% |
| 10,000 | 99.7% | 25% |

Without attention the model still works on normal genes (codon accuracy 70–76% vs. 94–96%), so this isn't "model broken".
It's specifically copying that dies. Surprise: even 100-letter gaps need attention, though the MR window (128 letters)
could in principle span them.

### Finding 2: block 30 dominates the output (hypothesis)

We split one prediction into "how much did each block's write push toward the right letter". **Block 30 (the last LI
block) got +8.607; every other block got 0.000.**

Most likely explanation: block 30's write is **enormously larger** than everyone else's. The final step normalizes the
log (rescales it to a fixed size), so if one entry is 10,000× bigger than the rest, the output effectively reads only
that entry. The other blocks still matter, but only *indirectly*, by shaping what block 30 computes. Our direct-credit
tool can't see indirect influence.

This explains three otherwise-strange results:
- **Stubbing "all LI" made the model guess uniformly at random** (log-prob −1.38 = log ¼). That stubbed block 30, the
  one write the output reads.
- **In the BRCA1 test, patching almost any late component "explained 100%" of the mutation's effect**, because every
  path runs through block 30.
- Block 30 itself only looks **~4 letters back** (its longest channel: 18). The final guess is assembled locally, from
  information earlier layers already gathered into the log.

Status: *hypothesis*. Round 2 measures the size of every write directly (R2.1).

### Finding 3: the "long" LI layers are mostly short

Reading the filters straight from the weights: typical LI channels reach only **4–7 letters**. A minority of channels
reach far (block 2: 4,528 letters; block 27: 1,303). MR layers use their whole 128-letter window. So "long implicit" is
a statement about *capacity*; in practice most LI channels act locally, and long reach lives in a few channels (plus
attention).

### Finding 4: first-layer motifs (block 0), explained

Block 0 is an SE layer reading raw letters. Its output at a position depends only on the **last 9 letters**: a
3-letter pre-filter feeds a 7-letter filter, and 3 + 7 − 1 = 9. We measured this (9) before relying on it. With only
4 letters, there are just 4⁹ = **262,144** possible inputs, so we ran every single one and recorded, for each of
block 0's 4,096 channels, which inputs make it fire hardest. That's a **complete, exact** description of the first
layer, with no sampling. (It's like unit-testing a pure function on its entire input domain.)

What we saw:
- **Many channels are sharp pattern detectors.** For example, channel 2409's top inputs all end in `AATTT`, and channel
  2202 fires on runs of T.
- **Channels for biologically meaningful 3-letter words exist:**
  - `ATG` is the **start codon**, where most genes begin: 46 channels.
  - `TAA`, `TAG`, `TGA` are the **stop codons**, where genes end: 145, 50 and 41 channels.
  - `AGGAG` is the **Shine-Dalgarno** sequence, where bacterial ribosomes grab the RNA just before a gene: 1 channel.

  Note: "N channels" means ≥ 80% of the channel's 50 strongest inputs contain that word.
- **Most channels focus on the last 5–6 of their 9 letters.** Each channel's "consensus" writes the letter it prefers
  at each position, or `N` if it doesn't care. Consensuses like `NNNNAATTT` mean positions 1–4 barely matter. So block
  0 technically sees 9 letters but mostly uses the nearest 5–6.

What was wrong with how we measured it:
- **The "68% of channels are sharp" number is meaningless.** We ranked each channel's top 100 inputs out of 262,144.
  Taking the top 0.04% of *anything* makes those inputs look alike, so almost every channel passes. The threshold
  doesn't separate real detectors from anything else.
- **The codon counts have no control.** A channel that simply likes A and T will have TAA in many of its top inputs
  by chance. "145 TAA channels" might mostly be "AT-rich channels". We need to compare against what chance predicts
  for a channel with the same A/C/G/T mix.
- **"Focus on the last 5–6 letters" was read off the N's**, which is an eyeball impression, not a measurement.

Round 2 fixes all three (R2.4): an exact per-position importance score from the complete enumeration, and a
composition-controlled motif count.

### What went wrong: stubbing whole families broke the model

The plan was: stub all SE mixers, then all MR, then all LI, then all attention, and see which job breaks. Results with
accuracy at the three codon positions (normal model: 94% / 96% / 70%):

| stubbed | codon accuracy | what it really means |
|---|---|---|
| all SE | **0% / 0% / 0%** | worse than guessing (25%): the model is broken, confidently wrong |
| all MR | 23% / 22% / 30% | ≈ guessing: broken |
| all LI | 39% / 18% / 30% | ≈ guessing: broken (this includes block 30, see finding 2) |
| all attention | 70% / 76% / 59% | still works: meaningful |

When the model is broken, **every** test fails at once: codons, copying, context. So those rows can't tell us which
family does codons or copying. It's like pulling out the database and concluding the database was responsible for
the login page, the search page and the checkout page. True, but not informative.

Why "average-response stubs" weren't enough: I switched from zeroing to mean-replacement *because* zeroing breaks
models. But these families are load-bearing: 9 blocks each, spread through the whole depth of the network. Replacing
all 9 with averages removes too much, and in LI's case it also removes the output bottleneck (block 30).

Consequences:
- **P3 (codon rhythm lives in SE)** can't be answered from round 1.
- **The LI half of P1 and P2** can't be answered either. We can't say "LI doesn't copy" when stubbing LI breaks
  everything.
- Only the **attention** rows are interpretable, because the model survives without attention. That's why finding 1
  is solid.

### Other design mistakes

- **BRCA1 patching at all positions** mostly measured the block-30 bottleneck (see finding 2), and the "harmless"
  variant's effect was so small (0.25) that its ±200% numbers were noise.
- **The context test used one gene** and found a tiny effect (+0.017 nats per letter). Stubbing attention erased it,
  which hints that attention, not LI, carries far context, the opposite of P2. But one gene isn't evidence.

### Changes for round 2 (`notebooks/marv_hyena_round2_colab.ipynb`)

| problem | fix |
|---|---|
| block-30 dominance is only a guess | R2.1: measure the size of every write in 3 genome regions; detect "bottleneck" blocks automatically |
| broken-model rows look like results | every stubbed condition now carries a **health check** (accuracy on ordinary genome; flagged `broken` near chance) |
| family stubbing too destructive | stub families **without the bottleneck**, and stub **one layer at a time** (32 single-layer runs) |
| motif threshold meaningless, no control | exact per-position importance; motif counts with a **composition control**; control motifs (CCC, GCG) for comparison |
| BRCA1 patching measured the bottleneck | patch **at the mutation site only**, only on variants with a real effect (\|effect\| ≥ 1); plus a 40-variant harmful-vs-harmless scoring check |
| one gene for context | 5 genes spread across the genome |

---

## 2026-09-21: Round 2, the redesigned experiments (Evo 2 7B, A100)

Setup ran clean (same cells as round 1). Smoke checks 8/8. Baseline health: 70% next-letter accuracy on ordinary
E. coli DNA (86% on the gene-dense window used for codons). Raw outputs: `results/round2/`.

### Finding 5: block 30 isn't just big, it's the only thing the output sees (confirmed)

Size of each block's write to the shared log, same in all three genome regions tested:

| write | size (L2 norm) |
|---|---|
| block 30 mixer (LI) | **~7 × 10¹¹ to 4 × 10¹²** |
| block 29 MLP | ~6 × 10⁶ |
| block 29 mixer | ~2 × 10⁵ |
| block 28 MLP | ~9 × 10³ |
| everything earlier | ~0.1 to 30 |
| block 30 MLP, block 31 (attention + MLP) | **~0** (10⁻¹⁵, 10⁻¹⁴, 0.4) |

Block 30's write is **~123,000× bigger** than the next one. The sizes also escalate through blocks 28 → 29 → 30.

What this means, in software terms: the log is stored as **bf16**, a 16-bit float with only ~3 significant digits.
Adding 6,000,000 to 700,000,000,000 in bf16 changes nothing. The smaller number is below the rounding step, like
adding one cent to a $10-billion balance stored with 3 significant digits. So **after block 30 writes, everything
written earlier is rounded away**, and the final guess is computed from block 30's output alone.

Two consequences, both measured:
- **Block 31 does nothing.** Switching off its attention gives *bit-identical* results to the normal model (same
  accuracy to every decimal). Block 30's MLP and block 31's MLP write essentially zero. In this inference setup, the
  last 1.5 blocks are dead weight.
- **Everything the model knows must be squeezed into block 30's input.** The other 30 blocks matter only through
  what they feed into block 30. So "direct credit" tools (our `trace`) will always say "block 30, 100%", and they are
  useless for this model. The right place to measure is **the input to block 30**.

Caveat: measured in the bf16 inference path (the official Evo 2 "light install" path for 7B, no Transformer Engine).
The weights are the same everywhere, and the huge magnitude comes from the weights, so this is very likely general.
The FP8 path wasn't tested.

Side effect on our tooling: the smoke check "all writes add back up to the final log" (error 8 × 10⁻⁶) is now weak.
It is dominated by block 30, so it couldn't notice a missing small write. The other checks (lag-band reconstruction
per block) are unaffected.

### Finding 6: five load-bearing layers, and the rest are individually expendable

Switching off **one mixer at a time** (32 runs). Health = accuracy on ordinary genome (normal: 0.859):

| switched off | health | verdict |
|---|---|---|
| L0 (SE, the first layer) | 0.288 | **breaks the model** |
| L1 (MR) | 0.367 | **breaks the model** |
| L9 (LI) | 0.254 | **breaks the model** |
| L29 (MR) | 0.248 | **breaks the model** |
| L30 (LI) | 0.288 | **breaks the model** (the bottleneck) |
| any of the other 27 | 0.70–0.86 | model keeps working |

This is why round 1's family ablations and round 2's "family minus block 30" still broke the model. **Every Hyena
family contains at least one load-bearing layer**: SE has L0, MR has L1 and L29, LI has L9 and L30. Attention has
none, which is why attention was the only family whose removal the model survived. The next family test must keep
all load-bearing layers on.

### Finding 7: which single layers copy (sharpens finding 1)

Second-copy accuracy with one layer off (normal: 1.000 at 1,000-letter gap, 0.994 at 10,000):

- **L3, the first attention layer: 0.939 / 0.633.** The main copier, especially at long range.
- **L2, the first LI layer: 0.972 / 0.822.** LI helps with long-range copying too. L2 is also the LI block whose
  filters reach furthest (one channel reaches 4,528 letters, finding 3). So the weight-read reach and the measured
  behaviour agree.
- Every other single layer: no effect on copying.
- **Surprise:** switching off L1 (MR) *breaks ordinary prediction* (health 0.367) but copying stays **perfect**
  (1.000 / 1.000). Copying is a separate circuit that survives when the "normal reading" machinery is damaged.

With the whole attention family off, copying dies completely (round 1), but L3 alone only partly. So attention layers
back each other up, with L3 doing most of the work.

### Finding 8: the codon rhythm is spread out, not in one layer (P9 refuted)

Rhythm = accuracy at codon positions 1–2 minus position 3 (normal: 0.251). No healthy single-layer switch-off halves
it. The biggest drops come from **L10 (attention) → 0.166** and **L5 (MR) → 0.191**, not from any SE layer.
(L0, SE, kills the rhythm, but it also kills the whole model.) The rhythm is distributed across many layers.

### Finding 9: far-away DNA helps a little, consistently, through attention

Benefit of 50,000 vs. 500 letters of upstream context, in nats per letter, 5 genes spread across the genome:

| | creD | gspE | ilvI | uup | yehQ |
|---|---|---|---|---|---|
| normal model | +0.016 | +0.024 | +0.017 | +0.008 | +0.016 |
| attention off (model still healthy) | −0.001 | −0.002 | −0.003 | −0.001 | −0.001 |

The benefit is small but positive in 5/5 genes, and switching off attention removes it in 5/5. LI couldn't be tested
(its family ablation breaks the model, finding 6). So the evidence says attention carries far context, the opposite of
P2's guess, with the LI side still open.

### Finding 10: first-layer motifs, after the controls

**Composition control** (a channel counts only if the motif is ≥ 3× more common in its top inputs than chance for its
letter mix):

| motif | round 1 count | after control |
|---|---|---|
| start ATG | 46 | **46** |
| stop TAA | 145 | **65** (80 were just AT-loving channels) |
| stop TAG | 50 | **50** |
| stop TGA | 41 | **41** |
| Shine-Dalgarno AGGAG | 1 | **1** |
| control: CCC | 264 | 8 |
| control: GCG | 95 | **53** |

The control did its job: it removed 80 fake TAA channels. But the control motif **GCG ends up with more detector
channels (53) than ATG (46)**. So "block 0 has ATG detectors" is true, but not special: block 0 seems to have dozens of
detectors for *many* 3-letter words, not specifically the biologically meaningful ones. To claim start/stop codons are
special, we'd need to count detectors for all 64 three-letter words and show ATG and the stops stand out.

**Position importance** (exact, from all 262,144 inputs): importance rises toward the current letter. Mean by
position, 8 letters back → current: 0.03, 0.04, 0.05, 0.05, 0.08, 0.12, 0.14, 0.13, **0.20**. 64% of channels put
≥ 80% of their importance in the last 6 positions, and 18% put ≥ 50% on the current letter alone. There are
exceptions: the best TAG channel (1523) matches TAG at 6–8 letters back.

**A flaw found in the new importance measure:** the best ATG channel (263) scored ~0 importance at *every*
position, even though its top inputs clearly all end in `ATGC`. The measure (a "main effect": how much the average
output changes with each letter at one position) misses channels that only fire on a **combination** of letters.
Hyena multiplies signals together, so "fires only when A-T-G-C all appear together" is natural for it. It's like
testing each feature flag separately when the bug only shows with a specific combination of flags. Fix: a
"total effect" measure, which varies one position while holding the others fixed, over the full enumeration.

### Finding 11: Evo 2 separates harmful from harmless BRCA1 variants (AUROC 0.88 on 40)

Harmful (loss-of-function) variants got lower scores: mean −0.0046 vs. −0.0010 for functional ones, AUROC **0.88**
(P11 predicted ≥ 0.65). Their effect on the following 200 letters was also 5× bigger (−22 vs. −4.4 nats). A small
sample, but in line with the Evo 2 paper.

**Patching at the mutation site** (which component's write *at the mutated position* carries the change onward):
for 2 of 3 variants, **no single write carries more than ~4%**. For the third (chr17:41219643 G>C), **L0's write
carries 64%**. Reading: the change is usually carried redundantly by the letter itself plus many writes at once, so
swapping one write barely matters. Better next tool: swap the **whole log** at the mutation site after each block,
which shows at what depth the information leaves that position.

### What went wrong or stayed open in round 2

- Family ablations are still broken, because each family has load-bearing members (finding 6). P8 is untestable again.
- The GCG control shows the motif test needs a proper baseline over all 64 three-letter words.
- The position-importance measure misses multiplicative channels (channel 263).
- Mutation-site patching of single writes is too fine-grained. Most variants show nothing.
- `trace` is useless for this model (finding 5). The bf16 dominance also weakens one smoke check.

### Plan for round 3

| question | method |
|---|---|
| does LI copy / carry context? | family ablations that **keep all five load-bearing layers on** (L0, L1, L9, L29, L30) |
| what does block 30 read? | measure at **block 30's input**: patch the whole log entering block 30, per position and per earlier block |
| where does a mutation's signal go? | **residual patching**: swap the entire log at the mutation site after each block |
| are start/stop codons special in block 0? | count controlled detectors for **all 64** three-letter words; see where ATG and the stops rank |
| channel importance | **total-effect** measure from the full enumeration (catches multiplicative channels) |
| is the dead-block finding general? | check that the rounding-away of earlier writes also happens in float32, i.e. is it arithmetic or weights |

---

## 2026-09-21: Novelty check and round 3 build

**Novelty (literature search, ~8 queries; not a full review).**
- Known prior work:
  - attention does recall that gated convolutions can't, on synthetic tasks (Zoology, arXiv 2312.04927; MAD, 2403.17844);
  - "massive activations" in transformers (Sun et al. 2024, 2402.17762);
  - Evo 2 SAE features at layer 26 (Goodfire; Arc);
  - attention-head and layer-ablation studies on genomic *transformers* (DNABERT etc.).
- Not found:
  - operator-level causal analysis of a trained StripedHyena / Evo 2;
  - the block-30 bottleneck (a whole-write blow-up near the end, dead final block; known massive activations are a
    few dimensions and fade at the end);
  - weight-read evidence that LI filters are mostly local, which contradicts common explainers that credit Hyena-LI
    with long-range integration;
  - exhaustive first-layer enumeration;
  - a load-bearing layer map.
- Before claiming novelty: check the Evo 2 paper supplement, ask whether Arc or Goodfire know about the block-30
  magnitudes, and replicate on a second checkpoint.

**Round 3 built** (`notebooks/marv_hyena_round3_colab.ipynb`, not yet run). New tools:
- `interface.block_input_attribution`: integrated gradients at block 30's input, with a completeness self-check;
- `variants.residual_patch_by_depth`: swap the whole residual at the mutation site after each block;
- `diagnostics.precision_check`: block 30 recomputed in float32;
- `diagnostics.find_load_bearing`;
- `motifs.rank_all_words`: all 64 three-letter words;
- total-effect importance (`enumerate_block0(keep_all=True)`).

35 tests pass on the tiny model, and the notebook dry-runs end to end. Predictions P12–P17 were registered before
running.

Found while building: the round-2 gene window (200,000–208,192) has no 60-letter intergenic stretch, so round 3 picks
its intergenic positions from a wider window.

---

## 2026-09-21: Round 3, measuring at the bottleneck (Evo 2 7B, **40 GB** A100)

Smoke checks 8/8. This session got the 40 GB A100 (earlier rounds had 80 GB), which caused one failure. Raw outputs:
`results/round3/`.

### What ran and what didn't

| step | status | why |
|---|---|---|
| R3.1 float32 check | ✅ ran | |
| R3.2 load-bearing map | ✅ ran | |
| R3.3 fair family tests: copying, codons | ✅ ran | |
| R3.3 far-context test (5 genes × 50k letters) | ❌ out of memory | Vortex builds the long-LI filter as one 4096 × 16 × 51,000 float32 array (12.5 GB). A chunked, bit-identical replacement was pasted in, but the failed cell was never rerun afterwards. Now built into the package (`memory.py`) |
| R3.4 what block 30 reads | ❌ crashed (my bug) | Vortex's loader converts weights to bf16 inside `torch.inference_mode()`, making them "inference tensors" that autograd refuses to use. The tiny test model never went through that loader. Fixed (`interface._autograd_safe`), with a test that reproduces the exact error |
| R3.5 mutation depth | ✅ ran | |
| R3.6 64-word ranking, total effect | ✅ ran | |

### Finding 12: the block-30 blow-up is in the weights (P12 confirmed)

Recomputing block 30 in float32 from the same input gives the same size (6.39 × 10¹¹ vs. 6.42 × 10¹¹ in bf16, within
0.5%). At full float32 precision, adding back every earlier write changes the logits by ~0.00002 on a scale of 12–24,
about 0.0001%. So earlier writes are irrelevant **at any precision**. It's not a bf16 artifact: the network's output
really is a function of block 30's output.

### Finding 13: fair family tests finally work (all healthy)

The load-bearing map on this run's health sequence: **L0, L1, L4, L9, L29, L30**. L4 is new: it scored 0.327 on this
genome stretch vs. 0.799 on round 2's gene window, so borderline layers depend on the test sequence. With those kept
on, **no family ablation broke the model** for the first time:

| switched off (load-bearing kept on) | health | copy @100 | @1k | @10k | codon rhythm |
|---|---|---|---|---|---|
| nothing | 0.696 | 1.000 | 1.000 | 0.997 | 0.251 |
| SE (7 layers) | 0.454 | 1.000 | 1.000 | 0.989 | 0.109 |
| MR (7 layers) | 0.425 | 1.000 | 1.000 | 1.000 | **−0.016** |
| LI (7 layers) | 0.528 | 0.986 | 0.944 | **0.556** | 0.177 |
| attention (5 layers) | 0.571 | **0.253** | **0.244** | **0.247** | 0.145 |

(The codon-rhythm health numbers come from the gene window: 0.859 / 0.571 / 0.461 / 0.624 / 0.670.)

- **Copying = attention (essential) + LI (helps at long range).** Without the non-load-bearing LI layers, copying
  at 10,000 letters drops to 0.556 while short-range copying survives. SE and MR play no part in copying: 1.000
  even though the model is degraded.
- **The codon rhythm (reading frame) lives mainly in MR**, the 128-letter layers. Removing them flattens accuracy
  across the three codon positions (0.450 / 0.472 / 0.477) while the model still predicts well above chance. SE
  removal halves the rhythm; LI and attention remove less. Round 1's guess (SE) was wrong. It fits biology loosely:
  MR's 128-letter window spans ~40 codons, enough context to lock onto the frame.

### Finding 14: a mutation's signal leaves the mutated position within the first ~8 blocks

Swap the entire residual at the mutation site from the mutated run into the normal run, after each block. The number
is the share of the mutation's downstream effect that moves with it:

| variant | depth −1 (the letter) | after block 0 | 2 | 4 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|---|
| 41256881 T>C (LOF) | 1.00 | **0.09** | 0.05 | 0.03 | 0.02 | 0.02 | 0.01 |
| 41256880 C>G (LOF) | 1.00 | **0.07** | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 |
| 41215936 A>C (LOF) | 1.00 | 0.88 | 0.76 | 0.58 | **0.28** | 0.12 | 0.09 |
| 41219643 G>C (LOF) | 1.00 | 0.88 | 0.88 | 0.79 | 0.74 | **0.04** | 0.05 |
| 41219636 A>G (FUNC) | 1.00 | 0.50 | 0.10 | −0.08 | −0.09 | −0.05 | −0.06 |

Reading, in software terms: once a block has run, the positions after the mutation have already *read* the mutated
letter and carry its effect themselves, so swapping the site afterwards transfers nothing new.
- For two variants, the hand-off happens **in block 0 itself**. Neighbouring positions read the mutated letter
  directly through block 0's 9-letter window.
- For the others, the site keeps carrying the signal through blocks 4–7, then hands it off.
- In all 5 cases the signal has left the site by block 7. Mutation effects are passed forward **early and locally**.

### Finding 15: block 0 is a generic 3-letter-word detector bank; codons aren't special

With the composition control, **every** 3-letter word has detector channels: median 46 across all 64 words. The
start codon ATG ranks **31/64** with 46, exactly the median. Stops: TAA #1 (65), TAG #26 (50), TGA #48 (41). Their
reverse complements (the same signals on the other strand): TTA #5, CTA #19, CAT #51, TCA #60. The top of the list
(TAA, GTG, AAT, CCT, TTA, ATA…) is mixed, and AT-rich words do well even after the control.

### Finding 16: 724 block-0 channels are dead, and round 2's "ATG channel" was one of them

Round 2's channel 263 (top inputs ending in `…ATGC`) has **zero total effect at every position**. Its output is
constant, or too small to register, across all 262,144 inputs. **724 channels (18%)** show zero effect this way. Their
"top inputs" were meaningless orderings of near-identical values. Round 2's "combinatorial channel" story was wrong:
the channel is dead. Implication: motif counts should exclude dead channels, which round 4 will do.

Also, the total-effect indices sum to **1.75 on average** (a purely additive channel sums to 1.0). So block 0's live
channels respond strongly to letter **combinations**, as expected from Hyena's multiplicative gating. Total effect
peaks at the current letter (0.31) and 2 letters back (0.30).

### What stayed open in round 3

- Far-context test: out of memory. Rerun with the chunked filter.
- What block 30 reads (R3.4): crashed on a bug, now fixed. Rerun.
- Motif counts include dead channels. Filter them.

---

## 2026-09-21: Literature review, what the papers taught us

We read the closest papers to see whether our work is new and what to borrow. Short version: **nobody has done causal,
operator-level analysis of Evo 2 or any trained Hyena model**, and a 2026 review says that's exactly the field's gap.
Several papers hand us concrete experiments and controls.

### What each paper says, and what we take from it

| Paper | What it found | What it means for us |
|---|---|---|
| **Evo 2** (Brixi et al., Nature 2026; bioRxiv 2025.02.18.638918) | Finds a 100-letter "needle" in 1M letters of random DNA; best zero-shot splice-variant prediction; SAE features at layer 26, including **f/24278, which fires on frameshifts and premature stop codons**. No mechanism, no ablations, nothing on activation sizes. | Our copying test is a mini needle-in-a-haystack, and we found the mechanism (attention, with LI at long range). f/24278 is a bridge to biological "why": check it on harmful BRCA1 variants, then switch it off. |
| **Goodfire, "Interpreting Evo 2"** | SAEs on several layers; layer 26 chosen because its features looked most biological. They *guess* late features appear because a 4-letter vocabulary needs only a few final layers for output. They say steering Evo 2 is much harder than steering LLMs. A methods paper is coming. | Our block-30 funnel is measured evidence for their guess. Hypothesis: the 28 → 30 funnel (×10⁸ growth, block 30 takes over) may be *why* steering from layer 26 is hard. Watch for their methods paper (possible overlap). |
| **"What Attention Recalls and Recurrence Controls"** (arXiv 2609.04434, text hybrids: Qwen3.5, Falcon-H1, Jamba) | Keep only attention's memory: exact lookup survives (64–98%). Keep only the recurrent state: lookup is 0%, but language and style survive (70–80%). Attention is an "addressable store"; recurrence is a "compressed prior". | The same split as our copying result, in text. **New experiment:** Vortex keeps separate generation states for attention and each Hyena type, so we can keep one and drop the other, or swap them between two organisms. Does Hyena carry "what kind of genome" (codon usage, GC content) while attention carries exact lookup? Their caveat applies to us: mixed states are unnatural, so a collapse can be partly artifact. |
| **"ICL Beyond Transformers"** (arXiv 2510.23006, Mamba/Hymba/Zamba2) | In hybrids, in-context learning is driven by specific *heads* in attention layers, mostly the middle ones, found by measuring each head's causal effect. | Go one level finer: **which heads in L3 do Evo 2's copying?** Are there induction heads in a DNA model? |
| **Massive activations** (Sun et al., COLM 2024, arXiv 2402.17762) | A few single values (~10³–10⁴, 1,000–100,000× typical) appear early (layer ~2) on special tokens, stay **constant regardless of input**, and fade at the end. Zeroing them breaks the model; replacing them with their mean is harmless (they act as hidden attention biases). Transformers and ViTs only. | The paper reviewers will compare block 30 to. Ours differs: the whole vector, late, every position measured, a Hyena model, and **mean-replacing block 30 breaks the model**, so it carries information and isn't a constant bias. **Missing checks:** how many dimensions carry block 30's size; whether it's a large constant part plus a small informative part; block 30 at position 0. |
| **"Massive activations are architecturally robust"** (arXiv 2606.20743, small transformers) | Given a separate output channel, huge values rebuild themselves in whatever representation the model decodes from, which points to a function, not an accident. | Doesn't cover our case (early layers, start token). But it supports a *functional* reading of block 30, which is Evo 2's decoding representation. |
| **Review: "What Do Biological Foundation Models Compute?"** (bioRxiv 2026.03.04.709491) | Three levels: representational → computational → causal and mechanistic. The field is "stuck at level 1"; causal patching exists in one protein study; Evo 2's architecture hasn't been analysed mechanistically. Calls for null models, non-circular validation, experimental evidence; SAE features are unstable (~30% survive a change of seed). | States our gap in writing: we work at levels 2–3. **Controls we still owe:** a random-weights baseline (would untrained Hyena gates produce 3-letter-word detectors anyway?); causal validation of any SAE feature we use; caution with genome-annotation matching. |

### Learnings

1. **The copying result is an extension, not a discovery.** "Attention does lookup" is established in text hybrids.
   Ours is the first in Hyena and DNA. Frame it as confirming and extending, and as the mechanism behind Evo 2's
   needle-in-a-haystack result.
2. **Block 30 must be separated carefully from massive activations.** Current evidence says it's different
   (information-carrying), but two cheap checks are needed first.
3. **Evo 2's MLPs are bilinear** (identity activation after block 0). There's a whole method for exactly this layer
   type: "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce et al., ICLR 2025, arXiv
   2410.08417), which reads features straight from the weights by eigendecomposition. Michael Pearce is also an
   Evo 2 co-author. This is the most direct route to MARV-style weight-space analysis of Evo 2's MLPs.
4. **The LI filter parameterisation (poles and residues)** comes from "Laughing Hyena Distillery" (Massaroli et al.,
   NeurIPS 2023, arXiv 2310.18780). That's the theory behind our weight-read reach measurements.
5. **Controls matter to reviewers:** random-weights baselines, SAE seed instability, circular annotation matching.

### New items for the full run

| Check | From |
|---|---|
| Block 30: dimensions carrying its size; constant vs. varying part; remove only the constant part; position 0 | Sun et al. |
| Random-weights baseline for the block 0 enumeration | the review |
| SAE feature f/24278 on harmful BRCA1 variants, then switch it off | the Evo 2 paper |
| Keep-only / swap generation states: attention vs. Hyena | arXiv 2609.04434 |
| Head-level search for copying heads in L3 | arXiv 2510.23006 |
| Weight-based analysis of Evo 2's bilinear MLPs | Pearce et al. |

---

## 2026-09-21: What the papers can't tell us; the cross-model plan

We read the full Hyena Hierarchy and HyenaDNA papers to see whether they already answer "do other Hyena models have a
funnel?" and "does attention do the copying everywhere?". They don't.

- **The papers show capability, not use.**
  - Small Hyena-only models *trained on a synthetic recall task* reach 97.2% recall at 131k tokens (Hyena
    Hierarchy).
  - Evo 2 finds a needle in 1M letters (Evo 2 paper).
  - HyenaDNA needs 450k–1M letters for species classification.

  None of them opens a large trained model to see *which part* does the work. Our Evo 2 result is the missing half:
  Hyena **can** copy, yet the trained hybrid **routes copying through attention**, with LI only helping at long range.
  "Capable but not used" is a finding in itself.
- **No paper reports per-layer activation sizes for any Hyena model**, so the block-30 funnel is unexamined everywhere.
  Evo 2's README says the 1B/20B/40B models need FP8 "for numerical accuracy", a hint at numerical issues that isn't
  explained.
- **Why:** these are capability and engineering papers (benchmarks, speed, synthetic tests on toy models).
  Interpretability of trained Hyena models is essentially undone.

### Cross-model plan: the same tests on other Hyena models

| Model | Architecture (checked) | Hardware | What it tests |
|---|---|---|---|
| **HyenaDNA** (0.44M–6.6M params, 2–8 layers, up to 1M context) | Hyena only, **no attention**; filters produced by a small network | T4, even CPU | Does a pure-Hyena DNA model copy at all, and which layer does it? Filter reach; funnel (few layers) |
| **Evo 1** (7B; `evo-1-8k-base`, `evo-1-131k-base`; Apache-2.0) | 32 blocks: **attention at 8, 16, 24 only**, 29 Hyena blocks of one long-filter type (modal, state size 8), GELU MLPs; trained on prokaryotes and phages | A100 (or a 24 GB card for short inputs); loads through HF `transformers` with custom code | The closest comparison to Evo 2: does it have a funnel? Does copying go through its 3 attention layers? Load-bearing map; mutation depth. E. coli is in-distribution. |
| **StripedHyena-7B** (text) | Same code family as Evo 1 | A100 | Is the operator split the same in text? |

Adapter effort: Evo 1 and StripedHyena-7B share one adapter (a different module layout from Vortex, same ideas).
HyenaDNA needs its own adapter, because its filters come from a small network rather than stored weights.

---

## Glossary

- **Residual stream**: the shared log every block appends to. The final guess reads it.
- **Write**: what one block appends to the log. Each block appends two: mixer and MLP.
- **Mixer**: the part of a block that looks at *other* positions in the DNA (SE / MR / LI / attention).
- **MLP**: the part of a block that processes the current position only.
- **Ablation / stubbing**: replacing a component's write with its average (mean-ablation), to see what breaks.
- **Patching**: run the normal DNA and the mutated DNA; copy one component's write from the mutated run into the
  normal run; if the output shifts toward the mutated result, that component carries the mutation's effect.
- **Direct vs. total effect**: direct = what a write contributes to the output by itself; total = including everything
  it changes downstream. Our "trace" tool measures direct; patching measures total.
- **Bottleneck**: a block whose write dominates the log, so the output depends almost only on it (block 30 is the
  suspect).
- **Health check**: does the model still predict ordinary DNA well under this intervention? Near 25% accuracy (or
  log-prob −1.386) = guessing = broken.
- **Codon**: DNA is read in 3-letter words inside genes. The **3rd position ("wobble")** matters least for which amino
  acid is made, so it's hardest to predict. That creates a 3-letter rhythm in prediction accuracy.
- **Start / stop codons**: `ATG` starts a gene; `TAA`, `TAG`, `TGA` end one.
- **Shine-Dalgarno**: a short `AGGAGG`-like sequence just before bacterial genes, where the ribosome binds.
- **nats / log-prob**: how confident the model was in the right letter. 0 = certain, −1.386 = pure guessing among 4.
- **AUROC**: how well a score separates two groups. 0.5 = no better than a coin flip, 1.0 = perfect.
- **bf16**: the 16-bit number format the model runs in. It explains the ~0.5% rounding in our self-checks.
- **Load-bearing layer**: a single layer whose removal alone breaks the model (L0, L1, L9, L29, L30 in Evo 2 7B).
- **Main effect vs. interaction**: a main effect is what one input position does *on average*; an interaction is an
  effect that only appears for a *combination* of positions (like a bug that needs two flags on at once).

