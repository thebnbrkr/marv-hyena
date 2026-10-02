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
3b. **The genetic code: under review.** Round 4 reported that a missense change disturbs the model more than a
   silent one at the same codon position in 68.9% of 119 sites, peaking in SE blocks 7/11/14 (finding 19). The
   2026-09-25 review found both halves confounded: the silent change was nearly always a transition and the missense
   a transversion (on the 17 sites where both were transversions, missense won 8/17), and the SE peak came from the
   argmax of a ratio. Round 5 (P25, P26) tests it properly. *(Findings 19, 23)*
4. **Look-up and copy (attention, mainly L3; LI helps at long range).** Exact repeats are recognised at any distance,
   from 100 to 10,000 letters. Attention is essential; the LI layers (especially L2) help only at long range.
   Copying is its own circuit: it survives even when ordinary reading is broken (e.g. with L1 removed).
   *(Findings 1, 7, 13)*
5. **Far context (unresolved).** 50,000 letters of upstream DNA help only slightly (0.016 nats per letter averaged
   over 5 genes). Removing attention erases that, but so does removing SE, which reaches 7 letters and cannot be
   the carrier: the test measures damage, not a pathway. Which operator carries far context, if any, is unknown.
   *(Findings 9, 17, 24)*
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
- Likely, one check owed: 1 (block 0 is learned — the random-weights null gave 0 vs 46, but through a hard 80% cutoff;
  P28 removes the cutoff).
- Under review (2026-09-25): 3b (genetic code in SE — letter-type confound, P25/P26). Withdrawn: 5 (far context via
  attention — its SE negative control failed, finding 24).
- **Bit-identical replication (round 3b):** the whole round-3 result set reproduced exactly on a different A100 SKU
  and CUDA version, so 2, 3 and 8 are no longer "measured once" in the numerical sense — though replication of an
  arithmetic result is not the same as replication across genomes or checkpoints.
- **Null model now run (round 4, finding 18):** a weight-shuffled block 0 produces zero detector channels for all
  64 words on 3 seeds, against median 46 for the trained model. Findings 4, 10, 15 and 16 stand, but the zero comes
  through one hard cutoff, so "learned" waits on P28.

**Not yet known:**
- whether Evo 2 represents amino acids, or only knows where the wobble position is (P25);
- whether the copying circuit is used on *less conserved* real repeats — round 4 answered only for 16S rRNA, which
  the model already predicts at 98.9% on first sight; IS2 and IS3 are the real test (round 5);
- whether 16S's predictability is conservation, low entropy, or memorisation of this locus;
- whether premature stops beat missense (P22 underpowered: 11 sites, 2 usable);
- what the rest of the middle layers compute — round 4 accounts for blocks 7/11/14 (the genetic code) only;
- whether LI carries far context (its condition keeps breaking);
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

## 2026-09-23: Round 3b, the two reruns (Evo 2 7B, 80 GB A100)

Round 3 lost two of its six steps: the far-context test ran out of memory on a 40 GB card, and the block-30
attribution crashed on the inference-tensor bug. Both fixes shipped (`memory.py`, `interface._autograd_safe`), and the
whole notebook was rerun clean — 19 code cells, executed 1 to 19 in order, zero errors, smoke checks 8/8, 37 tests
passed. Raw outputs: `results/round3b/`.

### The replication is the first result

This run drew an **80 GB A100 on CUDA 13.0**; round 3 had a **40 GB A100 on CUDA 12.8**. Same `vtx` 1.1.0, same
`evo2` 0.6.0. Every one of the ten result keys shared with round 3 came back **bit-identical**, down to baseline
health at 16 significant digits (0.6959707140922546 / −0.6843298673629761).

So findings 12 and 14–17 replicated exactly across two A100 SKUs and two CUDA versions. We had not claimed
reproducibility before; now we can, and for a paper it is worth stating plainly, because "measured once" was the
confidence label on half the findings.

One caveat on the memory fix: `memory.install()` is unconditional in `HyenaModel.load`, so the chunked filter *was*
exercised — but on an 80 GB card, where it did not have to save us. It has still never been shown to fit the original
50k-letter test into 40 GB.

### Finding 17: far context goes through attention (P13's context half, finally run)

Baseline benefit from 50,000 letters of upstream context versus 500: **0.0160 nats/letter**, over 5 genes.

| switched off (load-bearing kept on) | creD | gspE | ilvI | uup | yehQ | mean | of baseline |
|---|---|---|---|---|---|---|---|
| nothing | .0163 | .0236 | .0167 | .0078 | .0158 | **.0160** | 100% |
| SE | .0016 | .0020 | .0004 | .0080 | .0052\* | .0034 | 21% |
| MR | −.0022 | −.0123 | −.0173 | −.0031 | .0067 | −.0056 | −35% |
| LI | .0565 | −.0019 | .0715 | −.0188 | **−.8043\*** | −.1394 | −870% |
| attention | −.0009 | −.0020 | −.0027 | −.0006 | −.0006 | **−.0014** | −9% |

\* flagged broken

**The attention row is the cleanest result in the file.** All five genes go from positive to slightly negative, none
broken, and the ablated model still predicts well (−0.59 to −1.25, clear of guessing at −1.386). Removing attention
abolishes the far-context benefit. That upgrades finding 9 from *likely* to *measured*, and settles P2 (registered in
round 1 as "far context travels through LI") as refuted.

**The LI row is not a result, and we nearly reported it as one.** Mean −0.1394 looks like LI removing 970% of the
benefit. It is one broken run: yehQ under `-li` scored −2.18 nats/letter — *worse than uniform guessing* — and carries
the `broken` flag. Drop it and the other four genes average **+0.0268**, i.e. LI removes nothing. P13's own escape
clause ("if `-li` is still broken, UNTESTABLE") applies. The lesson from round 1 keeps recurring in new disguises: the
health flag is on every row for a reason, and a mean over a set containing a broken run is not a measurement.

**What this experiment still cannot support.** Two confounds, both worth fixing before any of it goes in a paper:

1. **Headroom.** Every ablated condition sits at −0.96 to −1.37 nats against baseline's −0.30 to −1.02. A model that
   close to guessing has almost no room left to show a 0.016-nat context gain, so "removed the benefit" is partly
   confounded with "degraded the model". `-mr` coming out *negative* on 4/5 genes is most likely this, not a finding
   about MR.
2. **Signal against noise.** The effect is 0.016 nats and the between-gene spread among healthy `-li` runs is
   +.0715 / +.0565 / −.0019 / −.0188, about four times larger. Five genes is too few. yehQ is the weakest gene at
   baseline and the one that breaks in two conditions — a bad probe that should be replaced.

### R3.4: the fix worked, the method did not

Integrated-gradients attribution at block 30's input now runs at all 6 positions. All 6 **fail the completeness
self-check at ~100%**: attribution sums to ~0.001 where the real change in `f` is up to 5.9. P14 pre-registered
"completeness error > 25% means the method fails here", so the answer is UNTESTABLE, and under invariant 1 nothing
underneath it gets reported — including a tempting regularity, L29's mixer (+0.206) cancelling its own MLP (−0.207)
at every single position.

What the failure *does* show is the funnel running backwards. Share of absolute attribution by block:

```
L29: 97.7-100.0%     L28: 0.0-2.3%     L27 and earlier: 0.0%
```

The gradient dies going backward at the same rate the activations grow going forward. Best explanation: block 30
amplifies to ~10¹² and the final RMSNorm divides that scale straight back out, so `f` depends on a *direction* that
can flip sharply at one point along the path from the embedding-only baseline to the real residual. 32 midpoint
samples step over the spike and integrate to nothing. The embedding-only baseline makes it worse — a residual with
zero writes is somewhere the model has never been.

This is worth stating as a finding rather than a failure, because it generalises: **the block-30 funnel defeats
gradient-based attribution.** Anything that wants to know what Evo 2 reads has to be causal. Round 4 opens with a
512-point forward-only scan of `f` along that path (P18) to confirm the shape, and then replaces the method with
mean-ablation of each write entering block 30.

---

## 2026-09-23: Round 4, biology instead of string statistics

Round 4 came out of showing the round-0-to-3 lab notebook to a working biologist. The criticism was not about any
number in it. It was about what the numbers are *of*: every test so far is a sequence-statistics test — copy a random
200-letter stretch, measure codon periodicity, score next-letter accuracy. None of them asks the model to do anything
a biologist would call a biological process.

Three concrete challenges came out of that conversation, and one of them is a control we already owed ourselves.

### The challenges

**1. "Attention mediates exact sequence retrieval, not a biological process."** Our headline copying result uses a
*random* insert. That proves the model *can* retrieve. It does not show the circuit is used when reading a genome —
the synthetic probe may recruit machinery real DNA rarely touches. Our own novelty check already conceded that
"attention does lookup" is established in text hybrids and that ours is an extension; if the extension only holds on
a synthetic task, it is a weaker extension than we have been claiming.

**2. "Make sure block 0 isn't learning — it's just a filter."** Block 0 gives detector channels to all 64 three-letter
words, median 46, mean 45.2. That distribution is suspiciously *flat*, and flat is exactly what an untrained gated
9-letter convolution might produce from random projections. The literature review already recorded this as owed
("a random-weights baseline: would untrained Hyena gates produce 3-letter-word detectors anyway?"), from the
bio-foundation-model review that calls null models the field's missing control. A biologist reached the same
objection independently, which is a good sign about the objection and a bad sign about our having deferred it.

**3. "Do a translation test."** Read as the genetic code: we have shown the model tracks the reading *frame* (finding
13, MR layers). We have never asked whether it knows what a codon *means*.

A fourth suggestion — protein folding — does not transfer, and the honest answer is to say so. Evo 2 is a nucleotide
model that never observes 3D structure. There is no well-posed "which part computes folding" question to ask it; any
test would be a long indirect chain through codon usage and translation rate. That is a category mismatch with the
model's input, not a limitation of the method.

### What round 4 runs

Ordered so that the test which can invalidate existing findings runs first. All of R4.1-R4.3 use data already in the
repo — no new downloads, and E. coli throughout.

| step | question | new code | prediction |
|---|---|---|---|
| R4.0 | is the IG failure a sharp path? then replace it with causal ablation at block 30's input | — | P18 |
| R4.1 | is block 0's word bank learned, or architectural? | `nullmodel.py` | P19 |
| R4.2 | does the copying circuit fire on real genomic repeats? | `genome.py` | P20 |
| R4.3 | does Evo 2 represent amino acids, or only letters? | `codons.py` | P21, P22 |

**R4.1, the null model.** `nullmodel.random_weights` swaps a block's parameters for random ones and restores them
exactly. The default mode *shuffles* each tensor, preserving the weight multiset exactly — same mean, variance, every
moment — so only the arrangement is destroyed and anything that survives is architecture rather than training. Then
the same 4⁹ enumeration runs on the scrambled block 0 and the two 64-word distributions are compared. Restoration is
checked twice (bit-identical logits in the test suite, `verify_restored` in the notebook) because everything measured
afterwards depends on it.

We expect this to go against us: P19 predicts the shuffled null *also* produces a flat bank for all 64 words. If so,
findings 4, 10, 15 and 16 describe a gated 9-letter convolution rather than anything Evo 2 learned, and the README
and the living summary both need rewording. Registering that expectation before running is the point of the exercise.

**R4.2, real repeats.** `genome.find_repeat_families` pulls E. coli's actual repeat families out of the GenBank
annotations — the seven rRNA operons, the IS elements — and `repeat_probes` builds the round-1 copy probe three ways
at matched geometry: the real repeat, the same repeat with its letters shuffled (composition held, biology gone), and
a random insert. The measured quantity is the **retrieval gain**, `second_lp − first_lp`, not the second copy's raw
score. That distinction is the whole design: a real rRNA copy already scores well on its *first* appearance because
the model knows rRNA, and only the gain isolates what retrieval added on top of prior knowledge.

**R4.3, translation.** The genetic code is redundant, so there is a clean controlled comparison: at one codon, change
the third letter two ways — silently (ATT→ATC, both isoleucine) and missense (ATT→ATG, isoleucine→methionine). Same
site, same codon position, same edit distance, same neighbours; the only difference is whether the protein changed.
Eight codon families admit both, which is what `codons.wobble_sites` looks for. Premature stops are excluded by
default: a stop is a far larger biological event than an amino-acid swap, and mixing it into the missense arm would
confound "the protein changed" with "the protein ended". Stops get their own arm (P22), which is also the behavioural
counterpart of Evo 2's published SAE feature f/24278.

`codons.divergence_by_block` then measures where the two arms part company, per block, relative to each block's own
write size so that block 30's ~10¹² does not swamp the comparison. That is the localisation: the depth at which the
genetic code gets applied.

### One methodological note that shapes all of it

The obvious way to answer "which part lights up when the model does biology" is attribution, and R3.4 just showed why
that will not work here: the gradient reaches exactly one block back before the funnel kills it. Every round-4
measurement is therefore causal — ablation, patching, matched substitutions — or representational at intermediate
blocks. None of them reads through the output. The block-30 bottleneck is not just a finding about Evo 2; it is a
constraint on what can be measured in it, and it is the reason the natural experiment design has to be abandoned.

### Writing this up

The goal these round-4 tests are pointed at is a conference paper on how Evo 2 reaches a decision. The ordering
reflects that: P19 can retract three existing findings, so it runs before anything is written; P20 decides whether
the copying result is about genomes or about our probe; P21 and P22 are the first tests in this project where a
positive result would be a statement about biology rather than about sequence statistics.

---

## 2026-09-23: Round 4, the null model and the biology (Evo 2 7B, 80 GB A100)

Smoke checks 8/8, 60/60 tests, all 19 cells clean. Load-bearing set came back `[0, 1, 4, 9, 29, 30]` for the third
consecutive run. Raw outputs: `results/round4/`.

Two predictions confirmed, two refuted, one untestable. Both refutations moved things forward rather than back.

### Finding 18: block 0's word bank is LEARNED (P19 refuted, in our favour)

This was the test that could have retracted four findings. It did the opposite.

| | median | mean | cv | range |
|---|---|---|---|---|
| trained | 46 | 45.2 | 0.280 | 6 – 65 |
| weight-shuffled, 3 seeds | **0** | **0** | — | 0 – 0 |

The shuffled null produces **no detector channels at all** — not for any of the 64 three-letter words, not on any of
the three seeds. We predicted the opposite: that a flat distribution over all 64 words was what an untrained gated
9-letter convolution would produce anyway, and that findings 4/10/15/16 described the architecture.

Checked before believing it: no NaNs, every value exactly 0, and the detector test is **rank-based** (does ≥80% of a
channel's top-50 9-mers share the motif, at ≥3× composition-controlled enrichment), so the change in output scale
that shuffling causes cannot explain a zero. The weights were verified restored afterwards.

So block 0's motif bank is a learned structure, and the *flatness across all 64 words is itself learned* rather than
an artifact of random projections. Findings 4, 10, 15 and 16 survive and are now stronger than any of them were
when measured: they carry the null model the bio-foundation-model review asks for by name ("SAEs trained on randomly
initialized models" — we ran the weight-space analogue).

Caveat worth keeping: the trained bottom five are all homopolymers (AAA 6, CCC 8, TTT 12, GGG 13). That is plausibly
the composition control penalising them, since a channel whose top k-mers are A-rich has a high expected AAA rate.
It does not affect the null comparison, which uses the identical measure on both sides.

### Finding 19: SE applies the genetic code, MR carries the frame (P21 confirmed)

At one codon site, change the third letter two ways — silently (ATT→ATC, both isoleucine) and missense (ATT→ATG,
isoleucine→methionine). Same site, same codon position, same edit distance, same neighbours. 119 sites.

| clause | result | predicted |
|---|---|---|
| missense more disruptive than silent, at the same site | **68.9%** | ≥ 65% |
| median effect ratio (n_usable = 48) | **1.319** | ≥ 1.20 |
| modal divergence block | **11** | > 7 |

**Evo 2 represents amino acids, not just letters.** Robust to the single outlier (68.6% without it). This is the
first result in the project that is a statement about biology rather than about sequence statistics.

The localisation was not predicted and is the more interesting half:

```
block  7  se   21 sites (17.6%)      block 18  se   2       block  3  attn  1
block 11  se   64 sites (53.8%)      block 28  se   2       block  6  li    1
block 14  se   19 sites (16.0%)      block 29  mr   9
```

**90.8% of sites peak in SE blocks**, 87.4% in blocks 7/11/14 alone. Set against finding 13 (MR carries the reading
frame), that is a clean division of labour:

- **MR**, 128 letters ≈ 40 codons — *where* the frame is;
- **SE**, 7 letters ≈ 2 codons — *what* the codon says.

It also sits mid-network, after the early hand-off of finding 14 and well before the funnel, which is a first partial
answer to "what do the redundant middle layers compute?"

### Finding 20: the funnel defeats gradient attribution; causal ablation inverts the answer (P18 partly confirmed)

A **single step out of 512** carries 97.4–99.9% of `f`'s entire range along the path from the embedding-only baseline
to block 30's real input, at every one of the six positions. `f` is a step function. The refutation condition (smooth,
no step above 10%) is nowhere close, so round 3b's ~100% completeness failure was never a bug in
`interface.block_input_attribution`.

The second clause failed: 80% of the *total variation* needs 25–46% of the path, not <10%. Not a contradiction —
`f` is a near-discontinuous jump **plus** heavy high-frequency noise along the whole path. Both defeat integrated
gradients, for different reasons: midpoint sampling steps over the jump, bf16 jitter swamps the gradient elsewhere.

Replacing it with mean-ablation of each write entering block 30 gives the opposite answer:

| | integrated gradients (round 3b) | causal ablation (round 4) |
|---|---|---|
| L29 | 97.7–100% | 12.0% |
| L28 | ≤ 2.3% | 8.3% |
| L27 and earlier | **0.0%** | the remaining ~80% |

Causal attribution reaches **block 0**. Top contributors: L29 (12.0%), L1 (10.3%), L28 (8.3%), L2 (7.9%), L9 (7.3%),
L0 (7.2%), L4 (6.4%) — essentially the load-bearing set plus L28 and L2, arrived at by a completely independent
route. **Block 30 reads from the whole network**; the gradient simply could not see it.

### Finding 21: on a real conserved repeat, retrieval is redundant (P20 refuted)

| arm | first_lp | first_acc | retrieval gain |
|---|---|---|---|
| **real 16S rRNA** | **−0.032** | **98.9%** | **0.028** |
| shuffled (same composition) | −1.401 | 28.3% | 1.392 |
| random insert | −1.431 | 26.1% | 1.421 |

The model predicts 16S rRNA at 98.9% accuracy **on its first appearance**, so there is no headroom for retrieval to
fill and the gain is 0.028 where we predicted ≥0.5. That meets P20's own fallback clause: the model predicts real
repeats from prior knowledge alone and never needs to retrieve.

Meanwhile `-attn` collapses the random/shuffled gain from 1.42 to 0.01, and `-li` keeps 69% of the gain at a
1,000-letter gap but only 29% at 10,000 — independently replicating findings 1, 7 and 13 on a new probe.

So the honest answer to the biologist's challenge is neither yes nor no: **the retrieval circuit is real and
attention-dependent, but on a conserved real repeat it is redundant, because prior knowledge gets there first.**

Two things blunt this result, and both are fixable:

1. **It ran on one family.** The notebook cloned before the repeat-finder fix landed (its output says
   `identity 0.995` and `60 passed`, both pre-fix), so it got 16S only instead of six families. 16S rRNA is the most
   conserved sequence in biology — the worst possible probe for detecting retrieval. **IS2 (7 copies) and IS3 (5
   copies) are the real test**, and they are available now.
2. `retrieval_gain` presumes headroom. When `first_lp ≈ 0` it cannot measure anything, by construction.

### P22: untestable, by our own gate

Only 2 of 11 nonsense sites were `usable`, against the ≥20 pre-registered. The direction is not subtle — **11/11** on
the sign test, mean effect **−38.1 nats** against −0.7 for missense — but it is not a reportable number.

A design flaw of ours contributed: `usable` gates on the **silent** arm's effect, which is right for P21 and wrong
for P22. At these sites the silent arm averaged +0.23, barely disturbing, which disqualified rows whose nonsense arm
was enormous. Fix both ways: widen the track for more sites, and gate P22 on its own arm.

### On "memorization", which we nearly wrote down as fact

The first reading of finding 21 was "the model has memorized 16S rRNA". That is not supported, and a number in the
same run argues against it: **baseline accuracy on ordinary E. coli is 0.696.** If the genome were memorised that
would be near 1.0. So 16S at 98.9% is not "this genome is in training" — it is that 16S specifically is far more
predictable than ordinary E. coli DNA, in the same model, on the same run.

Three explanations remain, and round 4 cannot separate them: conservation/generalisation (Evo 2 saw thousands of
bacterial 16S homologs), low intrinsic entropy (16S folds into a rigid structure, so bases are heavily constrained),
or genuine memorisation of this locus — now the least-supported of the three. The correct wording is **prior
knowledge**. One confound was checked and cleared: the probe background is 1.5–1.56 Mb and all seven rrn operons sit
at ~223 kb, 2.7, 3.4, 3.9, 4.0, 4.2 Mb, so no rRNA leaked into the background.

---

## 2026-09-23: Novelty re-check against the papers (revised; supersedes the round-3 check)

The round-3 novelty check was done with ~8 searches and no full reads. This one reads the sources. Two claims we were
making have to be softened, and one becomes much stronger.

### The architecture paper states our premise for us

We finally read **StripedHyena 2** ([arXiv 2503.01868](https://arxiv.org/html/2503.01868v1)), the paper that built
the architecture Evo 2 runs on. It contains **no layer ablation, no per-layer activation magnitudes, no numerical
stability discussion, and no interpretability of any kind** — no attribution, no probing, no analysis of learned
filters. No biology either.

Its operator specialisation claims are prose, sourced to prior *synthetic* work (Akyürek et al. 2024; Poli et al.
2024), with no new evidence:

> **Hyena-SE** "specializing in local multi-token recall" · **Hyena-MR** "tailored to efficient modeling across
> hundreds of tokens" · **Hyena-LI** "aggregate information over the entire sequence" · **Attention** "optimized for
> targeted recall of information across longer sequences"

Putting those next to what we measured is the sharpest framing this project has:

| operator | the architecture paper claims | we measured | verdict |
|---|---|---|---|
| **SE** | "local multi-token **recall**" | no part in recall (copying 1.000 / 1.000 / 0.989 with SE removed); it is where the **genetic code** is applied (90.8% of codon divergence peaks in SE 7/11/14) and its removal halves the codon rhythm | **wrong** |
| **MR** | "modeling across hundreds of tokens" | uses that window for one job: the **reading frame** (rhythm 0.251 → −0.016), copying untouched | **vague, now specific** |
| **LI** | "aggregate information over the **entire sequence**" | most channels reach **4–7 letters**; does **not** carry far context (attention does, 5/5 genes); assists only long-range copying | **wrong** |
| **attn** | "targeted recall across **longer** sequences" | essential at **every** distance including a 100-letter gap that MR's 128-letter window could span; also carries far context | **right, understated** |

**Three of four claims are wrong and the fourth is understated.** The architecture was built on a specialisation
story borrowed from synthetic benchmarks; in the trained model the operators do specialise, just not that way.

The one ablation they do run is architecture-level, not layer removal: Table 2.1 compares block *layouts*
(SE-SE-LI vs SE-MR-LI vs LI-LI-LI) at 7B/400B tokens and finds SE-MR-LI best on pretraining quality. That does not
scoop anything — it proves **the mix matters** and never asks why. It is the motivation section we were missing.
They also give no justification for the striping pattern or for why five attention layers.

### The load-bearing map is a replication, not a discovery

This is the correction the round-3 check needs most. We listed "a load-bearing layer map" as not-found. Not found
*for Hyena or Evo 2* — true, and confirmed by reading StripedHyena 2, the Evo 2 paper and the Goodfire report. But
the idea is established:

- [**ShortGPT**](https://arxiv.org/html/2403.03853v3) and *The Unreasonable Ineffectiveness of the Deeper Layers*
  built layer-importance maps for LLMs. Their pattern: **shallow layers crucial, middle-to-late redundant, initial
  and final layers important.** Ten of LLaMA-2-13B's 40 layers removed costs MMLU 55.0 → 52.2.
- [**Hegde, Nebel & Rahman, *Genes* 16(11):1358**](https://doi.org/10.3390/genes16111358) (Nov 2025) ablated **every
  layer** of DNABERT-2 (12 layers) and Nucleotide Transformer (24 layers), built "layer importance profiles", and
  pruned the redundant ones. Layer-importance mapping already exists for DNA language models.

Our map — L0, L1, L4, L9 early plus L29, L30 late, middle 10–28 expendable — *is ShortGPT's pattern*. Claim it as a
replication in a new architecture class. (The Genes paper is encoder-only transformers, a downstream task and an
efficiency goal, so no operator types exist in it to compare; the full text is paywalled to us, so its internal
details are not characterised here.)

What has no precedent is one level down: **every Hyena family contains a load-bearing layer and attention contains
none.** That asymmetry is new, and it is what made the fair family tests of round 3 possible at all.

### Three papers the round-3 review missed

| paper | what it means for us |
|---|---|
| [**PAS-ISP**](https://arxiv.org/abs/2608.12149) (Aug 2026) — massive activations in hybrids **spike immediately before full-attention layers**, then plateau through the intervening linear-attention layers | Looked like the live threat to the funnel, since block 30 (LI) sits immediately before block 31 (attn). **Checked and cleared — see finding 22 below.** |
| [**Mamba activation-subspace bottleneck**](https://arxiv.org/html/2602.22719) — a Layer-20 bottleneck with "low gradient sensitivity and extremely high post-ablation KL divergence" | Our exact finding-20 signature, in a different architecture. Demotes "gradients fail, ablation works" from discovery to convergent confirmation — which is still worth reporting, and worth citing. |
| [**Memorization in genomic LMs**](https://arxiv.org/html/2603.08913) — Evo **1** recovers 100% of planted canaries; their canaries are "random nucleotide strings carrying no biological structure" and they say it "remains an open question whether memorization of real sequences would manifest at comparable rates" | Our synthetic copy probe is their canary experiment; they show *that* Evo recovers, we show *which component does it*. Our 16S result sits precisely in their stated gap, and their framing is why "prior knowledge" is the right word, not "memorisation". |

Also upgraded: [**What Attention Recalls and Recurrence Controls**](https://arxiv.org/abs/2609.04434) is EMNLP 2026
Findings and is *causal* (split-prefill, state-swap on Qwen3.5/Falcon-H1: retrieval 64–98% through attention, **zero**
through recurrence). It is a stronger prior on our copying result than the round-3 entry credited. And
[Zoology](https://arxiv.org/abs/2312.04927) pretrained 17 models on the Pile and attributes **82%** of the
attention–gated-conv gap to recall; also stronger than credited. Both still stop short of opening a large trained
model.

### Where the novelty actually sits, after all that

**Strong, nothing comparable found:**
1. The block-0 **random-weights null** (trained 46/word vs shuffled 0/word). The bio-FM review asks for exactly this
   class of control; nobody has run it on a DNA model's first layer.
2. **SE applies the genetic code, MR carries the frame** — causal localisation of a biological computation to an
   operator *type*. Codon↔amino-acid structure is known in *codon* LMs and *protein* LMs, but those take codons or
   residues as input, and the results are representational clustering, not causal localisation.
3. The exhaustive 4⁹ first-layer dictionary.
4. The operator-family load-bearing asymmetry.

**Confirm-and-extend, and should be written that way:** copying/retrieval (Zoology, arXiv 2609.04434);
gradients-fail-ablation-works (Mamba subspace); the load-bearing map (ShortGPT, Genes 16:1358).

### Finding 22: the funnel is not a pre-attention spike

PAS-ISP measures `max_j |X_{t,j}|` — the single largest *dimension* of a token's hidden state — at attention-sink
token positions, in text hybrids built from linear attention (RetNet, HGRN, GLA, DeltaNet, GDN; Qwen3.5, Kimi
Linear, Nemotron-H, Zamba2). It covers no convolutional mixer, no Hyena, and no non-text domain, and it discusses
neither final layers, nor output logits, nor gradients. Its only intervention is modulating output gating; it never
deletes a spike.

Ours is the **L2 norm of the entire write**, at **every position measured**, in the **final** blocks, and
mean-ablating it breaks the model. Different quantity, different place, different architecture.

The decisive test was already in `results/round2/`. If block 30's magnitude were a PAS, the other four attention
layers (3, 10, 17, 24) should show spikes before them, at blocks 2, 9, 16 and 23. Each pre-attention block against
the mean of its nearest non-attention neighbours, in all three genome regions:

| block | region 0 | region 1 | region 2 |
|---|---|---|---|
| L2 | 0.88x | 0.83x | 0.90x |
| L9 | 1.41x | 1.43x | 1.58x |
| L16 | 1.87x | 1.88x | 1.88x |
| L23 | 1.14x | 0.98x | 1.04x |
| **L30** | **3.2e6 x** | **1.7e7 x** | **5.5e6 x** |

**Evo 2 has no consistent pre-attention spike.** L2 sits *below* its neighbours, L23 is flat, and L9/L16 are mildly
elevated in the way a depth trend produces. Block 30 is six to seven orders of magnitude past anything PAS predicts.
And in PAS-ISP the spike exists to serve the attention layer after it, whereas block 31's attention write is 0.38 and
removing it gives bit-identical output: our spike feeds a dead layer.

So PAS-ISP is related work, not a scoop, and the separation is quantitative rather than rhetorical. It arguably
strengthens the funnel: the known hybrid morphology is *absent* here, and something categorically different is
present instead.

**One framing fight worth having in print.** The bio-FM review's position — and essentially all bio-FM
interpretability — is that the unit of analysis is the **feature**, via SAEs. Our work is entirely
**component-level**: blocks and operator types. That is a disagreement, not a gap, and finding 19 is the best
evidence on our side: a component-level result with real biological content that no SAE has reported.

---

## 2026-09-25: A review of round 4, in plain language, and the design of round 5

A second pair of eyes went through the round-4 branch: the code, the raw result files, and the papers cited. The
code is sound (63 tests, then 83 with round 5's). Three of round 4's headline claims did not survive the review, and
one round-3b claim that had been withdrawn in a working copy was reinstated by mistake. This entry explains each one
from scratch, because the people evaluating this work include readers without a biology or machine-learning
background. The formal record is in `PREDICTIONS.md` under "Round 4 review".

### Five pieces of biology this entry needs

1. **DNA is a string over four letters**, A, C, G and T. Evo 2 reads it one letter at a time and predicts the next.
2. **A gene is read in three-letter words called codons.** Each codon stands for one **amino acid**; a chain of amino
   acids folds into a protein. So a gene is like source code, each codon an instruction, the protein the compiled
   program.
3. **The code has duplicates.** There are 64 codons but only 20 amino acids plus "stop", so several codons mean the
   same thing — usually differing only in the **third letter**. ATT, ATC and ATA all mean isoleucine. The third
   letter is called the **wobble position** because it can often change without changing the meaning.
4. **Three kinds of one-letter change inside a gene:**
   - **silent** (synonymous): the amino acid stays the same. ATT → ATC. The protein is identical.
   - **missense**: a different amino acid. ATT → ATG (isoleucine → methionine). The protein is altered.
   - **nonsense**: the codon becomes "stop". TAT → TAA. The protein is cut short, usually fatally for its function.
5. **Transitions and transversions.** The four letters come in two chemical families: **purines** (A, G — larger,
   two rings) and **pyrimidines** (C, T — smaller, one ring). A **transition** swaps a letter for the other member
   of its own family (A↔G or C↔T). A **transversion** swaps across families (A↔C, A↔T, G↔C, G↔T). Because of how DNA
   gets damaged and copied, transitions happen roughly two to three times more often in real genomes, even though
   there are twice as many possible transversions. A model trained on billions of letters will learn that
   transitions are "normal" variation and transversions are "surprising" — whether or not it knows anything about
   proteins.

### Finding 23: round 4's amino-acid result is mostly a letter-type result (so far)

Round 4 (P21) changed the third letter of the same codon two ways — silent and missense — and found the missense
change disturbed the model more at 68.9% of 119 sites. That was read as "Evo 2 knows amino acids".

The catch: the genetic code is built so that at the third letter, **silent changes are nearly always transitions and
missense changes nearly always transversions**. TTT and TTC are both phenylalanine (T↔C, a transition); TTA is leucine
(T→A, a transversion). So round 4's comparison was, in most cases, "transition vs transversion" wearing a biology
label. Splitting `results/round4/results_round4.json` by letter type:

| silent / missense letter type | sites | missense more disruptive | 95% interval |
|---|---|---|---|
| transition / transversion | 102 | **72.5%** | 63–80% |
| transversion / transversion (all isoleucine) | 17 | **47.1%** (8 of 17) | 26–69% |
| all (round 4's headline) | 119 | 68.9% | 60–77% |

When both changes are the same letter type, the effect is at coin-flip level.

**Why only 47% on those 17 — and why that does not prove the opposite either.** Three things are tangled together:
- **17 sites is too few.** The 95% interval runs from 26% to 69%: it cannot rule out a real amino-acid effect of the
  size round 4 claimed.
- **The silent arm lands on a rare codon.** The only way to make a silent *transversion* at the third letter is
  isoleucine ATT/ATC → **ATA**, and ATA is one of E. coli's rarest codons (about 7% of isoleucine codons). A rare
  codon is itself surprising, which pushes the silent arm to look more disruptive than it "should".
- **Isoleucine → methionine is one of the mildest amino-acid swaps**: both are oily (hydrophobic) and similar in
  size. Even a model that fully understands proteins might barely care.

So these 17 sites cannot say the model is blind to amino acids. They only say round 4's evidence does not show it
isn't. No part of the repo had explored this before the review. The Evo 2 paper itself (Fig. 2) already showed that
missense and stop changes lower Evo 2's likelihood more than silent ones across 36 species; its main text says
mutations were "introduced at each position" and reports no letter-type or codon-usage control that we could find.

**Gene-aware vs protein-aware.** Picture two spell-checkers. One has learned which letter slots in common words tend
to vary (colour/color) and flags odd changes in the slots that never vary: it knows *where* words bend, not what
they mean. The other knows that "cat → cot" changes the meaning and "colour → color" doesn't. Round 3 already showed
Evo 2 is at least the first kind — **gene-aware**: its MR layers track the reading frame, so it knows which letter is
the wobble position (finding 13). A gene-aware model that has learned "at the wobble position, transitions are
normal" would produce round 4's 68.9% without knowing a single amino acid. Whether Evo 2 is also **protein-aware** is
the open question round 5 is built to answer.

### The SE-peak claim used a statistic that finds small denominators

Round 4 also said that the missense/silent difference "peaks in SE blocks" (90.8% of sites; blocks 7, 11, 14). For
each site it took, in each of the 32 blocks, the ratio *missense disturbance ÷ silent disturbance*, and reported the
block with the largest ratio. A ratio explodes wherever the bottom number is tiny, and the maximum of 32 noisy ratios
finds exactly those blocks. The warning sign is in the data: at the winning blocks the ratio was 5–7×, while the
same comparison at the model's output is 1.32×. Round 4 saved only the winning block, not the 32 numbers behind it,
so this cannot be re-checked from the results file; round 5 keeps every block. A hint that the statistic can mislead:
on marv-hyena's own **untrained** tiny test model it puts 64% of peaks in SE blocks (base rate 25%) — and after
shuffling that model's weights, 62% land in LI instead. The statistic follows the particular weights, not biology.

### Block 0 "learned, 46 vs 0" went through a single cutoff

Round 4's null model (finding 18) scrambled block 0's weights and found **zero** word detectors, against a median of
46 per word in the trained model. The idea of the null is right and important (see Heap et al. below). But "detector"
means "at least 80% of the channel's 50 strongest inputs contain the word". An exact zero across all 64 words and all
3 seeds is what a cutoff produces when scrambled channels are *less* selective — for example, a random gated channel
whose top inputs split between two unrelated patterns, one pushing it up and one down, never reaches 80% even though
it responds sharply. Round 5 (P28) measures the selectivity itself, for every channel, and sweeps the cutoff.

### Finding 24: far context does not go through attention — restoring the round-3b reading

On 2026-09-23 two readings of the round-3b far-context rerun were written. One (in the round-4 branch, finding 17)
said "attention carries far context: removing it erases the benefit on 5/5 genes". The other (in an uncommitted
working copy, and in the published notebook page) said the test cannot tell. The data decide it
(`results/round3b/results_round3b.json`, broken rows excluded):

| removed | benefit of 50,000 vs 500 letters of context | per gene | health |
|---|---|---|---|
| nothing | +0.0160 | .017 .008 .016 .024 .016 | 0.696 |
| attention | −0.0014 | all five between −0.003 and −0.001 | 0.571 |
| **SE** | **+0.0030 (−81%)** | .000 .008 .002 .002 | 0.454 |
| MR | −0.0056 (erased) | −.017 −.003 +.007 −.012 −.002 | 0.425 |
| LI | +0.0268 (spread 0.038) | +.072 −.019 −.002 +.057 | 0.528 |

SE looks back 7 letters. It cannot carry 50,000 letters of context by any mechanism anyone has proposed, yet
removing it wiped out 81% of the benefit. It is a built-in **negative control**, and it failed: every removal damages
the model (health 0.70 → 0.43–0.57) by far more than the 0.016-nat effect being measured, so the test measures
*damage*, not a *pathway*. Finding 17's attention reading is withdrawn. What survives: LI is the only family whose
removal leaves the benefit intact. A fair redesign needs health-matched ablations (damage unrelated parts until health
matches attention's 0.571, then measure), genes with a larger native benefit, and the benefit scored relative to the
model's own log-probability.

*Lesson: a negative control only helps if someone acts on it when it fails. This one fired on the first run and was
read past once, because the headline condition behaved exactly as hoped.*

### A small fix: P20's −313%

Round 4 reported that removing attention "keeps −313%" of the retrieval gain on the real 16S repeat. The gain being
divided by was 0.028 nats — essentially nothing, because the model already predicts 16S rRNA at 98.9% on first
sight. `genome.summarize_repeat_test` now refuses (returns NaN) when the baseline gain is under 0.1 nats. Same lesson
as round 1's ±200% "fractions".

### How the model is built, for the non-specialist

Evo 2 is a *transformer-like* network in which most attention layers have been replaced by **convolutions**.

- **Attention** compares every letter with every earlier letter. For a sequence of *n* letters that is about n²/2
  comparisons: for 1,000,000 letters, half a trillion per layer, and a memory cache that grows with every letter.
- **A convolution** slides a fixed pattern (a filter) along the sequence. A short filter of 7 letters costs 7
  operations per letter; a filter as long as the whole sequence can be applied with the Fast Fourier Transform in
  about n·log₂n operations — for a million letters roughly 20 million instead of half a trillion. The long (LI)
  filters are also written as sums of decaying exponentials, which lets the model run them as a small running
  summary during generation, with memory that does not grow. That is why StripedHyena 2 can read a million letters
  at once, and why it keeps only 5 attention layers for the jobs that need exact look-up.

Evo 2 7B has 32 blocks. Each block has a **mixer** (looks along the DNA) and an **MLP** (processes the current letter
only), and each appends its output to a shared running total — the *residual stream*, "the log" in this project's
analogy. After block 31, the total is turned into four probabilities for the next letter.

```
 0 SE   1 MR   2 LI   3 ATTN
 4 SE   5 MR   6 LI   7 SE   8 MR   9 LI  10 ATTN
11 SE  12 MR  13 LI  14 SE  15 MR  16 LI  17 ATTN
18 SE  19 MR  20 LI  21 SE  22 MR  23 LI  24 ATTN
25 SE  26 MR  27 LI  28 SE  29 MR  30 LI  31 ATTN
```

SE looks 4–7 letters back, MR about 128, LI the whole sequence (fading), attention anything exactly.

**Is this like Titans?** Only in outline. Titans (Behrouz et al., arXiv 2501.00663) pairs attention (exact, short-term
memory) with a neural long-term memory that **keeps learning while it reads**: it updates its own weights at test
time, driven by how surprising each new token is. Evo 2 does **not** learn while reading. Its weights are frozen; the
residual stream is recomputed from scratch on every input; error rates only shape the weights during training. The
resemblance is the division of labour — attention for exact look-up, a compressed running summary (Hyena LI in Evo 2,
the memory module in Titans) for the rest — which is also the split the text-hybrid paper arXiv 2609.04434 measured.

### What the papers added (read 2026-09-25)

| paper | what it found | what it means here |
|---|---|---|
| **Induction Meets Biology** (Pomerants et al., ICML 2026, arXiv 2602.23179) | Protein language models (ESM-3, ESM-C) detect repeats in two stages: position heads and "amino-acid similarity" neurons build aligned representations, then **induction heads** in the middle-to-late layers attend from one copy to the other and promote the next token. Found with attribution patching + integrated gradients. | The closest precedent to our copying result (finding 1), in proteins rather than DNA, and a template for the head-level search in L3. Also a contrast: integrated gradients *works* in their transformers, and fails in Evo 2 because of the block-30 funnel (finding 20) — which makes the funnel the reason, not the method. Not previously cited here. |
| **The Mechanistic Invariance Test** (Cheng & Zhang, arXiv 2604.06549) | Five genomic models including **Evo2-1B**: apparent sensitivity to regulatory logic was driven by AT content (r = 0.78–0.96); composition effects beat positional ones 46-fold. | The same failure mode as P21: a letter-level property masquerading as biological understanding. Round 5 adds a G/C-change covariate for exactly this reason. |
| **Heap et al.** (arXiv 2501.17727) | Sparse autoencoders trained on **randomly initialised** transformers get interpretability scores about as good as those on trained ones. | Why random-weights nulls (P19, P28) are not optional: "it looks interpretable" is not evidence of learning. |
| **Position: beyond anecdotal evaluation** (Zhou et al., arXiv 2606.07607) | Genomic interpretability relies on cherry-picked examples; methods contradict each other and miss known motifs. Proposes tiered reporting. | Independent support for pre-registration and for shipping controls with every claim. |
| **Decode-gLM** (Maiwald et al., bioRxiv 2025.10.31.685860) | SAE-based tools to interpret and audit Nucleotide Transformer; found training-data leakage. | The nearest existing "audit" tool — feature-based, not a controls battery. |
| **Evo 2** (Brixi et al., Nature 2026) | Missense, stop and frameshift changes lower likelihood more than silent ones, 36 species. | P21/P22's *behaviour* is known; only the localisation would be new, and it needs the letter-type control. |
| **Goodfire, Interpreting Evo 2** | SAEs on layer 26, chosen because it "had the most interesting biologically-relevant features". No codon or reading-frame features reported on the page. | Unchanged: nobody has published component-level causal work on Evo 2. |

**Novelty after this re-read.** Searches found no operator-level causal analysis of Evo 2 or any StripedHyena model,
and no report of the block-30 funnel. The wiring findings (attention copies; the funnel; LI filters are short;
MR carries the frame; gradients fail at the funnel while ablation works) remain the strongest candidates. The two
round-4 claims ranked first in the 2026-09-23 novelty list — the block-0 null and "SE applies the genetic code" —
are the two with a control still owed, and should not lead until P25–P28 are in.

### Round 5: designs that could explain round 4 away

Built in `codons.paired_sites` / `design_sites` and `controls`, run by `notebooks/marv_hyena_round5_colab.ipynb`,
pre-registered as P23–P28. Every design makes two changes **at the same position**, chosen by (kind, letter type),
drawn from sites spread across the whole genome:

| design | change a | change b | what it isolates |
|---|---|---|---|
| `round4` | silent (mostly transition) | missense (mostly transversion) | round 4's exact sites, every block kept |
| `fourfold` | silent transition | silent transversion | letter type alone; protein unchanged (GCT → GCC vs GCA, all alanine) |
| `noncoding` | transition | transversion | letter type alone, no gene at all |
| `matched` | silent transversion | missense transversion | amino acid, letter type held fixed (Ile, Arg) |
| `flipped` | silent transversion | missense transition | amino acid, letter type pushing the *other* way (Arg AGG → CGG vs GGG; Ile ATA → ATT vs ATG) |
| `stop_matched` | stop, transversion | missense, transversion | stop vs amino-acid swap (Cys TGT → TGA vs TGG) |
| `stop_flipped` | stop, transition | missense, transversion | stop vs swap, letter type against the stop (Trp TGG → TGA vs TGT) |
| `round4_shuffled` | as `round4` | as `round4` | the SE-peak statistic on a weight-shuffled Evo 2 |

Then one regression pools the paired designs: the within-site difference against *did the amino acid change*, *did
the protein end*, *was it a transversion*, *did G/C content change* and *did the codon get rarer*, all at once.
Because both changes share the site, everything else about the site cancels.

How to read the outcome:
- `fourfold` ≈ round 4's rate and `matched` ≈ 50% → round 4 measured letter type; Evo 2 is gene-aware, not shown to
  be protein-aware.
- `matched` well above 50% and `flipped` ≥ 50% → Evo 2 cares about the amino acid even when letter type is held fixed
  or turned against it — the first result here that would be about biology rather than sequence statistics.
- SE still winning under `diff` but not under `fourfold` or `round4_shuffled` → finding 19's localisation survives.

Found while building it: isoleucine's rare codon ATA gives a second "flipped" site (ATA → ATT is a silent
transversion, ATA → ATG a missense transition). The hand-written expectation in the new test said only arginine
could; the test caught it.

### Other models: what each would teach

Not built yet; no adapter code exists. All numbers so far are **one checkpoint** (`evo2_7b`).

| model | what it is | the question it answers |
|---|---|---|
| `evo2_7b_base`, `evo2_7b_262k` | the same model at other training stages / context lengths | Do the findings survive a second checkpoint? No new code: rerun the notebooks with `MODEL_NAME` changed. The cheapest and most important next step. |
| **Evo2-1B** | smaller Evo 2 | The Mechanistic Invariance Test used it, so it would connect to that paper. Needs FP8, i.e. an H100; the A100s used so far cannot run it. |
| **Evo 1** (`evo-1-8k-base`, `evo-1-131k-base`) | 7B, trained on microbes; attention only at blocks 8, 16, 24; every other block one kind of long Hyena filter | Is the funnel a quirk of Evo 2 or of this model family? Does copying still go only through the few attention layers? Cannot test SE vs MR vs LI (it has one filter type). Needs an adapter (different module layout). |
| **StripedHyena-7B** | same family as Evo 1, trained on English text | Is the operator split DNA-specific or architectural? If text also copies through attention and has a funnel, we are describing the architecture; if not, we are describing what DNA training does. Shares Evo 1's adapter. |
| **HyenaDNA** | tiny (0.4–6.6M parameters), **no attention at all**, trained on the human genome | Can a Hyena-only model copy repeated DNA at all, and where? If yes, attention in Evo 2 is *preferred*, not *necessary*. Runs on a laptop CPU; needs its own adapter (its filters come from a small network); use human DNA, not E. coli. |

### Where this could go beyond the paper

**A controls toolkit.** Most of what the review did was not new interpretability — it was running the checks that
can explain a finding away: match the substitution types; compare against random weights; sweep the cutoff; refuse
ratios over tiny denominators; check whether the statistic picks the maximum of noisy numbers; check the metric has
room to move; replicate on a second checkpoint. `marv_hyena/controls.py` is the seed of that.

- *Who it is for:* less the model makers (Arc, Goodfire) than the people making claims with these models — paper
  authors, reviewers, and teams using Evo 2 scores to interpret patient variants or to design sequences, who need to
  know a score reflects biology and not letter statistics.
- *Is it Hyena-specific?* No. The substitution designs and statistics need only "score this sequence" — they apply
  to Evo 2, Evo2-1B, Nucleotide Transformer, DNABERT-2, Caduceus, HyenaDNA, and with a different genetic-code layer,
  protein models. Only the internals checks (per-block divergence, weight nulls) need model access, and any PyTorch
  model allows that. The operator decompositions in this repo *are* Hyena-specific.
- *What it would find:* the obvious first target is the Evo 2 paper's own silent-vs-missense figure — does it
  survive letter-type matching across all 36 species? — and the same question for every genomic model at once. That
  is a self-contained methods paper ("Do DNA language models know the genetic code? A matched-substitution audit").
- *Honest size:* a real gap (the MIT paper, the position paper and Heap et al. all say so), but a niche. The
  realistic form is an open-source library plus that paper; paid audit work only if teams deploying these models ask
  for sign-off. It is closer to a reusable component — the way HNSW is an algorithm many vector databases embed —
  than to a product: a matched-substitution generator plus a paired test is small enough to drop into anyone's
  evaluation pipeline.

**A venue.** The 2nd International Workshop on Trustworthy AI for Biomedical Discovery (IEEE BIBM 2026 workshops,
online, 1–4 Dec) lists foundation models, evaluation and reproducibility, and interpretability among its topics.
Submissions are due **27 September 2026**. What is ready for it today: the solid wiring findings, and three worked
cases where a control reversed a conclusion inside this project (the SE negative control in round 3b, the
letter-type split of P21, the ratio statistics). Round 5 would need to run on Colab within a day to be included.

*Lesson from the review: a pre-registered prediction protects against moving the goalposts, not against measuring
the wrong thing. P21 was registered, run and met — and measured letter type. Pre-registration needs a companion
question: "what else would produce this number?"*

---

## 2026-09-30: Round 5 results — the letter-type worry mostly did not hold up

Round 5 ran on 2026-09-25 (80 GB A100, commit `1f2b29d`, about 1.5 hours after its predictions were committed) and
is recorded here on 2026-09-30. Smoke checks 8/8, 83 tests, the same six load-bearing layers and the block-30
bottleneck for the fourth run in a row. 150 sites per design from the whole E. coli genome. The formal record is in
`PREDICTIONS.md` under "Round 5 outcomes"; raw files are in `results/round5/`.

**The short version.** The review of round 4 worried that "Evo 2 knows amino acids" was really "Evo 2 is surprised
by rare kinds of letter swaps" (transversions). Round 5 held the swap type fixed, and then set it against the
hypothesis. The amino-acid effect stayed: the review's worry was mostly wrong. Two things weaken that good news: the
test only exists for two amino acids, and the regression we registered turned out unable to answer the question.
Separately, a bug was found that undoes round 4's "block 0 is learned" result until it is rerun.

### Finding 25: with letter type controlled, a protein change still disturbs Evo 2 more

Every comparison changes **the same letter** of DNA two ways and asks which change disturbs the model more (a more
negative score for the next 200 letters).

| design | change A | change B | B disturbs more | what it means |
|---|---|---|---|---|
| round 4's sites, rerun | silent, common swap | missense, rare swap | **69.2%** | round 4's 68.9% reproduces |
| four-fold sites | silent, common swap | silent, rare swap | **54.7%** (p = 0.29) | swap type alone barely matters when the protein does not change |
| outside genes | common swap | rare swap | 60.7% (p = 0.01) | a small swap-type effect |
| `matched` | silent, rare swap | missense, rare swap | **72.0%** (p = 7e-8) | protein change wins with swap type equal |
| `flipped` | silent, rare swap | missense, common swap | **80.0%** (p = 6e-14) | protein change wins with swap type pushing the other way |

Read across the rows. If swap type were the explanation, the four-fold row would be near 69% and `matched` near 50%.
The opposite happened. P23 is **refuted**, and by its own registered clause, letter type is not a live explanation
for round 4's result.

Three limits, all real:
- **Only isoleucine and arginine.** These designs need a silent change and a missense change of the same swap type at
  the same letter, and the genetic code only allows that for these two amino acids. So the result is about two amino
  acids, not twenty.
- **Isoleucine's missense change always makes ATG**, which also means "start a protein here". Isoleucine sites still
  win (65–69%), but less strongly than arginine (93–100%), and in one `flipped` subgroup (ATA → ATC vs ATG, 23 sites)
  missense wins only 43%.
- **The regression we registered could not decide anything.** P25 also asked for a pooled regression whose
  "missense" coefficient had to be negative. It came out **positive**, so P25 is formally "in between". The design was
  the problem. Inside each design, "is it missense" and "is it a rare swap" never vary, so the regression just
  draws a straight line through four averages and extrapolates to a case no design contains. Its other coefficient
  claims rare swaps are *less* disruptive, which the direct rows above contradict. Refit on the direction of each
  difference only, the missense coefficient turns negative (−0.17, interval −0.36 to +0.02). The mistake was ours,
  made when we wrote the prediction down: we registered a statistic before checking that the designs could feed it.

### Finding 26: the SE layers light up when the protein changes, and not otherwise

Round 4 said the missense/silent difference shows up most in the short SE layers (blocks 7, 11, 14). The review
said that came from a statistic that favours small denominators. Round 5 ranked layers by plain *difference* and
kept every layer's numbers.

| what changed | share of sites whose biggest gap is in an SE layer |
|---|---|
| the amino acid (round 4's sites) | **85%** |
| the amino acid, swap type fixed (`matched`) | 86% |
| nothing about the protein (four-fold) | **11%** |
| no gene at all | 31% |
| (SE layers are 9 of 32, i.e. 28% by chance) | |

Same model, same statistic: SE layers carry the difference when the protein changes and not when it doesn't. That
is the strongest localisation evidence so far. The review's prediction (P26: the share would drop below 60%) failed.
Its last check, that a fully scrambled model should *not* also favour SE (it gave 62%), failed too. But that
scrambled model is close to dead: its layers barely react to a letter change (about 0.01, against 0.1–0.8 when
trained). So that check had nothing to measure.

### Finding 27: a premature stop disturbs Evo 2 6–9× more than an ordinary protein change

A change that makes a "stop" codon (cutting the protein short) beat an ordinary missense change at the same letter at
**93%** of 150 sites, and at **92%** when swap type argued against the stop. Median effect −21.9 vs −2.5 nats. P27
**confirmed**. The behaviour itself is in the Evo 2 paper. New here is that it survives a same-letter, swap-type
controlled comparison.

### Finding 28: "block 0 is learned" was measured with a bug, and is untested again

P28 compared the first layer's word detectors with copies whose weights were scrambled. Trained scored 0.98 and every
scrambled copy scored exactly **0.000**. That looked like a clean win, but 0 is impossible: every 9-letter input
contains some 3-letter word, so the lowest possible score is 1/64.

The cause is a software bug. Before listing all inputs, the code measures how many letters back block 0 can see.
It did that on the *scrambled* weights, decided the answer was 1–2 letters, and listed inputs 1–2 letters long, too
short to contain any 3-letter word. Think of a unit test that silently ran on an empty fixture and passed. The check
uses a fixed tolerance of 0.001; the scrambled block's output either barely moves or is small. Round 4's "46
detectors vs 0" (finding 18) went through the same code. **Finding 18 is withdrawn until rerun**: not false,
untested.

Fixed in code: `controls.best_word_share` now refuses inputs shorter than the word it counts. Round 5b
(`notebooks/marv_hyena_round5b_colab.ipynb`, predictions P29 and a gated P28 rerun registered first) pins the input
length at 9, checks with `nullmodel.block0_sensitivity` whether each scrambled block responds to its input at all,
and scrambles one weight tensor at a time to find which one breaks it.

### What changes in the picture

- **Upgraded:** "Evo 2 reacts to amino-acid changes beyond letter statistics" — for Ile and Arg, with letter type
  controlled both ways. "SE layers are where it shows up" — supported against the no-protein-change control.
- **Confirmed:** stops ≫ missense.
- **Withdrawn pending 5b:** block 0's word bank is learned.
- **Lesson:** two of this round's problems were in our own tools, not the model. One was a registered statistic that
  could not be identified from the designs. The other was a helper that silently re-measured a setting on the wrong
  weights. Both passed every test, because the tests passed the setting in explicitly and the notebook did not.
  *Before registering a statistic, run it on fake data where the answer is known; before trusting a null, check it
  still responds to its input.*

## 2026-09-30: Prior art for the funnel, and where round 5 leaves the project

### A paper that measured the magnitude growth first

`arXiv:2510.27629` (Wei et al., v4 Nov 2025) is a biosecurity evaluation paper about how durable Evo 2's
pre-training data filters are. Its subject is not interpretability, and nothing in its main results bears on this
project. One part does: to explain a result of their own, they measured **Evo 2's per-layer representation
magnitude** and reported the same explosion we call the funnel.

| what they report | where | our matching number |
|---|---|---|
| Representation magnitude grows drastically **beyond layer 28** | Fig. 4b, App. C | writes 28 → 29 → 30: 10^4 → 10^6 → 10^12 (round 2) |
| Two causes: **no normalisation of the residual stream** between blocks, and Hyena's **input-dependent gated** convolution multiplying two already-large tensors | App. C | consistent; we did not state a mechanism |
| The final RMSNorm rescales before the logits, so output quality depends on **direction, not magnitude** | App. C | consistent with our float32 check |
| Reproduced through the **Vortex** pipeline, so it is the architecture and not the inference path | App. C | we also use Vortex |
| Layer-wise probe accuracy **collapses after layer 28** | Fig. 4a | independent corroboration: downstream-readable information dies exactly where our funnel starts |

**What this costs us.** The novelty claim "nobody has reported Evo 2's late-layer magnitude explosion" is **wrong and
is withdrawn**. It was reported, with a mechanism, in a paper we had not found. Our literature checks looked under
interpretability and massive-activation keywords; this paper is filed under `cs.CR`, and the magnitude analysis is an
appendix supporting an unrelated result. *Lesson: a measurement can have prior art in a field that has no reason to
cite yours. Search by the phenomenon ("representation magnitude", "Evo 2 layer norm") as well as by your own framing.*

**What survives, and is sharper for the overlap.** They measured *that* magnitudes grow. We measured what it does:
- Block 30's write is not merely large, it is **the entire residual**: share 1.0000 of the final stream in every
  region tested, so earlier writes are rounded away in bf16.
- Block 31 is **inert**: removing it gives bit-identical logits.
- The same in **float32**, and restoring every earlier write moves the output by ~0.0001%, so this is not a
  precision artifact. Their appendix argues the model still works because direction survives normalisation; our
  float32 check tests that the *decision* is block 30's alone, which is a different claim.
- The consequence for method: **gradient attribution dies here** (one step out of 512 carries 97-100% of the
  change), and removal-based attribution has to replace it. Their Fig. 4a shows the same wall from the outside, as a
  probe collapse, without naming it a problem for attribution.
- Everything operator-level — the load-bearing map, copying through attention, MR and the reading frame, SE and the
  genetic code, the paired-substitution controls — has no counterpart in their work. They probe layers; they do not
  ablate operators.

So the honest framing changes from "we found the funnel" to **"the magnitude explosion is known; we show one block's
write is the whole residual, that this is in the weights, and what it breaks."** README and the notebook page are
updated to cite them.

### Where round 5 leaves the project, in one place

| claim | status after round 5 | strength |
|---|---|---|
| Attention does exact copying at every distance; LI helps at long range | holds, 3 runs | solid |
| Block 30's write is the whole residual; block 31 inert; in the weights | holds, with prior art for the magnitude itself | solid, now shared |
| Six load-bearing layers, one per Hyena family, none in attention | holds, 4 runs | solid |
| MR carries the reading frame | holds, one fair test | measured once |
| Gradients fail at the funnel; removal-based attribution reaches block 0 | holds | measured once |
| A protein-changing substitution disturbs the model more than a silent one, with substitution type controlled and reversed | **new in round 5**: 72% and 80% of 150 sites | two amino acids only |
| Substitution type alone, with no protein change, explains little | **new**: 54.7%, p = 0.29 | solid for four-fold sites |
| The difference appears in SE blocks when the amino acid changes (85%) and not when it does not (11%) | **new**: round 4's claim survives a fair ranking | one weak leg (shuffled model is near-dead) |
| A premature stop beats a missense change at the same site | **new**: 93% / 92% of 150 sites | behaviour known from the Evo 2 paper; the controlled version is ours |
| Block 0's word bank is learned | **withdrawn**: the null was enumerated over 1-2-letter inputs | untested, round 5b pending |
| Far context travels through a particular operator | withdrawn (round 3b) | untestable as designed |

## 2026-09-30: Round 5b — the first layer's word detectors are not learned

Round 5b ran the one test round 5 could not score. It took about 20 minutes on an 80 GB A100: 89 tests, smoke checks
8/8, block 0 restored after every scramble. Raw files in `results/round5b/`. Two outcomes, one of them a refutation
of our own earlier finding.

### Finding 29: the bug is confirmed, and it was not what we thought

The diagnosis held. On all six scrambled copies of block 0, the code that asks "how many letters back can this layer
see?" answered **1 letter**, so round 5 enumerated 1-letter inputs and no 3-letter word could appear. With the input
length pinned to the trained answer of 9, every null scores 0.90–0.94 instead of 0.000.

But the *reason* is not the one written in the round-5 entry. That entry said the scrambled block had stopped
responding to its input — a broken block rather than an untrained one. **That was wrong.** The scrambled block
responds almost exactly as the trained one does:

| | median movement per changed letter | letters it responds to | outputs unmoved |
|---|---|---|---|
| trained | 0.47 | offsets 0–8, zero from 9 | 1.1% |
| shuffled (3 seeds) | 0.41–0.42 | offsets 0–8, zero from 9 | 0.7% |

What differs is **size, not sensitivity**. The trained cascade's output has median magnitude 3.7e-4; the nulls' is
5.6e-9, about 65,000× smaller. The receptive-field check compares two outputs with an *absolute* tolerance of 1e-3.
Against numbers near 5e-9, everything looks equal, so the search stops at one letter. Scrambling one weight tensor at
a time puts the magnitude collapse in `projections.weight` alone; every other tensor leaves the measured reach at 9.

*So the bug was a fixed absolute tolerance meeting a quantity whose scale is not fixed. The round-5 entry diagnosed
the symptom correctly and the cause wrongly, and nothing in the run contradicted it, because both stories predict the
same "1 letter" output. The thing that separated them was a measurement nobody had made yet.*

### Finding 30: a scrambled first layer is as word-selective as the trained one (P28 refuted)

With the enumeration fixed, the comparison round 4 and round 5 both meant to run finally ran. For each live channel:
the largest share of its 50 strongest inputs that contain any single 3-letter word. 1.0 means every top input shares
one word; about 0.3 would mean no word stands out.

| | live channels | median share | channels ≥ 0.6 | channels ≥ 0.8 |
|---|---|---|---|---|
| trained | 3,372 | 0.980 | 3,269 | 2,735 |
| weight-shuffled, 3 seeds | 4,096 | 0.940 | 3,938–3,958 | 3,072–3,148 |
| Gaussian, 3 seeds | 4,096 | 0.900–0.920 | 3,918–3,934 | 2,871–2,972 |

The trained model is ahead by **0.04** on the median, against the 0.20 the prediction asked for and the 0.05 below
which it is refuted. And at every cutoff from 0.5 to 0.9, **every null has more selective channels than the trained
model**. Both refutation conditions fire.

**Round 4's finding 18 is refuted, not just withdrawn.** "A scrambled block 0 has zero word detectors against 46 when
trained" was entirely the bug. The honest statement is the one the prediction pre-committed to: *training sharpens a
selectivity the architecture already has.* A 9-letter gated convolution with random weights of the right scale already
gives most of its channels a dominant 3-letter word. That is what the architecture does, not what Evo 2 learned.

This is the outcome [Heap et al.](https://arxiv.org/abs/2501.17727) predicted in general: interpretability results on
randomly initialised networks often score about as well as on trained ones. We cited that paper as the reason the null
model was not optional, ran the null, and got an answer so clean it hid a bug for two rounds.

**What survives about block 0.** Round 3's description is untouched, because none of it depended on the null: all 64
three-letter words have detector channels (median 46), start and stop codons are not special, channels respond to
letter combinations rather than single positions, and 724 of 4,096 channels are dead. The dead channels are now the
one block-0 property that clearly separates trained from untrained — 724 in the trained model, **0 in every null**.
Training in this layer looks less like building detectors and more like switching some off and setting the scale of
the rest.

### What this costs elsewhere

**P26's shuffled leg is reopened.** Round 5 dismissed one leg of the SE-localisation check — a fully scrambled Evo 2
still put 62% of peaks in SE blocks — by calling that model near-dead, citing small per-block divergences. That is the
same argument that just failed here. Those divergences are relative rather than absolute, so this result does not
refute the dismissal, but it is no longer trustworthy. Before that leg is accepted or dismissed, the per-block
equivalent of `nullmodel.block0_sensitivity` has to be run on the fully scrambled model.

*Lesson, and it is the third of its kind in two rounds: when a null model produces a suspiciously clean number,
measure the null itself before believing the comparison. "Scrambled" is not a state we observed; it is a state we
assumed. Round 5b's whole contribution is one function that looks at it.*

## 2026-09-30: An outside review of the claim list, and three checks that strengthen the biology

Two review notes arrived on the claim list written for the Leuven abstract. Both are worth recording: between them they
caught two factual errors in how the claims were being stated, named prior work for three of them, and asked three
questions about the amino-acid result that turned out to be answerable from the existing result files. No new GPU run.

### Two errors in the claim list (not in the repo)

- **"Six load-bearing layers, one per Hyena family" is wrong: it is two per family.** L0 and L4 are SE, L1 and L29 are
  MR, L9 and L30 are LI. The repo's own wording ("every Hyena family contains a load-bearing layer, attention contains
  none") was always correct and is unchanged; the "one per family" slip was introduced in the summary written for the
  abstract and in version 8 of the notebook page, and is corrected there.
- **The health numbers were mixed across rounds.** "Baseline 0.696, broken 0.225–0.345, all others 0.70–0.86" takes the
  broken range from round 3 and the healthy range from round 2, whose baseline was 0.859. Round 3's own numbers:
  baseline **0.696**, the six broken layers **0.225–0.345**, the other 26 **0.568–0.696**.
- A third point in the review — that the LI reach claim misquotes the architecture paper as saying "thousands of
  nucleotides" — applies to the summary, not the repo. The paper says LI aggregates over the entire sequence; the
  "thousands of nucleotides" phrasing is a popular explainer's, and this log has always cited it as such.

### Prior work named for three claims

- **Copying through attention** has a direct precedent in trained text hybrids: ablating attention in RecurrentGemma
  and Jamba drops retrieval to zero, with no compensation from the recurrent layers. Our claim is an extension to DNA
  and to Hyena, which the log already said in general terms; now there is a specific result to cite. Two further
  papers (on where pretrained hybrids put the aggregation step, and on short convolutions carrying associative recall)
  were named through another paper's related-work section, so they need reading before citation.
- **LI filters decaying in finite time** is expected from Laughing Hyena Distillery. Our measurement is a confirmation
  in Evo 2, not a surprise, and should be presented that way.
- **"Load-bearing layers" is taken, in this exact subfield.** An ICML 2026 paper (Cho, Kim and Kim) studies NT-v2 and
  **Evo 2 7B**, finds a "load-bearing" layer by leave-one-layer-out ablation, and reports it deep in the hybrid. The
  method differs from ours — they ablate layers of a *classifier built on hidden states* for ClinVar variants, not the
  model's own forward pass, and they never touch operators — but the term collides and the conclusion overlaps. Either
  rename our concept or distinguish it explicitly. The specific layer numbers attributed to that paper came from a
  third-party summary, so they need checking in the paper itself, along with whether its layer numbering is 0- or
  1-based. A tempting follow-up — that our funnel explains their deep layer, since their feature is a norm and late
  norms explode — should **not** be claimed: if they standardise features per layer, magnitude washes out.

### Three checks on the amino-acid result, all answered in our favour

The review asked whether P25's 72% and 80% could be (1) carried entirely by arginine, with isoleucine at chance, (2) a
codon-frequency effect, or (3) a comparison between different populations of sites. Full numbers are in
`PREDICTIONS.md` under "Round 5 re-analysis". In short:

**1. The effect is graded by how drastic the substitution is, and isoleucine is not at chance.**

| | sites | protein-changing arm wins | p |
|---|---|---|---|
| matched, Ile→Met (a mild swap) | 124 | 66.9% | 2.0e-4 |
| matched, Arg→Gly / Arg→Trp (drastic) | 26 | 96.2% | 8.1e-7 |
| flipped, Ile | 57 | 63.2% | 0.063 |
| flipped, Arg | 93 | 90.3% | 2.2e-16 |

Round 4's worrying 8-of-17 at isoleucine was 17 sites; with 124 it is 66.9% and clearly above chance. The severity
ordering — mild swap two-thirds, drastic swap almost always — is what a protein-aware model predicts and what letter
statistics do not. It also explains the oddity the review flagged, that `flipped` (80%) beat `matched` (72%) even
though letter type works against `flipped`: `flipped` is 62% arginine sites and `matched` only 17%. Within each amino
acid, `matched` is the stronger design, as it should be.

**2. Codon frequency works against the result almost everywhere.** In **266 of 300** paired sites the *silent* arm is
the one landing on the rarer codon, so rarity pushes against our hypothesis for free. Restricted to those sites the
result is **75.9%, p = 7.7e-18**. The one subgroup where rarity helps is `flipped` isoleucine ATA → ATT, 34 sites — and
it is the subgroup the review predicted from the genetic code alone, without seeing the data. Dropping it strengthens
the result rather than weakening it. `controls.rarity_split` now ships this split, with a test.

**3. The letter-type baseline is not a different population.** `fourfold` samples 9 amino acids; restricted to the two
the matched designs use, it gives 55.6% (n = 27) against 54.7% for all 150. Same answer.

Also verified: P26's chance level of 28.1% is correct, because the peak is taken over 32 blocks rather than 64 writes
(the stored `kinds` vector has 32 entries), and the weight-shuffled null was run on 60 sites — the leg round 5b
reopened.

### What the reviews leave as the strongest material

Both notes independently landed on the same two things, which is worth recording because it is not what we would have
guessed at the start:

1. **SE layers carry the amino-acid signal** (85% of sites when the amino acid changes, 11% when it does not). Neither
   review found prior work localising amino-acid sensitivity to operator types in any model. This is also the result
   most exposed by round 5b, since its weight-shuffled leg is reopened.
2. **LI's long-range contribution to copying** is a genuine difference from the text hybrids, where the recurrent
   layers contributed nothing to retrieval. We had been treating it as a detail of claim 1; it is the part that is not
   an extension of someone else's result.

Taken together the operator map is the contribution, more than any single row of it: SE reads local codon meaning and
does no recall, which contradicts the architecture paper's "local multi-token recall"; MR carries the reading frame;
attention does the copying with LI assisting at long range, unlike text hybrids; block 30 is the readout.

*Neither review was systematic — about eight targeted searches each. They caught the obvious collisions, which is
exactly the failure mode this project keeps hitting, and they cost nothing compared with the runs they protect.*

## 2026-10-01: A second review, and the correction it forced on the headline claim

A second outside read of the log arrived. It accepted the three checks from 30 September and raised eight further
points. **Seven are valid**, one is right on substance but wrong about where a phrase came from, and one new check it
proposed turned out to matter more than anything else in the list. Full numbers in `PREDICTIONS.md` under "second
pass".

### The check that changed a number

The claim was: the missense-vs-silent difference peaks in SE blocks at 85% of sites when the amino acid changes and
11% when it does not. The review's objection: nothing real separates the two arms of the four-fold design, so its
differences are small and its peaks may land wherever *noise* peaks. The 85-vs-11 contrast could then be signal
against noise rather than meaning against no meaning.

Matching the designs on how big the per-block difference is, before asking where it peaks:

| restricted to peak difference ≥ 0.313 (matched's median) | sites | peak is in an SE block |
|---|---|---|
| amino acid changes (`matched`) | 75 | **89.3%** (CI 0.80–0.94) |
| no gene present (`noncoding`) | 51 | **51.0%** (CI 0.38–0.64) |
| no amino-acid change (`fourfold`) | 6 | too few to report |
| chance (9 of 32 blocks) | — | 28.1% |

z = 4.81, p = 1.5e-6. So **the honest claim is 89% against 51%, not 85% against 11%.** Part of the original contrast
was size: large differences of any kind favour SE blocks somewhat (51% against a 28% chance level). What survives is
that an amino-acid change is far more SE-localised than a size-matched difference with no gene. That is a weaker
headline and a much harder one to attack.

Two details worth keeping. Within each design the preference does *not* depend on size — `matched` holds 82–90%
across its own range, `fourfold` holds 11–16% across its — so neither is internally magnitude-driven. And the
four-fold design cannot enter the size-matched comparison at all: only 6 of 150 sites reach the threshold, which is
itself the finding, because silent-vs-silent differences are small.

### A statistic that was never defined where we used it

Checking the above surfaced something separate. `peak_blocks(..., "diff")` takes argmax(div_b − div_a). For the stop
designs that quantity is **≤ 0 at 111 of 150 sites** (median exactly 0), because the stop arm diverges more than the
missense arm at *every* block. The argmax then returns the least-negative block, which is not a peak at all. The
round-5 notebook prints a line about "where stop and missense separate most"; that line is **not interpretable** and
is withdrawn. The stop designs need the sign flipped, or a signed statistic.

*This is the fourth statistic in this project that computed something other than what its name said. The pattern is
consistent: each one was a reasonable formula applied to a case its author had not pictured. A statistic needs a test
on data whose answer is known — not only its code.*

### The seven other valid points

- **"SE does no recall at all" is unsupported.** The copying gaps are 100, 1,000 and 10,000 letters, all far beyond
  SE's 7-letter reach, so the test cannot speak to *local* recall. The claim that the map "contradicts the
  architecture paper in both directions" goes with it. What survives: SE plays no part in recall at the distances
  tested.
- **"A real divergence from text hybrids" overstates what the precedent did.** The text-hybrid study removed
  attention; it did not remove the recurrent layers. So it cannot be cited as showing they contribute nothing. The
  defensible form: *LI is needed for long-range copying, which the text-hybrid studies did not test.* We have also
  only read that work through search summaries, not in the original.
- **The "severity gradient" is confounded with codon position.** The isoleucine pairs change the **third** codon
  letter and the arginine pairs the **first** (confirmed in the stored rows: `I pos2`, `R pos0`). So "mild versus
  drastic" and "third versus first position" cannot be separated here. The defensible form: *much larger at arginine
  sites (96.2% vs 66.9%)*, with no claim about why. Flipped isoleucine (63.2%, p = 0.063) should not be leaned on.
- **"Part of the published transversion effect may be protein change" is not safe.** That variant set includes
  non-coding regions, where no protein changes. The defensible form: *within matched coding sites, letter type alone
  has at most a modest effect (54.7%, CI 0.47–0.62).*
- **"SE carries the signal" should be "peaks in SE"** until SE writes are ablated or patched on these pairs.
  `peak_blocks` is descriptive, not causal.
- **The controls-toolkit target was garbled.** It conflated two papers: the 36 species belong to the Evo 2 paper's
  silent-versus-missense result, not to the transversion study.
- **Smaller ones, all correct.** The notebook page said "after three rounds" when it is now five plus 5b. The
  round-2 table said "any of the other 26" when five layers broke there, so it is 27 (round 3's six is a different
  run). "Recovered by a different method" overstates it, since the load-bearing map and the block-30 attribution both
  use single-component mean-ablation and differ only in readout. And block 31's inertness was shown in **bf16**: the
  float32 check covers block 30's magnitude and the contribution of earlier writes (`precision` holds `norm_fp32` and
  `logit_change_from_earlier_writes_fp32`), not block 31.

### The one point that is right for the wrong reason

The review said that "training sharpens a selectivity the architecture already has" contradicts our own registered
criterion. It does not: that exact sentence is **the wording P28 pre-committed to on refutation**, written before the
run. But the substance is right — with a gap of 0.04 and every null ahead at every cutoff, "sharpens" implies
trained > null, which the data do not support. "**Matches**" is the accurate word, and the log and page now use it.
The registered sentence stays in `PREDICTIONS.md` as written, with this note beside it, because predictions are not
edited after the fact.

The review's related observation is a real asymmetry worth recording: the trained median is computed over **live**
channels only (3,372 of 4,096), while the nulls have no dead channels to exclude (4,096 of 4,096). Dead channels have
tie-ordered top-k lists, so including them would most likely *lower* the trained median — which would make the
refutation stronger, not weaker. That direction is reasoning, not measurement, and a rerun could settle it.

## 2026-10-01: Mathur & Sachidanandam, read in full — the closest paper, and not a collision

[Benchmarking DNA Foundation Models: Biological Blind Spots in Evo2 Variant-Effect Prediction](https://doi.org/10.64898/2026.03.10.710786)
(bioRxiv, posted 11 March 2026, not peer reviewed; Horace Mann School and New York Medical College). Read all 22
pages. This is the paper the second review cited for the transversion result, and it is the closest existing work to
round 5's biology. It does **not** pre-empt anything here, and it changes three things.

### What it is

A **black-box behavioural benchmark**. Everything runs through Evo 2's hosted API (`/generate`, `/forward`), scoring
variants by mean log-likelihood. No layers, no ablations, no operators, no internals of any kind — their Conclusion
asks for exactly that as future work (see below). Mitochondrial DNA is the test bed: compact, intron-free, and
exhaustively annotated. Their findings:

| test | result |
|---|---|
| Codon usage bias in wobble predictions | **Not internalised.** Preferred codon chosen at 24.4% of 966 positions, against 25% for a coin flip; mean JSD 0.254 |
| Mitochondrial start/stop codon reassignment | Not respected: all 26 valid mito start-preserving variants called pathogenic |
| Zero-shot mitochondrial pathogenicity | AUROC 0.896, balanced accuracy 87.6%, but APOGEE2 (supervised) beats it on AUROC, specificity and auPRC |
| Transversions vs transitions | Transversions score ~1.9× more negative (mean ΔL −0.0113 vs −0.0060) |
| tRNA cyclic permutation (a null manipulation) | Sensitivity collapses **65.8% → 5.1%** with the tRNA sequence unchanged |
| Gene completion across 10 species | 86.2% mean, but it does not track mutational constraint; the most constrained complex completes worst |
| NUMTs | Evo 2 prefers the mitochondrial allele at divergence sites: it treats the pseudogene as authentic mtDNA |
| PhyloP conservation | ρ = 0.77 overall, but conservation peaks do not line up with likelihood peaks |

### 1. Their transversion result is the one ours corrects, and it is weaker than it reads

The second review was right that their variant set spans non-coding regions — their own Figure 3 stratifies into
D-loop, RNA genes, synonymous and missense. Two further things the numbers show:

- The comparison rests on **46 transversions against 727 transitions**. Transversions are 6% of the set.
- It is **unmatched and unpaired**: different positions, different regions, benign and pathogenic pooled, compared as
  two group means rather than at the same site.

So their effect could be carried by which regions their 46 transversions happen to fall in. Our `fourfold` design
asks the same question with 150 **paired** comparisons at the same position, with the protein held constant, and
finds **54.7% (p = 0.29)**. That is a direct, citable refinement: *within matched coding sites, substitution type
alone has at most a modest effect.* It does not contradict their measurement; it shows what is left of it once
position and coding consequence are held fixed. Our round-4 numbers are the cautionary half of the same story —
unmatched, our own version of this comparison gave 68.9% and looked like biology.

### 2. Their codon-usage finding is convergent support for our rarity control, and sharpens our claim

Round 5's biggest worry was that codon rarity drove the amino-acid result, since a rare codon is surprising in
itself. The re-analysis found rarity pushing *against* the result at 266 of 300 sites. Their paper explains why that
confound is so weak: **Evo 2 does not appear to track codon usage frequencies at all** — its wobble predictions are
near-random against empirical codon frequencies.

Put together, the two results are a dissociation worth stating, and only someone holding both can state it:

> Evo 2 responds to **what a codon means** (a substitution changing the amino acid disturbs it more than a silent one
> at the same position, with substitution type controlled) while **not tracking which synonymous codon is preferred**
> (wobble predictions near-random against codon-usage tables).

Those are compatible — knowing the amino acid is not knowing codon frequencies — and together they are more
interesting than either alone.

### 3. They name our work as the extension of theirs

From their Conclusion: *"Mechanistic interpretability of Evo2 embeddings may clarify which biological features are
encoded and which are absent."* That is a second published gap statement pointing at this project, alongside the
bio-foundation-model review already cited. Their framing is **what is missing**; ours is **where things happen**.
Complementary, and the pairing is a natural way to motivate a paper.

### One strategic warning

Their paper is a benchmarking framework built on **null manipulations** — the tRNA cyclic permutation is exactly the
kind of negative control this log keeps arguing for. So the "controls toolkit for claims about DNA models" idea in
*Where this could go* is **no longer an empty niche**: someone is publishing in it, from the behavioural side. The
differentiator has to be the internals — operator-level ablation, per-block divergence, weight nulls — not the
controls framing by itself.

### No overlap at all

Nothing in their paper touches operators, layers, ablation, the write-magnitude funnel, copying, attention, the
load-bearing map, or block 0. Different question, different method, same model.

## 2026-10-02: Where this stands, after the 20B and 40B runs

Four days of back-and-forth with a third party running H100s, a verdict reversed, a second verdict
reversed back, and two bugs in our own tools. This entry says plainly what happened, what we now
believe, and what is still open — because the thread is hard to follow from the commit log alone.

### The one-paragraph version

Someone with 8×H100s ran our scaling notebooks on Evo 2's **20B** and **40B**. The 20B results are
good and confirm the funnel at a second scale. The 40B at first looked broken; it is not. What it
revealed instead is that **the 26 blocks sitting after the funnel have weights about a thousand times
smaller than the blocks before it** — and not small as in untrained, small as in *driven far below where
they started*. The best explanation is that the funnel starves them: their contribution is rounded away,
so they never affect the model's error, so they never get corrected, and ordinary weight decay shrinks
them to nothing. A one-cell check on the 7B would confirm it.

### What happened, in order

**1. The 20B replicated the funnel, and more.** One block's write is 1.07e13× the next largest, taking
100.0000% of the final residual — stronger than the 7B's 1.2e5×. Attention is still required for
copying at every gap (0.26 against a chance level of 0.25, with the model still healthy at 0.82).
Load-bearing layers in every Hyena family, **none in attention**. The amino-acid effect is stronger
than at 7B (82% against 72%). So the project's central findings are not a 7B quirk.

**2. We called the 40B invalid. We were wrong.** Its numbers were bit-identical to the 20B's wherever
both models have the block, including the embedding — which cannot happen with different weights. We
inferred a mislabelled checkpoint file. The runner inferred something else: that the two checkpoints
*share* their first 24 blocks. They were right. The 40B file is byte-complete at 82.3 GB with all 50
blocks, and the 20B and 40B have the same width with roughly double the depth — the signature of
depth-growth training. Of their 261 shared tensors, 191 are bit-identical.

*The lesson is not subtle: our smoke checks verify that a measurement is an exact decomposition of
whatever model is loaded. They say nothing about whether it is the model you asked for. Nothing in the
pipeline looked at weight values until we wrote something that did.*

**3. The consequence for the paper, which is real.** The 20B and 40B are **not independent
checkpoints**. "Replicates at 20B and 40B" is one replication, not two, and every shared-block number is
one measurement reported twice.

**4. Then the weights explained the dead blocks.** The runner re-ran a fixed checkpoint reader. In the
40B, blocks 24–49 carry weights around 7.7e-06 across MLP, output projection, normalisation and
attention, while blocks 0–23 sit at 1.2e-02. The number that matters: standard initialisation for an
8192-wide layer gives about **8.8e-03**. The trunk is at 1.4× that, where trained weights normally are.
The second half is at **0.00088×** — roughly **1,140× below where it started**. Weights do not fall
three orders of magnitude by being ignored; they fall that far by being driven down.

### What we now believe, and how confident to be

**The funnel starves what comes after it.** A block whose residual write is rounded away in 16-bit
arithmetic contributes nothing to the loss, so it receives no gradient through the output, and weight
decay then drives its parameters toward zero.

Three checkpoints agree, including the detail that is hardest to explain any other way:

| model | funnel | immediately downstream | what we see |
|---|---|---|---|
| 7B | block 30's mixer | **block 30's own MLP**, then block 31 | MLP writes ~1e−15; block 31 writes a **constant** on three unrelated stretches of DNA |
| 20B | block 23's mixer, the last block | **block 23's own MLP** | its `mlp.l1/l2/l3` are 7.76e−06 against a 1.23e−02 median |
| 40B | block 23's mixer, of 50 | 26 blocks | ~7.7e−06 throughout |

The MLP runs after the mixer *inside the same block*, so the funnel block's own MLP is the first thing
downstream — and it is suppressed in both the 7B and the 20B, neither of which was depth-extended. The
20B is the accidental control: its funnel sits at the very last block, so it has almost nothing
downstream, and almost nothing dead.

**Confidence: good, not settled.** Two things could still undo it. The checkpoint reader drops
Transformer Engine `_extra_state` blobs, and if part of a model were stored FP8-scaled with the scales
in those blobs, the comparison would not be like-for-like — though TE stores activation scaling rather
than weight quantisation, and 7.7e-06 is below FP8's range anyway. And the direction of causation is
assumed rather than shown: quiet blocks could in principle come first, with the funnel as the
consequence.

### What is open, and the cheap test

`notebooks/marv_hyena_weight_check_7b.ipynb` is one cell on a **free Colab CPU runtime**. The 7B has
five attention blocks (3, 10, 17, 24, 31) and only block 31 sits after the funnel, so four healthy
siblings make the comparison unambiguous — unlike the 40B, where the affected blocks are the majority.
The prediction is registered: block 31's attention weights and block 30's MLP weights are 100–1000×
below their siblings, refuted if block 31 is within 10×.

### Two bugs of ours the runner found

Both were real, and both came from code dry-run only against a tiny test model that has no Transformer
Engine state:

1. `torch.load(weights_only=True)` refuses real Evo 2 checkpoints, because of those `_extra_state`
   blobs. Fixed by allowlisting `io.BytesIO`, which keeps the protection that `weights_only=False`
   would throw away.
2. Our outlier rule compared each block against the **median** for its tensor. When more than half the
   blocks are affected — as in the 40B — the median sits inside the affected group and the rule
   declares the majority normal. The runner spotted it from the numbers. Fixed to compare against the
   **maximum**, the one order statistic that survives majority contamination.

*A tiny stand-in model catches logic errors. It cannot catch a file-format difference that only exists
in real checkpoints, and it cannot catch a statistic that fails only when most of the data is affected.*

## 2026-10-02: A correct critique of the funnel's mechanism, and an error of mine

A reviewer challenged the explanation of *why* the funnel forms — both the one we recorded and the one
we credited to Wei et al. They are right, and the correction matters because it changes what can be
claimed about cause.

### The argument, and why it holds

> The mixer only ever sees `pre_norm(u)`, and RMSNorm gives the same output if you multiply its input
> by any constant. So a huge residual cannot by itself make the next block's write bigger. Every
> pre-norm transformer, Llama included, leaves the residual unnormalized between blocks, and none grow
> to 10¹².

Checked against the block formula in `arch.py` and the transcription in `tests/tiny_hyena.py`:

```
z    = projections(pre_norm(u));  z = filter(z);  z_in = out_filter_dense(z) + u
out  = mlp(post_norm(z_in)) + z_in
RMSNorm(x) = scale * x / (||x|| * H**-0.5 + eps)
```

For large `||x||` the `eps` is negligible, so `RMSNorm(c·x) = RMSNorm(x)` exactly. **Both paths into a
block — mixer and MLP — are scale-invariant in the residual.** There is no bypass: the raw residual
reaches the next block only through the `+ u` skip, never through the mixer.

So **"no residual normalisation, therefore magnitudes compound" is wrong**, and that is the mechanism
Wei et al. state and this log repeated. A block's write magnitude is set by its own weights acting on a
normalised input, not inherited from upstream. What does compound across blocks is *direction*: block
30 reads a normalised vector whose direction is dominated by whatever wrote largest before it.

**This log's statement of the mechanism is withdrawn.** The magnitude *observation* stands, in their
paper and in ours; the explanation does not.

### Their proposed test, and why it is weaker than it looks

They suggest rescaling a real block-30 input to unit size, running blocks 28–30 alone, and checking
whether block 30 still writes ~1e11. Worth running, but the outcome is **determined analytically** by
the scale invariance above: `pre_norm` undoes the rescaling exactly, so the write will be unchanged to
numerical precision. It is a useful check that no bypass path exists, not a measurement of the gain.

The question the critique actually opens is better served by measuring gain directly: feed unit-norm
inputs into each block *in isolation* and record the output/input norm ratio, per block. If block 30's
ratio is ~1e11 while its neighbours are O(1), the funnel is entirely a property of block 30's own
weights, and "why that block" becomes the open question. Registered as a round-7 item.

### An error of mine the critique exposes indirectly

Reading the 7B weight table, this log noted that block 30's own weights are at or *below* the per-tensor
maximum (`projections` 7× below, `filter.D` 18×, `out_filter_dense` 46×) and treated that as a puzzle:
how can a block with unremarkable weights write 1e11?

**It is not a puzzle, because `|mean|` does not measure gain.** A matrix whose entries have small mean
absolute value can still have a large operator norm — especially if it is low-rank or if its rows align
with the input it receives. Mean absolute weight is the right statistic for the question it was built
for ("is this tensor suppressed toward zero?", where 1e-6 against 1e-2 is unambiguous) and the wrong
statistic for "does this block amplify?". The two questions need different measurements, and conflating
them was careless.

### What is unaffected

**The starvation result does not depend on any of this.** It says that everything downstream of the
dominant write has weights 10³–10⁴ below its siblings, with the boundary inside the funnel block. That
is a statement about weights, and it holds whatever makes the write large. The reviewer agrees it is the
original part:

> The general mechanism family is published (Wortsman et al., *Curse of Depth*, and the gradient-sink
> papers), but the within-block boundary in released weights appears to be yours.

**Three papers to read before claiming that**, and none has been read here — they go into
`RELATED_WORK.md` as `SECONDHAND`, which this project's own rule says may not be cited yet. Given that
two of this week's corrections came from unread sources, that rule earns its keep again.

*Lesson: a mechanism that explains an observation is not evidence for the mechanism. "No residual
normalisation" sounded sufficient, was repeated from a published paper, and survives about ten seconds
of contact with the actual block formula — which was sitting in our own `arch.py` docstring the whole
time.*

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
- **Transition / transversion**: a one-letter change within a chemical family (A↔G, C↔T) / across families. Transitions
  are the common kind in real genomes.
- **Purine / pyrimidine**: the two families of DNA letters: A and G (larger) / C and T (smaller).
- **Silent (synonymous) / missense / nonsense**: a change that keeps the amino acid / swaps it / turns the codon into
  "stop".
- **Wobble position**: the third letter of a codon, which can often change without changing the amino acid.
- **Gene-aware vs protein-aware**: knowing where codons and their wobble positions are, vs knowing what amino acid a
  codon encodes.
- **Negative control**: a condition that cannot produce the effect; if it does, the test is measuring something else.
- **Null model**: the same analysis on a model with its learned structure destroyed (e.g. shuffled weights).
- **Confound**: a second difference hiding behind the one you meant to test (letter type behind silent-vs-missense).
- **Induction head**: an attention head that finds an earlier copy of the current text and predicts what came next.
