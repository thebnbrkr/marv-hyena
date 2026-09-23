# marv-hyena

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv-hyena/blob/main/notebooks/marv_hyena_colab.ipynb)

**MARV for StripedHyena DNA models.** Open up Evo 2 and find out which part of
the model does which job, and why it gave a specific answer.

[MARV](../marv) turns a transformer's FFN weights into an inspectable index,
edits the live model and measures what the edit broke. [LARQL](../larql-main)
is the large Rust system MARV borrowed the vindex idea from. Neither can see
the convolution layers that make hybrid models different. LARQL's own LFM2
adapter says of the conv mixer: *"The conv layers keep refusing."*
marv-hyena covers that gap for Evo 2, a DNA model that is
mostly convolutions.

## What Evo 2 looks like inside

Evo 2 7B has 32 blocks. Each block runs one "mixer" (the operator that
moves information between positions) followed by an MLP, and both add their
output to the residual stream:

| type | blocks (7B) | how far it can look | how it looks |
|---|---|---|---|
| **SE** Hyena short explicit | 0 4 7 11 14 18 21 25 28 | 7 letters | fixed gated convolution |
| **MR** Hyena medium regularized | 1 5 8 12 15 19 22 26 29 | 128 letters | fixed gated convolution |
| **LI** Hyena long implicit | 2 6 9 13 16 20 23 27 30 | whole sequence, fading | sum of decaying exponentials |
| **attn** multi-head attention | 3 10 17 24 31 | whole sequence, exact | content-based lookup |

The Evo 2 architecture paper ([arXiv 2503.01868](https://arxiv.org/abs/2503.01868))
claims these operators specialize. It contains no interpretability analysis
that checks the claim. This repo is built to check it.

## What it does

| module | question it answers | kind of answer |
|---|---|---|
| `trace` | Which blocks pushed this prediction, and by how much? | **exact** split of the logit difference over every block's write (direct effects) |
| `distance` | Did the Hyena block use the letter right there, or DNA 30,000 letters back? | **exact** split of a Hyena block's write into lag bands (0, 1–8, 9–127, … 16k+) |
| `intervene` | What breaks if an operator type is switched off? What carries a mutation's effect? | mean-ablation, activation patching (total effects) |
| `filters` | How far does each Hyena block look, from the weights alone? | reach table; the CPU mode reads the checkpoint with no GPU |
| `motifs` | What does every first-layer channel detect? | **exact** motif dictionary: all 4⁹ possible inputs enumerated |
| `sae` | Which operator types build this SAE feature? Edit features in a live forward pass | Goodfire layer-26 SAE: decomposition and error-preserving edits |
| `vindex` | MARV's neuron index for Evo 2's MLPs; checkpoint diff; label neurons with biology | GenBank-annotation enrichment (LARQL's feature labels, with biology as the answer key) |
| `probes`, `variants` | Test inputs with a known right answer | copy test, context truncation, codon phase, variant windows |
| `nullmodel` | Is this finding learned, or just the architecture? | weight-shuffled null with exact restoration |
| `genome` | Does the copying circuit fire on **real** repeats? | rRNA operons / IS elements vs. shuffled vs. random |
| `codons` | Does the model represent **amino acids**, or only letters? | silent vs. missense at the same codon site |

**Every exact decomposition checks itself.** `trace` compares its sum
against the model's real logits. `distance` compares its reconstruction
against the real module output captured in the same forward pass. If a
check fails, the numbers mean nothing. Typically that means the installed
Vortex computes something differently from what this code assumes.

## What we've found so far (Evo 2 7B)

Short version, as of round 4. The reasoning, numbers and mistakes are in [`RESEARCH_LOG.md`](RESEARCH_LOG.md), and
predictions and outcomes are in [`PREDICTIONS.md`](PREDICTIONS.md).

- **One block decides.** Block 30's output is ~10⁵× larger than any other, so the prediction is a function of block
  30 alone. It's built into the weights (the same in float32), and block 31 has no effect.
- **Attention does exact copying** at every distance. The LI layers help at long range (without them, copying at
  10,000 letters drops from 99.7% to 55.6%).
- **The reading frame (codon rhythm) lives in the MR (medium) layers**, not the SE (short) ones.
- **Mutations are judged early and locally.** Their effect leaves the mutated position within blocks 0–7.
- **"Long" LI filters are mostly short** (4–7 letters), with a few long-reaching channels.
- **Block 0 is a generic bank of 3-letter-word detectors, and it is LEARNED.** Start and stop codons aren't special,
  and 18% of its channels are dead. A weight-shuffled block 0 produces **zero** detector channels for all 64 words
  (3 seeds) against a median of 46 for the trained model, so the bank is not an artifact of the architecture.
- **SE applies the genetic code; MR carries the frame.** A missense change disturbs the model more than a silent one
  at the same codon position in 68.9% of 119 sites, and 90.8% of those sites diverge most in an SE block (7/11/14).
  MR's 128-letter window tracks *where* the frame is; SE's 7-letter window reads *what* the codon says.
- **The funnel defeats gradient attribution.** Integrated gradients at block 30's input attributes zero (a single
  step out of 512 carries ~99% of the function's range). Causal ablation instead reaches all the way back to
  block 0 — so block 30 reads from the whole network, and anything measuring it has to be causal.
- **Far context goes through attention.** 50,000 letters of upstream DNA are worth 0.016 nats/letter, and removing
  attention erases all of it on 5/5 genes while the model stays healthy.
- **Six load-bearing layers** (L0, L1, L4, L9, L29, L30); every other single layer is individually expendable.
- **Round 3 replicated bit-identically** on a different A100 SKU and CUDA version (round 3b).

**How this compares to what the architecture paper claims.** StripedHyena 2
([arXiv 2503.01868](https://arxiv.org/html/2503.01868v1)) asserts operator specialization in prose, sourced to prior
*synthetic* work, with no ablations, no per-layer magnitudes and no interpretability of its own:

| operator | the architecture paper claims | we measured | |
|---|---|---|---|
| **SE** | "local multi-token **recall**" | no part in recall; applies the **genetic code** | wrong |
| **MR** | "modeling across hundreds of tokens" | tracks the **reading frame** specifically | made specific |
| **LI** | "aggregate over the **entire sequence**" | channels mostly reach **4–7 letters**; no far context | wrong |
| **attn** | "recall across **longer** sequences" | essential at **every** distance, plus far context | understated |

Caveats we hold ourselves to:

- **Everything is one checkpoint** (`evo2_7b`). Nothing is checked on `evo2_7b_262k` or `evo2_7b_base`.
- The real-repeat test ran on **one family** (16S rRNA), which the model already predicts at 98.9% on first sight,
  so it had no headroom to show retrieval. Less-conserved IS elements are the real test.
- The **load-bearing map replicates a known pattern** ([ShortGPT](https://arxiv.org/html/2403.03853v3): early layers
  crucial, middle redundant, last important). What is new is that every Hyena family contains a load-bearing layer
  while attention contains none.

## Run it on Colab

- **Round 1** (the first full pass): `notebooks/marv_hyena_colab.ipynb`.
- **Round 2** (redesigned after round 1; see [`RESEARCH_LOG.md`](RESEARCH_LOG.md)):
  [`notebooks/marv_hyena_round2_colab.ipynb`](https://colab.research.google.com/github/thebnbrkr/marv-hyena/blob/main/notebooks/marv_hyena_round2_colab.ipynb)
- **Round 3** (measuring at the bottleneck): `notebooks/marv_hyena_round3_colab.ipynb`. Results in `results/round3/`,
  and `results/round3b/` for the rerun of the two steps that failed.
- **Round 4** (null model and biology):
  [`notebooks/marv_hyena_round4_colab.ipynb`](https://colab.research.google.com/github/thebnbrkr/marv-hyena/blob/main/notebooks/marv_hyena_round4_colab.ipynb).
  Needs no new data — E. coli and the GenBank annotations already in the repo.


Open [`notebooks/marv_hyena_colab.ipynb`](notebooks/marv_hyena_colab.ipynb) with the badge above. Choose
**Runtime → Change runtime type → A100 GPU** and turn on **High-RAM**, then run the cells top to bottom. The notebook
installs Evo 2 and marv-hyena and downloads its data (the E. coli genome and the BRCA1 variants from the evo2 repo).
It then runs the smoke checks and every experiment in `PREDICTIONS.md`, with plots, and saves `results.json`.
An L4 may work with shorter sequences. A T4 won't, because it has no bfloat16 support.

**No flash-attn needed.** On Colab, `pip install flash-attn` usually finds no prebuilt wheel and compiles for
hours. `HyenaModel.load` detects that flash-attn is missing and runs Evo 2's attention through PyTorch's built-in
fused kernel instead (`marv_hyena/noflash.py`). If flash-attn *is* installed, it is used.

## Setup (Linux + NVIDIA GPU, e.g. one A100)

```bash
# 1. Evo 2 itself. Light install: 7B models in bf16, no Transformer Engine; this is the A100 path
pip install evo2                  # on Python 3.13 use: pip install --ignore-requires-python evo2==0.6.0
pip install flash-attn==2.8.0.post2 --no-build-isolation   # optional; skipped automatically if absent
# 2. this repo
cd marv-hyena && pip install -e .
python -m pytest -q          # 60 tests on a tiny CPU model, runs anywhere
```

A100s have no FP8, so only the 7B checkpoints run (`evo2_7b`, `evo2_7b_262k`,
`evo2_7b_base`, `evo2_7b_microviridae`). The Goodfire SAE was trained on
`evo2_7b_262k`.

## Run order on the A100

```bash
G=../evo2-main/notebooks/sparse_autoencoder/NC_000913.gb    # annotated E. coli genome, in the evo2 repo

python scripts/smoke_test.py --genome $G                       # 1. MUST pass before anything else is trusted
python scripts/filter_reach.py --model evo2_7b                 # 2. how far each Hyena block looks (P4)
python scripts/run_copy_test.py --genome $G --gaps 100 1000 10000 50000   # 3. who copies (P1)
python scripts/block0_motifs.py                                # 4. the first layer's motif dictionary (P6)
python scripts/explain_variant.py --genome <fasta> --pos <0-based> --ref A --alt G   # 5. why this mutation (P5)
```

`smoke_test.py` checks the assumptions this code makes about Vortex: that
every residual write is captured, that the trace sums to the real logits,
that lag bands reconstruct one block of each Hyena kind, the block-0
receptive field, and that hooks leave no trace. It exits non-zero on any
failure.

## Python quickstart

```python
import marv_hyena as mh

hm = mh.HyenaModel.load("evo2_7b")
seq = mh.probes.load_sequence("NC_000913.gb")[100_000:108_192]

# Why did the model predict letter 5000 the way it did?
d = mh.decompose_prediction(hm, seq, 5000)
d.show()                       # per-block rows, plus totals by type: se / mr / li / attn / mlp / embed

# How far back was each Hyena block looking when it made that push?
direction = mh.normed_direction(hm, seq, 5000)
mh.show_profile(mh.distance_profile(hm, seq[:5000], 4999), direction=direction)

# What carries a mutation's effect? (patch ALT components into the REF run)
v = mh.make_variant(seq, 4000, seq[4000], "T")
mh.explain_variant(hm, v).show()

# Switch off every LI mixer and see what changes
ids = hm.ids(seq)
with mh.mean_ablate(hm, mh.mean_writes(hm, ids, mh.mixers_of(hm, "li"))):
    logits = hm.logits(ids)
```

## Layout

```
marv_hyena/
  arch.py        HyenaModel: block kinds from the config, the two residual writes per block, tokens = ASCII
  hooks.py       capture_outputs / replace_outputs (every intervention goes through these)
  trace.py       capture_writes, decompose_prediction, normed_direction   (direct logit attribution)
  distance.py    hyena_distance, distance_profile                         (exact lag-band split)
  filters.py     lag_kernel, reach_table, reach_table_from_checkpoint     (weights only)
  intervene.py   mean_ablate, zero_ablate, patch_sweep, kind_groups, span metrics
  probes.py      copy_probe, truncation_curve, genbank_track, codon_phase_accuracy
  variants.py    make_variant, delta_logp, explain_variant
  sae.py         Goodfire SAE loader, feature_acts, decompose_feature, edit_features
  vindex.py      MLP vindex, diff, neuron_acts, label_units
  motifs.py      receptive_field, enumerate_block0
  noflash.py     run Evo 2 without the flash-attn package (PyTorch SDPA instead)
  memory.py      chunked long-LI filter build: bit-identical, fits 50k-letter inputs on a 40 GB GPU
  interface.py   integrated-gradients attribution at a bottleneck block's input (round 3)
  diagnostics.py write_norms, find_bottlenecks, health (round 2)
  nullmodel.py   random_weights: weight-shuffled null, restored exactly (round 4)
  genome.py      real repeat families from GenBank; three-arm copy probes (round 4)
  codons.py      genetic code; wobble sites; silent vs. missense divergence (round 4)
  checks.py      run_smoke_checks (shared by scripts/smoke_test.py and the notebook)
  experiments.py copy_test, codon_test, context_test (the PREDICTIONS.md experiments)
notebooks/       marv_hyena_colab.ipynb: the whole pipeline on a Colab A100
scripts/         smoke_test, filter_reach, run_copy_test, block0_motifs, explain_variant
tests/           tiny_hyena.py (Vortex's module names + math, CPU, float32) + 60 tests
PREDICTIONS.md   pre-registered predictions; outcomes get appended, never edited
RESEARCH_LOG.md  what we ran, what happened, what went wrong (plain language)
```

## Status and limits

- **Run on real Evo 2 7B** (rounds 1–3b, Colab A100). Written against the
  Vortex source (`vtx` 1.0.8, commit `8b00afe`; runs verified on 1.1.0) and
  tested on a tiny model that copies Vortex's module names and math.
  `smoke_test.py` is the bridge to the real model and has passed 8/8 on every
  round. Round 3's full result set reproduced **bit-identically** on a
  different A100 SKU and CUDA version.
- **Only `evo2_7b` so far.** Nothing has been checked on `evo2_7b_262k` or
  `evo2_7b_base`, so every finding is one checkpoint deep.
- **`distance` does not decompose attention blocks.** Flash attention
  doesn't expose its weights. Use patching and context truncation for
  attention.
- `trace` and `sae.decompose_feature` measure **direct** effects.
  `patch_sweep` measures total effects. The two numbers answer different
  questions.
- Do not enable Vortex's Triton kernels (`use_kernels=True`) while
  analysing. The reconstruction checks assume the reference code path.

## Scope

This repo is for understanding the model and measuring removals: which
parts do what, why an answer came out, and what breaks when something is
switched off. Inserting new capabilities, especially anything related to
viral or pathogen design, is out of scope.
