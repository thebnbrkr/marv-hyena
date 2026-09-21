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

**Every exact decomposition checks itself.** `trace` compares its sum
against the model's real logits. `distance` compares its reconstruction
against the real module output captured in the same forward pass. If a
check fails, the numbers mean nothing. Typically that means the installed
Vortex computes something differently from what this code assumes.

## Run it on Colab

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
pip install evo2
pip install flash-attn==2.8.0.post2 --no-build-isolation   # optional; skipped automatically if absent
# 2. this repo
cd marv-hyena && pip install -e .
python -m pytest -q          # 19 tests on a tiny CPU model, runs anywhere
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
  checks.py      run_smoke_checks (shared by scripts/smoke_test.py and the notebook)
  experiments.py copy_test, codon_test, context_test (the PREDICTIONS.md experiments)
notebooks/       marv_hyena_colab.ipynb: the whole pipeline on a Colab A100
scripts/         smoke_test, filter_reach, run_copy_test, block0_motifs, explain_variant
tests/           tiny_hyena.py (Vortex's module names + math, CPU, float32) + 19 tests
PREDICTIONS.md   pre-registered predictions; outcomes get appended, never edited
```

## Status and limits

- **Not yet run on real Evo 2.** It was written against the Vortex source
  (`vtx` 1.0.8, commit `8b00afe`) and tested on a tiny model that copies
  Vortex's module names and math. `smoke_test.py` is the bridge to the real
  model.
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
