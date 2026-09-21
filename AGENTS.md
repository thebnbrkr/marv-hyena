# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## What this is

MARV-style interpretability for **StripedHyena** DNA models (Evo 2, run
through the Vortex inference library). MARV and LARQL index transformer FFN
neurons. marv-hyena adds what neither can do: decompose, ablate and read
the **Hyena convolution operators** (SE / MR / LI) alongside attention.
`README.md` is the user-facing map. `PREDICTIONS.md` is the experiment
record.

## The model facts everything depends on

Read from Vortex (`vtx` 1.0.8, commit `8b00afe`) `vortex/model/model.py` and
`engine.py`. If Vortex changes, re-check these first.

- `model.blocks[i]` is an `AttentionBlock` or a `ParallelGatedConvBlock`.
  The kind comes from the config lists `hcs_layer_idxs` (se),
  `hcm_layer_idxs` (mr), `hcl_layer_idxs` (li), `attn_layer_idxs` (attn).
- Each block adds exactly two residual writes: the **mixer** (output of
  `out_filter_dense` or `inner_mha_cls`) and the **mlp** (output of `mlp`).
  final = embed + Σ writes; logits = unembed(norm(final)).
- Hyena operator: `y = x2 * (conv(k, x1*v) + D*x1*v)`, after a length-3
  featurizer FIR, `interleave`, and a split into x2 / x1 / v.
- **Lag conventions** (`filters.lag_kernel`): FIR filters under 128 taps go
  through `F.conv1d` (cross-correlation), so lag = len−1−tap. FIR filters of
  128 taps or more go through an FFT convolution, so lag = tap. LI uses
  `compute_filter`, with lag = t. Vortex chooses by **length**, not operator
  name.
- MLP: `l3(act(l1 x) * l2 x)`. The activation is GELU at block 0 and the
  **identity** after it, so from block 1 on, neurons are bilinear.
- Tokens are ASCII bytes. Logits at position j predict letter j+1.

## Invariants: do not break these

1. **Every exact decomposition ships its own check**: `Decomposition.actual`
   read from the real logits, `DistanceResult.rel_error` against the real
   module output, and `Writes.reconstruction_error()`. Never report numbers
   from a run whose check failed. Never remove a check to make something
   pass.
2. **`tests/tiny_hyena.py` transcribes Vortex's math. It does not
   re-derive it.** The analysis code is written independently and must
   reproduce it. When Vortex changes, update the tiny model from the Vortex
   source first, then fix the analysis.
3. **Capture positions, not whole sequences.** A full 32-block capture of
   100k letters is about 50 GB. Hooks slice to the requested positions,
   and `mean_writes` reduces inside the hook.
4. **Ablate with means, not zeros, by default.** Zeroing whole operator
   families pushes the residual off-distribution.
5. **Direct ≠ total.** `trace`, `distance` and `sae.decompose_feature` give
   direct effects. `patch_sweep` gives total effects. Label which one any
   reported number is.
6. **A probe shorter than an operator's reach cannot test it** (LARQL
   `AGENTS.md`). Long-range claims need gaps far beyond MR's 128 letters.
7. Keep Vortex's Triton kernels (`use_kernels`) off during analysis.
8. Predictions are pre-registered in `PREDICTIONS.md`. Append outcomes;
   never edit a prediction after its test has run.

## Build / test

```bash
python -m pytest -q                   # tiny CPU model, no network, no GPU
python scripts/smoke_test.py          # on the GPU box, real evo2_7b: run before any experiment
```

New behaviour that can be checked on the tiny model gets a test there,
preferably an exactness test (decomposition vs real output).

## Not in scope (say so if asked)

- Attention-weight-based distance decomposition. Flash attention hides the
  weights. It would need a manual attention recompute that includes
  Vortex's rotary embedding and column split.
- Training or fine-tuning (use Savanna or BioNeMo).
- Inserting capabilities into the model, in particular anything touching
  viral or pathogen design. This repo is for understanding the model and
  measuring removals.
