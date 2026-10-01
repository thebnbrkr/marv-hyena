# Running the Evo 2 20B / 40B scaling tests

**For whoever has the H100s.** You don't need to know anything about this project. Run a notebook, send
back what the last cell prints. That's it.

---

## The short version

| | |
|---|---|
| **What** | Two notebooks: `marv_hyena_scaling_20b_h100.ipynb` and `marv_hyena_scaling_40b_h100.ipynb` |
| **Hardware** | 20B: 1–2 H100 (80 GB). 40B: 2 minimum, 4–8 better. **Hopper required** — these checkpoints need FP8 |
| **Not Colab** | Colab has no multi-H100 node. Any Jupyter on a Hopper box, or `jupyter nbconvert --execute` |
| **Time** | 20B ≈ 2–4 h, 40B ≈ 4–8 h at defaults, plus the weight download (~40 GB / ~80 GB) |
| **Send back** | the `REPORT BACK` text the last cell prints, and `results_evo2_*.json` |
| **Safe?** | Read-only. It loads a published checkpoint, measures it, and writes only into its own output folder |

Start with **20B**. If it works, 40B is the same notebook with a bigger model.

---

## Why these runs are worth your GPU hours

This project takes Evo 2 apart to find which of its four layer types does which job. Everything we know
comes from the **7B** model, because that's all an A100 can load. Four patterns showed up:

1. **One late block's write is ~10⁵× larger than any other**, so in 16-bit arithmetic every other
   block rounds away and the prediction is effectively a function of that one block.
2. **The final block does nothing** — removing it leaves the output bit-identical.
3. **Attention is required for exact copying** at every distance tested, while the "long" convolution
   layers only help at long range.
4. **Every convolutional family contains a layer that breaks the model on its own; attention contains
   none.**

Nobody knows whether any of that is a property of the architecture or an accident of that one
checkpoint. 20B and 40B are the only way to find out, and they need hardware we don't have.

**Block indices will differ** — these models have different depths and attention layouts. Nothing is
hardcoded; the notebook reads the layout from the checkpoint and reports the pattern.

---

## Step by step

1. **Open the notebook** on a machine with the GPUs. Check the top cell's settings:
   - `QUICK = True` — leave it. First pass is a shakedown at reduced size.
   - `RUN_BIOLOGY = False` — leave it. That stage is the slowest.
   - `SEQ_LEN = 4096` — lower it if you hit out-of-memory.
   - Set `HF_HOME` to a disk with room if your default cache is small.
2. **Run the cells in order.**
3. **Stage 0 is a gate.** It checks that Transformer Engine works, that a Hopper GPU is present, and
   that the measurement code's own self-checks pass. If it stops there, send back what it printed and
   don't run further — everything after would be meaningless.
4. **Watch Stage 0's device table.** On a sharded model, blocks land on different GPUs. We believe the
   code handles that, but if anything later says `Expected all tensors to be on the same device`, that
   table plus the traceback is exactly what we need.
5. **If the quick pass completes**, set `QUICK = False` and run again. Each stage checkpoints, so
   re-running skips what's already done. Optionally set `RUN_BIOLOGY = True` for the last stage.
6. **Run the final cell** and send back its `REPORT BACK` block plus the JSON file it names.

A reduced run that finishes is far more useful than a full run that doesn't. If time is short, send the
`QUICK = True` results.

---

## What each stage does, in plain terms

| Stage | What it does | Cost |
|---|---|---|
| 0 | Checks the environment and that the measurements add up against the real model | minutes |
| 1 | Reads the layer layout from the checkpoint | seconds |
| 2 | Measures how big each layer's contribution is, then switches off the last layer to see if it matters | minutes |
| 3 | Switches off one layer at a time and checks whether the model can still read DNA | the bulk of the time |
| 4 | Inserts a repeated stretch of DNA and tests whether the model can copy it with attention switched off | significant |
| 5 | Measures the three-letter rhythm inside genes, with each layer family switched off | significant |
| 6 | Optional: makes two single-letter changes at the same spot — one that changes a protein, one that doesn't | slowest |

---

## Troubleshooting

| Symptom | What it means / what to do |
|---|---|
| `transformer_engine: NOT USABLE` | FP8 is unavailable, so the checkpoint can't load. Install Transformer Engine for your CUDA version, restart the kernel, re-run. |
| `No Hopper GPU found` | These checkpoints need H100 or newer. Nothing to do. |
| `SELF-CHECKS FAILED` | Likely a library-version difference on our side. Send Stage 0's full output; skip the rest. |
| `Expected all tensors to be on the same device` | A sharding bug on our side. Send the traceback **and** Stage 0's device table. |
| Out of memory | Lower `SEQ_LEN` (try 2048, then 1024) and re-run. |
| A `*** WARNING: no condition for [...]` line | Expected to be absent on these models; if it appears, that's a real finding — just include it. |
| Download is enormous / slow | Set `HF_HOME` to a big disk. 40B weights are ~80 GB. |
| Something else | Send the traceback. Don't try to fix it — a failure is a result. |

---

## What we'd conclude from your numbers

So you know why each line matters:

- **A single dominant late block, at any depth** → the funnel is architectural, not a 7B quirk. This is
  the result we most want.
- **No dominant block** → the funnel is specific to 7B, which is just as interesting and would change
  how we read everything else.
- **Attention ablated and copying drops to ~0.25** → the copying result holds at scale.
- **Load-bearing layers in every convolutional family and none in attention** → that pattern is
  architectural too.
- **Self-checks fail** → we have a bug to fix before any of this means anything, and finding that out
  costs you an hour instead of costing us a wrong paper.

---

## Provenance, so you can judge the risk

- Code: <https://github.com/thebnbrkr/marv-hyena> — the notebook clones it at run time.
- These two notebooks are **generated** by `scripts/gen_scaling_notebooks.py`, so they stay in step.
- **They have never run on real H100 hardware.** Their logic was dry-run end to end against a tiny
  CPU stand-in model, which catches Python and logic errors but not FP8, sharding or memory problems.
  Stage 0 exists because of that: it's designed to fail early and legibly rather than halfway through
  an expensive run.
- Predictions are registered as P36–P40 in `PREDICTIONS.md` **before** these run, so the thresholds
  can't move afterwards.

Thank you — this is the one experiment the project genuinely can't do on its own hardware.
