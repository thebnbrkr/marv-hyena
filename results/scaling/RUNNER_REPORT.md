# marv-hyena: Evo 2 20B / 40B scaling runs and checkpoint check

Run on AWS p5 H100s, 2026-10-01 (scaling runs) and 2026-10-02 (checkpoint check).

## Summary

| Pattern | 20B (24 blocks, 1 GPU) | 40B (50 blocks, 8 GPUs) | 7B reference |
|---|---|---|---|
| 1. Funnel (one block dominates) | **Yes**: block 23 (li), the *last* block, 1.07e13x the next, share 1.000 | **Weak**: block 23 (li), mid-depth, only 2.1x the next, share 0.927 | ~1.2e5x, share 1.0 |
| 2. Final block contributes nothing | **No**: ablating it changes logits by 16.5 | **Yes**: blocks 49 (attn) and 48 (li) are bit-identical when ablated | yes |
| 3. Copying needs attention | Yes: -attn gives ~0.26 at gaps 100 / 1k / 10k | Yes: same | yes |
| 4. Load-bearing layers (all 5 sets) | [0, 1, 2, 21, 23]: 2 se, 1 mr, 2 li, **0 attn** | same set | 2 se, 2 mr, 2 li, 0 attn |
| 6. Biology (amino-acid change more disruptive) | 82%, p = 7.6e-7 | 85%, p = 3.1e-8 | 72% |

Self-checks (Stage 0 gate) passed on both models in both QUICK and full mode. No notebook cell errored.

**Checkpoint check (H1 vs H2): not H1.** The merged `evo2_40b.pt` is byte-complete (82,253,491,694 bytes = sum of the
two HF shards), holds all 50 blocks with no gaps, and no probed block is zero. H2 holds only partly: of the 261 tensors
shared by the two files, 191 are bit-identical and 70 differ (first: `blocks.0.post_norm.scale`).

**Observation not flagged by the script:** the 40B attention blocks 24 and 49 have `Wqkv` |mean| of about 7.66e-6, which is
100-10,000x smaller than every other probed tensor and nearly identical to each other. That looks near-initialisation. The
script only flags values below 1e-8, so it reported them as "trained-looking".

## Environment

- **Hardware:** 40B ran on a p5.48xlarge (8x H100 80 GB), sharded across all 8 GPUs. 20B ran on a p5.4xlarge (1x H100 80 GB).
- **Software:** container `nvcr.io/nvidia/pytorch:25.04-py3` (torch 2.7.0a0+79aa17489c.nv25.04, Transformer Engine 2.2.0+c55e425,
  driver 595.91.07). `evo2==0.6.0` and marv-hyena were cloned at run time by the notebook.
- **Execution:** notebooks ran unattended with `jupyter nbconvert --execute --allow-errors`, with `HF_HOME=/work/hf_cache`.

## Checkpoint check: method and deviations

- **Command as given:** the two paths in the command were joined without a space, so they were passed as two arguments:
  `python check_checkpoint.py /work/hf_cache/evo2_20b.pt /work/hf_cache/evo2_40b.pt`.
- **40B file:** the original merged 40B file was on the 8x H100 box, which had been terminated when its block ended. The two shards were
  re-downloaded and merged with a verbatim copy of the evo2 0.6.0 merge loop (`models.py` lines 231-277): byte concatenation
  of `.part0`, `.part1` into `dirname(HF_HUB_CACHE)/evo2_40b.pt`.
- **20B file:** `/work/hf_cache/evo2_20b.pt` is a symlink to the HF snapshot file, because the 20B is not sharded.
- **The script as shipped fails twice on these checkpoints. Its file was not edited; a wrapper applied two load-time shims:**
  1. `torch.load(weights_only=True)` rejects the `io.BytesIO` objects in the file. The shim allowlists `io.BytesIO` via
     `torch.serialization.safe_globals`, which keeps the code-execution protection.
  2. `describe()` calls `.is_floating_point()` on every entry, including Transformer Engine `*._extra_state` BytesIO blobs
     (FP8 scaling metadata, not weights). The shim drops non-tensor entries: 123 in 20B and 258 in 40B.
  Both failures are reproduced below so the script can be fixed upstream.

### Final output (with shims)

```
=== /work/hf_cache/evo2_20b.pt
file size: 47.9 GB   (Evo 2 bf16: 7B ~14 GB, 20B ~40 GB, 40B ~80 GB)
[shim] dropped 123 non-tensor entries, e.g. ['blocks.0.out_filter_dense._extra_state', 'blocks.0.projections._extra_state']
tensors: 261
block indices present: 0..23  (count 24)

 block  tensors     abs mean  zero frac  sample key
     0       10   2.8637e-02      0.000  blocks.0.filter.h
     6       12   5.3358e-02      0.000  blocks.6.filter.D
    12       11   1.7007e-01      0.000  blocks.12.filter.D
    18       10   2.4037e-01      0.000  blocks.18.filter.h
    23       12   6.0531e-03      0.000  blocks.23.filter.D

All probed blocks carry trained-looking weights (H2 side):
    the file is not obviously truncated.

=== /work/hf_cache/evo2_40b.pt
file size: 82.3 GB   (Evo 2 bf16: 7B ~14 GB, 20B ~40 GB, 40B ~80 GB)
[shim] dropped 258 non-tensor entries, e.g. ['blocks.0.out_filter_dense._extra_state', 'blocks.0.projections._extra_state']
tensors: 537
block indices present: 0..49  (count 50)

 block  tensors     abs mean  zero frac  sample key
     0       10   2.8637e-02      0.000  blocks.0.filter.h
    12       11   1.7007e-01      0.000  blocks.12.filter.D
    23       12   6.0531e-03      0.000  blocks.23.filter.D
    24        9   7.6603e-06      0.000  blocks.24.inner_mha_cls.Wqkv.weight
    25       10   3.9924e-04      0.000  blocks.25.filter.h
    37       11   1.3684e-03      0.000  blocks.37.filter.D
    49        9   7.6559e-06      0.000  blocks.49.inner_mha_cls.Wqkv.weight

All probed blocks carry trained-looking weights (H2 side):
    the file is not obviously truncated.

=== comparing the two files
keys: 261 vs 537, shared 261
shared tensors BIT-IDENTICAL: 191/261   differing: 70
first differing key: blocks.0.post_norm.scale

Send this whole output back.
CKPT3_DONE
```

### First run, unmodified (crash 1)

```
=== /work/hf_cache/evo2_20b.pt
file size: 47.9 GB   (Evo 2 bf16: 7B ~14 GB, 20B ~40 GB, 40B ~80 GB)
Traceback (most recent call last):
  File "/work/check_checkpoint.py", line 160, in <module>
    sys.exit(main())
             ^^^^^^
  File "/work/check_checkpoint.py", line 152, in main
    out = [describe(p) for p in args.paths]
           ^^^^^^^^^^^
  File "/work/check_checkpoint.py", line 64, in describe
    sd = load_state_dict(path)
         ^^^^^^^^^^^^^^^^^^^^^
  File "/work/check_checkpoint.py", line 43, in load_state_dict
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1496, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL _io.BytesIO was not an allowed global by default. Please use `torch.serialization.add_safe_globals([_io.BytesIO])` or the `torch.serialization.safe_globals([_io.BytesIO])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
CKPT_CHECK_DONE
```

### With io.BytesIO allowlisted only (crash 2)

```
=== /work/hf_cache/evo2_20b.pt
file size: 47.9 GB   (Evo 2 bf16: 7B ~14 GB, 20B ~40 GB, 40B ~80 GB)
tensors: 384
block indices present: 0..23  (count 24)

 block  tensors     abs mean  zero frac  sample key
Traceback (most recent call last):
  File "/work/run_check_safe.py", line 5, in <module>
    runpy.run_path("check_checkpoint.py", run_name="__main__")
  File "<frozen runpy>", line 286, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "check_checkpoint.py", line 160, in <module>
    sys.exit(main())
             ^^^^^^
  File "check_checkpoint.py", line 152, in main
    out = [describe(p) for p in args.paths]
           ^^^^^^^^^^^
  File "check_checkpoint.py", line 87, in describe
    keys = sorted(k for k in by_block[b] if sd[k].is_floating_point()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "check_checkpoint.py", line 87, in <genexpr>
    keys = sorted(k for k in by_block[b] if sd[k].is_floating_point()
                                            ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: '_io.BytesIO' object has no attribute 'is_floating_point'
CKPT2_DONE
```

## REPORT BACK: full runs (QUICK = False, biology on)

### evo2_20b

```
==============================================================================
REPORT BACK  --  marv-hyena scaling test, evo2_20b
==============================================================================
mode: full   biology stage: run
blocks: 24  hidden: 8192  attention at: [3, 10, 17]
devices: 1  sharded: False  torch 2.7.0a0+79aa17489c.nv25.04  TE 2.2.0+c55e425
self-checks passed: True

1. FUNNEL: dominant block 23 (li), 0 from the end
   largest/second = 1.07e+13   (7B: ~1.2e5)
   share of final residual = 1.000000   (7B: 1.000000)
2. ABLATE block 23 (li): max logit change 1.650e+01
2. ABLATE block 22 (mr): max logit change 4.688e+00
2. ABLATE block 23 (li): max logit change 1.650e+01

4. LOAD-BEARING (health < 0.5 on all 5 sets): [0, 1, 2, 21, 23]
   by operator type: {'se': 2, 'mr': 1, 'li': 2}   (7B: 2 se, 2 mr, 2 li, 0 attn)
   baseline health: [0.944, 0.947, 0.951, 0.866, 0.943]

3. COPYING (mean second-copy accuracy; chance = 0.25)
   -attn                            gap    100  0.260  health 0.820
   -attn                            gap   1000  0.265  health 0.820
   -attn                            gap  10000  0.263  health 0.820
   -li (keep L0,1,2,21,23)          gap    100  0.995  health 0.689
   -li (keep L0,1,2,21,23)          gap   1000  0.983  health 0.689
   -li (keep L0,1,2,21,23)          gap  10000  0.943  health 0.689
   -mr (keep L0,1,2,21,23)          gap    100  1.000  health 0.655
   -mr (keep L0,1,2,21,23)          gap   1000  1.000  health 0.655
   -mr (keep L0,1,2,21,23)          gap  10000  1.000  health 0.655
   -se (keep L0,1,2,21,23)          gap    100  0.999  health 0.639
   -se (keep L0,1,2,21,23)          gap   1000  0.992  health 0.639
   -se (keep L0,1,2,21,23)          gap  10000  0.995  health 0.639
   none                             gap    100  1.000  health 0.944
   none                             gap   1000  1.000  health 0.944
   none                             gap  10000  1.000  health 0.944

5. READING FRAME (descriptive; read with the health column)
   -mr (keep L0,1,2,21,23)          periodicity +0.1248  health 0.506
   -se (keep L0,1,2,21,23)          periodicity +0.1412  health 0.514
   -li (keep L0,1,2,21,23)          periodicity +0.1621  health 0.506
   -attn                            periodicity +0.1748  health 0.554
   none                             periodicity +0.1753  health 0.783

6. BIOLOGY
   matched        n= 60  b_more=0.817  p=7.56e-07  SE share of peaks=0.5833333333333334
   fourfold       n= 60  b_more=0.583  p=0.245  SE share of peaks=0.13333333333333333
   stop_matched   n= 60  b_more=0.183  p=7.56e-07  SE share of peaks=0.75
==============================================================================
ALSO SEND: ./marv_hyena_evo2_20b_full_out/results_evo2_20b.json  (5812 bytes)
==============================================================================
```

### evo2_40b

```
==============================================================================
REPORT BACK  --  marv-hyena scaling test, evo2_40b
==============================================================================
mode: full   biology stage: run
blocks: 50  hidden: 8192  attention at: [3, 10, 17, 24, 31, 35, 42, 49]
devices: 8  sharded: True  torch 2.7.0a0+79aa17489c.nv25.04  TE 2.2.0+c55e425
self-checks passed: True

1. FUNNEL: dominant block 23 (li), 26 from the end
   largest/second = 2.1   (7B: ~1.2e5)
   share of final residual = 0.926809   (7B: 1.000000)
2. ABLATE block 49 (attn): max logit change 0.000e+00  BIT-IDENTICAL
2. ABLATE block 48 (li): max logit change 0.000e+00  BIT-IDENTICAL
2. ABLATE block 23 (li): max logit change 1.869e+01

4. LOAD-BEARING (health < 0.5 on all 5 sets): [0, 1, 2, 21, 23]
   by operator type: {'se': 2, 'mr': 1, 'li': 2}   (7B: 2 se, 2 mr, 2 li, 0 attn)
   baseline health: [0.944, 0.947, 0.951, 0.865, 0.942]

3. COPYING (mean second-copy accuracy; chance = 0.25)
   -attn                            gap    100  0.259  health 0.820
   -attn                            gap   1000  0.263  health 0.820
   -attn                            gap  10000  0.263  health 0.820
   -li (keep L0,1,2,21,23)          gap    100  0.994  health 0.690
   -li (keep L0,1,2,21,23)          gap   1000  0.983  health 0.690
   -li (keep L0,1,2,21,23)          gap  10000  0.943  health 0.690
   -mr (keep L0,1,2,21,23)          gap    100  1.000  health 0.656
   -mr (keep L0,1,2,21,23)          gap   1000  1.000  health 0.656
   -mr (keep L0,1,2,21,23)          gap  10000  1.000  health 0.656
   -se (keep L0,1,2,21,23)          gap    100  0.999  health 0.639
   -se (keep L0,1,2,21,23)          gap   1000  0.992  health 0.639
   -se (keep L0,1,2,21,23)          gap  10000  0.995  health 0.639
   none                             gap    100  1.000  health 0.944
   none                             gap   1000  1.000  health 0.944
   none                             gap  10000  1.000  health 0.944

5. READING FRAME (descriptive; read with the health column)
   -mr (keep L0,1,2,21,23)          periodicity +0.1262  health 0.507
   -se (keep L0,1,2,21,23)          periodicity +0.1406  health 0.513
   -li (keep L0,1,2,21,23)          periodicity +0.1617  health 0.506
   none                             periodicity +0.1739  health 0.784
   -attn                            periodicity +0.1748  health 0.553

6. BIOLOGY
   matched        n= 60  b_more=0.850  p=3.09e-08  SE share of peaks=0.48333333333333334
   fourfold       n= 60  b_more=0.633  p=0.0519  SE share of peaks=0.13333333333333333
   stop_matched   n= 60  b_more=0.167  p=1.62e-07  SE share of peaks=0.6166666666666667
==============================================================================
ALSO SEND: ./marv_hyena_evo2_40b_full_out/results_evo2_40b.json  (6037 bytes)
==============================================================================
```

## REPORT BACK: QUICK runs (first pass)

### evo2_20b

```
==============================================================================
REPORT BACK  --  marv-hyena scaling test, evo2_20b
==============================================================================
mode: quick   biology stage: skipped
blocks: 24  hidden: 8192  attention at: [3, 10, 17]
devices: 1  sharded: False  torch 2.7.0a0+79aa17489c.nv25.04  TE 2.2.0+c55e425
self-checks passed: True

1. FUNNEL: dominant block 23 (li), 0 from the end
   largest/second = 6.07e+12   (7B: ~1.2e5)
   share of final residual = 1.000000   (7B: 1.000000)
2. ABLATE block 23 (li): max logit change 1.650e+01
2. ABLATE block 22 (mr): max logit change 4.688e+00
2. ABLATE block 23 (li): max logit change 1.650e+01

4. LOAD-BEARING (health < 0.5 on all 2 sets): [0, 1, 2, 21, 23]
   by operator type: {'se': 2, 'mr': 1, 'li': 2}   (7B: 2 se, 2 mr, 2 li, 0 attn)
   baseline health: [0.951, 0.866]

3. COPYING (mean second-copy accuracy; chance = 0.25)
   -attn                            gap    100  0.261  health 0.760
   -attn                            gap   1000  0.264  health 0.760
   -li (keep L0,1,2,21,23)          gap    100  0.981  health 0.643
   -li (keep L0,1,2,21,23)          gap   1000  0.953  health 0.643
   -mr (keep L0,1,2,21,23)          gap    100  1.000  health 0.603
   -mr (keep L0,1,2,21,23)          gap   1000  1.000  health 0.603
   -se (keep L0,1,2,21,23)          gap    100  0.992  health 0.575
   -se (keep L0,1,2,21,23)          gap   1000  0.994  health 0.575
   none                             gap    100  1.000  health 0.951
   none                             gap   1000  1.000  health 0.951

5. READING FRAME (descriptive; read with the health column)
   -mr (keep L0,1,2,21,23)          periodicity +0.0807  health 0.506
   -attn                            periodicity +0.1336  health 0.554
   -se (keep L0,1,2,21,23)          periodicity +0.1375  health 0.514
   -li (keep L0,1,2,21,23)          periodicity +0.1559  health 0.506
   none                             periodicity +0.2226  health 0.783
==============================================================================
ALSO SEND: ./marv_hyena_evo2_20b_out/results_evo2_20b.json  (3853 bytes)
==============================================================================
```

### evo2_40b

```
==============================================================================
REPORT BACK  --  marv-hyena scaling test, evo2_40b
==============================================================================
mode: quick   biology stage: skipped
blocks: 50  hidden: 8192  attention at: [3, 10, 17, 24, 31, 35, 42, 49]
devices: 8  sharded: True  torch 2.7.0a0+79aa17489c.nv25.04  TE 2.2.0+c55e425
self-checks passed: True

1. FUNNEL: dominant block 23 (li), 26 from the end
   largest/second = 2.83   (7B: ~1.2e5)
   share of final residual = 0.923310   (7B: 1.000000)
2. ABLATE block 49 (attn): max logit change 0.000e+00  BIT-IDENTICAL
2. ABLATE block 48 (li): max logit change 0.000e+00  BIT-IDENTICAL
2. ABLATE block 23 (li): max logit change 1.869e+01

4. LOAD-BEARING (health < 0.5 on all 2 sets): [0, 1, 2, 21, 23]
   by operator type: {'se': 2, 'mr': 1, 'li': 2}   (7B: 2 se, 2 mr, 2 li, 0 attn)
   baseline health: [0.951, 0.865]

3. COPYING (mean second-copy accuracy; chance = 0.25)
   -attn                            gap    100  0.261  health 0.760
   -attn                            gap   1000  0.264  health 0.760
   -li (keep L0,1,2,21,23)          gap    100  0.981  health 0.643
   -li (keep L0,1,2,21,23)          gap   1000  0.953  health 0.643
   -mr (keep L0,1,2,21,23)          gap    100  1.000  health 0.603
   -mr (keep L0,1,2,21,23)          gap   1000  1.000  health 0.603
   -se (keep L0,1,2,21,23)          gap    100  0.992  health 0.573
   -se (keep L0,1,2,21,23)          gap   1000  0.994  health 0.573
   none                             gap    100  1.000  health 0.951
   none                             gap   1000  1.000  health 0.951

5. READING FRAME (descriptive; read with the health column)
   -mr (keep L0,1,2,21,23)          periodicity +0.0810  health 0.507
   -attn                            periodicity +0.1306  health 0.553
   -se (keep L0,1,2,21,23)          periodicity +0.1418  health 0.513
   -li (keep L0,1,2,21,23)          periodicity +0.1521  health 0.506
   none                             periodicity +0.2235  health 0.784
==============================================================================
ALSO SEND: ./marv_hyena_evo2_40b_out/results_evo2_40b.json  (4120 bytes)
==============================================================================
```

## Files in this folder

| File | What it is |
|---|---|
| `marv_hyena_evo2_{20b,40b}_full_out/results_evo2_*.json` | Full-run JSON, the file the README asks to send |
| `marv_hyena_evo2_{20b,40b}_full_out/stage_*.json` | Per-stage checkpoints, full run |
| `marv_hyena_evo2_{20b,40b}_out/` | Same, QUICK run |
| `executed_{20b,40b}{,_full}.ipynb` | Executed notebooks with all cell outputs |
| `check_checkpoint_output*.txt` | Raw checkpoint-check logs |
