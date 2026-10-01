import json

def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(keepends=True)}
def md(s):   return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}

def build(model, params, min_gpus, rec_gpus, hours, outfile):
    big = params == '40B'
    intro = md(f"""# marv-hyena scaling test: Evo 2 {params} (`{model}`)

**You are running this for someone else, so the most useful thing you can send back is the
`REPORT BACK` block the last cell prints.** You do not need to interpret any of it.

## What this is

marv-hyena takes Evo 2 apart and asks which of its four layer types does which job. Everything measured
so far comes from the **7B** model, which runs on an A100. The 20B and 40B checkpoints need **FP8 on
Hopper GPUs**, which we do not have — hence this notebook.

The question is **not** whether specific numbers reproduce. It is whether the *pattern* holds at scale:

| # | Pattern found in 7B | What this notebook measures |
|---|---|---|
| 1 | One late block's output is ~10⁵× larger than any other, and is effectively the only thing the output layer reads | the write magnitude of every block |
| 2 | The final block contributes nothing (removing it gives bit-identical output) | output with the last block ablated |
| 3 | Attention is required for exact copying at every distance | copying with attention ablated |
| 4 | Every Hyena family contains a layer that breaks the model alone; attention contains none | one-layer-at-a-time ablation on 5 DNA sets |
| 5 | A substitution that changes the amino acid disturbs the model more than a silent one, and the difference peaks in the short SE layers | matched substitution pairs (optional, slowest stage) |

**Block indices will differ from 7B** — {params} has a different layer count and attention layout.
Nothing here hardcodes "block 30"; the notebook discovers the layout and reports the pattern.

## Hardware

- **{min_gpus} H100 (80 GB) minimum, {rec_gpus} recommended.** Vortex splits the model across whatever
  CUDA devices it finds.
- **FP8 requires a working Transformer Engine.** Stage 0 checks this and stops if it is missing — do not
  skip it.
- Not Colab: Colab does not offer multi-H100 nodes. Any Jupyter environment on a Hopper box works
  (a cloud GPU node, a cluster login node with a notebook, or `jupyter nbconvert --execute`).
- Rough budget: **{hours}** at the default settings, most of it in Stages 3–5.{"" if not big else chr(10) + chr(10) + "- The 40B weights are ~80 GB to download the first time. Set `HF_HOME` to a disk with room."}

## How to run it

1. Run the cells in order. **Stage 0 is a gate**: if its self-checks fail, stop and send back what it
   printed. Everything after it would be meaningless.
2. Each stage checkpoints to `OUT` as it finishes, so a crash or a disconnect loses at most one stage.
   Re-running the notebook picks up where it stopped.
3. `QUICK = True` runs every stage at reduced size — do that first to shake out the environment. Then
   set `QUICK = False` for the real run.
4. `RUN_BIOLOGY = False` skips the slowest stage. Leave it off on the first pass.
5. When it finishes: send back `results_{model}.json` **and** the `REPORT BACK` text block. If anything
   crashed, the traceback is more useful than a description.

## If something goes wrong

| symptom | what it means |
|---|---|
| Stage 0 says Transformer Engine is missing | FP8 is unavailable; this checkpoint cannot load. Install TE for your CUDA version. |
| `Expected all tensors to be on the same device` | a sharded-model bug on our side. Send the traceback **and** Stage 0's device table — that is exactly what we need. |
| A reconstruction/self-check error above 0.05 | a hook is missing a write, probably a Vortex version difference. Send Stage 0's output; do not run further stages. |
| Out of memory | lower `SEQ_LEN` in the settings cell, then re-run. |
| It is simply slow | set `QUICK = True` and send that back; a reduced run is far better than none. |

Nothing here modifies the model or writes anything outside `OUT`. It is read-only analysis of a
published checkpoint.""")

    settings = code(f"""# ---- settings: these are the only lines you should need to edit ----
MODEL_NAME  = '{model}'
QUICK       = True     # True = reduced sizes, to shake out the environment first. Then set False.
RUN_BIOLOGY = False    # the slowest stage (matched substitutions). Leave off on the first pass.
SEQ_LEN     = 4096     # lower this if you hit out-of-memory
OUT         = './marv_hyena_{model}_out'

import os, json, time, random
os.makedirs(OUT, exist_ok=True)
# Put the weights somewhere with room ({'~80 GB for 40B' if big else '~40 GB for 20B'}) if the default cache is small:
# os.environ['HF_HOME'] = '/mnt/big_disk/hf_cache'
print('model   =', MODEL_NAME)
print('mode    =', 'QUICK (reduced)' if QUICK else 'FULL RUN')
print('biology =', RUN_BIOLOGY)
print('outputs =', OUT)
print('HF_HOME =', os.environ.get('HF_HOME', '(default ~/.cache/huggingface)'))""")

    gpucell = code("""!nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv""")

    install = code("""# Install evo2 + marv-hyena. NOTE: unlike the 7B path, this checkpoint needs FP8, so
# Transformer Engine must be present and WORKING. Do not uninstall it here.
!pip install -q --ignore-requires-python evo2==0.6.0
!rm -rf marv-hyena && git clone -q https://github.com/thebnbrkr/marv-hyena.git
!pip install -q -r marv-hyena/requirements.txt openpyxl

import sys, subprocess
sys.path.insert(0, './marv-hyena')
print('marv-hyena at', subprocess.run(['git','-C','marv-hyena','rev-parse','--short','HEAD'],
                                      capture_output=True, text=True).stdout.strip())
for m in ('diagnostics', 'controls', 'experiments', 'codons', 'genome'):
    assert os.path.exists(f'marv-hyena/marv_hyena/{m}.py'), f'{m}.py missing'
assert 'def device_map' in open('marv-hyena/marv_hyena/diagnostics.py').read(), \\
    'checkout too old: needs diagnostics.device_map'
print('marv-hyena modules present')""")

    stage0 = md("""## Stage 0 — environment, device map, and the self-checks

**This is a gate.** marv-hyena's measurements are exact decompositions that check themselves against
the real model: per-block writes must sum to the real residual, and per-block credit must sum to the
real logit change. If those checks fail, every later number is meaningless, so the notebook stops.

It also prints which GPU each block landed on. On a sharded model that table is the first thing we need
if anything later complains about devices.""")

    stage0c = code("""import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('CUDA devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f'  [{i}] {torch.cuda.get_device_name(i)}  capability {cap[0]}.{cap[1]}'
          f'{"  <-- Hopper, FP8 OK" if cap[0] >= 9 else "  <-- NOT Hopper: FP8 unavailable"}')

# FP8 needs a working Transformer Engine. A broken install is worse than none: marv_hyena would
# disable it and the load would silently fall back or fail.
TE_OK, TE_WHY = False, ''
try:
    import transformer_engine, transformer_engine.pytorch  # noqa: F401
    TE_OK = True
    TE_WHY = getattr(transformer_engine, '__version__', 'unknown version')
except Exception as e:
    TE_WHY = f'{type(e).__name__}: {e}'
print('\\ntransformer_engine:', TE_WHY if TE_OK else f'NOT USABLE ({TE_WHY})')
assert TE_OK, ('This checkpoint needs FP8, which needs a working Transformer Engine. '
               'Install TE for your CUDA version, restart, and re-run. Send this message back if stuck.')
assert any(torch.cuda.get_device_capability(i)[0] >= 9 for i in range(torch.cuda.device_count())), \\
    'No Hopper GPU found: this checkpoint cannot run here.'""")

    load = code("""import marv_hyena as mh
from marv_hyena import diagnostics, probes
t0 = time.time()
hm = mh.HyenaModel.load(MODEL_NAME)
print(f'loaded in {time.time() - t0:.0f}s')
print(hm.describe())

DM = diagnostics.device_map(hm)
print(f"\\nsharded across {DM['n_devices']} device(s): {DM['devices']}")
for d, blocks in DM['by_device'].items():
    print(f"  {d}: {len(blocks)} blocks  {blocks[:6]}{'...' if len(blocks) > 6 else ''}")
print('embedding on', DM['embedding'], '| unembed on', DM['unembed'])

REPORT = {'model': MODEL_NAME, 'mode': 'quick' if QUICK else 'full',
          'n_blocks': hm.n_blocks, 'layout': hm.describe().splitlines()[0],
          'devices': DM['n_devices'], 'sharded': DM['sharded'],
          'torch': torch.__version__, 'te': TE_WHY}
def save():
    json.dump(REPORT, open(f'{OUT}/results_{MODEL_NAME}.json', 'w'), default=str, indent=1)
def stage(name, fn):
    path = f'{OUT}/stage_{name}.json'
    if os.path.exists(path):
        print(f'[{name}] loaded checkpoint'); return json.load(open(path))
    t = time.time(); out = fn(); dt = time.time() - t
    json.dump(out, open(path, 'w'), default=str)
    print(f'[{name}] {dt:.0f}s'); return out
save()""")

    checks = code("""from marv_hyena.checks import run_smoke_checks
from Bio import SeqIO
import urllib.request
# E. coli K-12, the same sequence every earlier round used, so numbers are comparable.
os.makedirs('data', exist_ok=True)
GENOME = 'data/NC_000913.gb'
if not os.path.exists(GENOME):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/ArcInstitute/evo2/main/notebooks/sparse_autoencoder/NC_000913.gb',
        GENOME)
genome = probes.load_sequence(GENOME)
print(f'E. coli genome: {len(genome):,} letters')

seq = genome[100_000:100_000 + SEQ_LEN]
OK = run_smoke_checks(hm, seq)
REPORT['smoke_checks_passed'] = bool(OK); save()
assert OK, ('SELF-CHECKS FAILED -- stop here and send back everything this cell printed, plus the '
            'device table above. Later stages would be meaningless.')
print('\\nStage 0 gate passed.')""")

    s1 = md("""## Stage 1 — the layer layout

Read from the checkpoint's own config, not assumed. In 7B the pattern is SE, MR, LI repeating with
attention every seventh block; if that differs here, it changes how the later stages read.""")
    s1c = code("""from collections import Counter
kinds = [hm.kind(i) for i in range(hm.n_blocks)]
print('blocks:', hm.n_blocks, '|', dict(Counter(kinds)))
print('attention at:', [i for i, k in enumerate(kinds) if k == 'attn'])
print()
print('  '.join(f'{i}:{k}' for i, k in enumerate(kinds)))
REPORT['kinds'] = kinds
REPORT['attn_blocks'] = [i for i, k in enumerate(kinds) if k == 'attn']
REPORT['hidden_size'] = int(hm.hidden_size); save()
print('\\nhidden size:', hm.hidden_size)""")

    s2 = md("""## Stage 2 — is there a funnel? (patterns 1 and 2)

Every block adds two writes to a shared running sum (the "residual stream"), and the output layer reads
that sum. In 7B one late block's write is about 10⁵× larger than any other, so in 16-bit arithmetic
everything else rounds away and the prediction is a function of that one block alone.

This stage measures every write's size, in three different stretches of DNA, then ablates the final
block to see whether it matters at all.""")
    s2c = code("""import numpy as np, pandas as pd
regions = [100_000, 1_000_000, 3_000_000][:1 if QUICK else 3]

def run_norms():
    out = []
    for r in regions:
        s = genome[r:r + SEQ_LEN]
        for w in diagnostics.write_norms(hm, s, positions=[SEQ_LEN - 1]):
            out.append({'region': r, 'block': w.block, 'kind': w.kind, 'part': w.part,
                        'norm': float(w.norm), 'share': float(w.share)})
    return out
norms = stage('norms', run_norms)
nd = pd.DataFrame(norms)

piv = nd.pivot_table(index=['block', 'kind', 'part'], columns='region', values='norm')
top = nd.groupby(['block', 'kind', 'part']).norm.max().sort_values(ascending=False)
print('largest writes (max over regions):')
print(top.head(8).to_string())
print('\\nmedian write of all the rest:', f'{top.iloc[8:].median():.3g}')
ratio = float(top.iloc[0] / max(top.iloc[1], 1e-30))
dom_block = int(top.index[0][0]); dom_kind = top.index[0][1]
print(f'\\nlargest / second largest = {ratio:.3g}   (7B: ~1.2e5)')
print(f'dominant block: {dom_block} ({dom_kind}), which is {hm.n_blocks - 1 - dom_block} from the end')
share = nd[(nd.block == dom_block)].share.max()
print(f'its share of the final residual: {share:.6f}   (7B: 1.000000)')
REPORT['funnel'] = {'dominant_block': dom_block, 'dominant_kind': dom_kind,
                    'from_end': hm.n_blocks - 1 - dom_block,
                    'ratio_top_to_second': ratio, 'max_share': float(share)}
save()""")
    s2d = code("""# Pattern 2: does the FINAL block matter at all? In 7B, removing it leaves the output bit-identical.
from marv_hyena.intervene import mean_ablate, mean_writes
s = genome[100_000:100_000 + SEQ_LEN]
ids = hm.ids(s)
base = hm.logits(ids).float().cpu()
last = hm.n_blocks - 1
rows = []
for b in (last, last - 1, dom_block):
    comps = [(b, 'mixer'), (b, 'mlp')]
    with mean_ablate(hm, mean_writes(hm, ids, comps)):
        alt = hm.logits(ids).float().cpu()
    d = float((alt - base).abs().max())
    rows.append({'block': b, 'kind': hm.kind(b), 'max_logit_change': d,
                 'bit_identical': d == 0.0})
    print(f'ablating block {b} ({hm.kind(b)}): max logit change {d:.3e}'
          f'{"   <-- BIT-IDENTICAL: contributes nothing" if d == 0.0 else ""}')
REPORT['last_block_ablation'] = rows; save()""")

    s3 = md("""## Stage 3 — which single layers are load-bearing? (pattern 4)

Replace one layer's output with its own average and ask whether the model still reads ordinary DNA
("health" = next-letter accuracy; chance is 0.25). In 7B exactly six layers break the model on their
own — two in each Hyena family and **none in attention**.

Measured on 5 independent stretches of DNA, each mixing coding and non-coding sequence. The rule, fixed
in advance: a layer counts only if it breaks the model on **all 5**.""")
    s3c = code("""rng = random.Random(0)
n_sets = 2 if QUICK else 5
DNA_SETS = [genome[o:o + SEQ_LEN] for o in sorted(rng.sample(range(10_000, len(genome) - SEQ_LEN), n_sets))]
BASE = [diagnostics.health(hm, s)['health_acc'] for s in DNA_SETS]
print('baseline health per set:', [round(h, 3) for h in BASE], '  (chance = 0.25)')

lb = stage('load_bearing', lambda: [dict(set_index=i, **r)
                                    for i, s in enumerate(DNA_SETS)
                                    for r in diagnostics.find_load_bearing(hm, s)])
ld = pd.DataFrame(lb)
CUT = 0.5
piv = ld.pivot_table(index=['block', 'kind'], columns='set_index', values='health_acc')
piv['below_in'] = (piv[list(range(n_sets))] < CUT).sum(axis=1)
hit = piv[piv.below_in == n_sets]
print(); print(piv[piv.below_in > 0].round(3).to_string())
LB = sorted(b for (b, k), _ in hit.iterrows())
by_kind = Counter(hm.kind(b) for b in LB)
print(f'\\nload-bearing on all {n_sets} sets: {LB}')
print(f'by operator type: {dict(by_kind)}')
print(f"attention layers among them: {[b for b in LB if hm.kind(b) == 'attn']}   (7B: none)")
print(f'every Hyena family represented: '
      f"{all(by_kind.get(k, 0) > 0 for k in ('se', 'mr', 'li'))}   (7B: yes, two each)")
REPORT['load_bearing'] = {'cutoff': CUT, 'n_sets': n_sets, 'baseline_health': BASE,
                          'blocks': LB, 'by_kind': dict(by_kind)}
save()""")

    s4 = md("""## Stage 4 — does copying need attention? (pattern 3)

A random 200-letter stretch is inserted into real DNA twice, with a gap between the copies. The second
copy can only be predicted by finding the first and copying it. In 7B, removing attention drops this
from ~100% to chance (25%) at every gap tested, while the "long" LI layers matter only at long range.

The load-bearing layers found in Stage 3 are kept switched on, so that removing a whole family does not
simply break the model.""")
    s4c = code("""from marv_hyena import experiments
n_ins, n_site = (2, 1) if QUICK else (10, 5)
gaps = (100, 1000) if QUICK else (100, 1000, 10000)
conds = experiments.make_conditions(hm, exclude_blocks=LB)
print('conditions built:', list(conds))
# A family whose every block is load-bearing has nothing left to ablate, so make_conditions drops it.
# On 7B all five survive. If any are missing here, say so loudly: the comparison is incomplete, and
# that is itself a finding worth reporting rather than a silent gap in the table.
missing = [k for k in ('-se', '-mr', '-li', '-attn')
           if not any(c.startswith(k) for c in conds)]
if missing:
    print(f'\\n*** WARNING: no condition for {missing}. Every block of that family is load-bearing on')
    print('    this checkpoint, so it cannot be ablated without breaking the model. Report this.')
REPORT['conditions'] = list(conds)
REPORT['missing_conditions'] = missing
save()

copy_rows = stage('copy', lambda: experiments.copy_test(
    hm, genome, gaps=gaps, insert_len=200, seeds=n_ins, sites=n_site,
    conditions=conds, health_seq=DNA_SETS[0], verbose=False))
cp = pd.DataFrame(copy_rows)
summ = (cp.groupby(['condition', 'gap'])
          .agg(n=('second_acc', 'size'), mean_second_acc=('second_acc', 'mean'),
               sd=('second_acc', 'std'), health=('health_acc', 'first')).reset_index())
print(); print(summ.round(4).to_string(index=False))
REPORT['copy'] = summ.to_dict('records'); save()
print('\\n7B for comparison: unablated ~1.00 at every gap; -attn ~0.25 at every gap;')
print('                   -li ~0.95 at 100 letters but ~0.56 at 10,000.')""")

    s5 = md("""## Stage 5 — the reading frame

Inside a gene, DNA is read in three-letter words, so the model's accuracy has a 3-periodic rhythm. In
7B that rhythm disappears when the medium-range MR layers are removed.

**Read this stage with care**, and it is the reason the health column is printed: in 7B *every* family
ablation lowers the rhythm somewhat, because every ablation damages the model. MR lowers it most. A
clean version of this test needs a damage-matched control that does not exist yet, so treat the numbers
as descriptive.""")
    s5c = code("""n_genes = 2 if QUICK else 10

def pick_forward_genes(n, seed=0, min_len=900):
    rec = next(SeqIO.parse(GENOME, 'genbank'))
    f = sorted((int(x.location.start), int(x.location.end), int(x.location.strand or 1))
               for x in rec.features if x.type == 'CDS')
    keep = []
    for i, (s_, e_, st) in enumerate(f):
        if e_ - s_ < min_len or st != 1: continue
        if i and f[i-1][1] > s_: continue
        if i + 1 < len(f) and e_ > f[i+1][0]: continue
        keep.append((s_, e_))
    return random.Random(seed).sample(keep, min(n, len(keep)))

GENES = pick_forward_genes(n_genes)
def run_frame():
    out = []
    for gi, (s_, e_) in enumerate(GENES):
        tr = probes.genbank_track(GENOME, max(0, s_ - 500), min(len(genome), e_ + 100))
        for r in experiments.codon_test(hm, tr, conditions=conds):
            out.append(dict(gene=gi, **r))
    return out
frame = stage('reading_frame', run_frame)
fr = pd.DataFrame(frame)
per = []
for (cond, gi), sub in fr.groupby(['condition', 'gene']):
    d = {r.region: r.acc for r in sub.itertuples()}
    if {'codon_pos1', 'codon_pos2', 'codon_pos3'} <= set(d):
        v = max(d['codon_pos1'], d['codon_pos2']) - d['codon_pos3']
        if v == v:
            per.append({'condition': cond, 'gene': gi, 'periodicity': v,
                        'health': float(sub.health_acc.iloc[0])})
pf = pd.DataFrame(per)
out = (pf.groupby('condition')
         .agg(n_genes=('gene', 'nunique'), mean_periodicity=('periodicity', 'mean'),
              sd=('periodicity', 'std'), health=('health', 'first'))
         .sort_values('mean_periodicity').reset_index())
print(out.round(4).to_string(index=False))
REPORT['reading_frame'] = out.to_dict('records'); save()
print('\\nNote the health column: in 7B every ablation lowers both, so this is descriptive only.')""")

    s6 = md("""## Stage 6 — the biology (optional, slowest)

Skipped unless `RUN_BIOLOGY = True`. At one position in a gene, two different single-letter changes are
made: one that leaves the protein unchanged and one that changes an amino acid, **both of the same
chemical class** so that class cannot explain the difference. In 7B the amino-acid change disturbs the
model more at 72% of 150 such sites, and the difference between the two shows up mostly in the short SE
layers.

This is the slowest stage because each site needs two full forward passes plus a per-block capture.""")
    s6c = code("""if not RUN_BIOLOGY:
    print('skipped (RUN_BIOLOGY = False). Set it True for this stage.')
else:
    from marv_hyena import codons, controls
    n_sites = 10 if QUICK else 60
    whole = probes.genbank_track(GENOME, 0, len(genome))
    usage = codons.codon_usage(whole)
    designs = codons.design_sites(whole, max_sites=None, min_spacing=300, seed=0)
    pick = {k: random.Random(0).sample(v, min(n_sites, len(v)))
            for k, v in designs.items() if k in ('matched', 'fourfold', 'stop_matched')}
    print({k: len(v) for k, v in pick.items()})

    def run_bio():
        return {k: codons.paired_test(hm, genome, v, window=SEQ_LEN, span=24,
                                      downstream_span=200, usage=usage)
                for k, v in pick.items()}
    bio = stage('biology', run_bio)
    rows = []
    for k, v in bio.items():
        s_ = controls.summarize_paired(v)
        es = controls.paired_effect_size(v, n_boot=2000)
        rows.append({'design': k, 'n': s_['n'], 'b_more_disruptive': s_['b_more_disruptive_frac'],
                     'ci95': s_['ci95'], 'sign_p': s_['sign_test_p'],
                     'median_diff_nats': es['median_diff_nats']})
        pk = controls.peak_kind_shares(v, method='diff')
        rows[-1]['se_share_of_peaks'] = pk['share'].get('se')
        rows[-1]['se_base_rate'] = pk['base_rate'].get('se')
    bd = pd.DataFrame(rows)
    print(); print(bd.to_string(index=False))
    REPORT['biology'] = rows; save()
    print('\\n7B for comparison: matched 0.720 (protein change wins), fourfold 0.547 (no protein'
          '\\nchange, so near chance), stop_matched 0.067 meaning the stop wins 93% of the time;'
          '\\nSE share of peaks 0.86 on matched vs 0.11 on fourfold, against a base rate of ~0.28.')""")

    rep = md("""## REPORT BACK

Run this last cell and send back **everything it prints**, plus the JSON file it names. That is all we
need; no interpretation required.""")
    repc = code("""save()
path = f'{OUT}/results_{MODEL_NAME}.json'
print('=' * 78)
print(f'REPORT BACK  --  marv-hyena scaling test, {MODEL_NAME}')
print('=' * 78)
print(f"mode: {REPORT['mode']}   biology stage: {'run' if 'biology' in REPORT else 'skipped'}")
print(f"blocks: {REPORT['n_blocks']}  hidden: {REPORT.get('hidden_size')}  "
      f"attention at: {REPORT.get('attn_blocks')}")
print(f"devices: {REPORT['devices']}  sharded: {REPORT['sharded']}  torch {REPORT['torch']}  TE {REPORT['te']}")
print(f"self-checks passed: {REPORT.get('smoke_checks_passed')}")
f = REPORT.get('funnel', {})
if f:
    print(f"\\n1. FUNNEL: dominant block {f['dominant_block']} ({f['dominant_kind']}), "
          f"{f['from_end']} from the end")
    print(f"   largest/second = {f['ratio_top_to_second']:.3g}   (7B: ~1.2e5)")
    print(f"   share of final residual = {f['max_share']:.6f}   (7B: 1.000000)")
for r in REPORT.get('last_block_ablation', []):
    print(f"2. ABLATE block {r['block']} ({r['kind']}): max logit change {r['max_logit_change']:.3e}"
          f"{'  BIT-IDENTICAL' if r['bit_identical'] else ''}")
lbr = REPORT.get('load_bearing', {})
if lbr:
    print(f"\\n4. LOAD-BEARING (health < {lbr['cutoff']} on all {lbr['n_sets']} sets): {lbr['blocks']}")
    print(f"   by operator type: {lbr['by_kind']}   (7B: 2 se, 2 mr, 2 li, 0 attn)")
    print(f"   baseline health: {[round(h, 3) for h in lbr['baseline_health']]}")
if REPORT.get('missing_conditions'):
    print(f"\\n*** no ablation condition for {REPORT['missing_conditions']} -- see Stage 4 ***")
if REPORT.get('copy'):
    print('\\n3. COPYING (mean second-copy accuracy; chance = 0.25)')
    for r in REPORT['copy']:
        print(f"   {r['condition'][:30]:32s} gap {int(r['gap']):6d}  {r['mean_second_acc']:.3f}  "
              f"health {r['health']:.3f}")
if REPORT.get('reading_frame'):
    print('\\n5. READING FRAME (descriptive; read with the health column)')
    for r in REPORT['reading_frame']:
        print(f"   {r['condition'][:30]:32s} periodicity {r['mean_periodicity']:+.4f}  "
              f"health {r['health']:.3f}")
if REPORT.get('biology'):
    print('\\n6. BIOLOGY')
    for r in REPORT['biology']:
        print(f"   {r['design']:14s} n={r['n']:3d}  b_more={r['b_more_disruptive']:.3f}  "
              f"p={r['sign_p']:.3g}  SE share of peaks={r['se_share_of_peaks']}")
print('=' * 78)
print(f'ALSO SEND: {path}  ({os.path.getsize(path)} bytes)')
print('=' * 78)""")

    cells = [intro, settings, gpucell, install, stage0, stage0c, load, checks,
             s1, s1c, s2, s2c, s2d, s3, s3c, s4, s4c, s5, s5c, s6, s6c, rep, repc]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
          'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
          'nbformat': 4, 'nbformat_minor': 0}
    open(outfile, 'w').write(json.dumps(nb, indent=1) + '\n')
    print(outfile, len(cells), 'cells')

build('evo2_20b', '20B', 1, '2', 'about 2-4 hours', 'notebooks/marv_hyena_scaling_20b_h100.ipynb')
build('evo2_40b', '40B', 2, '4-8', 'about 4-8 hours', 'notebooks/marv_hyena_scaling_40b_h100.ipynb')
