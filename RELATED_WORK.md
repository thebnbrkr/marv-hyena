# Related work, and what each paper does to our claims

Written 2026-10-01. Companion to `PREDICTIONS.md` (what we predicted and what happened) and
`RESEARCH_LOG.md` (how we got there). This file exists because three claims in this project turned
out to be published already, and in two cases we found out only after writing them up as novel.

**Every entry carries how it was read.** That column is the point of the document. Two of this
session's mistakes came from citing work we had seen only through a summary:

| tag | meaning |
|---|---|
| `FULL` | read end to end in this project, with specific numbers taken from the text |
| `READ` | read in an earlier session and recorded in `RESEARCH_LOG.md`; not re-verified since |
| `ABSTRACT` | only the abstract or a listing page; claims are as that page states them |
| `SUMMARY` | only a search-engine summary. **Do not cite without reading.** |
| `SECONDHAND` | seen only through another paper's related-work section. **Do not cite at all yet.** |

---

## 1. The model, and what it claims about itself

### Evo 2 — Brixi et al. `READ`
*Genome modeling and design across all domains of life with Evo 2.*
[Nature 2026](https://www.nature.com/articles/s41586-026-10176-5) ·
[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full)

**Claims:** genome-scale sequence generation; that the model captures biological features across
species without supervision; and zero-shot pathogenicity prediction for coding and non-coding
variants that beats the state of the art (BRCA1 AUROC 0.891 for the 7B base model). Mechanistic
interpretability analyses report learned features for exon–intron boundaries, transcription-factor
binding sites, protein structural elements and prophage regions. Figure 2 reports that missense,
premature-stop and frameshift changes lower likelihood more than synonymous ones, across 20
prokaryotic and 16 eukaryotic species. An SAE feature (f/24278) fires on frameshifts and premature
stops.

**What it does to us:** it is the model under test, and it pre-empts the *behaviour* behind two of
our biology claims. Missense-beats-silent and stop-beats-missense are its results, not ours. What is
left to us is **where** that happens inside the network, and the matched designs that separate it
from substitution-type statistics. Its main text reports no substitution-type or codon-usage control
that we could find — which is the gap round 5 was built for. Its BRCA1 AUROC 0.891 is also the number
our round-2 replication (0.88 on 40 variants) sits against.

### StripedHyena 2 — Ku et al. `READ`
*Systems and algorithms for convolutional multi-hybrid language models at scale.*
[arXiv 2503.01868](https://arxiv.org/html/2503.01868v1)

**Claims, in prose, with no ablations and no interpretability of its own:**

| operator | what the paper says it is for |
|---|---|
| Hyena-SE (short explicit) | local motifs and nucleotide patterns, e.g. transcription-factor binding sites; "local multi-token recall" |
| Hyena-MR (medium regularized) | patterns spanning an entire exon or intron; modelling across hundreds of tokens |
| Hyena-LI (long implicit) | aggregation over the **entire sequence** |
| attention | recall across longer sequences |

**What it does to us:** this is the claim our operator map is measured against. Three corrections.
SE plays no part in recall at the distances tested (100–10,000 letters — note our probes never test
SE's own 7-letter range, so we cannot speak to *local* recall). LI's filters mostly reach 4–7 letters.
Attention is essential at *every* distance tested, not only long ones.

**Careful with the quote.** The paper says LI aggregates over the entire sequence. The phrase
"integrates regulatory elements thousands of nucleotides away" is from a popular
[explainer](https://medium.com/meta-multiomics/evo2-demystified-the-ultimate-technical-guide-to-genomic-language-modeling-a75b0afe7b87),
not the paper. A summary written for the Leuven abstract misattributed it; the error is recorded in
`RESEARCH_LOG.md` (2026-10-01).

---

## 2. Prior art that takes something off our list

### BioRiskEval — Wei et al. `FULL`
*Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models.*
[arXiv 2510.27629v4](https://arxiv.org/html/2510.27629v4) (cs.CR, Nov 2025)

A biosecurity evaluation paper. Its main subject is unrelated to this project and is not discussed
here. **Appendix C and Figure 4b are what matter:** to explain a result of their own they measured
Evo 2's per-layer representation magnitude and reported the same explosion we had been calling the
funnel. Their stated mechanism: the residual stream carries no layer normalisation between blocks,
and Hyena's input-dependent gated convolutions multiply two already-large tensors, so magnitude
compounds. The final RMSNorm rescales before the logits, so output quality depends on direction
rather than magnitude. They reproduced it through **Vortex**, the same inference library we use, and
their layer-wise probe accuracy collapses after layer 28.

**What it does to us:** **the novelty claim on the magnitude growth is withdrawn.** It was published,
with a mechanism, in a venue our interpretability-keyword searches never touched. What survives as
ours: that *one* block's write is 100.00% of the residual; that block 31 is bit-identically inert
(shown in bf16); that the magnitude and the irrelevance of earlier writes hold in float32; and that
both direct and gradient attribution break on it. Their probe collapse after layer 28 is the same
wall seen from outside, and is worth one sentence as independent corroboration.

*Lesson recorded: search by the phenomenon ("representation magnitude", "Evo 2 layer norm"), not only
by your own framing.*

### Representative vs. Load-bearing Layers — Cho, Kim & Kim `ABSTRACT`
*A Dissociation in Genomic Foundation Models.* [ICML 2026](https://icml.cc/virtual/2026/70729)

**Claims (as the listing page states them):** studies NT-v2 500M (masked LM) and **Evo 2 7B**
(causal, hyena/attention hybrid) on 8,008 ClinVar single-nucleotide variants. Distinguishes the
*representative* layer (peak single-feature AUROC) from the *load-bearing* layer (largest drop under
leave-one-layer-out ablation of a joint multi-layer classifier). Representative layers sit
mid-network in both models; load-bearing depth is mid-shallow in the MLM and **deep in the CLM
hybrid**. Their feature is the L2 norm of the per-layer hidden-state shift at the variant token.

**What it does to us:** a **terminology collision inside our exact subfield**, on our exact model.
Their ablation removes layers from a *classifier built over hidden states*; ours removes a component
from the model's own forward pass and measures whether the model still reads DNA. Different
operation, different meaning, same name. Either rename our concept or distinguish it explicitly in
one sentence. That both methods land deep in the network is worth noting.

**Do not claim** that our funnel explains their deep load-bearing layer. Their feature is a norm and
late-layer norms in Evo 2 explode, which is suggestive — but if they standardise features per layer,
magnitude washes out, and nobody has checked. Specific layer numbers attributed to this paper
elsewhere came from a third-party summary; read the paper, and check whether its layer numbering
starts at 0 or 1, before citing anything numeric.

### Mathur & Sachidanandam `FULL`
*Benchmarking DNA Foundation Models: Biological Blind Spots in Evo2 Variant-Effect Prediction.*
[bioRxiv 2026.03.10.710786](https://doi.org/10.64898/2026.03.10.710786) (posted 11 March 2026, not
peer reviewed)

A **black-box behavioural benchmark**, run entirely through Evo 2's hosted API (`/generate`,
`/forward`), scoring variants by mean log-likelihood. No layers, no ablations, no operators, no
internals. Mitochondrial DNA is the test bed. Their results:

| their test | their result |
|---|---|
| Codon usage bias in wobble predictions (966 codons of TTN exon 305) | **Not internalised.** Preferred codon chosen at 24.4% of positions against 25% for a coin flip; mean probability on the preferred codon 28.5%; mean JSD 0.254 |
| Mitochondrial start/stop codon reassignment | Not respected: 26 of 26 valid mito start-preserving variants called pathogenic; 16 of 22 stop-preserving ones |
| Zero-shot mitochondrial pathogenicity (130 pathogenic, 623 benign) | AUROC 0.896, balanced accuracy 87.6%, highest MCC (0.631) of all tools tested — but APOGEE2 beats it on AUROC (0.950), specificity, balanced accuracy and auPRC |
| Performance by region | Worst specificity in the D-loop (34.9% of benign called pathogenic); 32.1% of benign missense misclassified |
| By disease severity | 100% of *mild* pathogenic variants correct, progressively worse for moderate and severe — the inverse of what clinical use needs |
| Transitions vs transversions | Transversions ~1.9× more negative (mean ΔL −0.0113 vs −0.0060) |
| tRNA cyclic permutation (a null manipulation) | Sensitivity collapses **65.8% → 5.1%** with tRNA sequences unchanged and only flanking context moved |
| Gene completion, 10 species | 86.2% mean base accuracy, but it does not track mutational constraint: the most constrained OXPHOS complex completes worst (85.0%) |
| NUMTs | Evo 2 prefers the **mitochondrial** allele at divergence sites — it treats the pseudogene as authentic mtDNA |
| PhyloP conservation (250 bp of mt-RNR1) | ρ = 0.77 overall, but conservation peaks do not line up with likelihood peaks |

**What it does to us — three things.**

1. **Their transversion result is the one our `fourfold` design refines.** It rests on **46
   transversions against 727 transitions** (6% of the set), unmatched and unpaired, pooled across
   D-loop, RNA genes, synonymous and missense variants, with benign and pathogenic mixed. So it could
   be carried by which regions those 46 fall in. Our design asks the same question with **150 paired
   same-site comparisons holding the protein constant**, and gets **54.7% (p = 0.29)**. Phrase it as a
   refinement, not a contradiction: *within matched coding sites, substitution type alone has at most
   a modest effect.* Our own round-4 numbers are the cautionary half — unmatched, our version of this
   comparison gave 68.9% and read as biology.
2. **Their codon-usage finding supports our rarity control and sharpens our claim.** Round 5's main
   worry was that codon rarity drove the amino-acid result. The re-analysis found rarity pushing
   *against* it at 266 of 300 sites; their result explains why that confound is weak. Together the two
   give a dissociation neither paper can state alone: **Evo 2 responds to what a codon *means* while
   not tracking which synonymous codon is *preferred*.**
3. **They name our work as the extension of theirs.** From their conclusion: *"Mechanistic
   interpretability of Evo2 embeddings may clarify which biological features are encoded and which are
   absent."*

**Strategic warning.** Their framework is built on **null manipulations** — the tRNA permutation is
exactly the kind of negative control this project keeps arguing for. The "controls toolkit" idea in
`RESEARCH_LOG.md` is therefore **no longer an empty niche**. Our differentiator has to be the
internals (operator ablation, per-block divergence, weight nulls), not the controls framing alone.

### Heap et al. `READ`
*Sparse Autoencoders Can Interpret Randomly Initialized Transformers.*
[arXiv 2501.17727](https://arxiv.org/abs/2501.17727)

**Claims:** interpretability tools applied to randomly initialised transformers score about as well
as on trained ones.

**What it does to us:** it predicted round 5b's outcome in general, and we cited it as the reason the
null model was not optional — then ran a null whose result was so clean it hid a bug for two rounds.
Round 5b's finding (a weight-shuffled block 0 is as word-selective as the trained one) is a
domain-specific instance of exactly this.

---

## 3. Precedents we extend

### Some Attention is All You Need for Retrieval — Michalak & Abreu `SUMMARY`
(2025) — **read before citing.**

**Claims, as summarised:** in trained hybrids (RecurrentGemma-2B/9B, Jamba-Mini-1.6), ablating
attention drops retrieval to 0%, and the recurrent layers show no compensation.

**What it does to us:** our "attention does exact copying" result is an **extension to DNA and to
Hyena**, not a discovery. Two cautions. First, we have this only through a search summary. Second,
and this matters: they removed **attention**; they did not remove the recurrent layers. So they cannot
be cited as showing recurrent layers contribute nothing. Our LI result is therefore *a question they
did not test*, not a divergence from their finding. An earlier draft of our novelty ledger claimed
divergence; corrected 2026-10-01.

### Zoology — `READ`
*Measuring and Improving Recall in Efficient Language Models.*
[arXiv 2312.04927](https://arxiv.org/abs/2312.04927)

**Claims:** on synthetic tasks, attention performs associative recall that gated convolutions cannot.

**What it does to us:** the original motivation for the copying experiments. Theirs is synthetic;
ours is inside a trained 7B model on real DNA.

### Laughing Hyena Distillery — `READ`
[arXiv 2310.18780](https://arxiv.org/abs/2310.18780)

**Claims:** pretrained Hyena filters typically decay to zero in finite time; gives the pole/residue
parameterisation behind LI.

**What it does to us:** our "LI filters mostly reach 4–7 letters" measurement is **expected** from
this, and should be presented as a confirmation in Evo 2 rather than a surprise.

### ShortGPT — `READ`
[arXiv 2403.03853](https://arxiv.org/html/2403.03853v3)

**Claims:** many transformer layers are redundant; a depth-importance profile shows early layers
critical, middle redundant, late important.

**What it does to us:** our load-bearing map repeats this known pattern. What is left to us is the
operator-level reading: every Hyena family contains load-bearing layers (**two** each — L0/L4 SE,
L1/L29 MR, L9/L30 LI) while attention contains none.

### DNABERT-2 / Nucleotide Transformer pruning study — `SUMMARY`
*Reaping the Fruits of LLM Pruning.* [Genes 16:1358](https://doi.org/10.3390/genes16111358)

**Claims, as summarised:** layer ablation in DNABERT-2 and NT builds layer-importance profiles;
pruned models match full ones on variant-effect prediction with much less compute.

**What it does to us:** layer ablation in DNA models is established. Ours differs in ablating
**operators within a hybrid** and in measuring whether the model still reads DNA at all.

### Massive activations — Sun et al. `READ`
[arXiv 2402.17762](https://arxiv.org/abs/2402.17762), with follow-ups
[2605.08504](https://arxiv.org/html/2605.08504) and
[2606.20743](https://arxiv.org/html/2606.20743v1)

**Claims:** a few hidden dimensions take enormous constant values, appearing early and on special
tokens; mean-replacing them is harmless, so they act as a hidden bias. The follow-up reports that
such values rebuild in whatever representation the model decodes from.

**What it does to us:** the comparison a reviewer will reach for. Block 30 differs on every axis — the
whole write rather than a few dimensions, late rather than early, every position rather than special
tokens, and mean-replacing it breaks the model. With BioRiskEval now holding the magnitude result,
this comparison matters less than it did, but it is still the right paragraph to pre-empt.

### Induction Meets Biology — `READ`
[arXiv 2602.23179](https://arxiv.org/abs/2602.23179) (ICML 2026)

**Claims:** protein models (ESM-3, ESM-C) detect repeats in two stages — alignment-like heads, then
**induction heads** that find the earlier copy and predict what follows.

**What it does to us:** the closest precedent for the copying result, and a template for finding the
copying heads in block 3. Their gradient method works on transformers; ours fails in Evo 2 because of
the funnel.

### Attention recalls, recurrence controls — `READ`
[arXiv 2609.04434](https://arxiv.org/html/2609.04434v1) · and
*ICL beyond transformers* [arXiv 2510.23006](https://arxiv.org/html/2510.23006v2)

**Claims:** in text hybrids, attention-only memory preserves exact lookup (64–98%) while
recurrence-only preserves language and style (70–80%); specific attention *heads* drive in-context
learning in hybrids.

**What it does to us:** suggests the division-of-labour experiment for Evo 2 (keep or swap attention
vs Hyena generation states), and a head-level follow-up in block 3.

---

## 4. Methods papers that shaped our controls

### Mechanistic Invariance Test — `READ`
[arXiv 2604.06549](https://arxiv.org/abs/2604.06549)

**Claims:** across five DNA models including Evo 2 1B, apparent "regulatory understanding" was driven
by AT content (r = 0.78–0.96).

**What it does to us:** the same trap as round 4's amino-acid claim — a letter-level property dressed
as understanding. Round 5 added G/C change as a covariate because of this paper.

### Position: beyond anecdotal evaluation — `READ`
[arXiv 2606.07607](https://arxiv.org/abs/2606.07607)

**Claims:** genomic interpretability leans on cherry-picked examples, and methods contradict each
other.

**What it does to us:** independent support for pre-registration and for shipping a control with every
claim.

### Bilinear MLPs enable weight-based interpretability — Pearce et al. `READ`
[arXiv 2410.08417](https://arxiv.org/abs/2410.08417) (ICLR 2025)

**Claims:** MLPs with no activation function admit features read directly from the weights.

**What it does to us:** Evo 2's MLPs are bilinear from block 1 on (`AGENTS.md`), so this is the natural
route into them, and Pearce is an Evo 2 co-author. Not yet attempted here.

### Genomic heterogeneity inflates variant pathogenicity predictions — Lu et al. `SECONDHAND`
bioRxiv 2025.09.05.674459 — seen only through BioRiskEval's discussion. **Do not cite yet.**

### Bick et al. 2025; Arora et al. 2025 `SECONDHAND`
Seen only through another paper's related-work section. As summarised: pretrained hybrids hand the
aggregation step of retrieval to attention (Bick); associative recall in recurrent models is carried
by short convolutions rather than the recurrence (Arora). **Read the originals before citing.** If
Arora holds in Evo 2, it is directly relevant to what SE and MR do.

---

## 5. Gap statements that motivate the work

### What do bio-foundation models compute? — `READ`
[bioRxiv 2026.03.04.709491](https://www.biorxiv.org/content/10.64898/2026.03.04.709491v1.full)

**Claims:** the field is stuck at representation-level analysis; causal work is missing; asks
specifically for null models and non-circular validation.

**What it does to us:** states our gap in writing, and is the reason round 4 ran a random-weights
baseline at all.

### Mathur & Sachidanandam's conclusion — `FULL`
Asks for "mechanistic interpretability of Evo2 embeddings" to clarify which biological features are
encoded. A second published gap statement pointing here, from the behavioural side.

---

## 6. Adjacent, and one threat we cleared

### Goodfire — Interpreting Evo 2 `READ`
[goodfire.com](https://www.goodfire.com/research/interpreting-evo-2) ·
[SAE weights](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed) ·
[Tree of life](https://www.goodfire.ai/research/phylogeny-manifold)

**Claims:** sparse autoencoders at layer 26 recover biologically meaningful features (exons, TF
motifs); layer 26 was chosen because it had the most interesting features on inspection, and they
hypothesise that a four-letter vocabulary needs fewer final layers to select and calibrate the next
token. They report that steering Evo 2 is considerably harder than steering a language model.

**What it does to us:** our block-30 result is measured evidence for their stated hypothesis about the
final layers, and may explain why steering is hard. Their framing (low vocabulary) and ours (one block
owns the readout) are different explanations of a shared observation.

### Decode-gLM — `READ`
[bioRxiv 2025.10.31.685860](https://www.biorxiv.org/content/10.1101/2025.10.31.685860v4)

**Claims:** tools to interpret and audit Nucleotide Transformer through sparse features; found
training-data leakage.

**What it does to us:** the nearest existing "audit" tool — feature-based rather than a battery of
controls.

### PAS-ISP — `FULL`, and cleared
[arXiv 2608.12149](https://arxiv.org/abs/2608.12149)

**Claims:** in linear-attention text hybrids, massive activations spike immediately **before**
full-attention layers, then plateau.

**What it does to us:** nothing, and we checked. If block 30's magnitude were this effect, blocks 2,
9, 16 and 23 should spike before the attention layers at 3, 10, 17 and 24. Against nearest
non-attention neighbours across three genome regions: L2 0.83–0.90×, L9 1.4–1.6×, L16 1.9×, L23
0.98–1.14× — no consistent spike, and L2 is *below* its neighbours, while L30 is 3.2e6–1.7e7×. Six to
seven orders of magnitude past what PAS predicts. Recorded as finding 22.

### Mamba bottleneck — `READ`
[arXiv 2602.22719](https://arxiv.org/html/2602.22719)

**Claims:** gradient attribution fails at a bottleneck where removal-based attribution works.

**What it does to us:** our "gradients fail, ablation works" result matches a known pattern, so it is
new only in that it happens in Evo 2, and why.

### Titans — `READ`
[arXiv 2501.00663](https://arxiv.org/abs/2501.00663)

**Claims:** pairs attention with a memory module that keeps updating its own weights while reading.

**What it does to us:** nothing directly — Evo 2's weights are frozen at inference. Kept because the
resemblance (exact lookup plus a compressed running summary) is the first thing readers ask about.

---

## 7. Claim-by-claim cross-reference

| # | Our claim | Status | Whose prior work bears on it |
|---|---|---|---|
| 1 | Attention is required for exact copying at every gap tested (100–10k) | extension | Michalak & Abreu `SUMMARY`; Zoology |
| 2 | LI is required for long-range copying (99.7% → 55.6% at 10k) | **appears new** | a question the text-hybrid studies did not test |
| 3 | MR carries the reading frame | localisation new | Evo 2 paper (periodicity); owes a damage-curve control |
| 4 | Six load-bearing layers, two per Hyena family, none in attention | pattern known | ShortGPT; ICML 2026 **name collision** |
| 5 | LI filters mostly reach 4–7 letters | confirmation | Laughing Hyena Distillery |
| 6 | Block 30's write is 100.00% of the residual | **appears new** | BioRiskEval holds the magnitude itself |
| 7 | Block 31 is bit-identically inert (bf16) | **appears new** | — |
| 8 | Magnitude and the irrelevance of earlier writes hold in float32 | **appears new** | — |
| 9 | Gradient and direct attribution both break here | known class | Mamba bottleneck |
| 10–11 | A protein-changing substitution disturbs Evo 2 more than a silent one at the same site, substitution type held equal (72.0%) and reversed (80.0%) | **appears new** (design) | behaviour in the Evo 2 paper |
| 12 | Substitution type alone, protein unchanged: 54.7% (p = 0.29) | **appears new**, and refines | Mathur & Sachidanandam's transversion result |
| — | Evo 2 tracks what a codon means but not which synonymous codon is preferred | **appears new** (joint) | second half is Mathur & Sachidanandam |
| 13 | The difference peaks in SE blocks: 89% vs 51% size-matched (chance 28%) | **appears new**; 1 check open | no prior operator-level localisation found |
| 14 | A premature stop beats a missense change at the same site (93% / 92%) | incremental | behaviour in the Evo 2 paper |
| 15 | Block 0's word selectivity is architectural, not learned | **appears new** (negative) | Heap et al. predicted it in general |
| — | 724 of 4,096 block-0 channels are dead; nulls have none | **appears new** | — |

---

## 8. Still owed before any of this is published

1. **Read `SUMMARY` and `SECONDHAND` entries in the original.** Michalak & Abreu, the ICML 2026
   paper, the pruning study, Bick, Arora, Lu.
2. **Check the ICML paper's layer numbering** (0- or 1-based) before citing any layer number, and
   rename our "load-bearing" concept or distinguish it in one sentence.
3. **Close the reopened scrambled-null leg** behind claim 13.
4. **Make claim 13 causal**: ablate or patch SE writes on the same pairs. Until then the verb is
   "peaks in", not "carries".
5. **A second checkpoint** (`evo2_7b_base`). Needs no new code, and every reviewer will ask.
6. **A damage-curve control for claim 3**, since removing MR is the most damaging family ablation.
7. **Do not claim** the funnel explains the ICML paper's deep load-bearing layer.
