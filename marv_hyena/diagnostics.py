"""Round-2 diagnostics, added after round 1 (see RESEARCH_LOG.md):

- write_norms: how big is each block's residual write? Round 1's trace gave
  ALL direct credit to block 30 (the last LI block). The hypothesis is that
  block 30 writes something so much larger than every other block that the
  final norm + unembedding effectively read only it. This measures it.
- find_bottlenecks: blocks whose write is most of the final residual. Ablating
  a bottleneck breaks the whole model, so ablation families must exclude them.
- health: is the model still working under an intervention? Next-letter
  accuracy / log-prob on ordinary genome. Round 1 misread "the model is broken"
  as "this component does X"; every ablation row now carries a health flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .arch import HyenaModel
from .intervene import token_logprobs
from .trace import capture_writes

UNIFORM_LP = -1.3863  # log(1/4): a model guessing uniformly over A/C/G/T


@dataclass
class WriteNorm:
    block: int | None
    kind: str
    part: str
    norm: float  # mean L2 norm of the write over the captured positions
    share: float  # norm / norm of the final residual


def write_norms(hm: HyenaModel, seq: str, positions: list[int]) -> list[WriteNorm]:
    w = capture_writes(hm, hm.ids(seq), positions)
    final = float(w.final.norm(dim=-1).mean())
    rows = [WriteNorm(None, "embed", "embed", float(w.embed.norm(dim=-1).mean()), 0.0)]
    for (b, part), x in sorted(w.parts.items()):
        rows.append(WriteNorm(b, hm.kind(b), part, float(x.norm(dim=-1).mean()), 0.0))
    for r in rows:
        r.share = r.norm / final
    return rows


def show_write_norms(rows: list[WriteNorm], k: int = 10) -> None:
    print(f"{'component':<18} {'norm':>12} {'share of final':>15}")
    for r in sorted(rows, key=lambda r: -r.norm)[:k]:
        name = "embed" if r.block is None else f"L{r.block} {r.kind} {r.part}"
        print(f"{name:<18} {r.norm:>12.1f} {r.share:>14.1%}")


def find_bottlenecks(hm: HyenaModel, seq: str, positions: list[int], min_share: float = 0.5) -> list[int]:
    """Blocks with a write carrying >= min_share of the final residual's norm."""
    return sorted({r.block for r in write_norms(hm, seq, positions) if r.block is not None and r.share >= min_share})


def find_load_bearing(hm: HyenaModel, seq: str, parts=("mixer",)) -> list[dict]:
    """Round 3: ablate each single component (mean-ablation) and report health.
    Round 2 found five mixers whose removal alone breaks Evo 2 7B (L0, L1, L9,
    L29, L30); family ablations must keep these on or they just measure a
    broken model."""
    from .intervene import mean_ablate, mean_writes

    ids = hm.ids(seq)
    out = []
    for b in range(hm.n_blocks):
        for p in parts:
            with mean_ablate(hm, mean_writes(hm, ids, [(b, p)])):
                h = health(hm, seq)
            out.append({"block": b, "kind": hm.kind(b), "part": p, **h})
    return out


@torch.no_grad()
def precision_check(hm: HyenaModel, seq: str, block: int, position: int | None = None) -> dict:
    """Round 3: is the bottleneck block's huge write a property of the WEIGHTS
    or of bf16 arithmetic? Recompute that block's mixer write in float32 from
    the same input and compare norms; then recompute the logits in float32 with
    and without every earlier write, to see whether those writes would matter
    at full precision."""
    blk = hm.block(block)
    position = len(seq) - 1 if position is None else position
    store = {}

    def pre(_m, args):
        store["u"] = args[0].detach()

    def out_hook(_m, _i, out):
        store["w"] = out.detach()

    h1 = blk.register_forward_pre_hook(pre)
    h2 = hm.component(block, "mixer").register_forward_hook(out_hook)
    try:
        hm.model(hm.ids(seq))
    finally:
        h1.remove()
        h2.remove()
    u, w_bf = store["u"], store["w"]

    dtypes = {n: p.dtype for n, p in blk.named_parameters()}
    try:
        blk.float()
        z = blk.projections(blk.pre_norm(u.float()))
        z = z[0] if isinstance(z, tuple) else z
        y, _ = blk.filter(z)
        w32 = blk.out_filter_dense(y)
    finally:
        for n, p in blk.named_parameters():
            p.data = p.data.to(dtypes[n])

    W = hm.unembed_weight().float()
    scale, eps = hm.final_norm_params()

    def logits_of(r):
        r = r.float()
        return (scale.float() * r / (r.norm() * hm.hidden_size ** -0.5 + eps)) @ W.T

    full = logits_of(u[0, position].float() + w32[0, position])
    alone = logits_of(w32[0, position])
    return {
        "norm_bf16": float(w_bf[0, position].float().norm()),
        "norm_fp32": float(w32[0, position].norm()),
        "norm_input_residual": float(u[0, position].float().norm()),
        "logit_change_from_earlier_writes_fp32": float((full - alone).abs().max()),
        "logit_scale": float(full.abs().max()),
    }


@torch.no_grad()
def health(hm: HyenaModel, seq: str) -> dict:
    """Next-letter accuracy and mean log-prob on `seq` (use ordinary genome).
    Evo 2 7B scores roughly 0.6-0.9 accuracy on E. coli; uniform guessing is
    0.25 / -1.386. `broken` = within 0.1 nats of uniform guessing."""
    ids = hm.ids(seq)
    logits = hm.logits(ids)
    lp = float(token_logprobs(logits, ids).mean())
    acc = float((logits[:-1].argmax(-1).cpu() == ids[0, 1:].cpu()).float().mean())
    return {"health_acc": acc, "health_lp": lp, "broken": lp < UNIFORM_LP + 0.1}
