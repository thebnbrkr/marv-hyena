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


def device_map(hm: HyenaModel) -> dict:
    """Which CUDA device each block's parameters live on.

    Vortex splits the 20B and 40B checkpoints across whatever CUDA devices it
    finds, so a sharded model has blocks on several devices while `hm.device`
    reports only the embedding's. Everything in this package is written to
    survive that -- `capture_writes` moves each write to CPU float32 before
    combining them, and `mean_ablate` sends each replacement back with
    `.to(x.device)` -- but a new hook that forgets is an easy mistake, and the
    error it raises ("Expected all tensors to be on the same device") is much
    easier to read with this table beside it.

    Returns the embedding, per-block and unembed devices, plus `sharded`.
    """
    def dev_of(module):
        for p in module.parameters():
            return str(p.device)
        for b in module.buffers():
            return str(b.device)
        return "none"

    blocks = {i: dev_of(hm.block(i)) for i in range(hm.n_blocks)}
    seen = sorted(set(blocks.values()) | {str(hm.device)})
    return {
        "embedding": str(hm.device),
        "blocks": blocks,
        "unembed": str(hm.unembed_weight().device),
        "devices": seen,
        "n_devices": len(seen),
        "sharded": len(seen) > 1,
        "by_device": {d: [i for i, x in blocks.items() if x == d] for d in seen},
    }


def model_fingerprint(hm: HyenaModel, n_tensors: int = 12) -> dict:
    """A cheap identity check for the loaded weights.

    Two runs of two different checkpoints must not produce the same fingerprint.
    If they do, the same weights were loaded twice, whatever the config said --
    which is exactly what happened in the 20B/40B scaling runs of 2026-10-01,
    where a mislabelled merged `.pt` gave a 50-block skeleton the 24 blocks of
    another model's weights and left the rest uninitialised. The package's own
    smoke checks could not catch it: they verify that the *measurement* is an
    exact decomposition of whatever model is loaded, not that the model is the
    one you asked for.

    Cheap enough to print on every run: it touches the embedding and a spread of
    block tensors, and reads a scalar from each.
    """
    import hashlib

    def sig(t: torch.Tensor) -> str:
        f = t.detach().float()
        return f"{float(f.sum()):.6e}/{float(f.abs().mean()):.6e}/{tuple(t.shape)}"

    parts = [f"embed:{sig(hm.model.embedding_layer.weight)}"]
    step = max(1, hm.n_blocks // n_tensors)
    for b in range(0, hm.n_blocks, step):
        for name, p in hm.block(b).named_parameters():
            parts.append(f"b{b}.{name}:{sig(p)}")
            break  # one tensor per block is enough to separate checkpoints
    blob = "|".join(parts)
    return {"fingerprint": hashlib.sha256(blob.encode()).hexdigest()[:16],
            "embedding": sig(hm.model.embedding_layer.weight),
            "n_tensors": len(parts)}


@torch.no_grad()
def find_dead_tail(hm: HyenaModel, seq: str, rel_floor: float = 1e-6) -> dict:
    """Blocks whose residual write is negligible against the largest write.

    A trained network has a few such blocks at most (7B has one: its final
    attention block, downstream of the funnel). A long contiguous dead tail
    means those blocks' weights were never loaded -- the failure mode behind the
    2026-10-01 "40B" run, where blocks 24-49 of 50 wrote ~1e-13 and ablating any
    of them left accuracy untouched.

    `dead_tail` counts how many of the LAST blocks are dead; `suspicious` is
    True once that exceeds a tenth of the network.
    """
    from .trace import capture_writes

    ids = hm.ids(seq) if isinstance(seq, str) else seq
    n = ids.shape[-1]
    W = capture_writes(hm, ids, positions=[n - 1])
    norms = {}
    for (b, part), v in W.parts.items():
        norms[b] = max(norms.get(b, 0.0), float(v.norm()))
    biggest = max(norms.values()) if norms else 0.0
    dead = sorted(b for b, n in norms.items() if biggest > 0 and n < rel_floor * biggest)
    tail = 0
    for b in range(hm.n_blocks - 1, -1, -1):
        if b in dead:
            tail += 1
        else:
            break
    return {"largest_write": biggest, "dead_blocks": dead, "n_dead": len(dead),
            "dead_tail": tail, "suspicious": tail > max(2, hm.n_blocks // 10),
            "reconstruction_error": W.reconstruction_error()}
