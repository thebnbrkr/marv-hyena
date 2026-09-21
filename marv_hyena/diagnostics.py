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
