"""TRACE: split one prediction exactly into what each block wrote.

The residual stream is a sum:  final = embed + sum_i (mixer_i + mlp_i).
Logits are unembed(norm(final)), and the final RMSNorm is linear once its
denominator is frozen at the value it actually took. So for any direction d
in logit space (e.g. "logit of G minus mean logit of A/C/T")

    d . logits  =  sum over components c of  (scale * write_c / denom) . W_U^T d

exactly -- every row of the table below is one term of that sum, and the
rows add up to the model's real logit difference (`Decomposition.actual`).
This is direct logit attribution (LARQL's `TRACE ... DECOMPOSE`) extended to
the four StripedHyena operator types. It measures DIRECT effects only: if an
SE block feeds an LI block, the credit lands on the LI block. Use
`patch.patch_sweep` for total (direct + indirect) effects.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .arch import BASES, HyenaModel, base_id
from .hooks import capture_outputs


@dataclass
class Writes:
    """Every residual write at the captured positions, float32 on CPU."""

    positions: list[int]
    embed: torch.Tensor  # (P, H)
    parts: dict[tuple[int, str], torch.Tensor]  # (block, part) -> (P, H)
    final: torch.Tensor  # (P, H) the real residual entering the final norm
    logits: torch.Tensor  # (P, vocab)

    def reconstruction_error(self) -> float:
        """Relative error of embed + sum(writes) vs the real final residual.
        Should be ~1e-2 or better in bf16; much larger means a hook is missing
        a write (e.g. a Vortex version changed the block structure)."""
        total = self.embed + sum(self.parts.values())
        return float((total - self.final).norm() / self.final.norm())


@torch.no_grad()
def capture_writes(hm: HyenaModel, ids: torch.Tensor, positions: list[int], blocks=None) -> Writes:
    blocks = range(hm.n_blocks) if blocks is None else blocks
    comps = [(b, p) for b in blocks for p in ("mixer", "mlp")]
    store: dict[str, torch.Tensor] = {}

    def embed_hook(_m, _i, out):
        store["embed"] = out[0, positions].detach().float().cpu()

    def final_hook(_m, inputs):
        store["final"] = inputs[0][0, positions].detach().float().cpu()

    h1 = hm.model.embedding_layer.register_forward_hook(embed_hook)
    h2 = hm.model.norm.register_forward_pre_hook(final_hook)
    try:
        with capture_outputs(hm, comps, positions=positions) as parts:
            logits, _ = hm.model(ids)
    finally:
        h1.remove()
        h2.remove()
    return Writes(
        positions=list(positions),
        embed=store["embed"],
        parts=dict(parts),
        final=store["final"],
        logits=logits[0, positions].float().cpu(),
    )


def logit_direction(hm: HyenaModel, target: str, baseline: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (direction, token_weights). token_weights is a vector over the
    vocab defining the quantity -- logit[target] - logit[baseline], or minus
    the mean of the other three bases when baseline is None -- and direction
    = W_U^T token_weights is the same quantity as a hidden-space direction."""
    W = hm.unembed_weight().detach().float().cpu()
    tw = torch.zeros(W.shape[0])
    tw[base_id(target)] = 1.0
    others = [baseline] if baseline is not None else [b for b in BASES if b != target]
    for b in others:
        tw[base_id(b)] -= 1.0 / len(others)
    return tw @ W, tw


def _frozen_norm_factor(hm: HyenaModel, final: torch.Tensor) -> torch.Tensor:
    """Per-position vector s such that norm(x) = s * x at the captured x.
    Mirrors vortex RMSNorm: scale * x / (||x|| * H**-0.5 + eps)."""
    scale, eps = hm.final_norm_params()
    denom = final.norm(dim=-1, keepdim=True) * hm.hidden_size ** -0.5 + eps
    return scale.detach().float().cpu() / denom  # (P, H)


@dataclass
class Row:
    block: int | None  # None for the embedding
    kind: str  # se | mr | li | attn | embed
    part: str  # mixer | mlp | embed
    value: float


@dataclass
class Decomposition:
    position: int
    label: str
    rows: list[Row]
    actual: float  # the model's real value of the direction at this position
    extras: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(r.value for r in self.rows)

    def by_kind(self) -> dict[str, float]:
        """Sum of direct contributions per operator type (mixers) plus 'mlp' and 'embed'."""
        out: dict[str, float] = defaultdict(float)
        for r in self.rows:
            key = r.kind if r.part == "mixer" else r.part
            out[key] += r.value
        return dict(out)

    def top(self, k: int = 10) -> list[Row]:
        return sorted(self.rows, key=lambda r: -abs(r.value))[:k]

    def show(self, k: int = 12) -> None:
        print(f"{self.label} @ position {self.position}: actual={self.actual:+.3f} "
              f"sum-of-rows={self.total:+.3f}")
        agg = self.by_kind()
        print("  by type: " + "  ".join(f"{key}={val:+.3f}" for key, val in sorted(agg.items())))
        for r in self.top(k):
            where = "embed" if r.block is None else f"L{r.block:<2} {r.kind:<4} {r.part}"
            print(f"    {where:<18} {r.value:+.3f}")


def decompose_writes(hm: HyenaModel, writes: Writes, direction: torch.Tensor, index: int, label: str,
                     token_weights: torch.Tensor | None = None) -> Decomposition:
    """Project every write at captured slot `index` onto `direction` through
    the frozen final norm. `actual` is read from the model's real logits when
    token_weights is given, so sum-of-rows vs actual is an independent check."""
    s = _frozen_norm_factor(hm, writes.final[index:index + 1])[0]
    d = s * direction  # write . d  ==  norm(write) . direction

    rows = [Row(None, "embed", "embed", float(writes.embed[index] @ d))]
    for (b, part), w in sorted(writes.parts.items()):
        rows.append(Row(b, hm.kind(b), part, float(w[index] @ d)))
    if token_weights is not None:
        actual = float(writes.logits[index] @ token_weights)
    else:
        actual = float(writes.final[index] @ d)
    return Decomposition(position=writes.positions[index], label=label, rows=rows, actual=actual)


def normed_direction(hm: HyenaModel, seq: str, index: int, target: str | None = None,
                     baseline: str | None = None) -> torch.Tensor:
    """The logit-difference direction for the letter at `index`, pre-multiplied
    by the frozen final-norm factor at that position -- dot any residual write
    at position index-1 with this to get its direct logit contribution. Feed it
    to distance.show_profile(direction=...)."""
    target = target or seq[index]
    writes = capture_writes(hm, hm.ids(seq[:index]), [index - 1], blocks=[])
    direction, _ = logit_direction(hm, target, baseline)
    return _frozen_norm_factor(hm, writes.final)[0] * direction


def decompose_prediction(hm: HyenaModel, seq: str, index: int, target: str | None = None,
                         baseline: str | None = None) -> Decomposition:
    """Explain the model's prediction FOR THE LETTER AT `index` (read from the
    logits at index-1). target defaults to the real letter; baseline defaults
    to the mean of the other three bases.

        d = decompose_prediction(hm, seq, 5000)
        d.show()      # which blocks / operator types pushed the right letter up
    """
    if index < 1:
        raise ValueError("index must be >= 1 (letter 0 has no prediction)")
    target = target or seq[index]
    pos = index - 1
    writes = capture_writes(hm, hm.ids(seq), [pos])
    direction, tw = logit_direction(hm, target, baseline)
    label = f"logit[{target}] - " + (f"logit[{baseline}]" if baseline else "mean(other bases)")
    dec = decompose_writes(hm, writes, direction, 0, label, token_weights=tw)
    dec.extras["reconstruction_error"] = writes.reconstruction_error()
    return dec
