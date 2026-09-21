"""Round 3: measure at the INPUT of the bottleneck block.

Round 2 showed Evo 2 7B's output is a function of block 30's output alone:
its write is ~1e5x every other write, so in bf16 everything written before it
is rounded away (RESEARCH_LOG.md, finding 5). Direct logit attribution
therefore always says "block 30: 100%". The informative question is what
block 30 READS: the residual stream entering it,

    u = embed + sum over blocks b < L of (mixer_b + mlp_b)

We attribute f(u) = logit difference for one letter to those writes with
INTEGRATED GRADIENTS along the straight path from a baseline (embedding only)
to the real u. Because u is a sum of writes, the attribution splits exactly
over writes and positions:

    attr(write w at position p) = w[p] . avg_grad[p],   avg_grad = mean of df/du along the path

and sum(attr) ~= f(u) - f(baseline) ("completeness", reported as a check).
Blocks L.. are run explicitly as the function f, so this is exact up to the
path discretisation -- unlike round 1's direct attribution, it sees INDIRECT
influence through the bottleneck's nonlinearity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .arch import HyenaModel
from .trace import Row, logit_direction


def _autograd_safe(hm: HyenaModel, layer: int) -> int:
    """Vortex's load_checkpoint casts weights to bf16 inside torch.inference_mode(),
    which turns every parameter into an 'inference tensor' -- and autograd refuses
    to save those for backward ("Inference tensors cannot be saved for backward").
    Replace such parameters/buffers in the modules the gradient flows through
    with ordinary copies (same values). Returns how many tensors were replaced."""
    mods = [hm.block(b) for b in range(layer, hm.n_blocks)] + [hm.model.norm, hm.model.embedding_layer]
    n = 0
    for top in mods:
        for m in top.modules():
            for name, p in list(m._parameters.items()):
                if p is not None and p.is_inference():
                    m._parameters[name] = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
                    n += 1
            for name, b in list(m._buffers.items()):
                if b is not None and b.is_inference():
                    m._buffers[name] = b.clone()
                    n += 1
    return n


def _tail(hm: HyenaModel, u: torch.Tensor, layer: int) -> torch.Tensor:
    """Run blocks layer.. + final norm + unembed on residual u (1, L, H) -> logits (1, L, V)."""
    for b in range(layer, hm.n_blocks):
        blk = hm.block(b)
        u = u.to(next(blk.parameters()).device)
        u, _ = blk(u, inference_params=None, padding_mask=None)
    u = u.to(hm.device)
    return hm.model.unembed(hm.model.norm(u))


@dataclass
class InterfaceAttribution:
    layer: int
    position: int  # residual position read (predicts letter position+1)
    rows: list[Row]  # per (block < layer, part) + embedding(0 by construction)
    by_distance: dict[int, float]  # letters back from `position` -> total attribution at that position
    f_actual: float
    f_baseline: float
    extras: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(r.value for r in self.rows)

    @property
    def completeness_error(self) -> float:
        """|sum(attr) - (f(u) - f(base))| / |f(u) - f(base)|: integrated-gradients self-check."""
        d = self.f_actual - self.f_baseline
        return abs(self.total - d) / max(abs(d), 1e-9)

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rows:
            key = r.kind if r.part == "mixer" else r.part
            out[key] = out.get(key, 0.0) + r.value
        return out

    def top(self, k: int = 10) -> list[Row]:
        return sorted(self.rows, key=lambda r: -abs(r.value))[:k]

    def show(self, k: int = 10) -> None:
        print(f"block-{self.layer} input attribution @ position {self.position}: f(u)={self.f_actual:+.3f} "
              f"f(base)={self.f_baseline:+.3f} sum(attr)={self.total:+.3f} completeness_err={self.completeness_error:.1%}")
        print("  by type: " + "  ".join(f"{a}={b:+.3f}" for a, b in sorted(self.by_kind().items())))
        for r in self.top(k):
            print(f"    L{r.block:<2} {r.kind:<4} {r.part:<5} {r.value:+.3f}")
        near = sorted(self.by_distance.items())[:12]
        print("  by letters back: " + " ".join(f"{d}:{v:+.2f}" for d, v in near))


def block_input_attribution(hm: HyenaModel, seq: str, index: int, layer: int, target: str | None = None,
                            baseline_letter: str | None = None, steps: int = 32,
                            max_distance: int = 64) -> InterfaceAttribution:
    """Attribute the model's preference for the letter at `index` (logit of the
    real letter minus mean of the other three) to every write entering block `layer`."""
    if index < 1:
        raise ValueError("index must be >= 1")
    target = target or seq[index]
    pos = index - 1
    ids = hm.ids(seq[: index])  # causal: nothing after pos matters
    comps = [(b, p) for b in range(layer) for p in ("mixer", "mlp")]

    # pass 1: residual entering `layer` and the embedding (the baseline)
    store = {}

    def pre_hook(_m, args):
        store["u"] = args[0].detach()

    def emb_hook(_m, _i, out):
        store["e"] = out.detach()

    h1 = hm.block(layer).register_forward_pre_hook(pre_hook)
    h2 = hm.model.embedding_layer.register_forward_hook(emb_hook)
    try:
        with torch.no_grad():
            hm.model(ids)
    finally:
        h1.remove()
        h2.remove()
    u, e = store["u"], store["e"].to(store["u"].device)

    _autograd_safe(hm, layer)
    u, e = u.clone(), e.clone()  # hooks ran under no_grad; make sure neither is an inference tensor
    direction, tw = logit_direction(hm, target, baseline_letter)
    tw = tw.to(hm.device)

    def f(x):
        return (_tail(hm, x, layer)[0, pos].float() * tw).sum()

    # integrated gradients along e -> u (midpoint rule)
    grad_sum = torch.zeros_like(u, dtype=torch.float32)
    for s in range(steps):
        a = (s + 0.5) / steps
        x = (e + a * (u - e)).detach().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(f(x), x)
        grad_sum += g.float()
    avg_grad = grad_sum / steps
    with torch.no_grad():
        f_u, f_e = float(f(u)), float(f(e))

    # pass 2: per-write attribution computed inside hooks (no full-sequence storage)
    attr: dict[tuple[int, str], float] = {}
    handles = []

    def make(c):
        def hook(_m, _i, output):
            w = (output[0] if isinstance(output, tuple) else output)[0].float()
            attr[c] = float((w * avg_grad[0].to(w.device)).sum())
        return hook

    try:
        for c in comps:
            handles.append(hm.component(*c).register_forward_hook(make(c)))
        with torch.no_grad():
            hm.model(ids)
    finally:
        for h in handles:
            h.remove()

    rows = [Row(b, hm.kind(b), p, attr[(b, p)]) for (b, p) in comps]
    per_pos = ((u - e).float() * avg_grad)[0].sum(-1)  # (L,)
    by_distance = {d: float(per_pos[pos - d]) for d in range(0, min(max_distance, pos) + 1)}
    return InterfaceAttribution(layer, pos, rows, by_distance, f_u, f_e,
                                extras={"far_total": float(per_pos[: max(0, pos - max_distance)].sum())})
