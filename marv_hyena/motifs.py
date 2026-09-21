"""Block 0 is a Hyena-SE block reading raw letters. Its output at position i
depends only on the last few letters: the short featurizer FIR (length 3)
feeds the SE filter (length 7), so the receptive field is 3 + 7 - 1 = 9
letters. With only 4 possible letters, that is 4**9 = 262,144 possible
inputs -- few enough to run ALL of them and get an exact, complete
dictionary of what every block-0 channel detects. Nothing is sampled.

`receptive_field` measures the field empirically first, so the enumeration
never relies on the arithmetic above being right for the installed model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .arch import BASES, HyenaModel


@torch.no_grad()
def _block0_filter_out(hm: HyenaModel, ids: torch.Tensor) -> torch.Tensor:
    """HyenaCascade output of block 0 (before out_filter_dense), (N, L, H)."""
    blk = hm.block(0)
    x = hm.model.embedding_layer(ids)
    z = blk.projections(blk.pre_norm(x))
    z = z[0] if isinstance(z, tuple) else z
    y, _ = blk.filter(z)
    return y


@torch.no_grad()
def receptive_field(hm: HyenaModel, max_len: int = 32, seed: int = 0) -> int:
    """Smallest k such that letters more than k-1 back never change block 0's
    output at the last position (checked by randomising the earlier letters)."""
    if hm.kind(0) != "se":
        raise ValueError(f"block 0 is {hm.kind(0)}, not se; enumeration assumes a short filter")
    g = torch.Generator().manual_seed(seed)
    letters = torch.tensor([ord(b) for b in BASES])
    base = letters[torch.randint(0, 4, (1, max_len), generator=g)].to(hm.device)
    ref = _block0_filter_out(hm, base)[0, -1]
    for k in range(1, max_len + 1):
        alt = base.clone()
        alt[0, : max_len - k] = letters[torch.randint(0, 4, (max_len - k,), generator=g)].to(hm.device)
        if not torch.allclose(_block0_filter_out(hm, alt)[0, -1], ref, atol=1e-3, rtol=1e-3):
            continue
        # confirm with a few more random prefixes
        ok = True
        for _ in range(4):
            alt[0, : max_len - k] = letters[torch.randint(0, 4, (max_len - k,), generator=g)].to(hm.device)
            ok &= torch.allclose(_block0_filter_out(hm, alt)[0, -1], ref, atol=1e-3, rtol=1e-3)
        if ok:
            return k
    raise RuntimeError(f"receptive field larger than {max_len}")


@dataclass
class MotifDictionary:
    k: int
    top_kmers: list[list[str]]  # per channel, strongest positive responders
    top_vals: np.ndarray  # (H, n_top)
    bottom_kmers: list[list[str]]  # strongest negative responders
    bottom_vals: np.ndarray
    # (k, H): how much each input position matters to each channel. Exact
    # main-effect variance over the COMPLETE enumeration: for position p, the
    # variance across the 4 letters of the mean output given that letter.
    # Columns sum to 1 (share of the channel's position-attributable variance).
    position_importance: np.ndarray | None = None

    def pwm(self, channel: int, n: int = 100, negative: bool = False) -> np.ndarray:
        """(k, 4) letter frequencies over the channel's top-n k-mers (A, C, G, T)."""
        kmers = (self.bottom_kmers if negative else self.top_kmers)[channel][:n]
        m = np.zeros((self.k, 4))
        for s in kmers:
            for i, ch in enumerate(s):
                m[i, BASES.index(ch)] += 1
        return m / max(1, len(kmers))

    def consensus(self, channel: int, n: int = 100) -> str:
        m = self.pwm(channel, n)
        return "".join(BASES[j] if m[i, j] >= 0.5 else "N" for i, j in enumerate(m.argmax(1)))


@torch.no_grad()
def enumerate_block0(hm: HyenaModel, k: int | None = None, n_top: int = 200, batch: int = 4096) -> MotifDictionary:
    """Run every k-mer through block 0 and keep, per channel, the n_top
    strongest positive and negative responses at the last position."""
    k = k or receptive_field(hm)
    letters = torch.tensor([ord(b) for b in BASES], device=hm.device)
    H = hm.hidden_size
    top_v = torch.full((H, n_top), -float("inf"), device=hm.device)
    top_i = torch.zeros((H, n_top), dtype=torch.long, device=hm.device)
    bot_v = torch.full((H, n_top), float("inf"), device=hm.device)
    bot_i = torch.zeros((H, n_top), dtype=torch.long, device=hm.device)
    total = 4 ** k
    powers = 4 ** torch.arange(k - 1, -1, -1, device=hm.device)
    letter_sums = torch.zeros((k, 4, H), device=hm.device, dtype=torch.float64)
    for s in range(0, total, batch):
        idx = torch.arange(s, min(s + batch, total), device=hm.device)
        digits = (idx[:, None] // powers[None, :]) % 4
        out = _block0_filter_out(hm, letters[digits])[:, -1].float()  # (B, H)
        onehot = torch.nn.functional.one_hot(digits, 4).to(torch.float64)  # (B, k, 4)
        letter_sums += torch.einsum("bpl,bh->plh", onehot, out.double())
        v = torch.cat([top_v, out.T], 1)
        i = torch.cat([top_i, idx[None].expand(H, -1)], 1)
        top_v, sel = v.topk(n_top, dim=1)
        top_i = i.gather(1, sel)
        v = torch.cat([bot_v, out.T], 1)
        i = torch.cat([bot_i, idx[None].expand(H, -1)], 1)
        bot_v, sel = (-v).topk(n_top, dim=1)
        bot_v = -bot_v
        bot_i = i.gather(1, sel)

    def to_str(n: int) -> str:
        return "".join(BASES[(n // 4 ** (k - 1 - j)) % 4] for j in range(k))

    means = letter_sums / (total // 4)  # every letter appears total/4 times at each position
    imp = means.var(dim=1, unbiased=False)  # (k, H)
    imp = (imp / imp.sum(0, keepdim=True).clamp_min(1e-30)).float().cpu().numpy()

    return MotifDictionary(
        k=k,
        top_kmers=[[to_str(int(n)) for n in row] for row in top_i.cpu()],
        top_vals=top_v.cpu().numpy(),
        bottom_kmers=[[to_str(int(n)) for n in row] for row in bot_i.cpu()],
        bottom_vals=bot_v.cpu().numpy(),
        position_importance=imp,
    )


@dataclass
class MotifHit:
    channel: int
    observed: float  # share of the channel's top-n k-mers containing the motif
    expected: float  # chance of that, given only the channel's letter composition
    enrichment: float


def _p_contains(motif: str, comp: np.ndarray, k: int) -> float:
    """P(a random k-mer with letter frequencies `comp` contains `motif`),
    approximating occurrences at different offsets as independent."""
    q = float(np.prod([comp[BASES.index(ch)] for ch in motif]))
    return 1.0 - (1.0 - q) ** max(1, k - len(motif) + 1)


def motif_channels(md: MotifDictionary, motif: str, n_top: int = 50, min_frac: float = 0.8,
                   min_enrichment: float = 3.0) -> tuple[list[MotifHit], list[MotifHit]]:
    """Channels whose top k-mers contain `motif`, with a composition control.

    Round 1 counted a channel as a 'TAA channel' if >= 80% of its top k-mers
    contained TAA. But an AT-rich channel's top k-mers contain TAA often by
    chance. Here `expected` is how often TAA would show up in random k-mers
    with the channel's OWN letter composition (conservative: that composition
    includes the motif's letters). Returns (raw_hits, controlled_hits);
    controlled also requires observed/expected >= min_enrichment.
    """
    raw, ctrl = [], []
    for c in range(len(md.top_kmers)):
        kmers = md.top_kmers[c][:n_top]
        obs = float(np.mean([motif in s for s in kmers]))
        if obs < min_frac:
            continue
        joined = "".join(kmers)
        comp = np.array([joined.count(b) for b in BASES], dtype=float) / len(joined)
        exp = _p_contains(motif, comp, md.k)
        hit = MotifHit(c, obs, exp, obs / max(exp, 1e-9))
        raw.append(hit)
        if hit.enrichment >= min_enrichment:
            ctrl.append(hit)
    return raw, ctrl
