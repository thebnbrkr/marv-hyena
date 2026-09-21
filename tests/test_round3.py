"""Round-3 tools, each checked against an exact or brute-force answer on the tiny model."""
import itertools

import numpy as np
import torch

import marv_hyena as mh
from marv_hyena import diagnostics, interface, motifs


def test_interface_attribution_is_complete(hm, seq):
    """Integrated gradients: sum of attributions == f(u) - f(embedding-only) (up to path discretisation),
    and the per-write split sums exactly to the per-position split."""
    a = interface.block_input_attribution(hm, seq, 200, layer=6, steps=64)
    assert a.completeness_error < 0.02, a.completeness_error
    assert len(a.rows) == 2 * 6
    per_pos_total = sum(a.by_distance.values()) + a.extras["far_total"]
    assert abs(per_pos_total - a.total) < 1e-3 * max(1.0, abs(a.total))


def test_interface_tail_matches_model(hm, seq):
    ids = hm.ids(seq[:100])
    store = {}
    h = hm.block(5).register_forward_pre_hook(lambda m, args: store.setdefault("u", args[0].detach()))
    full = hm.logits(ids)
    h.remove()
    tail = interface._tail(hm, store["u"], 5)[0]
    assert torch.allclose(tail.float(), full, atol=1e-4)


def test_residual_patch_by_depth_endpoints(hm, seq):
    v = mh.make_variant(seq, 150, seq[150], "A" if seq[150] != "A" else "C", window=300)
    rows = mh.variants.residual_patch_by_depth(hm, v, span=60)
    assert rows[0]["depth"] == -1 and abs(rows[0]["fraction"] - 1.0) < 1e-5  # patching the letter itself = ALT run
    assert len(rows) == hm.n_blocks + 1


def test_find_load_bearing_and_precision(hm, seq):
    lb = diagnostics.find_load_bearing(hm, seq[:200])
    assert len(lb) == hm.n_blocks and all("broken" in r for r in lb)
    pc = diagnostics.precision_check(hm, seq, block=6)
    assert abs(pc["norm_fp32"] - pc["norm_bf16"]) / pc["norm_bf16"] < 1e-4  # tiny model is fp32 already


def test_total_effect_matches_brute_force_and_catches_interactions(hm):
    k = 4
    md = motifs.enumerate_block0(hm, k=k, n_top=5, batch=64, keep_all=True)
    kmers = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    ids = torch.tensor([[ord(ch) for ch in s] for s in kmers])
    y = motifs._block0_filter_out(hm, ids)[:, -1].double().numpy()  # (4^k, H)
    grid = y.reshape(*([4] * k), -1)
    ref = np.stack([grid.var(axis=p).mean(axis=tuple(range(k - 1))) for p in range(k)]) / y.var(axis=0)
    assert np.allclose(md.position_total_effect, ref, atol=2e-3)  # float16 storage
    assert (md.position_total_effect >= 0).all() and (md.position_total_effect.sum(0) > 0).all()


def test_rank_all_words():
    rng = np.random.default_rng(0)
    top = [["".join(rng.choice(list("ACGT"), 9)) for _ in range(50)] for _ in range(3)]
    top[0] = ["CC" + "ATG" + "".join(rng.choice(list("ACGT"), 4)) for _ in range(50)]
    md = motifs.MotifDictionary(9, top, np.zeros((3, 50)), [[], [], []], np.zeros((3, 50)))
    rows = motifs.rank_all_words(md)
    assert len(rows) == 64 and rows[0]["rank"] == 1
    assert next(r for r in rows if r["word"] == "ATG")["controlled"] >= 1
