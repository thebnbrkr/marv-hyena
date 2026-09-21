"""Test 4 (PREDICTIONS.md P4): how far each Hyena block looks, from weights only.

CPU-only mode (works on a laptop, no CUDA, no model instantiation):
    python scripts/filter_reach.py --checkpoint ~/.cache/huggingface/evo2_7b.pt \
        --config ../evo2-main/evo2/configs/evo2-7b-1m.yml

Live mode (on the GPU box, after the model is loaded anyway):
    python scripts/filter_reach.py --model evo2_7b

The evo2 loader merges checkpoint shards into <HF_HOME>/evo2_7b.pt; that file
is what --checkpoint wants.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import marv_hyena as mh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--model")
    ap.add_argument("--max-len", type=int, default=1_000_000)
    ap.add_argument("--out", default="filter_reach.json")
    args = ap.parse_args()

    if args.checkpoint:
        import torch
        import yaml

        sd = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        cfg = yaml.safe_load(open(args.config))
        rows = mh.reach_table_from_checkpoint(sd, cfg, max_len=args.max_len)
    elif args.model:
        rows = mh.reach_table(mh.HyenaModel.load(args.model), max_len=args.max_len)
    else:
        ap.error("give --checkpoint + --config, or --model")

    mh.show_reach(rows)
    Path(args.out).write_text(json.dumps(
        [{"block": r.block, "kind": r.kind, "reach50": r.reach50, "reach90": r.reach90, "reach99": r.reach99,
          "max_reach99": r.max_reach99, "per_channel99": r.per_channel99.tolist()} for r in rows]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
