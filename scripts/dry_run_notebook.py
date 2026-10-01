#!/usr/bin/env python3
"""Run a Colab notebook's logic locally, against the tiny CPU model.

Evo 2 needs Linux + CUDA, so the real model only runs on a GPU box. But most of
what breaks in a notebook is not the model: it is a typo, a wrong column name, a
dict key collision, a missing variable, an f-string that never terminates. Those
cost a GPU session each to find, and nothing to find here.

This swaps `HyenaModel.load` for `tests/tiny_hyena.py`'s stand-in, skips the
install and GPU cells, and executes the rest in order. The numbers it prints are
meaningless -- the stand-in is small and untrained -- but if it reaches the end,
the notebook's code is sound.

Bugs this caught before any GPU time was spent: a missing QUICK setting, genes
not filtered to the forward strand, NaN periodicity, two dict key collisions, a
wrong column name, a case where every ablation condition silently vanished, and
a real newline inside an f-string.

Usage
-----
    python scripts/dry_run_notebook.py notebooks/marv_hyena_round6_colab.ipynb
    python scripts/dry_run_notebook.py notebooks/marv_hyena_round6_colab.ipynb N_SITES=5 QUICK=True
    python scripts/dry_run_notebook.py notebooks/marv_hyena_scaling_20b_h100.ipynb --fake-gpu

Any `NAME=value` argument rewrites a top-level assignment of that name, so you
can shrink a run without editing the notebook. `--fake-gpu` additionally
neutralises the Hopper and Transformer-Engine asserts in the scaling notebooks,
which cannot pass on a laptop.

Needs the E. coli GenBank file the notebooks download. Pass --data DIR to point
at a directory that already has `data/NC_000913.gb`, or let it fetch one into a
scratch directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENOME_URL = ("https://raw.githubusercontent.com/ArcInstitute/evo2/main/"
              "notebooks/sparse_autoencoder/NC_000913.gb")

# Cells that only make sense on a GPU box with network access.
SKIP_MARKERS = ("nvidia-smi", "pip install", "git clone", "noflash.prepare", "pytest")


def prepare_workdir(data_dir: str | None) -> Path:
    work = Path(data_dir) if data_dir else REPO / ".dry_run"
    (work / "data").mkdir(parents=True, exist_ok=True)
    gb = work / "data" / "NC_000913.gb"
    if not gb.exists():
        print(f"fetching {gb} ...")
        urllib.request.urlretrieve(GENOME_URL, gb)
    return work


def patch(src: str, overrides: dict[str, str], fake_gpu: bool, work: Path) -> str:
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!"))
    # Colab's /content does not exist locally, and is read-only on macOS.
    src = src.replace("'/content/", f"'{work}/").replace('"/content/', f'"{work}/')
    if fake_gpu:
        src = src.replace("assert TE_OK, (", "TE_OK = True; assert TE_OK, (")
        src = src.replace("assert any(torch.cuda.get_device_capability",
                          "assert True or any(torch.cuda.get_device_capability")
        src = src.replace("print('CUDA devices:', torch.cuda.device_count())",
                          "print('CUDA devices: 0 (dry run, no GPU)')")
    for name, value in overrides.items():
        src = re.sub(rf"^{re.escape(name)}\s*=.*$", f"{name} = {value}", src, flags=re.M)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("overrides", nargs="*", metavar="NAME=VALUE",
                    help="rewrite a top-level assignment, e.g. QUICK=True N_SITES=5")
    ap.add_argument("--fake-gpu", action="store_true",
                    help="neutralise the Hopper / Transformer-Engine asserts (scaling notebooks)")
    ap.add_argument("--data", default=None, help="working directory holding data/NC_000913.gb")
    # parse_known_args so NAME=VALUE overrides may appear in any order, before or
    # after the flags; argparse otherwise refuses them after a store_true flag.
    args, extra = ap.parse_known_args()
    tokens = list(args.overrides) + list(extra)
    bad = [t for t in tokens if "=" not in t]
    if bad:
        ap.error(f"expected NAME=VALUE overrides, got {bad}")
    overrides = dict(t.split("=", 1) for t in tokens)

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tests"))
    import matplotlib
    matplotlib.use("Agg")  # no display, and no blocking on plt.show()

    from tiny_hyena import make_tiny  # noqa: E402  (needs the path above)

    import marv_hyena as mh  # noqa: E402

    # The whole point: never touch the real checkpoint.
    mh.HyenaModel.load = staticmethod(lambda name="tiny", **kw: mh.HyenaModel(make_tiny()))

    work = prepare_workdir(args.data)
    os.chdir(work)
    print(f"dry run: {args.notebook}\n  cwd {work}\n  overrides {overrides or '(none)'}\n")

    nb = json.loads((REPO / args.notebook).read_text() if not os.path.isabs(args.notebook)
                    else Path(args.notebook).read_text())
    ns: dict = {}
    failed = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(m in src for m in SKIP_MARKERS):
            print(f"--- cell {i}: skipped (install / GPU / tests)")
            continue
        src = patch(src, overrides, args.fake_gpu, work)
        print(f"--- cell {i}")
        sys.stdout.flush()
        t0 = time.time()
        try:
            exec(compile(src, f"cell{i}", "exec"), ns)  # noqa: S102 -- that is the job
        except Exception:
            import traceback
            traceback.print_exc()
            failed = i
            break
        print(f"--- cell {i} ok ({time.time() - t0:.1f}s)")

    if failed is None:
        print("\nDRY RUN COMPLETE -- the notebook's code runs end to end.")
        print("The numbers above are meaningless: the stand-in model is tiny and untrained.")
        return 0
    print(f"\nDRY RUN FAILED in cell {failed}. Fix it before spending GPU time.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
