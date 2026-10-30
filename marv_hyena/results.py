"""RESULTS: one append-only table of measurements, one row per number.

Every claim this repo makes is an effect of some intervention on some
component, measured on some task and dataset. Recording each one with its
size, interval, sample size and provenance, as it is computed, lets any
later question ("every LI result on real genomes, with CIs") be a filter
over one file instead of a search through notebooks.

The file is JSON Lines: rows are only ever appended, never rewritten, and it
diffs cleanly in git. Colab: write to Drive or download it with the
notebook's other results.

    t = ResultsTable("results/table.jsonl")
    t.add(model="evo2_7b", component=(30, "mixer"), intervention="mean_ablate",
          task="copy@10000", dataset="E. coli K-12", stats=paired_effect_size(rows))
    t.where(intervention="mean_ablate", task="copy@10000")
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ("model", "component", "intervention", "task", "dataset", "n", "effect")


def _num(x):
    """JSON-safe number: numpy/torch scalars to float, non-finite to None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _component(c) -> str:
    """(block, part) -> 'L30 mixer'; strings (groups such as 'all li mixers') pass through."""
    if isinstance(c, (tuple, list)) and len(c) == 2:
        return f"L{int(c[0])} {c[1]}"
    return str(c)


def git_commit(path: str | Path | None = None) -> str | None:
    """HEAD of the repo containing `path` (default: this package), with '+dirty'
    if it has uncommitted changes. None outside a git checkout (e.g. a pip
    install on Colab): then pass the commit explicitly."""
    where = str(Path(path) if path else Path(__file__).resolve().parent)
    try:
        head = subprocess.run(["git", "-C", where, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", where, "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return head + ("+dirty" if dirty else "")


def _from_stats(stats: dict) -> dict:
    """Pick effect / ci / n out of controls.py's outputs (paired_effect_size,
    cluster_bootstrap, summarize-style dicts)."""
    effect = next((stats[k] for k in ("estimate", "median_diff_nats", "effect") if k in stats), None)
    n = next((stats[k] for k in ("n", "n_rows") if k in stats), None)
    return {"effect": effect, "ci": stats.get("ci95"), "n": n}


class ResultsTable:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, *, model: str, component, intervention: str, task: str, dataset: str,
            n: int | None = None, effect: float | None = None, ci: tuple[float, float] | None = None,
            stats: dict | None = None, unit: str | None = None, checkpoint: str | None = None,
            seed: int | None = None, commit: str | None = "auto", notes: str | None = None,
            **extra) -> dict:
        """Append one measurement and return the row. `stats` (a dict from
        controls.py) fills effect, ci and n unless they are given explicitly.
        Extra keyword arguments are kept under 'extra'."""
        if stats is not None:
            s = _from_stats(stats)
            effect = s["effect"] if effect is None else effect
            ci = s["ci"] if ci is None else ci
            n = s["n"] if n is None else n
        if n is None or effect is None:
            raise ValueError("a result needs n and effect (directly or through stats=)")
        from . import __version__

        row = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model, "checkpoint": checkpoint, "component": _component(component),
            "intervention": intervention, "task": task, "dataset": dataset,
            "n": int(n), "effect": _num(effect), "unit": unit,
            "ci_low": _num(ci[0]) if ci is not None else None,
            "ci_high": _num(ci[1]) if ci is not None else None,
            "seed": seed, "commit": git_commit() if commit == "auto" else commit,
            "marv_hyena": __version__, "notes": notes, "extra": extra or None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True, default=_num) + "\n")
        return row

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def where(self, **match) -> list[dict]:
        """Rows whose fields equal every given value; component accepts (block, part)."""
        if "component" in match:
            match["component"] = _component(match["component"])
        return [r for r in self.rows() if all(r.get(k) == v for k, v in match.items())]
