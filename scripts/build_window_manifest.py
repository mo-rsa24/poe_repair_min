#!/usr/bin/env python
"""Collect the timing sweep into one JSON the inspector's window tab reads.

The tab lets you drag the window across the denoising run and watch the picture
change. To do that it needs, for every (pair, seed, window): the image on disk,
what the scorer said about it, and where that window sits on the compose-rate
curve. This script joins those three, which live in different places, into one
file.

It never generates and never scores. Images come from the sweep
(scripts/mechanism_study/run_window_sweep.sh) and verdicts come from the scorer
(scripts/plot_window_curves.py, which writes window_curves.json). A cell with an
image but no score shows up as scored=false in the manifest rather than being
dropped, so a half-scored sweep is visible in the UI instead of looking complete.

Usage:
    python scripts/build_window_manifest.py
    python scripts/build_window_manifest.py --root <dir> --out <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term import window_grid as wg

WINDOW_ROOT = paths.resolve(paths.WINDOW)
DEFAULT_ROOT = WINDOW_ROOT / "pairs"
DEFAULT_CURVES = WINDOW_ROOT / "window_curves.json"
DEFAULT_OUT = WINDOW_ROOT / "window_inspector_manifest.json"
WIN_RE = re.compile(r"_w(\d+)-(\d+)")

MIN_STEPS = 40


def collect(root: Path, *, min_steps: int = MIN_STEPS) -> tuple[dict, list[dict]]:
    """Walk the sweep output. Returns (cells, skipped).

    cells is {pair: {seed: {"a-b": record}}}. Each record carries the absolute
    image path, the window, and the step count it was sampled at; the caller
    joins the scorer's verdict onto it.
    """
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    skipped: list[dict] = []
    if not root.is_dir():
        return cells, skipped
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir():
            continue
        for seed_dir in sorted(pair_dir.glob("seed_*")):
            seed = seed_dir.name.split("_")[1]
            for run_dir in sorted(seed_dir.glob("teacher_residual_*_w*")):
                m = WIN_RE.search(run_dir.name)
                if not m:
                    continue
                png = run_dir / f"{run_dir.name}.png"
                summary = run_dir / f"summary_{run_dir.name}.json"
                if not png.exists() or not summary.exists():
                    continue
                meta = json.loads(summary.read_text())
                steps = int(meta.get("num_inference_steps", 0))
                key = f"{int(m.group(1))}-{int(m.group(2))}"
                if steps < min_steps:
                    skipped.append({"pair": pair_dir.name, "seed": seed,
                                    "window": key, "num_inference_steps": steps})
                    continue
                cells.setdefault(pair_dir.name, {}).setdefault(seed, {})[key] = {
                    "window": [int(m.group(1)), int(m.group(2))],
                    "image": str(png),
                    "num_inference_steps": steps,
                    "scored": False,
                }
    return cells, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS)
    args = ap.parse_args()

    cells, skipped = collect(args.root, min_steps=args.min_steps)
    if not cells:
        print(f"no sweep images under {args.root}\n"
              f"run scripts/mechanism_study/run_window_sweep.sh first",
              file=sys.stderr)
        return 2

    curve: dict = {}
    n_scored = 0
    if args.curves.is_file():
        cw = json.loads(args.curves.read_text())
        curve = {
            "centres": cw.get("window_centres", []),
            "compose_rate": cw.get("compose_rate", []),
            "n": cw.get("n", []),
            "peak_centre": cw.get("peak_centre"),
            "peak_band": cw.get("peak_band"),
            "scorer": cw.get("scorer"),
        }
        for row in cw.get("scores", []):
            a, b = row["window"]
            rec = (cells.get(row["pair"], {})
                        .get(str(row["seed"]), {})
                        .get(f"{a}-{b}"))
            if rec is not None:
                rec["scored"] = True
                rec["n_instances"] = int(row["n_instances"])
                rec["compose"] = int(row["compose"])
                n_scored += 1
    else:
        print(f"[warn] no scores at {args.curves}; the tab will show images "
              f"with no verdicts. Run scripts/plot_window_curves.py.",
              file=sys.stderr)

    windows = wg.windows()
    n_cells = sum(len(w) for p in cells.values() for w in p.values())
    expected = len(windows) * len(wg.PAIRS) * len(wg.SEEDS)

    manifest = {
        "experiment": "interaction_term/window",
        "num_steps": wg.NUM_STEPS,
        "width": wg.WIDTH,
        "stride": wg.STRIDE,
        "fork_step": wg.FORK_STEP,
        "windows": [list(w) for w in windows],
        "window_keys": [f"{a}-{b}" for a, b in windows],
        "pairs": sorted(cells),
        "seeds_by_pair": {p: sorted(cells[p], key=int) for p in sorted(cells)},
        "curve": curve,
        "cells": cells,
        "n_cells_on_disk": n_cells,
        "n_cells_scored": n_scored,
        "n_cells_planned": expected,
        "skipped_short_runs": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    print(f"{n_cells} of {expected} planned cells on disk, {n_scored} scored")
    if skipped:
        print(f"{len(skipped)} short run(s) left out (under {args.min_steps} steps)")
    print(f"manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
