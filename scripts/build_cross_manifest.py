#!/usr/bin/env python
"""Collect the crossed grid and its per-step frames into one JSON for the tab.

Joins three things that live apart: the images the cross sweep wrote, the frames
decoded from their saved trajectories, and the grid definition that says which
cells were supposed to exist. A cell that has not been sampled yet is reported
as missing rather than omitted, so a half-filled grid looks half-filled in the
UI instead of looking complete.

Usage:
    python scripts/build_cross_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term import cross_grid as cg

ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cross")
DEFAULT_OUT = ROOT / "cross_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    jobs = cg.jobs()
    cells: dict[str, dict[str, dict]] = {}
    n_on_disk = n_with_frames = 0

    for j in jobs:
        d = (args.root / "pairs" / j["pair"] / f"seed_{j['seed']}" / j["cell_id"])
        img = d / "image.png"
        if not img.exists():
            continue
        n_on_disk += 1
        rec = {
            "cell_id": j["cell_id"], "block": j["block"],
            "cond_tag": j["cond_tag"], "corr": j["corr"],
            "cond_window": (None if j["cond_window"] is None
                            else list(j["cond_window"])),
            "corr_window": (None if j["corr_window"] is None
                            else list(j["corr_window"])),
            "lambda_max": j["lambda_max"],
            "image": str(img),
            "frames": [],
        }
        fj = d / "frames" / "frames.json"
        if fj.exists():
            fr = json.loads(fj.read_text())
            rec["frames"] = fr.get("frames", [])
            rec["frame_source"] = fr.get("source", "x0")
            n_with_frames += 1
        cells.setdefault(j["pair"], {}).setdefault(str(j["seed"]), {})[
            j["cell_id"]] = rec

    manifest = {
        "experiment": "interaction_term/cross",
        "num_steps": cg.NUM_STEPS,
        "fork_step": cg.FORK_STEP,
        "width": cg.WIDTH,
        "pairs": sorted(cells),
        "seeds_by_pair": {p: sorted(cells[p], key=int) for p in sorted(cells)},
        "cond_schedules": [
            {"tag": c.tag,
             "window": (None if c.window is None else list(c.window)),
             "outside": c.outside,
             "describe": c.describe()}
            for c in cg.COND_SCHEDULES
        ],
        "corr_columns": list(cg.CROSS_CORRECTIONS),
        "dense_windows": [list(w) for w in cg.dense_windows()],
        "cells": cells,
        "n_cells_planned": len(jobs),
        "n_cells_on_disk": n_on_disk,
        "n_cells_with_frames": n_with_frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"{n_on_disk} of {len(jobs)} cells on disk, {n_with_frames} with frames")
    print(f"manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
