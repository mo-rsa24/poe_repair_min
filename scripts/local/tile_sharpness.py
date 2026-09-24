#!/usr/bin/env python3
"""Edge energy per fidelity-harness treatment, so "sharper" is a number rather than an opinion.

Laplacian variance on the greyscale render, the same measure the noise-search finding is judged
on, so a ratio here sits on the scale that finding's 1.10 bar was written for. It counts edges,
not blur: a picture with more contrast or more drawn texture scores higher without holding more
detail, which is why the finding reads it beside the pictures rather than instead of them.

    python3 scripts/local/tile_sharpness.py <tiles_dir> <out.json>

<tiles_dir> is the figure_candidates output tree, <pair>/<seed>/<stem>.png.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

KERNEL = {(-1, 0): 1.0, (1, 0): 1.0, (0, -1): 1.0, (0, 1): 1.0, (0, 0): -4.0}


def laplacian_variance(path: Path) -> float:
    """Variance of the 4-neighbour Laplacian, in units of 1e-4 on a 0-to-1 greyscale."""
    g = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    out = np.zeros_like(g)
    for (dy, dx), w in KERNEL.items():
        out += w * np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    return float(out[1:-1, 1:-1].var()) * 1e4


def treatment_of(stem: str) -> str:
    if stem in ("mono", "poe"):
        return stem
    m = re.match(r"ours_(.+?)_w\d+(?:_c\d+)?$", stem)
    return m.group(1) if m else stem


def main(tiles: Path, out: Path) -> None:
    per_cell: dict[str, dict[str, float]] = {}
    for p in sorted(tiles.rglob("*.png")):
        if "sheets" in p.parts:
            continue
        cell = f"{p.parent.parent.name}/{p.parent.name}"
        per_cell.setdefault(treatment_of(p.stem), {})[cell] = laplacian_variance(p)

    plain = float(np.median(list(per_cell["plain"].values()))) if "plain" in per_cell else None
    summary = {}
    for t, cells in per_cell.items():
        vals = np.array(sorted(cells.values()))
        summary[t] = {"cells": len(vals), "median": round(float(np.median(vals)), 2),
                      "ratio_to_plain": None if plain is None else round(float(np.median(vals)) / plain, 3),
                      "per_cell": {k: round(v, 2) for k, v in sorted(cells.items())}}
    out.write_text(json.dumps({"measure": "Laplacian variance x 1e4, greyscale 0 to 1",
                               "bar": "1.10x the adapter, the bar the noise-search finding used",
                               "treatments": summary}, indent=1))
    for t in ("mono", "poe", "plain", "guid10", "guid12", "reweight", "refine25", "refine40",
              "tail20", "bo8"):
        if t in summary:
            s = summary[t]
            print(f"{t:10s} n={s['cells']:2d} median {s['median']:7.2f}  {s['ratio_to_plain']}x")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
