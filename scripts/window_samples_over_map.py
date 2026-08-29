#!/usr/bin/env python
"""The timing figure in one piece: sample strip above, the all-pairs map below.

The strip and the heatmap were separate images rendered by separate scripts, so
their columns only lined up approximately. This renders both on one canvas with
one shared x axis, so the picture above any count is the picture that count
scored. Top: the final image for one pair (default a_cat__x__a_dog) at seeds
9 to 12, one column per window position. A green frame marks images whose
scored verdict in window_curves.json is composed (instance count >= 2 after
NMS and the size floor); frameless images scored as one animal. Bottom: for
all pairs in the grid, the number of seeds out of 4 scored as composed at each
window start. A dagger marks cells verified by eye as detector error (one
animal boxed twice); the caption carries that sentence.

Reads images and the scored grid only; it does not sample and it does not
score, so this figure and the window curves can never disagree.

    python scripts/window_samples_over_map.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair import paths

WINDOW_ROOT = paths.resolve(paths.WINDOW)
SRC = WINDOW_ROOT / "window_curves.json"
OUT_DIR = Path("paper/iclr/figures/when-the-correction-arrives/poe")
FIG_NAME = "samples-over-the-window-map"

STRIP_PAIR = "a_cat__x__a_dog"
STRIP_SEEDS = [9, 10, 11, 12]
THUMB = 256  # px per cell in the strip mosaic

# Cells verified by eye as detector error: one animal, two kept boxes.
# Verdict and paths in this folder's README.md.
DAGGER_CELLS = {("a_frog__x__a_toad", (35, 45)), ("a_frog__x__a_toad", (40, 50))}

GREEN = "#2e8b57"


def pretty(slug: str) -> str:
    def strip_art(side: str) -> str:
        side = side.replace("_", " ")
        for art in ("an ", "a "):
            if side.startswith(art):
                return side[len(art):]
        return side
    return " x ".join(strip_art(s) for s in slug.split("__x__"))


def image_path(pair: str, seed: int, win: tuple[int, int]) -> Path:
    name = f"teacher_residual_const_lam100_w{win[0]}-{win[1]}"
    return WINDOW_ROOT / "pairs" / pair / f"seed_{seed}" / name / f"{name}.png"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    if not SRC.exists():
        raise SystemExit(f"no scored window grid at {SRC}")
    d = json.loads(SRC.read_text())
    scores = d["scores"]

    width = d.get("width", 10)
    windows = sorted({tuple(s["window"]) for s in scores if s["window"][1] - s["window"][0] == width})
    pairs = sorted({s["pair"] for s in scores})
    # Order rows by how many seeds compose at the earliest window, strongest first,
    # so the map's left edge reads as a ranking rather than an alphabet.
    first = windows[0]
    pairs.sort(key=lambda p: -sum(s["compose"] for s in scores
                                  if s["pair"] == p and tuple(s["window"]) == first))
    ncol = len(windows)

    verdict = {(s["pair"], tuple(s["window"]), s["seed"]): s["compose"] for s in scores}

    counts = defaultdict(int)
    seen = defaultdict(int)
    for s in scores:
        w = tuple(s["window"])
        if w not in windows:
            continue
        seen[(s["pair"], w)] += 1
        counts[(s["pair"], w)] += s["compose"]

    # ---- the strip mosaic ----
    rows = []
    missing = []
    for seed in STRIP_SEEDS:
        row = []
        for win in windows:
            p = image_path(STRIP_PAIR, seed, win)
            if p.exists():
                with Image.open(p) as im:
                    row.append(np.asarray(im.convert("RGB").resize((THUMB, THUMB))))
            else:
                missing.append(str(p))
                row.append(np.full((THUMB, THUMB, 3), 220, dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    mosaic = np.concatenate(rows, axis=0)

    nrow_strip = len(STRIP_SEEDS)
    nrow_map = len(pairs)

    fig_w = 0.92 * ncol
    fig_h = 0.92 * nrow_strip + 0.30 * nrow_map + 0.85
    fig, (ax_strip, ax_map) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [0.92 * nrow_strip, 0.30 * nrow_map]},
        constrained_layout=True,
    )

    ax_strip.imshow(mosaic, extent=(0, ncol, 0, nrow_strip), aspect="auto",
                    interpolation="lanczos")
    for r, seed in enumerate(STRIP_SEEDS):
        y = nrow_strip - 1 - r
        for c, win in enumerate(windows):
            if verdict.get((STRIP_PAIR, win, seed)) == 1:
                ax_strip.add_patch(Rectangle((c + 0.02, y + 0.02), 0.96, 0.96,
                                             fill=False, edgecolor=GREEN, linewidth=2.0))
        ax_strip.text(-0.08, y + 0.5, f"seed {seed}", ha="right", va="center",
                      fontsize=8, transform=ax_strip.transData)
    ax_strip.set_xlim(0, ncol)
    ax_strip.set_ylim(0, nrow_strip)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for side in ax_strip.spines.values():
        side.set_visible(False)
    ax_strip.set_title(f"{pretty(STRIP_PAIR)}, final image per window position",
                       fontsize=9, loc="left")

    # ---- the map ----
    grid = np.zeros((nrow_map, ncol))
    for i, pair in enumerate(pairs):
        for j, win in enumerate(windows):
            grid[i, j] = counts.get((pair, win), 0)
    ax_map.imshow(grid, extent=(0, ncol, 0, nrow_map), aspect="auto",
                  cmap="Blues", vmin=0, vmax=4, origin="upper")
    for i, pair in enumerate(pairs):
        y = nrow_map - 1 - i if False else i  # origin upper: row i drawn top-down
        for j, win in enumerate(windows):
            n = int(grid[i, j])
            dag = "†" if (pair, win) in DAGGER_CELLS else ""
            ax_map.text(j + 0.5, nrow_map - i - 0.5, f"{n}{dag}",
                        ha="center", va="center", fontsize=8,
                        color="white" if n >= 3 else "#333333")
        ax_map.text(-0.08, nrow_map - i - 0.5, pretty(pair), ha="right", va="center",
                    fontsize=8, transform=ax_map.transData)
    ax_map.set_xlim(0, ncol)
    ax_map.set_ylim(0, nrow_map)
    ax_map.set_xticks([j + 0.5 for j in range(ncol)])
    ax_map.set_xticklabels([str(w[0]) for w in windows], fontsize=8)
    ax_map.set_yticks([])
    for side in ax_map.spines.values():
        side.set_visible(False)
    ax_map.set_xlabel("first step of the ten-step window the correction was applied in",
                      fontsize=9)
    ax_map.set_title("seeds composed, of 4", fontsize=9, loc="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=200)
    sidecar = {
        "source": str(SRC),
        "strip_pair": STRIP_PAIR,
        "strip_seeds": STRIP_SEEDS,
        "windows": [list(w) for w in windows],
        "pairs": pairs,
        "cells_per_point": {f"{p} {w}": seen[(p, tuple(w))] for p in pairs for w in windows},
        "counts": {f"{p} {w}": counts[(p, tuple(w))] for p in pairs for w in windows},
        "green_frame_rule": "compose == 1 in window_curves.json, i.e. instance count >= 2 "
                            "after NMS and the size floor",
        "dagger_cells": [f"{p} {w}" for (p, w) in sorted(DAGGER_CELLS)],
        "missing_images": missing,
    }
    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png/.pdf/.json"
          + (f"  ({len(missing)} missing images)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
