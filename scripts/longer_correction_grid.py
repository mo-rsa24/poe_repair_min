#!/usr/bin/env python
"""Does more of the corrected start raise the ceiling?

One seed, drawn as a 5x5 triangular matrix. Rows are the five cutoffs
(0-10, 0-20, 0-30, 0-40, 0-50); columns are the run's five ten-step chunks
(0-10, 10-20, ..., 40-50). Unlike F4a's grid, a cell is not that row's
finished picture: it is what the model's own running estimate of the
finished picture looked like right after that column's chunk of steps, for
that row's run (x̂₀, decoded from the saved trajectory; the last column uses
the real scored image, since a chunk ending at step 50 is the finished
picture, not an estimate of it). A green border marks the chunks that fell
inside the corrected prefix for that row: row 0-10 borders one chunk, row
0-50 borders all five, so the border pattern grows into a triangle while the
pictures show the run's own progress column by column.

Frames come from scripts/decode_trajectory_frames.py (the two cells this
grid shares with F4a) and scripts/recover_growing_window_frames.py (the new
cells, whose x̂₀ is recovered in closed form from the saved noisy latents,
since the current composer no longer saves it directly); this script decodes
nothing itself. The compose-rate curve that backs the quantitative claim
across all 8 pairs is a separate, diagnostic-only plot: see
scripts/plot_growing_window_curves.py and growing_window_curves.json.

    python scripts/longer_correction_grid.py
    python scripts/longer_correction_grid.py --seed 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair.experiments.interaction_term import window_grid as wg  # noqa: E402

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "samples-as-the-correction-runs-longer"

PAIR = "a_cat__x__a_dog"
SEED = 12
INK = "#222222"
CORRECTED_EDGE = "tab:green"
PLAIN_EDGE = "#aaaaaa"

CELL = 0.62
LEFT_GUTTER = 0.95
TOP_BAND = 0.46
RIGHT_PAD = 0.06
BOTTOM_PAD = 0.08
ROW_AXIS_TITLE = "corrected window"
COL_AXIS_TITLE = "picture decoded after step"

CHUNKS = [(i, i + 10) for i in range(0, wg.NUM_STEPS, 10)]  # (0,10) .. (40,50)


def cell_dir(seed: int, w0: int, w1: int) -> Path:
    tag = f"teacher_residual_const_lam100_w{w0}-{w1}"
    return WINDOW_ROOT / "pairs" / PAIR / f"seed_{seed}" / tag


def final_png(seed: int, w0: int, w1: int) -> Path:
    d = cell_dir(seed, w0, w1)
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def frame_png(seed: int, w0: int, w1: int, step: int) -> Path:
    """That row's own running x̂₀ estimate right after this many steps.

    Step 50 is the finished picture itself, not an estimate of it, so it
    reads the real scored image rather than a decoded frame.
    """
    if step >= wg.NUM_STEPS:
        return final_png(seed, w0, w1)
    p = cell_dir(seed, w0, w1) / "frames" / f"step_{step:03d}.png"
    if not p.is_file():
        raise SystemExit(
            f"missing frame {p}\n"
            f"run scripts/decode_trajectory_frames.py or "
            f"scripts/recover_growing_window_frames.py first"
        )
    return p


def contains(window: tuple[int, int], chunk: tuple[int, int]) -> bool:
    return chunk[0] >= window[0] and chunk[1] <= window[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = wg.prefix_windows()
    seed = args.seed

    missing = [w for w in rows if not cell_dir(seed, *w).is_dir()]
    if missing:
        raise SystemExit(f"seed {seed} missing cells: {missing}")

    nrow, ncol = len(rows), len(CHUNKS)
    fig_w = LEFT_GUTTER + ncol * CELL + RIGHT_PAD
    fig_h = TOP_BAND + nrow * CELL + BOTTOM_PAD
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, row_win in enumerate(rows):
        for j, chunk in enumerate(CHUNKS):
            ax = fig.add_axes([
                (LEFT_GUTTER + j * CELL) / fig_w,
                (BOTTOM_PAD + (nrow - 1 - i) * CELL) / fig_h,
                CELL / fig_w, CELL / fig_h,
            ])
            ax.imshow(plt.imread(frame_png(seed, *row_win, chunk[1])))
            ax.set_xticks([]); ax.set_yticks([])
            corrected = contains(row_win, chunk)
            for sp in ax.spines.values():
                sp.set_linewidth(2.2 if corrected else 0.5)
                sp.set_color(CORRECTED_EDGE if corrected else PLAIN_EDGE)

        ax_lab = fig.add_axes([0.24 / fig_w, (BOTTOM_PAD + (nrow - 1 - i) * CELL) / fig_h,
                               (LEFT_GUTTER - 0.24) / fig_w, CELL / fig_h])
        ax_lab.axis("off")
        ax_lab.text(0.92, 0.5, f"{row_win[0]}–{row_win[1]}", ha="right",
                    va="center", fontsize=8, family="serif", color=INK)

    fig.text(0.10 / fig_w, BOTTOM_PAD / fig_h + (nrow * CELL / fig_h) / 2,
             ROW_AXIS_TITLE, ha="center", va="center", rotation=90,
             fontsize=8, family="serif", color=INK)

    for j, chunk in enumerate(CHUNKS):
        ax_top = fig.add_axes([(LEFT_GUTTER + j * CELL) / fig_w,
                               (BOTTOM_PAD + nrow * CELL) / fig_h,
                               CELL / fig_w, 0.22 / fig_h])
        ax_top.axis("off")
        ax_top.text(0.5, 0.05, str(chunk[1]), ha="center", va="bottom",
                    fontsize=8, family="serif", color=INK)

    fig.text(LEFT_GUTTER / fig_w + (ncol * CELL / fig_w) / 2, 1 - 0.10 / fig_h,
             COL_AXIS_TITLE, ha="center", va="top",
             fontsize=8, family="serif", color=INK)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"grid       {nrow} rows x {ncol} step-columns, seed {seed}")
    print(f"size       {fig_w:.2f} x {fig_h:.2f} in")
    print(f"wrote      {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
