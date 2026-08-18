#!/usr/bin/env python
"""F4h: one seed, one row per cutoff, a track showing exactly what ran where.

The converse of F4g. Five rows, one growing uncorrected prefix each: row 1 is
plain PoE for steps 0-10 then correction for 11-50, row 2 is plain PoE for
0-20 then correction for 21-50, and so on to row 5, which never corrects at
all. Each row pairs the real sampled image with a track of the 50 denoising
steps: grey where it ran plain PoE, green where the correction was on. Same
track drawing as scripts/make_f4g_strip.py and scripts/window_strip.py, so
green always means the same thing across the three figures.

Single seed by design; the pooled compose-rate curve across all 8 pairs is
the population evidence, kept separate at
outputs/interaction_term/window/growing_window_suffix_curve.png.

    python scripts/make_f4h_strip.py
    python scripts/make_f4h_strip.py --seed 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair.experiments.interaction_term import window_grid as wg  # noqa: E402

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "F4h-too-late-to-fix"

PAIR = "a_cat__x__a_dog"
SEED = 12
INK = "#222222"
ROW_IN = 1.55
TRACK_IN = 0.26
HEADER_IN = 0.42
FIG_W = 1.95


def cell_dir(seed: int, w0: int, w1: int) -> Path:
    tag = f"teacher_residual_const_lam100_w{w0}-{w1}"
    return WINDOW_ROOT / "pairs" / PAIR / f"seed_{seed}" / tag


def cell_png(seed: int, w0: int, w1: int) -> Path:
    d = cell_dir(seed, w0, w1)
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    windows = wg.suffix_windows()
    seed = args.seed

    missing = [w for w in windows if not cell_dir(seed, *w).is_dir()]
    if missing:
        raise SystemExit(f"seed {seed} missing cells: {missing}")

    n = len(windows)
    fig_h = HEADER_IN + n * (ROW_IN + TRACK_IN)
    fig, axes = plt.subplots(
        n * 2, 1, figsize=(FIG_W, fig_h),
        gridspec_kw={"height_ratios": [ROW_IN, TRACK_IN] * n, "hspace": 0.12},
    )

    for i, win in enumerate(windows):
        ax, tr = axes[2 * i], axes[2 * i + 1]
        ax.imshow(Image.open(cell_png(seed, *win)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        label = "off" if win[0] >= wg.NUM_STEPS else f"{win[0]}–50"
        ax.set_ylabel(label, fontsize=8, family="serif", color=INK,
                     rotation=0, ha="right", va="center", labelpad=6)

        tr.set_xlim(0, wg.NUM_STEPS); tr.set_ylim(0, 1)
        tr.set_xticks([]); tr.set_yticks([])
        for s in tr.spines.values():
            s.set_visible(False)
        tr.add_patch(Rectangle((0, 0.2), wg.NUM_STEPS, 0.6,
                               facecolor="0.85", edgecolor="none"))
        start = min(win[0], wg.NUM_STEPS)
        if start < wg.NUM_STEPS:
            tr.add_patch(Rectangle((start, 0.2), wg.NUM_STEPS - start, 0.6,
                                   facecolor="tab:green", edgecolor="none"))
        if i == n - 1:
            tr.text(0, -0.9, "0", fontsize=6, family="serif", color=INK,
                    ha="left", va="top")
            tr.text(wg.NUM_STEPS, -0.9, str(wg.NUM_STEPS), fontsize=6,
                    family="serif", color=INK, ha="right", va="top")

    fig.suptitle(f"seed {seed}, cat × dog\ngrey = plain PoE, green = correction on",
                fontsize=8, family="serif", color=INK, y=1 - 0.06 / fig_h)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
