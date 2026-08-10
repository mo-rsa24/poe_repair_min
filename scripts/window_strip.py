#!/usr/bin/env python
"""The picture half of the timing figure: one cell, every window position.

The curve says a rate; this says what the rate looks like. Each column is the
same pair, the same seed, the same starting noise, differing only in which
stretch of denoising steps the correction was allowed to act in. Under each
column is a small track showing where that window sat, and the scorer's verdict
for that exact picture.

Reads images; it does not sample and it does not score. Verdicts come from
window_curves.json, so a picture in this strip and a point on the curve can
never disagree.

Usage:
    python scripts/window_strip.py --pair a_cat__x__a_dog --seed 9
    python scripts/window_strip.py --pairs a_cat__x__a_dog,a_frog__x__a_toad --seed 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term import window_grid as wg

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
DEFAULT_ROOT = WINDOW_ROOT / "pairs"
DEFAULT_CURVES = WINDOW_ROOT / "window_curves.json"


def image_path(root: Path, pair: str, seed: int, win: tuple[int, int]) -> Path:
    name = f"teacher_residual_const_lam100_w{win[0]}-{win[1]}"
    return root / pair / f"seed_{seed}" / name / f"{name}.png"


def load_verdicts(curves: Path) -> dict[tuple[str, int, tuple[int, int]], dict]:
    if not curves.is_file():
        return {}
    out = {}
    for row in json.loads(curves.read_text()).get("scores", []):
        a, b = row["window"]
        out[(row["pair"], int(row["seed"]), (a, b))] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", help="one pair slug")
    ap.add_argument("--pairs", help="comma-separated slugs, one row each")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.pairs:
        pairs = args.pairs.split(",")
    elif args.pair:
        pairs = [args.pair]
    else:
        pairs = [wg.PAIRS[6]]  # a_cat__x__a_dog, the pair the smoke used

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    windows = wg.windows()
    verdicts = load_verdicts(args.curves)

    n_rows, n_cols = len(pairs), len(windows)
    # Each cell is the picture plus a thin track under it showing the window.
    # The header needs a fixed amount of room whatever the row count, so it is
    # added in inches and converted to a fraction rather than guessed at.
    header_in = 0.85
    row_in = 2.0
    fig_h = header_in + row_in * n_rows
    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(1.55 * n_cols, fig_h),
        gridspec_kw={"height_ratios": [1, 0.13] * n_rows,
                     "hspace": 0.55, "wspace": 0.04},
        squeeze=False,
    )

    missing = 0
    for r, pair in enumerate(pairs):
        for c, win in enumerate(windows):
            ax, tr = axes[2 * r][c], axes[2 * r + 1][c]
            for a in (ax, tr):
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_visible(False)

            p = image_path(args.root, pair, args.seed, win)
            if p.is_file():
                ax.imshow(Image.open(p))
            else:
                missing += 1
                ax.text(0.5, 0.5, "not run", ha="center", va="center",
                        fontsize=7, color="0.6", transform=ax.transAxes)
                ax.set_facecolor("0.94")

            v = verdicts.get((pair, args.seed, win))
            if v is not None:
                composed = bool(v["compose"])
                ax.set_title(
                    f"{'composes' if composed else 'blended'} · "
                    f"{v['n_instances']} inst",
                    fontsize=6.5, pad=2.5,
                    color="#1a7a3c" if composed else "#a33",
                )
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("#1a7a3c" if composed else "#a33")
                    s.set_linewidth(1.6)
            elif p.is_file():
                ax.set_title("not scored", fontsize=6.5, pad=2.5, color="0.55")

            # The window itself, drawn on the 50 steps.
            tr.set_xlim(0, wg.NUM_STEPS); tr.set_ylim(0, 1)
            tr.add_patch(Rectangle((0, 0.15), wg.NUM_STEPS, 0.7,
                                   facecolor="0.88", edgecolor="none"))
            tr.add_patch(Rectangle((win[0], 0.15), win[1] - win[0], 0.7,
                                   facecolor="tab:green", edgecolor="none"))
            tr.axvline(wg.FORK_STEP, color="k", lw=0.9, ls="--")
            tr.set_xlabel(f"{win[0]}–{win[1]}", fontsize=7, labelpad=1.5)

        axes[2 * r][0].set_ylabel(pair.replace("__x__", "\n× "), fontsize=7)

    fig.suptitle(
        f"The correction injected in one window only, seed {args.seed}\n"
        f"green bar = steps where it acts; dashed line = fork step "
        f"{wg.FORK_STEP}; box colour = the scorer's verdict",
        fontsize=9, y=1 - 0.10 / fig_h,
        verticalalignment="top",
    )
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=0.06, left=0.055, right=0.995,
    )

    out = args.out or (
        WINDOW_ROOT / f"window_strip_{'_'.join(pairs)[:60]}_seed{args.seed}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"{n_rows * n_cols} cells, {missing} not yet run")
    print(f"strip: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
