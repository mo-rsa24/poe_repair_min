#!/usr/bin/env python
"""The correction's size follows how far through the run you are.

One panel, one thing to look at: the seventeen pool pairs collapse onto one
shape. The median line and two grey bands against the denoising step. No
frames, no named pairs: the reader looks at the band, the caption draws the
conclusion.

Three choices in here are not cosmetic, and the reasons are in
paper/iclr/what-each-figure-argues.md:

- **The axis is the step, not log-SNR.** Only DDIM is used, so the sampler-
  independent axis buys a reader the paper does not have. Every cell runs the
  same 50-step schedule, so the curves stack with no interpolation at all.
- **Every pair is on the page.** The bands hold all seventeen; nothing is
  chosen for display, so nothing is cherry-picked.
- **No spread number on the plot.** A reader cannot act on a percentage while
  looking at a curve. Those live in the appendix table, printed by
  `scripts/snr_collapse.py --pool --x-axis step`.

The measure itself is imported from snr_collapse rather than repeated, so this
figure cannot drift from the numbers in the appendix table.

Cache-only, no GPU.

Usage:
    python scripts/correction_size_vs_run_position.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import CACHE_ROOT  # noqa: E402
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import curve_for, iter_cells  # noqa: E402

POOL = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"
# Adjacent steps differ by more than the shape does, so the raw curves read as
# jitter and the shared shape is invisible. A rolling median over five steps is
# applied to every drawn line INCLUDING the band, so nothing is smoothed
# relative to anything else, and the window is written on the figure. The
# unsmoothed curves stay in cache_analyses/step_collapse.json.
SMOOTH = 5
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "correction-size-over-the-denoising-run-across-17-pairs"


def smooth(y, w=SMOOTH):
    """Rolling median, window w, edges shrunk rather than padded.

    Padding would invent values at step 0 and step 49, which are the two places
    the figure is read hardest.
    """
    y = np.asarray(y, dtype=float)
    h = w // 2
    return np.array([np.median(y[max(0, i - h):min(len(y), i + h + 1)])
                     for i in range(len(y))])


def pair_curves(pairs, *, max_seeds=2):
    """{slug: (per_seed [n_seeds, 50], mean over seeds [50])}."""
    out = {}
    for slug, seed in iter_cells(CACHE_ROOT, list(pairs), max_seeds):
        out.setdefault(slug, []).append(curve_for(slug, seed)[3])
    return {s: (np.stack(v), np.stack(v).mean(0)) for s, v in out.items()}


def main() -> int:
    pool = load_pool(POOL)
    pool_pairs = pool.train + pool.heldout()
    population = pair_curves(pool_pairs)
    if not population:
        print("no cached pool cells found", file=sys.stderr)
        return 2

    stack = np.stack([m for _, m in population.values()])   # [17, 50]
    steps = np.arange(stack.shape[1], dtype=float)
    p10, q25, med, q75, p90 = np.percentile(stack, [10, 25, 50, 75, 90], axis=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(5.5, 2.9))

    ax.fill_between(steps, smooth(p10), smooth(p90), color="0.88", lw=0,
                    label="10th to 90th", zorder=1)
    ax.fill_between(steps, smooth(q25), smooth(q75), color="0.72", lw=0,
                    label="middle half", zorder=2)
    ax.plot(steps, smooth(med), color="0.25", lw=1.8, zorder=3, label="median")

    ax.axhline(1.0, color="0.55", lw=0.6, ls=":", zorder=0)
    ax.set_xlim(steps[0], steps[-1])
    # Set from the data so nothing runs off the top unlabelled: the outer band
    # peaks under 1.5.
    ax.set_ylim(0.1, 1.65)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("relative correction size")
    ax.text(0.0, -0.24, "noise", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.text(1.0, -0.24, "image", transform=ax.transAxes, fontsize=7, color="0.4",
            ha="right")
    ax.text(0.985, 0.03, f"rolling median over {SMOOTH} steps",
            transform=ax.transAxes, fontsize=6, color="0.5", ha="right")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5,
              borderaxespad=0.2, title=f"{stack.shape[0]} pairs")
    ax.get_legend().get_title().set_fontsize(7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR / FIG_NAME}.pdf and .png")

    # What was drawn, so a caption can be checked against it later without
    # rerunning anything.
    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "population_pairs": sorted(population),
        "n_population_pairs": len(population),
        "seeds_per_pair": {s: int(v[0].shape[0]) for s, v in population.items()},
        "x_axis": "denoising step, 0 to 49, DDIM, no interpolation",
        "y_axis": "||r_t|| / ||eps_PoE||, each curve divided by its own median; "
                  "the axis label shortens this to 'relative correction size' "
                  "and the caption carries the definition",
        "smoothing": f"rolling median over {SMOOTH} steps, all drawn elements",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
