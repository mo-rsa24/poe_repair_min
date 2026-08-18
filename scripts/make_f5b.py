#!/usr/bin/env python
"""Register slot F5b: the correction steers the image's meaning from the
start, and the fork step is the midpoint of that drift, not its onset.

One panel plus a frame strip. The curve is the two arms' separation along each
cell's own PoE-to-Mono endpoint axis in CLIP space (the manifold-walk
construction, extended over time): 0 means the arms are indistinguishable,
1 means the full endpoint distance. The seventeen pool pairs are two grey
bands and a median line, same convention as F3. Cat × dog is drawn on top from
the fine 50-step grid, elephant × penguin from the 20-step grid. The fork band
(steps 13 to 20 of 50, measured on raw latents) is shaded: the reader sees the
curves pass through it at roughly half their final separation, already moving.

The x-axis is the schedule timestep, noise on the left, because the population
runs a 20-step schedule and the named example a 50-step one, and the timestep
is the one axis both live on with no interpolation.

Above the curve, two rows of decoded frames from cat × dog seed 9: the
correction-on arm (reproduces Mono) above the correction-off arm (pure PoE),
at steps 0, 16, 33, 49 of the 50-step run. They anchor the axis and show the
two pictures being near-identical early and different late; they are not
evidence for the y-value.

Cache-only, no GPU. Reads the two JSONs written by
scripts/trajectory_divergence.py and scripts/dose_trajectory_divergence.py.

Usage:
    python scripts/make_f5b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANALYSIS_DIR = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses"
    "/trajectory_divergence")
CROSS_FRAMES = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cross/pairs"
    "/a_cat__x__a_dog/seed_9")
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "F5b-gradual-commitment"

NAMED = ("a_cat__x__a_dog", "an_elephant__x__a_penguin")
FRAME_STEPS = (0, 16, 33, 49)
# The 50-step DDIM schedule places step s at timestep 981 - 20s; the fork band
# steps 13..20 and fork step 16 in timestep terms:
FORK_BAND_T = (981 - 20 * 20, 981 - 20 * 13)   # (581, 721)
FORK_T = 981 - 20 * 16                          # 661
SMOOTH_FINE = 5    # 50-point curves, same window as F3
SMOOTH_COARSE = 3  # 20-point curves; 5 would flatten a quarter of the run
SPACE = "clip"


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    h = w // 2
    return np.array([np.median(y[max(0, i - h):min(len(y), i + h + 1)])
                     for i in range(len(y))])


def main() -> int:
    cross = json.loads((ANALYSIS_DIR / "divergence.json").read_text())
    dose = json.loads((ANALYSIS_DIR / "dose_divergence.json").read_text())

    # Population: mean over seeds per pair, pool = every dose pair not named.
    per_pair: dict[str, list[np.ndarray]] = {}
    dose_t = None
    for row in dose["cells"]:
        dose_t = np.array(row["timesteps"], dtype=float)
        per_pair.setdefault(row["pair"], []).append(
            np.array(row[SPACE]["separation"], dtype=float))
    pool = {p: np.stack(v).mean(0) for p, v in per_pair.items()
            if p not in NAMED}
    if len(pool) != 17:
        print(f"expected 17 pool pairs, found {len(pool)}; drawing what is "
              f"there and recording the count in the sidecar", file=sys.stderr)
    stack = np.stack([smooth(c, SMOOTH_COARSE) for c in pool.values()])
    p10, q25, med, q75, p90 = np.percentile(stack, [10, 25, 50, 75, 90], axis=0)

    # Cat × dog from the fine 50-step grid: per-seed faint, mean bold.
    cross_t = None
    cat_curves = []
    for row in cross["cells"]:
        if row["pair"] != "a_cat__x__a_dog":
            continue
        cross_t = 981.0 - 20.0 * np.array(row["steps"], dtype=float)
        cat_curves.append(smooth(row[SPACE]["separation"], SMOOTH_FINE))
    cat_curves = np.stack(cat_curves)

    eleph = smooth(np.stack(per_pair["an_elephant__x__a_penguin"]).mean(0),
                   SMOOTH_COARSE)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7,
    })
    fig = plt.figure(figsize=(5.5, 4.6))
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1, 2.6), hspace=0.12,
                          wspace=0.04)

    for r, (arm, label) in enumerate(
            (("call__rall", "correction on\n(= Mono)"),
             ("call__roff", "correction off\n(PoE)"))):
        for c, s in enumerate(FRAME_STEPS):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(Image.open(CROSS_FRAMES / arm / "frames"
                                 / f"step_{s:03d}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"step {s}", fontsize=7, pad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=6)

    ax = fig.add_subplot(gs[2, :])
    ax.fill_between(dose_t, p10, p90, color="0.88", lw=0,
                    label="pool, 10th to 90th", zorder=1)
    ax.fill_between(dose_t, q25, q75, color="0.72", lw=0,
                    label="pool, middle half", zorder=2)
    ax.plot(dose_t, med, color="0.25", lw=1.6, zorder=3, label="pool median")
    for c in cat_curves:
        ax.plot(cross_t, c, color="C0", lw=0.7, alpha=0.35, zorder=4)
    ax.plot(cross_t, cat_curves.mean(0), color="C0", lw=1.8, zorder=5,
            label="cat × dog (50-step run)")
    ax.plot(dose_t, eleph, color="C1", lw=1.8, zorder=5,
            label="elephant × penguin")
    ax.axvspan(*FORK_BAND_T, color="0.82", alpha=0.55, zorder=0)
    ax.axvline(FORK_T, color="0.45", lw=0.8, ls="--", zorder=6)
    ax.text(FORK_T, 0.06, "fork step,\nfrom raw latents", fontsize=6.5,
            color="0.35", ha="center")
    ax.axhline(0.5, color="0.55", lw=0.6, ls=":", zorder=0)
    ax.axhline(1.0, color="0.55", lw=0.6, ls=":", zorder=0)

    ax.set_xlim(1000, 0)   # noise on the left, image on the right
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("schedule timestep")
    ax.set_ylabel("arm separation along the\nPoE → Mono axis (1 = final)")
    ax.text(0.0, -0.26, "noise", transform=ax.transAxes, fontsize=7,
            color="0.4")
    ax.text(1.0, -0.26, "image", transform=ax.transAxes, fontsize=7,
            color="0.4", ha="right")
    ax.text(0.985, 0.03,
            f"rolling median, {SMOOTH_FINE} steps fine / {SMOOTH_COARSE} coarse",
            transform=ax.transAxes, fontsize=6, color="0.5", ha="right")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5,
              borderaxespad=0.2, title=f"{len(pool)} pool pairs, CLIP space")
    ax.get_legend().get_title().set_fontsize(7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {OUT_DIR / FIG_NAME}.pdf and .png")

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "space": SPACE,
        "population_pairs": sorted(pool),
        "n_population_pairs": len(pool),
        "population_grid": "20-step DDIM, timesteps 951..1, seeds 9 and 10 "
                           "averaged per pair, x0 recovered in closed form",
        "named": {"a_cat__x__a_dog": "50-step grid, seeds 9/10/11, per-seed "
                                     "faint + mean bold",
                  "an_elephant__x__a_penguin": "20-step grid, 2 seeds meaned"},
        "x_axis": "schedule timestep, 1000 (noise) to 0 (image); both grids "
                  "live on it with no interpolation",
        "y_axis": "pos(on) - pos(off) along the cell's own PoE-to-Mono CLIP "
                  "endpoint axis; 0 = indistinguishable, 1 = final separation",
        "fork_band_timesteps": list(FORK_BAND_T),
        "fork_timestep": FORK_T,
        "frames": {"cell": "a_cat__x__a_dog seed_9, 50-step run",
                   "steps": list(FRAME_STEPS),
                   "rows": ["call__rall (= Mono)", "call__roff (PoE)"]},
        "smoothing": {"fine_50_step": SMOOTH_FINE,
                      "coarse_20_step": SMOOTH_COARSE},
        "sources": [str(ANALYSIS_DIR / "divergence.json"),
                    str(ANALYSIS_DIR / "dose_divergence.json")],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
