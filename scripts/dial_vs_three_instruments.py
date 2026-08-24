#!/usr/bin/env python
"""One dial, three instruments that share no machinery.

F2's whole result is read through one object detector, so a reviewer can
reasonably ask whether lambda improves pictures or improves detectability.
This answers with three measurements that use none of the detector's
machinery, so agreement between them is agreement between instruments.

    (a) where the picture sits, in image space. Each generated picture is
        placed on the axis running from that cell's own PoE render to its own
        Mono render in CLIP image space, and the position is read against the
        dose. Same three injected vectors as F2's first three rows.
    (b) what the picture is called, in language space. Each picture is matched
        against a bank of captions and the panel plots how often the best
        match is a two-animal caption rather than a single-hybrid one.
    (c) which way the correction pushes, in the sampler's own space. Not a
        dose curve: for each cell, the correction is compared with the
        direction the sampler is already travelling, summed over the run. One
        point per cell, with three controls beside it.

Panels (a) and (b) share the lambda axis. Panel (c) does not have one, because
the quantity it measures does not depend on the dose, and drawing it on a
lambda axis would imply a relationship that was never measured.

Reads only measurements that already exist, each with its bars written before
the run:
    cache_analyses/manifold_slide_clip.json
    cache_analyses/caption_readback.json
    cache_analyses/plausibility_climb.json

    python scripts/dial_vs_three_instruments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair import paths

CACHE = paths.resolve(paths.CACHE_ANALYSES)
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "F5-one-dial-three-instruments"

# The same colours F2 gives these three injected vectors, so a reader who has
# met them once does not have to learn them again.
OWN, RANDOM, WRONG = "#1f77b4", "#d62728", "#ff7f0e"
ROWS = (("oracle", OWN, "o-", "own $r_t$"),
        ("wrong_pair", WRONG, "^--", "other pair's $r_t$"),
        ("random", RANDOM, "s--", "random, same size"))


def main() -> int:
    for f in ("manifold_slide_clip", "caption_readback", "plausibility_climb"):
        if not (CACHE / f"{f}.json").exists():
            raise SystemExit(f"missing {CACHE / f}.json")
    walk = json.loads((CACHE / "manifold_slide_clip.json").read_text())
    caps = json.loads((CACHE / "caption_readback.json").read_text())
    climb = json.loads((CACHE / "plausibility_climb.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})

    # (a) where the picture sits
    ax = axes[0]
    for row, colour, style, label in ROWS:
        c = walk["curves"][row]
        ax.plot(c["lambdas"], c["mean_projection"], style, color=colour,
                lw=1.4, ms=3.5, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["PoE", "half", "Mono"])
    ax.set_title("a. where the picture sits", fontsize=8, loc="left")
    ax.set_ylabel("position in image space")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.15)

    # (b) what the picture is called
    ax = axes[1]
    for row, colour, style, label in ROWS:
        c = caps["curves"][row]
        ax.plot(c["lambdas"], c["two_rate"], style, color=colour, lw=1.4,
                ms=3.5)
    c = caps["curves"]["oracle"]
    ax.plot(c["lambdas"], c["blend_rate"], ":", color=OWN, lw=1.2, alpha=0.7)
    ax.annotate("called one blended animal", (0.5, c["blend_rate"][2]),
                xytext=(0.06, 0.80), textcoords="data", fontsize=6.2,
                color=OWN, alpha=0.85,
                arrowprops=dict(arrowstyle="-", color=OWN, lw=0.5, alpha=0.6))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title("b. what it is called", fontsize=8, loc="left")
    ax.set_ylabel("called two animals")

    for ax in axes[:2]:
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("$\\lambda$")
        ax.grid(alpha=0.22, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # (c) which way the correction pushes. One point per cell, so the reader
    # sees the spread rather than a bar hiding it.
    ax = axes[2]
    cells = climb["cells"]
    series = [("normalised", "own $r_t$", OWN),
              ("control_shuffled_vs_dx", "wrong step", "#9467bd"),
              ("control_random_vs_dx", "random", RANDOM)]
    rng = np.random.default_rng(0)      # jitter only, no statistic uses it
    for i, (key, label, colour) in enumerate(series):
        v = np.array([c[key] for c in cells], dtype=float)
        y = len(series) - 1 - i
        ax.plot(v, y + rng.uniform(-0.13, 0.13, len(v)), ".", color=colour,
                ms=2.5, alpha=0.35, markeredgewidth=0)
        ax.plot([float(np.median(v))], [y], "o", color=colour, ms=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax.axvline(0.0, color="0.75", lw=0.7)
    ax.set_yticks(range(len(series))[::-1])
    ax.set_yticklabels([s[1] for s in series], fontsize=7)
    ax.set_xlim(-0.55, 1.05)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("agrees with the step")
    # Centred, unlike the other two: this panel's category labels sit outside
    # its axes, so a left-aligned title starts too far right and runs off.
    ax.set_title("c. which way it pushes", fontsize=8, loc="center")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)

    fig.tight_layout(w_pad=1.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    v = np.array([c["normalised"] for c in cells])
    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "panels": {
            "a": {"source": "cache_analyses/manifold_slide_clip.json",
                  "n_cells": walk["n_cells"], "axis": walk["axis"],
                  "bars": walk["bars"], "verdicts": walk["verdicts"]},
            "b": {"source": "cache_analyses/caption_readback.json",
                  "n_cells": caps["n_cells"], "bars": caps["bars"],
                  "verdicts": caps["verdicts"]},
            "c": {"source": "cache_analyses/plausibility_climb.json",
                  "n_cells": climb["n_cells"],
                  "measure": climb["measure"], "caveat": climb["caveat"],
                  "median": float(np.median(v)),
                  "n_negative": int((v < 0).sum())},
        },
        "caption_caps": [
            "panel a: the wrong pair's correction still travels 44% of the "
            "distance the right one travels, against a 50% bar written "
            "beforehand. It passes, barely, and the caption says so.",
            "lambda=1 reproduces the joint prediction by construction "
            "(endpoint drift 1.9 grey levels of 255), so the lambda=1 point "
            "carries no evidence and every dose comparison is read at 0.75.",
            "panel c does not depend on lambda and is not drawn against it; "
            "it is measured along the cached PoE path, so it is the push at "
            "the states PoE visits, not the climb along a corrected path.",
        ],
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    print(f"  a: oracle travels to {walk['curves']['oracle']['mean_projection'][-1]:.3f}, "
          f"wrong pair reaches "
          f"{walk['verdicts']['wrong_pair_fraction_of_oracle_travel']:.1%} of it")
    print(f"  b: two-animal readback {caps['curves']['oracle']['two_rate'][0]:.1%} "
          f"to {caps['curves']['oracle']['two_rate'][-1]:.1%}, crossover at "
          f"lambda {caps['verdicts']['crossover_lambda']}")
    print(f"  c: median {np.median(v):+.3f}, {int((v<0).sum())} of {len(v)} negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
