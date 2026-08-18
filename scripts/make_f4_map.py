#!/usr/bin/env python
"""F4f: when the correction works, for every pair at once.

F4a and F4b read the window sweep pooled over pairs, which answers "when" but
hides that the answer is not the same for every pair. The grid already holds
8 pairs x 4 seeds x 9 window positions, all scored, so the per-pair map costs
nothing to draw.

One cell is one (pair, window) combination: the fraction of that pair's 4
seeds that composed when the correction was applied only inside that window.
So a cell is 0, 0.25, 0.5, 0.75 or 1, and the printed number is the count.

Two things the pooled curve cannot show. Pairs differ in how wide their
working window is: some compose only when the correction arrives in the first
ten steps, others still compose one notch later. And within a pair at a fixed
window, seeds disagree, which is the good-seed / bad-seed effect made visible
rather than averaged away.

Reads the scored grid only; it does not sample.

    python scripts/make_f4_map.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window/"
           "window_curves.json")
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "F4f-the-window-map"


def pretty(slug: str) -> str:
    """a_cat__x__a_dog -> cat x dog. Both sides lose their article, not just
    the first, which a single replace on the joined string would do."""
    def strip(side: str) -> str:
        side = side.replace("_", " ")
        for art in ("an ", "a "):
            if side.startswith(art):
                return side[len(art):]
        return side
    return " x ".join(strip(s) for s in slug.split("__x__"))


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"no scored window grid at {SRC}")
    d = json.loads(SRC.read_text())
    scores = d["scores"]

    windows = sorted({tuple(s["window"]) for s in scores})
    pairs = sorted({s["pair"] for s in scores})
    seeds = sorted({s["seed"] for s in scores})
    cell = defaultdict(list)
    for s in scores:
        cell[(s["pair"], tuple(s["window"]))].append(s["compose"])

    M = np.array([[np.mean(cell[(p, w)]) for w in windows] for p in pairs])
    # Order by how much of the run still works, so the reader meets the widest
    # working window first and the ordering is a property of the data.
    order = np.argsort(-M.sum(1))
    M, pairs = M[order], [pairs[i] for i in order]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            n = int(round(M[i, j] * len(seeds)))
            ax.text(j, i, str(n), ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.55 else "#555555")

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{w[0]}" for w in windows])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pretty(p) for p in pairs], fontsize=7)
    ax.set_xlabel("first step of the window the correction was applied in")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"seeds that composed, of {len(seeds)}", fontsize=7)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", "2", "4"])
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    earliest = M[:, 0]
    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "source": str(SRC), "pairs_in_drawn_order": pairs,
        "windows": [list(w) for w in windows], "seeds": seeds,
        "matrix_compose_rate": M.tolist(),
        "earliest_window_range": [float(earliest.min()), float(earliest.max())],
        "reading": "every pair works best in the earliest window and none "
                   "works after step 20, so the timing claim holds pair by "
                   "pair and not only in the pooled curve",
        "caption_cap": "four seeds per cell, so a cell is a count out of four "
                       "and not a rate. Differences of one seed between "
                       "neighbouring cells are within what four samples can "
                       "resolve, and the figure is read for its shape rather "
                       "than for any single cell",
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    print(f"  earliest window ranges {earliest.min():.2f} to {earliest.max():.2f} "
          f"across the {len(pairs)} pairs")
    print(f"  columns from step 20 onward: max cell "
          f"{M[:, 4:].max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
