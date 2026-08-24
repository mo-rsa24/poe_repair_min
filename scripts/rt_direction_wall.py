#!/usr/bin/env python
"""The zero wall: r_t's direction is shared nowhere except inside a single run.

Candidate panel B for register slot F3. Panel A (correction_size_vs_run_position.py) shows the
correction's SIZE follows one schedule for every pair. This panel shows the
complement that stops a reader over-reading it: the correction's DIRECTION is
not shared at all. Same pair re-run from different noise: cosine 0. Different
pairs: cosine 0. The only direction agreement anywhere is between adjacent
steps of one run, and even that is pair-dependent (eagle x hawk is a smooth
field, cat x dog alternates sign mid-run). Together the two panels say: what
transfers is the rule that produces the correction, never the vector itself.

Three groups, cosine at matched denoising steps, fp32 (fp16 cache upcast
before any arithmetic, per the F6 note):

    same run, adjacent steps   step t vs t+1 within one trajectory
    same pair, different runs  seed 9 vs seed 13, matched steps
    different pairs            all combos of the 8 dose pairs, seed 9

One large dot per comparison (its median over steps), faint cloud behind it
(every per-step value). Reads only the training cache; no GPU, no sampling.

    python scripts/rt_direction_wall.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import load_cell

# The dose sweep's 8 pairs: the same population every F2 number is read over.
PAIRS = (
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus", "a_goose__x__a_swan", "a_cow__x__a_buffalo",
    "a_cat__x__a_dog", "an_elephant__x__a_penguin",
)
SEEDS = (9, 13)          # 13 = the wrong-seed control's donor (seed + 4)
OUT_DIR = Path("outputs/interaction_term/direction_wall")

BLUE = "#1f77b4"         # cat x dog, the running example, as in F2/F3
RED = "#d62728"          # elephant x penguin, the not-alike pair, as in F3
INK = "#333333"

SHORT = {p: p.replace("a_", "").replace("an_", "").replace("__x__", " x ").replace("_", " ")
         for p in PAIRS}


def stack(pair: str, seed: int) -> torch.Tensor:
    return load_cell(pair, seed).r_t().float().flatten(1)   # [T, D]


def cos(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.cosine_similarity(a, b, dim=1).numpy()


def main() -> int:
    stacks = {(p, s): stack(p, s) for p in PAIRS for s in SEEDS}

    groups = {}   # name -> list of (label, per-step values)
    groups["same run,\nadjacent steps"] = [
        (SHORT[p], cos(stacks[(p, s)][:-1], stacks[(p, s)][1:]))
        for p in PAIRS for s in SEEDS
    ]
    groups["same pair,\ndifferent runs"] = [
        (SHORT[p], cos(stacks[(p, 9)], stacks[(p, 13)])) for p in PAIRS
    ]
    groups["different pairs"] = [
        (f"{SHORT[a]} / {SHORT[b]}", cos(stacks[(a, 9)], stacks[(b, 9)]))
        for a, b in combinations(PAIRS, 2)
    ]

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    rng = np.random.default_rng(0)   # jitter only; no statistic depends on it
    ys, names = [], []
    for gi, (gname, rows) in enumerate(groups.items()):
        y = len(groups) - 1 - gi
        ys.append(y); names.append(gname)
        for label, vals in rows:
            # A comparison ACROSS two pairs belongs to neither, so the named
            # colours apply only where the whole comparison is that pair.
            color = INK if "/" in label else (
                BLUE if label.startswith("cat x dog") else
                RED if label.startswith("elephant") else INK)
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.plot(vals, y + jitter, ".", color=color, markersize=2,
                    alpha=0.10, markeredgewidth=0, zorder=2)
            med = float(np.median(vals))
            ax.plot([med], [y], "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # Direct labels on the comparisons a reader will ask about, no legend.
    top = len(groups) - 1
    for pair, seed_ix, dy, ha in (("an_eagle__x__a_hawk", 0, 0.32, "center"),
                                  ("a_cat__x__a_dog", 0, 0.32, "center")):
        vals = cos(stacks[(pair, 9)][:-1], stacks[(pair, 9)][1:])
        med = float(np.median(vals))
        color = BLUE if pair == "a_cat__x__a_dog" else INK
        ax.annotate(SHORT[pair], (med, top + dy), fontsize=7, color=color,
                    ha=ha, va="bottom")

    ax.axvline(0.0, color="#999999", linewidth=0.8, zorder=1)
    ax.text(-0.04, -0.38, "no direction shared", ha="right", va="center",
            fontsize=7, color="#777777")

    ax.set_yticks(ys[::-1])
    ax.set_yticklabels(list(groups.keys())[::-1], fontsize=8)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.55, len(groups) - 0.1)
    ax.set_xlabel("direction agreement of $r_t$ (cosine, matched steps)",
                  fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "direction_wall.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    sidecar = {
        "pairs": PAIRS, "seeds": SEEDS, "dtype": "fp32 upcast from fp16 cache",
        "groups": {
            g: [{"label": lab, "median": float(np.median(v)),
                 "n_steps": int(len(v))} for lab, v in rows]
            for g, rows in groups.items()
        },
    }
    (OUT_DIR / "direction_wall.json").write_text(json.dumps(sidecar, indent=2))
    for g, rows in groups.items():
        meds = [float(np.median(v)) for _, v in rows]
        print(f"{g.replace(chr(10), ' '):28s} n={len(rows):2d}  "
              f"median of medians {np.median(meds):+.3f}  "
              f"range [{min(meds):+.3f}, {max(meds):+.3f}]")
    print(f"figure   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
