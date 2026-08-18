#!/usr/bin/env python
"""The opening figure: one prompt, one seed, three ways of asking for it.

Three renders side by side, all of cat x dog, all from the same noise draw:

    left    one model given the joint prompt "a cat and a dog". This is the
            dose sweep's lam=1.00 render, which reproduces the joint
            prediction to within 1.9 grey levels of 255, so it is what the
            base model can already do.
    middle  the plain product of the two single-animal experts, the dose
            sweep's lam=0.00 render. One fused animal.
    right   that same product with a rank-8 LoRA attached, trained on cat x
            dog itself (pooled across its own seeds 1-4), evaluated on seed
            9, which the adapter never trained on. This is a same-pair,
            held-out-seed claim, not the cross-pair transfer F8a measures:
            this adapter has seen this pair during training, just not this
            noise draw. Run k04__ep2000_resumed, project
            prime_lab/poe-repair-cross-seed.

Samples nothing. Every panel is a PNG already on disk.

One draft per seed, because which seed is shown is a real choice and it is
made by looking, in the open, rather than by taking whichever seed came
first. The seed shown in the paper is named in the caption together with the
seeds that were looked at.

    python scripts/make_main_row.py

Writes paper/iclr/figures/drafts/main-row-seed{09,10,11,12}.{png,pdf} and one
stacked contact sheet of all four for choosing between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12)
STEP_DIR = "epoch_2000_step_100000"

DOSE_ROOT = Path("outputs/interaction_term/dose/pairs")
LORA_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/"
                 "task_b_learning_curve/k04__ep2000_resumed/samples/per_epoch")
OUT_DIR = Path("paper/iclr/figures/drafts")

PAPER = "#ffffff"
INK = "#2b2b2b"
MUTED = "#6b6b6b"

LABEL_PT = 8.0
TAG_PT = 6.0

COLUMNS = (
    ("Joint prompt", "mono"),
    ("Product of experts", "poe"),
    ("LoRA adapter", "lora"),
)


def panel_paths(seed: int) -> dict[str, Path]:
    """The three files one row reads, for one seed."""
    stem = DOSE_ROOT / PAIR / f"seed_{seed}"
    return {
        "mono": stem / "teacher_residual_const_lam100"
                     / "teacher_residual_const_lam100.png",
        "poe": stem / "teacher_residual_const_lam000"
                    / "teacher_residual_const_lam000.png",
        "lora": LORA_ROOT / STEP_DIR / f"sample_seed_{seed:02d}.png",
    }


def draw_row(axes, seed: int, *, column_labels: bool, row_tag: str | None) -> None:
    paths = panel_paths(seed)
    for ax, (label, key) in zip(axes, COLUMNS):
        ax.imshow(imread(paths[key]))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        if column_labels:
            ax.set_xlabel(label, fontsize=LABEL_PT, family="serif", color=INK,
                          labelpad=4)
    if row_tag is not None:
        axes[0].set_ylabel(row_tag, fontsize=LABEL_PT, family="serif",
                           color=MUTED, labelpad=6)


def one_seed(seed: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))
    fig.patch.set_facecolor(PAPER)
    draw_row(axes, seed, column_labels=True, row_tag=None)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.135,
                        wspace=0.035)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"main-row-seed{seed:02d}"
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=PAPER)
    fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}.png and .pdf")


def contact_sheet() -> None:
    fig, grid = plt.subplots(len(SEEDS), 3, figsize=(5.5, 7.5))
    fig.patch.set_facecolor(PAPER)
    for row, seed in enumerate(SEEDS):
        draw_row(grid[row], seed, column_labels=(row == len(SEEDS) - 1),
                 row_tag=f"seed {seed}")
    fig.text(0.995, 0.006, "four candidates for the opening row, choose one",
             fontsize=TAG_PT, family="serif", color=MUTED, ha="right",
             va="bottom")
    fig.subplots_adjust(left=0.045, right=0.992, top=0.995, bottom=0.05,
                        wspace=0.035, hspace=0.035)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "main-row-four-seeds"
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}.png")


def main() -> int:
    missing = [p for seed in SEEDS for p in panel_paths(seed).values()
               if not p.exists()]
    if missing:
        for p in missing:
            print(f"missing: {p}", file=sys.stderr)
        return 1
    for seed in SEEDS:
        one_seed(seed)
    contact_sheet()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
