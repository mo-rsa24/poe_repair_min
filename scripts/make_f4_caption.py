#!/usr/bin/env python
"""Register slot F4.3: the timing cliff read in language instead of by counting.

Experiment 5 of the F4 set. The object detector is the validated read, and it has
been caught twice being wrong on these pictures: it missed a cat beside a dog on
cat x dog seed 12, and it drew three boxes on one fused body for frog x toad at
lambda 0. So the timing result currently rests on a counter with known failures.

This scores the same 36 window images a second way, with no sampling. Each image
is compared against two families of captions, "a cat and a dog" and "a hybrid of
cat and dog", using the same templates and the same CLIP embedding as
`caption_readback.py`, imported rather than restated. The number plotted is the
margin: how much more the picture reads as two animals than as one blended one.
Positive means language calls it two animals.

**What a flat curve would and would not mean.** `caption_readback.py` records that
whole-image CLIP was already nulled in this repo as a way to tell a blend from a
composition, and that anchoring to text is a different use of the space that may
still come back flat. A flat curve here is a limitation of the instrument, not
evidence that the pictures did not change. Only a fall that tracks the counted
one corroborates anything. The asymmetry is deliberate: this experiment can
confirm the cliff and cannot refute it.

    python scripts/make_f4_caption.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_readback import bank_for  # noqa: E402  the same caption templates

WINDOW_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
CURVES = WINDOW_ROOT / "window_curves.json"
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "F4c-the-cliff-in-language"

PAIR = "a_cat__x__a_dog"
PAIR_LABEL = "a cat and a dog"
PROMPT_A, PROMPT_B = "a cat", "a dog"
SEEDS = (9, 10, 11, 12)
SEED_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
INK = "#222222"


def cell_png(seed: int, w0: int, w1: int) -> Path:
    d = WINDOW_ROOT / "pairs" / PAIR / f"seed_{seed}" / \
        f"teacher_residual_const_lam100_w{w0}-{w1}"
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    # The same two helpers caption_readback.py uses. Both return L2-normalised
    # features already, so nothing here renormalises them.
    from poe_repair.experiments.residual_between_mono_and_poe.metrics import (
        clip_image_embed, clip_text_embed,
    )

    doc = json.loads(CURVES.read_text())
    windows = sorted({tuple(r["window"]) for r in doc["scores"]})
    counted = {(r["seed"], tuple(r["window"])): int(r["compose"])
               for r in doc["scores"] if r["pair"] == PAIR}

    bank = bank_for(PROMPT_A, PROMPT_B)
    # Averaged within each caption family, then renormalised: the mean of unit
    # vectors is not a unit vector, and the margin is a difference of two
    # similarities that must be on the same scale.
    with torch.no_grad():
        t_two = clip_text_embed(bank["two"]).mean(0)
        t_blend = clip_text_embed(bank["blend"]).mean(0)
    t_two = t_two / t_two.norm()
    t_blend = t_blend / t_blend.norm()
    print(f"captions   two: {bank['two'][0]!r}")
    print(f"           blend: {bank['blend'][0]!r}")

    paths, keys = [], []
    for seed in SEEDS:
        for win in windows:
            paths.append(str(cell_png(seed, *win)))
            keys.append((seed, win))
    with torch.no_grad():
        feats = torch.cat([clip_image_embed(paths[i:i + 32])
                           for i in range(0, len(paths), 32)])
    margin = (feats @ t_two - feats @ t_blend).cpu().numpy()
    print(f"scored     {len(paths)} images, margin range "
          f"{margin.min():+.3f} to {margin.max():+.3f}")

    by = {k: float(m) for k, m in zip(keys, margin)}
    centres = [(w[0] + w[1]) / 2 for w in windows]

    early = np.mean([by[(s, w)] for s in SEEDS for w in windows if w[0] <= 5])
    late = np.mean([by[(s, w)] for s in SEEDS for w in windows if w[0] >= 20])
    print(f"margin     early windows {early:+.3f}, late windows {late:+.3f}, "
          f"fall {early - late:+.3f}")

    fig = plt.figure(figsize=(5.50, 2.75))
    ax = fig.add_axes([0.165, 0.325, 0.815, 0.545])
    for seed, colour in zip(SEEDS, SEED_COLOURS):
        y = [by[(seed, w)] for w in windows]
        ax.plot(centres, y, color=colour, lw=1.3, marker="o", markersize=2.6,
                label=f"seed {seed}")
        hit = [(c, v) for c, v, w in zip(centres, y, windows) if counted[(seed, w)]]
        if hit:
            ax.plot([c for c, _ in hit], [v for _, v in hit], linestyle="none",
                    marker="o", markersize=6.5, markerfacecolor="none",
                    markeredgecolor=colour, markeredgewidth=1.0)

    ax.axhline(0.0, color="#999999", lw=0.6, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlim(0, 49)
    ax.set_xlabel("denoising step the correction was applied during",
                  fontsize=8, family="serif", color=INK)
    ax.set_ylabel("two animals minus blend\n(CLIP similarity)",
                  fontsize=8, family="serif", color=INK)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper right",
              handlelength=1.4, columnspacing=1.1, borderaxespad=0.2)
    ax.set_title(f"{PAIR_LABEL}: the same cliff, scored in language",
                 fontsize=9, family="serif", color=INK, loc="left", pad=5)
    ax.text(0.0, -0.30,
            "Above the dashed line CLIP prefers \"a cat and a dog\"; below it, "
            "\"a hybrid of cat and dog\". Values are small because the two caption\n"
            "families differ only in their relational words. Ring: the detector "
            "also called that cell two animals.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, family="serif", color="#777777")
    ax.tick_params(labelsize=7, length=2.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family("serif")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6); ax.spines[side].set_color("#444444")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (args.out_dir / f"{args.name}.json").write_text(json.dumps({
        "pair": PAIR, "seeds": list(SEEDS),
        "windows": [list(w) for w in windows],
        "captions": {"two": bank["two"], "blend": bank["blend"]},
        "measure": "mean CLIP similarity to the two-animal captions minus the "
                   "same to the blend captions, image embedding L2-normalised",
        "margin": {f"seed_{s}": [by[(s, w)] for w in windows] for s in SEEDS},
        "detector_composed": {f"seed_{s}": [counted[(s, w)] for w in windows]
                              for s in SEEDS},
        "early_mean": float(early), "late_mean": float(late),
        "limitation": "a flat curve is a CLIP limitation, not evidence the images "
                      "did not change; only a fall corroborates the counted cliff",
    }, indent=2))
    print(f"wrote      {out}, {out.with_suffix('.pdf')} and the sidecar json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
