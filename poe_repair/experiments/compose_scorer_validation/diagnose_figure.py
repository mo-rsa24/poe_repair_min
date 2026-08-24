"""Diagnostic figure for the scorer validation (compose-scorer plan 03, F1 slot).

Because the gate FAILED, this figure diagnoses WHY rather than celebrating a pass.
Two panels:

  (top) A grouped bar chart of d_joint vs min(d_a,d_b) per output, coloured by
        ground truth. The read assumes composes sit NEAR the joint and blends
        collapse onto a single. The bars show the assumption is violated for the
        hard wolf×husky pair: its blends are as-near or nearer the joint than the
        true cat×dog composes are.

  (bottom) The wolf×husky blame case, shown: the corrected blend output beside its
        three anchors, so the eye can see the blend and the joint anchor occupy the
        same single-canine look — which is exactly why whole-image CLS distance
        cannot separate them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs" / "compose_scorer"
ANCHOR_ROOT = OUT_DIR / "anchors"


def main() -> int:
    agr = json.loads((OUT_DIR / "agreement_table.json").read_text())
    rows = agr["rows"]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.15)

    # ---- top: grouped bars, d_joint vs min_single, per output, per space ----
    ax = fig.add_subplot(gs[0, :])
    ids = [r["id"].replace("catdog_compose_compose_seed_", "cd_compose_s")
                    .replace("wolfhusky_sample_seed_", "wh_blend_s")
                    .replace("catdog_poe_blend", "cd_poe_blend") for r in rows]
    x = np.arange(len(rows))
    w = 0.2
    dj_dino = [r["dino"]["d_joint"] for r in rows]
    ms_dino = [min(r["dino"]["d_a"], r["dino"]["d_b"]) for r in rows]
    dj_clip = [r["clip"]["d_joint"] for r in rows]
    ms_clip = [min(r["clip"]["d_a"], r["clip"]["d_b"]) for r in rows]
    ax.bar(x - 1.5 * w, dj_dino, w, label="DINO d_joint", color="#2b6cb0")
    ax.bar(x - 0.5 * w, ms_dino, w, label="DINO min(d_a,d_b)", color="#90cdf4")
    ax.bar(x + 0.5 * w, dj_clip, w, label="CLIP d_joint", color="#c05621")
    ax.bar(x + 1.5 * w, ms_clip, w, label="CLIP min(d_a,d_b)", color="#fbd38d")
    ax.set_xticks(x)
    truth = [r["truth"] for r in rows]
    ax.set_xticklabels([f"{i}\n({t})" for i, t in zip(ids, truth)], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("cosine distance")
    ax.set_title("Scorer null: for a COMPOSE the joint should be nearer than either single "
                 "(d_joint < min). It is NOT for the wolf×husky blends — they sit as near the "
                 "joint as the true cat×dog composes.", fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.3)

    # ---- bottom: the wolf×husky blame case beside its anchors ----
    wh = ANCHOR_ROOT / "a_wolf__x__a_husky"
    blame_out = REPO / ("artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/"
                        "heldout_pair/a_wolf__x__a_husky/sample_seed_09.png")
    panels = [
        (blame_out, "corrected output\n(truth: BLEND — one animal)"),
        (wh / "anchor_a_alone.png", "anchor: a wolf"),
        (wh / "anchor_b_alone.png", "anchor: a husky"),
        (wh / "anchor_joint.png", "anchor: a wolf and a husky"),
    ]
    for i, (p, cap) in enumerate(panels):
        axi = fig.add_subplot(gs[1, i])
        axi.imshow(Image.open(p).convert("RGB"))
        axi.set_title(cap, fontsize=8)
        axi.axis("off")

    fig.suptitle("F1 (null read): the compose-scorer cannot separate the wolf×husky blend "
                 "in either DINOv2 or CLIP — blend and joint anchor share the single-canine look.",
                 fontsize=10, y=0.98)
    out = OUT_DIR / "F1_scorer_null.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
