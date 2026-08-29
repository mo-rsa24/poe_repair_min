#!/usr/bin/env python
"""Show what the detector saw on the two late-window frog x toad cells.

The all-pairs window map has two cells that break its pattern: frog x toad,
seed 10, windows 35-45 and 40-50, each scored composed (instance count 2). An
eyeball read says one animal per image. This renders the detector's own
evidence: every "animal" detection GroundingDINO returns on those two images,
with the scorer's filters applied in the open. Solid boxes are the kept
detections (confidence >= 0.30, longer side >= a quarter of the image, NMS at
IoU 0.5), dashed boxes were rejected, each labelled with its confidence and
longer side. If two solid boxes sit on one body, the cells are instrument
error and the caption's dagger sentence stands.

Runs the same detector and filters as the scorer (count_instances in
detection_scorer.py); it does not change any stored score.

    python scripts/frog_toad_late_window_boxes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair import paths
from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    MIN_BOX_FRACTION, _box_side,
)
from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics

PAIR = "a_frog__x__a_toad"
SEED = 10
WINDOWS = [(35, 45), (40, 50)]
CONF, NMS_IOU = 0.30, 0.5

OUT_DIR = Path("artifacts/results/did-the-detector-miscount-the-late-frog-toad-cells")


def image_path(win: tuple[int, int]) -> Path:
    name = f"teacher_residual_const_lam100_w{win[0]}-{win[1]}"
    return (paths.resolve(paths.WINDOW) / "pairs" / PAIR / f"seed_{SEED}" / name
            / f"{name}.png")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(7.2 * len(WINDOWS), 7.6))
    record = []
    for ax, win in zip(axes, WINDOWS):
        p = image_path(win)
        with Image.open(p) as im:
            img = im.convert("RGB")
            min_side = MIN_BOX_FRACTION * max(im.size)
        dets = vmetrics.detect_boxes(p, ["animal"])
        dets.sort(key=lambda d: -d["confidence"])
        kept, rejected = [], []
        for d in dets:
            ok = d["confidence"] >= CONF and _box_side(d["box"]) >= min_side
            if ok and all(vmetrics.box_iou(d["box"], k["box"]) < NMS_IOU for k in kept):
                kept.append(d)
            else:
                rejected.append(d)
        ax.imshow(img)
        for d, style in [(d, "kept") for d in kept] + [(d, "rej") for d in rejected]:
            x0, y0, x1, y1 = d["box"]
            solid = style == "kept"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="#d62728" if solid else "#888888",
                                   linewidth=2.6 if solid else 1.4,
                                   linestyle="-" if solid else "--"))
            ax.text(x0 + 6, y0 + 26,
                    f"{'kept' if solid else 'rejected'}  conf {d['confidence']:.2f}, "
                    f"side {int(_box_side(d['box']))}px",
                    fontsize=9, color="white",
                    bbox=dict(facecolor="#d62728" if solid else "#666666",
                              edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_title(f"{PAIR.replace('__x__', ' x ')}, seed {SEED}, window {win[0]}-{win[1]}: "
                     f"{len(kept)} kept detection(s)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        record.append({"window": list(win), "image": str(p),
                       "kept": [{"box": d["box"], "conf": d["confidence"]} for d in kept],
                       "rejected": [{"box": d["box"], "conf": d["confidence"]} for d in rejected]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "frog-toad-late-windows-with-detector-boxes"
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=170)
    (OUT_DIR / "detections.json").write_text(json.dumps({
        "detector": "IDEA-Research/grounding-dino-tiny, query 'animal'",
        "filters": {"conf": CONF, "min_box_fraction": MIN_BOX_FRACTION, "nms_iou": NMS_IOU},
        "cells": record,
    }, indent=2))
    print(f"wrote {out}.png and detections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
