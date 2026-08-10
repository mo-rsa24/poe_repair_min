"""Instance-count compose/blend scorer (the instance-level rework).

Two reads were tried and rejected before this one:
  1. Whole-image embedding (DINOv2/CLIP, scorer.py): NULLED. For two similar animals,
     "one chimera" and "two separate animals" sit in the same feature region.
  2. Per-query box-IoU regime (classify_detection_regime): also fails on the hard
     pair. Two similar canines standing close have coinciding boxes whether it is one
     blended animal or two real ones, so a genuine wolf×husky compose and a wolf×husky
     blend BOTH read "both_overlapping". Using IoU would under-count real composes on
     exactly the hard pairs the downstream experiment cares about (a measurement bias).

The read that works is a genuine INSTANCE COUNT: a chimera blend has one head/body; a
real composition has two, even when the two animals touch. We query GroundingDINO with
a generic instance term ("animal") and count distinct boxes after NMS. On the decisive
validation images this gives the exact ground-truth count and separates the wolf×husky
good-compose (2) from the wolf×husky blend (1) — the case no other read could do both
ways, and with no spatial-separation bias.

So: COMPOSE iff distinct-instance-count >= 2. Primary read. The per-query confidences
and the embedding distances are kept only as secondary diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from poe_repair.experiments.residual_diagnostics import metrics as vmetrics

COMPOSE_REGIME = "both_distinct"
INSTANCE_QUERY = "animal"

# A detection must span at least this fraction of the image's longer side to count
# as an animal. Chosen by looking at the boxes, not at the resulting rate: on the
# an_elephant__x__a_penguin seed 10 strip the detector put a 220px box at confidence
# 0.60 on a shadow beside a single fused creature, turning a control-row blend into
# a "compose". Confidence cannot separate that case (0.60 sits above every real
# animal's floor at 0.54), and size can: every genuine animal in that strip spans
# 458px or more, every spurious box 220px or less, on a 1024px image.
#
# 0.25 is the same line scripts/dose_strip.py already draws when it colours boxes
# yellow or magenta, so the diagnostic picture and the scorer agree by construction.
# The filter can only remove detections, so it can only lower compose rates: it
# cannot manufacture the result the paper claims.
MIN_BOX_FRACTION = 0.25


def count_instances(
    image_path: Path,
    *,
    query: str = INSTANCE_QUERY,
    device: torch.device | None = None,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
    conf: float = 0.30,
    nms_iou: float = 0.5,
    min_box_fraction: float = MIN_BOX_FRACTION,
) -> tuple[int, list[dict]]:
    """Count distinct animal instances via a generic detector query + greedy NMS.

    Returns (count, kept_boxes). count >= 2 ⇒ a real multi-animal composition;
    count <= 1 ⇒ a single-animal blend.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        min_side = min_box_fraction * max(im.size)

    dets = vmetrics.detect_boxes(
        image_path, [query],
        box_threshold=box_threshold, text_threshold=text_threshold, device=device,
    )
    ds = [d for d in dets if d["confidence"] >= conf]
    ds = [d for d in ds if _box_side(d["box"]) >= min_side]
    ds.sort(key=lambda d: -d["confidence"])
    keep: list[dict] = []
    for d in ds:
        if all(vmetrics.box_iou(d["box"], k["box"]) < nms_iou for k in keep):
            keep.append(d)
    return len(keep), keep


def _box_side(box) -> float:
    """The box's longer side in pixels, matching dose_strip.py's yellow/magenta rule."""
    x0, y0, x1, y1 = box
    return max(x1 - x0, y1 - y0)


@dataclass
class InstanceScore:
    label: str                 # "compose" | "blend"
    n_instances: int           # distinct animal instances (query "animal" + NMS)
    n_instances_head: int      # corroborating count via "animal head"
    conf_a: float              # per-query conf for the pair's A prompt (diagnostic)
    conf_b: float


def score_output_instances(
    output_path: Path,
    query_a: str,
    query_b: str,
    *,
    device: torch.device | None = None,
) -> InstanceScore:
    n_animal, _ = count_instances(output_path, query="animal", device=device)
    n_head, _ = count_instances(output_path, query="animal head", device=device)
    # Per-query confidences (diagnostic only; not used for the label).
    dets = vmetrics.detect_boxes(output_path, [query_a, query_b], device=device)
    return InstanceScore(
        label="compose" if n_animal >= 2 else "blend",
        n_instances=n_animal,
        n_instances_head=n_head,
        conf_a=_best_conf(dets, query_a),
        conf_b=_best_conf(dets, query_b),
    )


def instance_score_to_dict(s: InstanceScore) -> dict:
    return asdict(s)


@dataclass
class DetectionScore:
    regime: str
    label: str                 # "compose" | "blend"
    conf_a: float
    conf_b: float
    iou: float                 # IoU of the best A and B boxes (nan if <2 detected)
    n_boxes: int


def _best_conf(dets, label: str) -> float:
    cands = [d["confidence"] for d in dets if d.get("label", "").strip().lower() == label.strip().lower()]
    return float(max(cands)) if cands else 0.0


def _best_box(dets, label: str, threshold: float):
    cands = [d for d in dets if d.get("label", "").strip().lower() == label.strip().lower()
             and d.get("confidence", 0.0) >= threshold]
    return max(cands, key=lambda d: d["confidence"]) if cands else None


def score_output_detection(
    output_path: Path,
    query_a: str,
    query_b: str,
    *,
    device: torch.device | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    regime_conf_threshold: float = 0.35,
    iou_overlap_threshold: float = 0.4,
) -> DetectionScore:
    dets = vmetrics.detect_boxes(
        output_path, [query_a, query_b],
        box_threshold=box_threshold, text_threshold=text_threshold, device=device,
    )
    regime = vmetrics.classify_detection_regime(
        dets, queries=(query_a, query_b),
        threshold=regime_conf_threshold, iou_overlap_threshold=iou_overlap_threshold,
    )
    ba = _best_box(dets, query_a, regime_conf_threshold)
    bb = _best_box(dets, query_b, regime_conf_threshold)
    iou = vmetrics.box_iou(ba["box"], bb["box"]) if (ba and bb) else float("nan")
    return DetectionScore(
        regime=regime,
        label="compose" if regime == COMPOSE_REGIME else "blend",
        conf_a=_best_conf(dets, query_a),
        conf_b=_best_conf(dets, query_b),
        iou=iou,
        n_boxes=len(dets),
    )


def detection_score_to_dict(s: DetectionScore) -> dict:
    return asdict(s)
