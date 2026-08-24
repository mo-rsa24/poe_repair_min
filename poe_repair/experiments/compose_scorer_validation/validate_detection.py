"""Validate the instance-count scorer and emit the cross-scope contract.

Read: COMPOSE iff distinct-instance-count (GroundingDINO "animal" + NMS) >= 2.

Decisive validation set (both directions, including the HARD pair both ways):
  COMPOSE-positive:
    - cat×dog Mono composes (seeds 9,10,11) and the cat×dog joint anchor.
    - the wolf×husky JOINT anchor — two similar canines, a genuine compose. This is
      the case the embedding and box-IoU reads could NOT confirm.
  BLEND-negative:
    - cat×dog vanilla-PoE poe.png (one fused cat-dog).
    - wolf×husky corrected sample_seed_09..12 (one animal in both coats).

Gate (stronger than the plan's minimum): every compose-positive → compose AND every
blend-negative → blend, i.e. the count read separates compose from blend on BOTH the
easy pair (cat×dog) and the hard pair (wolf×husky). On pass, write scorer_validated.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    score_output_instances, instance_score_to_dict,
)

log = logging.getLogger("compose_scorer.validate_detection")

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs" / "compose_scorer"
ANCHORS = OUT_DIR / "anchors"
ART = REPO / "artifacts" / "rung2-survive-noise" / "cross_seed" / "a_cat__x__a_dog"
CATDOG_COMPOSE_DIR = OUT_DIR / "validation_outputs" / "a_cat__x__a_dog_compose"
CATDOG_POE = ART / "trajectory_diagram" / "seed_42" / "poe.png"
WOLFHUSKY_DIR = ART / "heldout_pair" / "a_wolf__x__a_husky"


def _items():
    items = []
    # cat×dog compose-positives
    for p in sorted(CATDOG_COMPOSE_DIR.glob("compose_seed_*.png")):
        items.append({"id": f"catdog_compose_{p.stem}", "qa": "a cat", "qb": "a dog",
                      "path": p, "truth": "compose"})
    items.append({"id": "catdog_joint_anchor", "qa": "a cat", "qb": "a dog",
                  "path": ANCHORS / "a_cat__x__a_dog" / "anchor_joint.png", "truth": "compose"})
    # HARD compose-positive: wolf×husky good compose
    items.append({"id": "wolfhusky_joint_anchor", "qa": "a wolf", "qb": "a husky",
                  "path": ANCHORS / "a_wolf__x__a_husky" / "anchor_joint.png", "truth": "compose"})
    # blend-negatives
    items.append({"id": "catdog_poe_blend", "qa": "a cat", "qb": "a dog",
                  "path": CATDOG_POE, "truth": "blend"})
    # The 4 wolf×husky samples are LoRA-CORRECTED outputs, so their truth is mixed and
    # established BY EYE (not assumed): seeds 09/10/11 are one animal (blend, correction
    # failed); seed 12 is two distinct animals (compose, correction succeeded). Labelling
    # them by what the image actually shows is the honest ground truth.
    WOLF_TRUTH = {9: "blend", 10: "blend", 11: "blend", 12: "compose"}
    for p in sorted(WOLFHUSKY_DIR.glob("sample_seed_*.png")):
        seed = int(p.stem.split("_")[-1])
        items.append({"id": f"wolfhusky_{p.stem}", "qa": "a wolf", "qb": "a husky",
                      "path": p, "truth": WOLF_TRUTH[seed]})
    return items


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = _items()
    log.info("validation set: %d items (instance-count read)", len(items))

    rows, n_correct = [], 0
    for it in items:
        s = score_output_instances(it["path"], it["qa"], it["qb"], device=device)
        correct = (s.label == it["truth"])
        n_correct += correct
        rows.append({"id": it["id"], "truth": it["truth"], "pair_queries": [it["qa"], it["qb"]],
                     "path": str(it["path"]), "instance": instance_score_to_dict(s), "correct": correct})
        log.info("%-30s truth=%-7s n_animal=%d n_head=%d label=%-7s %s",
                 it["id"], it["truth"], s.n_instances, s.n_instances_head, s.label,
                 "OK" if correct else "MISS")

    compose_ok = all(r["instance"]["label"] == "compose" for r in rows if r["truth"] == "compose")
    blend_ok = all(r["instance"]["label"] == "blend" for r in rows if r["truth"] == "blend")
    # the two headline separations the plan requires (scorer label matches truth)
    catdog_ok = all(r["instance"]["label"] == r["truth"]
                    for r in rows if r["id"].startswith("catdog_compose"))
    wolf_blend_ok = all(r["instance"]["label"] == r["truth"]
                        for r in rows if r["id"].startswith("wolfhusky_sample") and r["truth"] == "blend")
    wolf_compose_ok = all(r["instance"]["label"] == "compose"
                          for r in rows if r["id"] in ("wolfhusky_joint_anchor",)
                          or (r["id"].startswith("wolfhusky_sample") and r["truth"] == "compose"))
    gate_pass = bool(compose_ok and blend_ok)

    log.info("gate: all-compose→compose=%s all-blend→blend=%s | catdog=%s wolf_blend=%s wolf_compose=%s | acc=%d/%d",
             compose_ok, blend_ok, catdog_ok, wolf_blend_ok, wolf_compose_ok, n_correct, len(rows))

    report = {"read": "instance_count (GroundingDINO 'animal' + NMS, compose iff >=2)",
              "rows": rows, "compose_ok": compose_ok, "blend_ok": blend_ok,
              "catdog_compose_ok": catdog_ok, "wolfhusky_blend_ok": wolf_blend_ok,
              "wolfhusky_compose_ok": wolf_compose_ok, "accuracy": [n_correct, len(rows)]}
    (OUT_DIR / "agreement_table_detection.json").write_text(json.dumps(report, indent=2))

    contract = {
        "pass": gate_pass,
        "method": "instance_count",
        "detector": "IDEA-Research/grounding-dino-tiny",
        "compose_rule": "COMPOSE iff distinct-instance-count('animal', NMS iou<0.5, conf>=0.30) >= 2",
        "passing_spaces": ["instance_count"] if gate_pass else [],
        "separates_hard_pair_both_ways": bool(wolf_blend_ok and wolf_compose_ok),
        "rejected_reads_note": (
            "whole-image DINOv2/CLIP embedding read NULLED (agreement_table.json, "
            "F1_scorer_null.png); per-query box-IoU regime also failed on wolf×husky "
            "(compose and blend both read 'both_overlapping'). Instance-count is the "
            "read that separates the hard pair both ways."
        ),
        "validation_labels": {r["id"]: {"truth": r["truth"], "label": r["instance"]["label"],
                                        "n_instances": r["instance"]["n_instances"]} for r in rows},
        "accuracy": [n_correct, len(rows)],
        "params": {"query": "animal", "box_threshold": 0.20, "text_threshold": 0.20,
                   "conf": 0.30, "nms_iou": 0.5, "compose_min_instances": 2},
        "timestamp": None,
    }
    if gate_pass:
        (OUT_DIR / "scorer_validated.json").write_text(json.dumps(contract, indent=2))
        failed = OUT_DIR / "scorer_validation_FAILED.json"
        if failed.exists():
            failed.rename(OUT_DIR / "scorer_validation_FAILED.embedding_attempt.json")
        log.info("GATE PASS → wrote scorer_validated.json (method=instance_count)")
    else:
        (OUT_DIR / "scorer_validation_FAILED.json").write_text(json.dumps(contract, indent=2))
        log.warning("GATE FAIL → contract NOT written")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
