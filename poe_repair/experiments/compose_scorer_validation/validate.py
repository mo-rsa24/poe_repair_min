"""Validate the scorer, pick the empirical threshold, emit the agreement table
and the cross-scope contract (compose-scorer plans 02 tasks 3-4 + plan 03).

Validation set (ground truth by eye):
  COMPOSE-positive:
    - cat×dog Mono composes at seeds 9,10,11 (two clearly separate animals).
      Non-circular: distinct from the seed-42 joint anchor.
  BLEND-negative:
    - cat×dog vanilla-PoE poe.png (a single fused cat-dog).
    - wolf×husky LoRA-corrected sample_seed_09..12.png (one animal in both coats;
      the case that fools the eye — the whole reason the scorer exists).

Threshold: the scorer's compose_margin = min(d_a,d_b) - d_joint. A COMPOSE has a
positive margin (nearer the joint); a BLEND has a small/negative margin (nearest a
single). We pick the margin threshold that maximally separates the two labelled
groups, per space, then record it. Default margin 0.0 is the natural boundary; we
report both the 0.0 read and the separation.

Gate (plan 03): cat×dog compose-positives → compose AND wolf×husky → blend, in a
space, ⇒ that space passes. Contract written only if a space passes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from poe_repair import paths
from poe_repair.experiments.compose_scorer_validation.scorer import (
    _Embedders, score_output, SPACES, space_score_to_dict,
)

log = logging.getLogger("compose_scorer.validate")

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = paths.resolve(paths.COMPOSE_SCORER_VALIDATION)
ANCHOR_ROOT = OUT_DIR / "anchors"

ART = paths.resolve(paths.HELD_OUT_SEEDS) / "a_cat__x__a_dog"
CATDOG_COMPOSE_DIR = OUT_DIR / "validation_outputs" / "a_cat__x__a_dog_compose"
CATDOG_POE = ART / "trajectory_diagram" / "seed_42" / "poe.png"
WOLFHUSKY_DIR = ART / "heldout_pair" / "a_wolf__x__a_husky"


def _anchors(pair: str) -> dict[str, Path]:
    d = ANCHOR_ROOT / pair
    return {k: d / f"anchor_{k}.png" for k in ("a_alone", "b_alone", "joint")}


def _validation_items() -> list[dict]:
    """Each item: {id, pair, output_path, truth}."""
    items = []
    for p in sorted(CATDOG_COMPOSE_DIR.glob("compose_seed_*.png")):
        items.append({"id": f"catdog_compose_{p.stem}", "pair": "a_cat__x__a_dog",
                      "output_path": p, "truth": "compose"})
    items.append({"id": "catdog_poe_blend", "pair": "a_cat__x__a_dog",
                  "output_path": CATDOG_POE, "truth": "blend"})
    for p in sorted(WOLFHUSKY_DIR.glob("sample_seed_*.png")):
        items.append({"id": f"wolfhusky_{p.stem}", "pair": "a_wolf__x__a_husky",
                      "output_path": p, "truth": "blend"})
    return items


def _best_threshold(margins_compose: list[float], margins_blend: list[float]) -> tuple[float, float]:
    """Threshold on compose_margin that best separates the two groups.
    Returns (threshold, balanced_accuracy_at_threshold). label=compose iff margin>thr."""
    cand = sorted(set(margins_compose + margins_blend + [0.0]))
    mids = [(-1e9)] + [(cand[i] + cand[i + 1]) / 2 for i in range(len(cand) - 1)] + [1e9]
    best_thr, best_acc = 0.0, -1.0
    for thr in mids:
        tp = sum(1 for m in margins_compose if m > thr)
        tn = sum(1 for m in margins_blend if m <= thr)
        sens = tp / max(1, len(margins_compose))
        spec = tn / max(1, len(margins_blend))
        acc = 0.5 * (sens + spec)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = _Embedders(device=device)
    items = _validation_items()
    log.info("validation set: %d items", len(items))

    # Score every output in every space at the natural boundary margin=0.0,
    # keeping the raw compose_margin so we can pick the empirical threshold after.
    rows = []
    for it in items:
        anchors = _anchors(it["pair"])
        scored = score_output(it["output_path"], anchors, embedders=emb, margin=0.0)
        row = {"id": it["id"], "pair": it["pair"], "truth": it["truth"],
               "output_path": str(it["output_path"])}
        for sp in SPACES:
            row[sp] = space_score_to_dict(scored[sp])
        rows.append(row)
        log.info("%-28s truth=%-7s dino=%-7s clip=%-7s",
                 it["id"], it["truth"], rows[-1]["dino"]["label"], rows[-1]["clip"]["label"])

    # Empirical threshold per space from the labelled margins.
    thresholds = {}
    for sp in SPACES:
        mc = [r[sp]["compose_margin"] for r in rows if r["truth"] == "compose"]
        mb = [r[sp]["compose_margin"] for r in rows if r["truth"] == "blend"]
        thr, acc = _best_threshold(mc, mb)
        thresholds[sp] = {"threshold": thr, "separated_balanced_acc": acc,
                          "compose_margins": mc, "blend_margins": mb}
        log.info("space=%s empirical threshold=%.4f balanced_acc=%.3f", sp, thr, acc)

    # Re-label at the empirical threshold and evaluate the gate per space.
    def label_at(margin_val: float, thr: float) -> str:
        return "compose" if margin_val > thr else "blend"

    space_pass = {}
    for sp in SPACES:
        thr = thresholds[sp]["threshold"]
        correct = all(label_at(r[sp]["compose_margin"], thr) == r["truth"] for r in rows)
        # The load-bearing gate specifically:
        catdog_ok = all(label_at(r[sp]["compose_margin"], thr) == "compose"
                        for r in rows if r["pair"] == "a_cat__x__a_dog" and r["truth"] == "compose")
        wolf_ok = all(label_at(r[sp]["compose_margin"], thr) == "blend"
                      for r in rows if r["pair"] == "a_wolf__x__a_husky")
        space_pass[sp] = {"all_correct": correct, "catdog_compose": catdog_ok,
                          "wolfhusky_blend": wolf_ok, "gate_pass": bool(catdog_ok and wolf_ok)}
        log.info("space=%s gate: catdog_compose=%s wolfhusky_blend=%s PASS=%s",
                 sp, catdog_ok, wolf_ok, space_pass[sp]["gate_pass"])

    # Agreement table (plan 02 task 4).
    agreement = {"rows": rows, "thresholds": thresholds, "space_pass": space_pass}
    (OUT_DIR / "agreement_table.json").write_text(json.dumps(agreement, indent=2))
    log.info("wrote agreement_table.json")

    # Contract decision (plan 03). Pass if ANY space passes the gate; record which.
    passing_spaces = [sp for sp in SPACES if space_pass[sp]["gate_pass"]]
    both = len(passing_spaces) == len(SPACES)
    any_pass = len(passing_spaces) > 0
    disagree = 0 < len(passing_spaces) < len(SPACES)

    contract = {
        "pass": bool(any_pass),
        "passing_spaces": passing_spaces,
        "both_spaces_pass": both,
        "spaces_disagree": disagree,
        "thresholds": {sp: thresholds[sp]["threshold"] for sp in SPACES},
        "validation_labels": {
            r["id"]: {"truth": r["truth"],
                      "dino": r["dino"]["label"], "clip": r["clip"]["label"]}
            for r in rows
        },
        "spaces_used": list(SPACES),
        "excluded_space_note": (
            "trajectory MDS/latent space excluded by design: it projects 65536-dim "
            "epsilon/latent residual tensors, not rendered images, so it cannot embed "
            "an output PNG. DINOv2 + CLIP are the two independent image spaces."
        ),
        "timestamp": None,  # left for caller to stamp
    }
    if any_pass:
        (OUT_DIR / "scorer_validated.json").write_text(json.dumps(contract, indent=2))
        log.info("GATE PASS (spaces=%s) → wrote scorer_validated.json", passing_spaces)
    else:
        (OUT_DIR / "scorer_validation_FAILED.json").write_text(json.dumps(contract, indent=2))
        log.warning("GATE FAIL → wrote scorer_validation_FAILED.json (contract NOT written)")

    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
