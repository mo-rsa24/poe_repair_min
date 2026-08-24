"""Score the trainer's inline samples with the instance-count scorer and log the
transfer analysis to W&B (compose-rate vs training step).

The trainer saves inline samples to
  <run>/samples/per_epoch/epoch_XXXX_step_YYYYYY/<quadrant>/<pair>/seed_NN.png
This walks those, scores each (COMPOSE iff >=2 animal instances), and produces:
  - compose_rate vs step, per quadrant (in_in = train pairs, out_out = held-out pairs)
  - compose_rate vs step, per held-out pair (the transfer curves)
  - a final per-pair table at the last checkpoint
logged to the same W&B run (by run-id) and saved as JSON.

Idempotent and re-runnable during training. Run:
  $PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.compose_rate \
      --run-dir outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k [--wandb]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from poe_repair.experiments.compose_scorer_validation.detection_scorer import score_output_instances

log = logging.getLogger("animals_compose_transfer.compose_rate")

REPO = Path(__file__).resolve().parents[3]
SCOPE = REPO / "outputs" / "animals_compose_transfer"
PROMPTS = yaml.safe_load((SCOPE / "pair_prompts.yaml").read_text())
STEP_RE = re.compile(r"epoch_(\d+)_step_(\d+)")
SEED_RE = re.compile(r"seed_(\d+)")


def _queries(slug: str) -> tuple[str, str]:
    p = PROMPTS[slug]
    return p["prompt_a"], p["prompt_b"]


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--wandb", action="store_true", help="log curves to the W&B run")
    ap.add_argument("--wandb-project", default="poe-repair-animals-compose")
    ap.add_argument("--run-id", default=None, help="W&B run id to resume (default: run-dir name)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_epoch = run_dir / "samples" / "per_epoch"
    if not per_epoch.exists():
        log.error("no inline samples at %s (has training sampled yet?)", per_epoch)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step -> quadrant -> list[(pair, is_compose)]
    by_step: dict[int, dict[str, list[tuple[str, bool]]]] = defaultdict(lambda: defaultdict(list))
    snap_dirs = sorted(per_epoch.glob("epoch_*_step_*"))
    log.info("scoring %d snapshots", len(snap_dirs))
    for snap in snap_dirs:
        m = STEP_RE.search(snap.name)
        if not m:
            continue
        step = int(m.group(2))
        # flat files: <quadrant>__<pairslug>__seedNN.png  (pairslug contains __x__)
        for png in snap.glob("*__seed*.png"):
            stem = png.stem
            if stem.startswith("in_in__"):
                quadrant, rest = "in_in", stem[len("in_in__"):]
            elif stem.startswith("out_out__"):
                quadrant, rest = "out_out", stem[len("out_out__"):]
            else:
                continue
            pair, _, _seed = rest.rpartition("__seed")
            if pair not in PROMPTS:
                continue
            qa, qb = _queries(pair)
            sc = score_output_instances(png, qa, qb, device=device)
            by_step[step][quadrant].append((pair, sc.label == "compose"))

    # aggregate compose-rate per step per quadrant, and per held-out pair
    curves = {"per_step_quadrant": {}, "per_step_heldout_pair": defaultdict(dict)}
    rows_for_wandb = []
    for step in sorted(by_step):
        curves["per_step_quadrant"][step] = {}
        for quad, items in by_step[step].items():
            rate = sum(c for _, c in items) / max(1, len(items))
            curves["per_step_quadrant"][step][quad] = {"compose_rate": rate, "n": len(items)}
        # per held-out pair (out_out quadrant)
        pair_hits: dict[str, list[bool]] = defaultdict(list)
        for pair, c in by_step[step].get("out_out", []):
            pair_hits[pair].append(c)
        for pair, hits in pair_hits.items():
            curves["per_step_heldout_pair"][pair][step] = sum(hits) / max(1, len(hits))
        rows_for_wandb.append((step, curves["per_step_quadrant"][step]))

    out = run_dir / "compose_rate.json"
    out.write_text(json.dumps(curves, indent=2, default=float))
    # console summary
    for step in sorted(by_step):
        q = curves["per_step_quadrant"][step]
        log.info("step=%d  train(in_in)=%.2f  heldout(out_out)=%.2f",
                 step,
                 q.get("in_in", {}).get("compose_rate", float("nan")),
                 q.get("out_out", {}).get("compose_rate", float("nan")))
    log.info("wrote %s", out)

    if args.wandb:
        import wandb
        run_id = args.run_id or run_dir.name
        run = wandb.init(project=args.wandb_project, id=run_id, resume="allow")
        for step, q in rows_for_wandb:
            logd = {}
            for quad, v in q.items():
                logd[f"transfer/compose_rate/{quad}"] = v["compose_rate"]
            for pair, steps in curves["per_step_heldout_pair"].items():
                if step in steps:
                    logd[f"transfer/heldout/{pair}"] = steps[step]
            wandb.log(logd, step=step)
        run.finish()
        log.info("logged transfer curves to W&B run %s", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
