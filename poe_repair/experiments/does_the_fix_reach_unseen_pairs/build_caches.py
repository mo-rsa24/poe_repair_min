"""Build training-cache cells for the animals pool (prereq for LoRA training + LoRA sampling).

Loads SDXL ONCE and builds every cell in-process (the old per-cell subprocess reloaded
the model ~88s each time — hours wasted). Supports sharding so several workers can run
concurrently on the same big GPU and use the spare VRAM.

  # single worker, all train cells:
  $PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.build_caches --which train
  # 4 parallel workers (launch 4 processes, shards 0..3):
  for i in 0 1 2 3; do $PY -m ...build_caches --which train --shard $i --num-shards 4 & done; wait
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from poe_repair.run import make_ctx
from scripts.build_training_cache import build_cell
from poe_repair import paths

log = logging.getLogger("animals_compose_transfer.build_caches")

REPO = Path(__file__).resolve().parents[3]
SCOPE = REPO / "outputs" / "animals_compose_transfer"
POOL = SCOPE / "pair_pool.yaml"
PROMPTS = SCOPE / "pair_prompts.yaml"
CACHE_ROOT = paths.resolve(paths.TRAINING_CACHE)
# Seeds match seed_pool.yaml: train pairs at 1-8, held-out pairs at 9-16.
SEEDS_TRAIN = list(range(1, 9))
SEEDS_HELDOUT = list(range(9, 17))
STEPS = 50


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("train", "heldout_eval"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    pool = yaml.safe_load(POOL.read_text())
    prompts = yaml.safe_load(PROMPTS.read_text())

    if args.which == "train":
        slugs, split, seeds = list(pool["train"]), "train", SEEDS_TRAIN
    else:
        slugs, split, seeds = list(pool["heldout"]), "heldout", SEEDS_HELDOUT

    # Build the flat cell list, then take this worker's shard (round-robin).
    cells = [(slug, s) for slug in slugs for s in seeds]
    mine = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
    log.info("shard %d/%d: %d of %d cells (split=%s)", args.shard, args.num_shards, len(mine), len(cells), split)

    ctx = make_ctx(num_inference_steps=STEPS)  # SDXL loaded ONCE
    n_done = n_fail = 0
    for slug, s in mine:
        p = prompts[slug]
        try:
            build_cell(prompt_a=p["prompt_a"], prompt_b=p["prompt_b"],
                       joint_prompt_text=p["joint_prompt"], seed=s, split=split,
                       out_root=CACHE_ROOT, overwrite=False, ctx=ctx)
            n_done += 1
        except Exception as exc:
            n_fail += 1
            log.warning("FAILED %s seed %d: %s", slug, s, exc)
    log.info("shard %d done: built/cached=%d failures=%d", args.shard, n_done, n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
