#!/usr/bin/env python
"""SuperDiff sampling with one of our adapters attached, on named cells.

SuperDiff combines the two concept models by re-weighting them each step so neither dominates,
where our correction adds a learned residual instead. This runs the two together: SuperDiff's
sampler through the adapted model, which is the question "does the correction help a sampler that
already balances the concepts".

The adapter's shape is read from its checkpoint, so a cross-and-self adapter attaches to all 420
projections rather than silently to half of them. SuperDiff loads its own copy of the base weights,
cached at module scope, which is the UNet the adapter lands on.

    python scripts/superdiff_with_adapter.py --checkpoint <pt> --tag plain --seeds 9 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Run by file name, sys.path[0] is scripts/, so the package would not import.
sys.path.insert(0, str(Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))))

import torch

from poe_repair.composers import superdiff
from poe_repair.experiments import _adapter_shape
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/sampling_grid/superdiff")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="names the run in the output path")
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--seeds", nargs="+", type=int, default=[9, 10])
    ap.add_argument("--lambda-value", type=float, default=1.0,
                    help="how much of the adapter's correction the sampler uses")
    ap.add_argument("--kappa", type=float, default=0.5,
                    help="SuperDiff's mixing weight, held fixed rather than clamped per step")
    ap.add_argument("--steps", type=int, default=50,
                    help="50 matches every other render in this comparison; the preview used 200")
    ap.add_argument("--out-root", type=Path, default=OUT)
    a = ap.parse_args()

    device, dtype = infer_device(None), infer_dtype("fp16", infer_device(None))
    models = superdiff._load_superdiff_models(device, dtype)
    info = _adapter_shape.attach(models["unet"], a.checkpoint, adapter_name="lora")
    print(f"attached {info['n_matched']} modules at rank {info['rank']} "
          f"({', '.join(info['target_modules'])}) from step {info['checkpoint_step']}", flush=True)

    # Create the tree before the composer writes into it: on this NFS a fresh path can be missing
    # under a job that never raced anyone, and the composer's own mkdir is not retried.
    for _try in range(5):
        (a.out_root / a.tag).mkdir(parents=True, exist_ok=True)
        if (a.out_root / a.tag).is_dir():
            break
        time.sleep(2)

    t0 = time.perf_counter()
    for seed in a.seeds:
        cell = PairSeedCell(pair_dir=None, pair_slug=a.pair, prompt_a=a.prompt_a,
                            prompt_b=a.prompt_b, seed=seed, regime="sampling_grid",
                            height=1024, width=1024, grid_assets={})
        ctx = MethodCtx(models={}, scheduler=None, output_root=a.out_root, device=device,
                        dtype=dtype, guidance_scale=7.5, num_inference_steps=a.steps)
        path = superdiff.run(cell, ctx, exp_name=a.tag, kappa_clamp=False,
                             kappa_override=a.kappa, lora_adapter_name="lora",
                             lambda_value=a.lambda_value, lora_tag=a.tag)
        print(f"seed {seed}: {path}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        (Path(path).with_suffix(".adapter.json")).write_text(json.dumps(
            {**info, "lambda_value": a.lambda_value, "kappa": a.kappa, "steps": a.steps,
             "seed": seed}, indent=1, default=str))
    print("done", flush=True)
