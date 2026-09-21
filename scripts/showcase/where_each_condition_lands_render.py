#!/usr/bin/env python
"""Render the three single-prompt reference conditions for the "where does each
condition land" figure: "a cat" alone, "a dog" alone, and the joint prompt
"a cat and a dog", one render per held-out seed, on the same pinned initial
latents the rank-32 PoE renders used (training cache, heldout split).

Sampler settings match scripts/showcase/lambda_boundary_probe.py exactly
(50 DDIM steps, guidance 7.5, 1024x1024, fp16 with the repo's upcast rule),
so every point in the figure differs from every other only by its prompt
and by whether the correction was injected.

Writes:
    <out-root>/<condition>/seed_<n>.png
    <out-root>/render_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.training_cache import CellPath  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents  # noqa: E402

TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands")

PAIR_SLUG = "a_cat__x__a_dog"
PROMPTS = {
    "solo_a": "a cat",
    "solo_b": "a dog",
    "joint": "a cat and a dog",
}
HELD_OUT_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
EULER_INIT_NOISE_SIGMA = 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--seeds", default=",".join(str(s) for s in HELD_OUT_SEEDS))
    ap.add_argument("--conditions", default=",".join(PROMPTS))
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c for c in args.conditions.split(",") if c]
    for c in conditions:
        if c not in PROMPTS:
            raise SystemExit(f"unknown condition {c!r}; expected one of {list(PROMPTS)}")

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    encoded = {
        c: encode_prompt_sdxl(PROMPTS[c], models=ctx.models, device=ctx.device, dtype=ctx.dtype)
        for c in conditions
    }

    manifest = []
    for seed in seeds:
        cell = CellPath.from_root(PAIR_SLUG, seed, split="heldout", cache_root=TRAINING_CACHE)
        init_latents = load_pinned_init_latents(
            cell, device=ctx.device, dtype=ctx.dtype, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
        )
        for c in conditions:
            img_path = out_root / c / f"seed_{seed}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if img_path.exists():
                print(f"[skip] {img_path} exists", flush=True)
                manifest.append({"seed": seed, "condition": c, "prompt": PROMPTS[c],
                                 "png": str(img_path), "reused": True})
                continue
            t0 = time.time()
            seq, pool = encoded[c]
            out = run_cfg(
                init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                seq_cond=seq, pool_cond=pool, seq_e=seq_e, pool_e=pool_e,
                guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                height=1024, width=1024, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
                device=ctx.device, dtype=ctx.dtype,
            )
            write_decoded_image(out.image, img_path)
            manifest.append({"seed": seed, "condition": c, "prompt": PROMPTS[c],
                             "png": str(img_path), "reused": False,
                             "elapsed_s": round(time.time() - t0, 1)})
            print(f"[{c}] seed={seed} -> {img_path} ({time.time() - t0:.1f}s)", flush=True)
            torch.cuda.empty_cache()

    (out_root / "render_manifest.json").write_text(json.dumps({
        "pair_slug": PAIR_SLUG, "prompts": PROMPTS, "seeds": seeds,
        "num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE,
        "init_latents": "training cache heldout split step_000 x_t, same as lambda_boundary_probe",
        "renders": manifest,
    }, indent=2))
    print(f"[done] {len(manifest)} renders, manifest at {out_root / 'render_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
