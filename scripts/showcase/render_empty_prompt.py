#!/usr/bin/env python
"""What does the empty prompt draw?

The composition subtracts the empty branch once and the guided sampler reads it
again on its own, so it carries the largest coefficient of the three branches.
Nothing in this project has ever looked at what it actually renders.

Three conditions, one seed, on the same pinned initial latents every other
render in the showcase uses:

    empty      ""        at guidance 1
    cat_w1     "a cat"   at guidance 1   the control: is the mush the prompt, or the dial?
    cat_w75    "a cat"   at guidance 7.5 the same prompt as the project renders it

Guidance does nothing to the empty condition: run_cfg computes
eps_uncond + w*(eps_cond - eps_uncond), and for the empty prompt the
conditional branch IS the unconditional one, so the guidance term is exactly
zero at every w. The w=1 label is the honest one.

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
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/empty_prompt")

PAIR_SLUG = "a_cat__x__a_dog"
NUM_INFERENCE_STEPS = 50
EULER_INIT_NOISE_SIGMA = 1.0

# condition -> (prompt, guidance_scale)
CONDITIONS = {
    "empty":   ("",      1.0),
    "cat_w1":  ("a cat", 1.0),
    "cat_w75": ("a cat", 7.5),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--seeds", default="9")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c for c in args.conditions.split(",") if c]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; expected one of {list(CONDITIONS)}")

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=7.5)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    encoded = {
        c: encode_prompt_sdxl(CONDITIONS[c][0], models=ctx.models,
                              device=ctx.device, dtype=ctx.dtype)
        for c in conditions
    }

    manifest = []
    for seed in seeds:
        cell = CellPath.from_root(PAIR_SLUG, seed, split="heldout", cache_root=TRAINING_CACHE)
        init_latents = load_pinned_init_latents(
            cell, device=ctx.device, dtype=ctx.dtype,
            euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
        )
        for c in conditions:
            prompt, w = CONDITIONS[c]
            img_path = out_root / c / f"seed_{seed}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if img_path.exists():
                print(f"[skip] {img_path} exists", flush=True)
                continue
            t0 = time.time()
            seq, pool = encoded[c]
            out = run_cfg(
                init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                seq_cond=seq, pool_cond=pool, seq_e=seq_e, pool_e=pool_e,
                guidance_scale=w, num_inference_steps=NUM_INFERENCE_STEPS,
                height=1024, width=1024, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
                device=ctx.device, dtype=ctx.dtype,
            )
            write_decoded_image(out.image, img_path)
            manifest.append({
                "seed": seed, "condition": c, "prompt": prompt, "guidance_scale": w,
                "png": str(img_path), "steps": NUM_INFERENCE_STEPS,
                "elapsed_s": round(time.time() - t0, 1),
            })
            print(f"[{c}] prompt={prompt!r} w={w} seed={seed} -> {img_path} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            torch.cuda.empty_cache()

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {len(manifest)} renders -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
