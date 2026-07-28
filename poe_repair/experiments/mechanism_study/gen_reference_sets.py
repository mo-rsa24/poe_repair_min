"""Generate pure-cat and pure-dog reference images for the manifold plot.

Plan 04 task 4 needs "cat cloud" and "dog cloud" anchors. We make them by
running the same PoE sampler but with BOTH prompts set to the same concept
(A=B="a cat" → a pure cat; A=B="a dog" → a pure dog), across the same 12 pinned
seeds, at λ=0 (no LoRA needed for a single-concept reference). This keeps the
references in the same init-latent distribution as the cat×dog samples.

Output: <out>/pure_cat/seed_<N>.png and <out>/pure_dog/seed_<N>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject, write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir, infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

DEFAULT_OUT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/manifold/references"
)


def _parse_seeds(a):
    out = []
    for part in a.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_reference_sets")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog",
                    help="pair whose pinned init latents to reuse")
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--out", default=None)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args(argv)

    out_root = ensure_dir(Path(args.out) if args.out else DEFAULT_OUT)
    seeds = _parse_seeds(args.seeds)
    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype)
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")

    for concept, prompt in [("pure_cat", "a cat"), ("pure_dog", "a dog")]:
        cdir = ensure_dir(out_root / concept)

        class _P:
            prompt_a = prompt
            prompt_b = prompt
            joint_prompt = prompt

        class _C:
            cell = _P()

        emb = encode_all_prompts(_C(), models, device, dtype)
        for s in seeds:
            cell = CellPath.from_root(args.pair_slug, int(s),
                                      cache_root=DEFAULT_CACHE_ROOT)
            init = load_pinned_init_latents(
                cell, device=device, dtype=dtype,
                euler_init_noise_sigma=float(args.euler_sigma))
            out = run_lora_residual_inject(
                init_latents=init, models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_j=emb["seq_j"], pool_j=emb["pool_j"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(args.guidance_scale),
                num_inference_steps=int(args.num_inference_steps),
                height=1024, width=1024,
                euler_init_noise_sigma=float(args.euler_sigma),
                device=device, dtype=dtype, lambda_value=0.0)
            write_decoded_image(out.image, cdir / f"seed_{int(s)}.png")
            print(f"[gen_ref] {concept} seed {s}", flush=True)
    print(f"[gen_ref] done → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
