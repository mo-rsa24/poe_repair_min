"""Decode the plain-PoE sample per seed from the pinned latent (overlay base).

The capture run did not save PNGs. This decodes the deterministic λ=0 sample
for each seed so the attention viewer has an image to ground maps on. Writes
``seed_<N>/sample.png`` under the regime dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

DEFAULT_ATTN_ROOT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
)


def _parse_seeds(arg: str) -> list[int]:
    out: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="decode_samples")
    ap.add_argument("--regime", default="plain_poe")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--lambda", dest="lambda_value", type=float, default=0.0,
                    help="0 = plain PoE; 1 = LoRA-corrected (needs --checkpoint)")
    ap.add_argument("--checkpoint", default=None,
                    help="lora_step_NNNNNN.pt — required for --lambda 1")
    args = ap.parse_args(argv)

    if float(args.lambda_value) != 0.0 and not args.checkpoint:
        raise SystemExit("--lambda 1 requires --checkpoint")

    root = DEFAULT_ATTN_ROOT / args.regime / args.pair_slug
    seeds = _parse_seeds(args.seeds)
    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype,
    )
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")

    adapter_name = "lora"
    if float(args.lambda_value) != 0.0:
        from poe_repair.experiments.mechanism_study.capture_attention import (
            _maybe_attach_lora,
        )
        adapter_name = _maybe_attach_lora(models["unet"], args.checkpoint)

    class _P:
        prompt_a, prompt_b, joint_prompt = (
            args.prompt_a, args.prompt_b, args.joint_prompt
        )

    class _C:
        cell = _P()

    emb = encode_all_prompts(_C(), models, device, dtype)
    for s in seeds:
        cell = CellPath.from_root(
            args.pair_slug, int(s), cache_root=DEFAULT_CACHE_ROOT
        )
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(args.euler_sigma),
        )
        out = run_lora_residual_inject(
            init_latents=init, models=models, scheduler=scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"],
            seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_j=emb["seq_j"], pool_j=emb["pool_j"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=7.5, num_inference_steps=50,
            height=1024, width=1024,
            euler_init_noise_sigma=float(args.euler_sigma),
            device=device, dtype=dtype,
            lambda_value=float(args.lambda_value),
            lora_adapter_name=adapter_name,
        )
        png = root / f"seed_{int(s)}" / "sample.png"
        write_decoded_image(out.image, png)
        print(f"[decode_samples] seed {s} → {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
