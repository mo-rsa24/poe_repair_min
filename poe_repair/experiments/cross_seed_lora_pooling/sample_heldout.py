"""Sample a trained LoRA on each held-out seed.

Loads a checkpoint produced by ``train_pooled`` (or by the legacy
``lora.main`` single-seed trainer), attaches it to a fresh SDXL UNet,
and runs ``run_lora_residual_inject`` on each held-out seed's pinned
init latent. Outputs one PNG per seed.

Optionally records per-step guided ε in the LoRA-on path (Task D's
``v_lora``) via ``--record-eps``.

Usage::

    python -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
        --checkpoint outputs/cross_seed_lora_pooling/.../lora_step_010000.pt \
        --out-dir   outputs/cross_seed_lora_pooling/.../samples/heldout \
        [--seeds 9,10,11,12] [--record-eps] [--lambda 1.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool
from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


log = logging.getLogger(__name__)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sample_heldout")
    ap.add_argument("--checkpoint", required=True,
                    help="path to lora_step_NNNNNN.pt")
    ap.add_argument("--out-dir", required=True,
                    help="where to write sample_seed_NN.png + metadata")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated list; defaults to seed_pool.held_out")
    ap.add_argument("--lambda", dest="lambda_value", type=float, default=1.0,
                    help="1.0 = full LoRA; 0.0 = canary frozen PoE")
    ap.add_argument("--record-eps", action="store_true",
                    help="dump per-step guided ε for Task D")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--heldout-pair", default=None,
                    help="evaluate the trained LoRA on a different pair "
                         "slug (e.g. a_wolf__x__a_husky). Init latents and "
                         "embeddings are read from that pair's cache. When "
                         "set, --prompt-a/-b/--joint-prompt should match.")
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--seed-pool-path", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_SEED_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)
    pool = load_seed_pool(p.seed_pool_path)
    seeds = (
        [int(x) for x in p.seeds.split(",") if x.strip()]
        if p.seeds is not None else list(pool.held_out)
    )
    out_dir = ensure_dir(Path(p.out_dir))

    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    log.info("device=%s dtype=%s seeds=%s", device, dtype, seeds)
    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)

    # Attach a fresh LoRA shell and load weights from the checkpoint.
    ckpt = torch.load(p.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    rank = int(cfg_dict.get("lora", {}).get("rank", 8))
    alpha = int(cfg_dict.get("lora", {}).get("alpha", rank))
    target_modules = tuple(
        cfg_dict.get("lora", {}).get("target_modules",
                                     ("attn2.to_q", "attn2.to_k", "attn2.to_v"))
    )
    adapter_name = str(cfg_dict.get("lora", {}).get("adapter_name", "lora"))
    log.info("ckpt config: rank=%d alpha=%d targets=%s", rank, alpha, target_modules)

    from peft import LoraConfig
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=list(target_modules), init_lora_weights=True,
    )
    unet = models["unet"]
    unet.add_adapter(lora_cfg, adapter_name=adapter_name)
    with torch.no_grad():
        for name, prm in unet.named_parameters():
            if "lora_" in name:
                prm.data = prm.data.to(torch.float32)
                prm.requires_grad_(False)
            else:
                prm.requires_grad_(False)
    lora_trainer.load_lora_state(unet, ckpt["lora_state"])
    log.info("loaded %d LoRA tensors from %s",
             len(ckpt["lora_state"]), p.checkpoint)

    # Encode prompts once.
    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt
    class _CfgShim:
        cell = _PromptShim()
    embeddings = encode_all_prompts(_CfgShim(), models, device, dtype)

    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT
    train_pair_slug = str(cfg_dict.get("cell", {}).get("pair_slug", pool.pair_slug))
    pair_slug = str(p.heldout_pair) if p.heldout_pair else train_pair_slug
    if pair_slug != train_pair_slug:
        log.info("held-out-pair eval: train_pair=%s eval_pair=%s",
                 train_pair_slug, pair_slug)

    manifest: list[dict] = []
    for s in seeds:
        cell = CellPath.from_root(pair_slug, int(s), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(p.euler_sigma),
        )
        png_path = out_dir / f"sample_seed_{int(s):02d}.png"
        eps_path = (out_dir / "eps_records" / f"eps_seed_{int(s):02d}.pt"
                    if p.record_eps else None)
        log.info("sampling seed=%d → %s%s", int(s), png_path,
                 f" (+ε log → {eps_path})" if eps_path else "")
        unet.eval()
        out = run_lora_residual_inject(
            init_latents=init, models=models, scheduler=scheduler,
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
            seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
            guidance_scale=float(p.guidance_scale),
            num_inference_steps=int(p.num_inference_steps),
            height=int(p.height), width=int(p.width),
            euler_init_noise_sigma=float(p.euler_sigma),
            device=device, dtype=dtype,
            lambda_value=float(p.lambda_value),
            lora_adapter_name=adapter_name,
            record_eps_path=eps_path,
        )
        write_decoded_image(out.image, png_path)
        manifest.append({
            "seed": int(s),
            "png": str(png_path),
            "eps_record": (None if eps_path is None else str(eps_path)),
            "delta_norm_sum": float(sum(out.extras["delta_norm_per_step"])),
        })

    write_json(out_dir / "manifest.json", {
        "checkpoint": str(p.checkpoint),
        "lambda_value": float(p.lambda_value),
        "pair_slug": pair_slug,
        "train_pair_slug": train_pair_slug,
        "prompt_a": p.prompt_a,
        "prompt_b": p.prompt_b,
        "joint_prompt": p.joint_prompt,
        "seeds": seeds,
        "samples": manifest,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
