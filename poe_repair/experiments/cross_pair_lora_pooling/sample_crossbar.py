"""Four-quadrant sampler for the cross-pair × cross-seed LoRA experiment.

Loads a trained pooled-LoRA checkpoint and, for each (pair, seed) cell
selected by the quadrant policy, runs ``run_lora_residual_inject``
once. Writes one PNG per cell plus a ``cells.jsonl`` manifest consumed
by the contact-sheet renderer and the Δ̄_t bridge.

Quadrant policy (matches Plan 15 §C1):

    Quadrant  Pair source   Seed source                  n_cells
    in/in     pair_pool.train  seed_pool.train_pool[0]   5
    in/out    pair_pool.train  seed_pool.held_out (4)    20
    out/in    pair_pool.heldout seed_pool.train_pool[0:2] 10
    out/out   pair_pool.heldout seed_pool.held_out (4)   20

Usage::

    python -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \\
        --checkpoint outputs/cross_pair_lora_pooling/all_groups/main/checkpoints/lora_step_<final>.pt \\
        --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \\
        --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \\
        --pair-prompts outputs/cross_pair_lora_pooling/pair_prompts.yaml \\
        --out-dir outputs/cross_pair_lora_pooling/all_groups/main/samples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair.experiments.cross_pair_lora_pooling.pair_prompts import (
    load_pair_prompts,
)
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import (
    load_seed_pool,
)
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

QUADRANTS = ("in_in", "in_out", "out_in", "out_out")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sample_crossbar")
    ap.add_argument("--checkpoint", required=True,
                    help="path to lora_step_*.pt from train_pooled")
    ap.add_argument("--pair-pool", required=True)
    ap.add_argument("--seed-pool-path", required=True)
    ap.add_argument("--pair-prompts", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="root for {quadrant}/{pair}/seed_{S}/lora.png + cells.jsonl")
    ap.add_argument("--quadrants", default=",".join(QUADRANTS),
                    help="comma-separated subset of: in_in,in_out,out_in,out_out")
    ap.add_argument("--out-in-train-seeds", type=int, default=2,
                    help="for out_in: number of train-pool seeds to sample per held pair")
    ap.add_argument("--record-eps", action="store_true",
                    help="dump per-step guided ε for Task D bridge")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--lambda", dest="lambda_value", type=float, default=1.0,
                    help="1.0 = full LoRA; 0.0 = frozen PoE")
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    return ap


def _attach_lora_from_ckpt(unet, ckpt_path: Path) -> tuple[dict, str]:
    """Mirror sample_heldout's attach-then-load recipe."""
    from peft import LoraConfig
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    rank = int(cfg_dict.get("lora", {}).get("rank", 8))
    alpha = int(cfg_dict.get("lora", {}).get("alpha", rank))
    target_modules = tuple(
        cfg_dict.get("lora", {}).get(
            "target_modules", ("attn2.to_q", "attn2.to_k", "attn2.to_v"),
        )
    )
    adapter_name = str(cfg_dict.get("lora", {}).get("adapter_name", "lora"))
    log.info("ckpt config: rank=%d alpha=%d targets=%s", rank, alpha, target_modules)
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=list(target_modules), init_lora_weights=True,
    )
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
             len(ckpt["lora_state"]), ckpt_path)
    return cfg_dict, adapter_name


def _build_cell_plan(
    pair_pool, seed_pool, *, quadrants: list[str], out_in_train_seeds: int,
) -> list[tuple[str, str, int]]:
    """Return list of (quadrant, pair_slug, seed)."""
    plan: list[tuple[str, str, int]] = []
    if "in_in" in quadrants:
        seed = int(seed_pool.train_pool[0])
        for pair in pair_pool.train:
            plan.append(("in_in", pair, seed))
    if "in_out" in quadrants:
        for pair in pair_pool.train:
            for s in seed_pool.held_out:
                plan.append(("in_out", pair, int(s)))
    if "out_in" in quadrants:
        chosen = seed_pool.train_pool[: int(out_in_train_seeds)]
        for pair in pair_pool.heldout:
            for s in chosen:
                plan.append(("out_in", pair, int(s)))
    if "out_out" in quadrants:
        for pair in pair_pool.heldout:
            for s in seed_pool.held_out:
                plan.append(("out_out", pair, int(s)))
    return plan


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_PAIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argparser().parse_args(argv)
    pair_pool = load_pair_pool(args.pair_pool)
    seed_pool = load_seed_pool(args.seed_pool_path)
    prompts = load_pair_prompts(args.pair_prompts, pair_pool=pair_pool)
    out_dir = ensure_dir(Path(args.out_dir))
    quadrants = [q.strip() for q in args.quadrants.split(",") if q.strip()]
    for q in quadrants:
        if q not in QUADRANTS:
            raise ValueError(f"unknown quadrant {q!r}; choose from {QUADRANTS}")

    plan = _build_cell_plan(
        pair_pool, seed_pool,
        quadrants=quadrants,
        out_in_train_seeds=int(args.out_in_train_seeds),
    )
    log.info("crossbar plan: %d cells across quadrants=%s",
             len(plan), quadrants)

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    cfg_dict, adapter_name = _attach_lora_from_ckpt(
        models["unet"], Path(args.checkpoint),
    )

    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT

    # Encode embeddings once per unique pair appearing in the plan.
    unique_pairs = sorted({pair for _, pair, _ in plan})
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]] = {}
    for pair in unique_pairs:
        pp = prompts[pair]
        class _PromptShim:
            prompt_a = pp.prompt_a
            prompt_b = pp.prompt_b
            joint_prompt = pp.joint_prompt
        class _CfgShim:
            cell = _PromptShim()
        embeddings_by_pair[pair] = encode_all_prompts(
            _CfgShim(), models, device, dtype,
        )
        log.info("encoded prompts for %s", pair)

    cells_path = out_dir / "cells.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    manifest: list[dict] = []
    unet = models["unet"]
    unet.eval()
    t_start = time.time()
    for i, (quadrant, pair, seed) in enumerate(plan, 1):
        cell_dir = ensure_dir(out_dir / quadrant / pair / f"seed_{seed:02d}")
        png_path = cell_dir / "lora.png"
        eps_path = (cell_dir / "eps_record.pt"
                    if args.record_eps else None)
        log.info("[%d/%d] quadrant=%s pair=%s seed=%d → %s",
                 i, len(plan), quadrant, pair, seed, png_path)
        try:
            cell = CellPath.from_root(pair, int(seed), cache_root=cache_root)
        except FileNotFoundError as exc:
            log.warning("cache miss for %s/seed_%d: %s", pair, seed, exc)
            continue
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(args.euler_sigma),
        )
        emb = embeddings_by_pair[pair]
        out = run_lora_residual_inject(
            init_latents=init, models=models, scheduler=scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"],
            seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_j=emb["seq_j"], pool_j=emb["pool_j"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            height=int(args.height), width=int(args.width),
            euler_init_noise_sigma=float(args.euler_sigma),
            device=device, dtype=dtype,
            lambda_value=float(args.lambda_value),
            lora_adapter_name=adapter_name,
            record_eps_path=eps_path,
        )
        write_decoded_image(out.image, png_path)
        row = {
            "quadrant": quadrant,
            "pair_slug": pair,
            "seed": int(seed),
            "png_path": str(png_path),
            "eps_record_path": (None if eps_path is None else str(eps_path)),
            "cache_root": str(cell.root),
            "delta_norm_sum": float(sum(out.extras["delta_norm_per_step"])),
        }
        manifest.append(row)
        with cells_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    elapsed = time.time() - t_start
    log.info("done: %d cells in %.0f s (%.1f s/cell)",
             len(manifest), elapsed,
             elapsed / max(1, len(manifest)))
    write_json(out_dir / "manifest.json", {
        "checkpoint": str(args.checkpoint),
        "n_cells_planned": len(plan),
        "n_cells_sampled": len(manifest),
        "quadrants": quadrants,
        "lambda_value": float(args.lambda_value),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
