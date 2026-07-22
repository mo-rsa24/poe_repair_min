"""Plan-15 backfill — sample mono / solo_a / solo_b / poe_no_lora /
poe_plus_lora on a list of seeds, recording **both** per-step latents
(z_t) and per-step guided eps (ε̂_t).

Stores arrays in the plan-15 manifold_cache layout:

    outputs/manifold_cache/<pair_slug>/
      mono_bank/{z_t,eps_t}/seed_NNN.npy
      poe_no_lora/{z_t,eps_t}/seed_NNN.npy
      poe_plus_lora/lambda_LL.LL/{z_t,eps_t}/seed_NNN.npy
      solo_a/{z_t,eps_t}/seed_NNN.npy
      solo_b/{z_t,eps_t}/seed_NNN.npy

Per cell:
  - z_t  is shape (num_steps + 1, 4, latent_H, latent_W), float16
         (i.e. (51, 4, 128, 128) at the default 1024² × 50 steps).
  - eps_t is shape (num_steps, 4, latent_H, latent_W), float16,
         storing the *guided* ε used to step (PoE eps for poe_no_lora,
         PoE+λ·R eps for poe_plus_lora, the CFG-guided eps for mono /
         solo_*).
  - decoded.png — the final image, written next to the arrays for
         a free visual sanity check.
  - meta.json — sampler settings, prompts, checkpoint path, λ, seed.

Idempotent: skips cells whose `z_t.npy` already exists unless
``--overwrite`` is set.

Pinned init latents come from the existing training_cache layout
(``poe_repair.training_cache``); no sampling is done from scratch
noise — the z_T is the same one the inspector / probe / sample_heldout
pipelines all use, so trajectories are byte-comparable across pipelines.

Usage (single process — models load once):

    python -m scripts.manifold.sample_with_trajectory \
        --seeds 1,2,3,4,5,6,7,8,9,10,11,12 \
        --conditions mono,poe_no_lora,poe_plus_lora \
        --lambdas 0.00,0.50,1.00 \
        --lora-checkpoint outputs/cross_seed_lora_pooling/task_b_learning_curve/k01_pick1__ep1600/checkpoints/lora_step_080000.pt

Add ``--conditions solo_a,solo_b`` if those bands are wanted (not
required by plan-15 Claim A).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool
from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_cfg,
    run_cfg_poe,
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


log = logging.getLogger("manifold.sample")


PAIR_SLUG_DEFAULT = "a_cat__x__a_dog"
PROMPT_A_DEFAULT = "a cat"
PROMPT_B_DEFAULT = "a dog"
JOINT_PROMPT_DEFAULT = "a cat and a dog"

VALID_CONDITIONS = ("mono", "solo_a", "solo_b", "poe_no_lora", "poe_plus_lora")


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

def cell_dir(out_root: Path, condition: str,
             lambda_value: float | None) -> Path:
    if condition == "poe_plus_lora":
        if lambda_value is None:
            raise ValueError("poe_plus_lora requires a λ value")
        return out_root / "poe_plus_lora" / f"lambda_{lambda_value:.2f}"
    if condition == "mono":
        return out_root / "mono_bank"
    return out_root / condition


def cell_paths(out_root: Path, condition: str,
               lambda_value: float | None, seed: int) -> dict[str, Path]:
    d = cell_dir(out_root, condition, lambda_value)
    seed_tag = f"seed_{seed:03d}"
    return {
        "dir": d,
        "z_t": d / "z_t" / f"{seed_tag}.npy",
        "eps_t": d / "eps_t" / f"{seed_tag}.npy",
        "png": d / "decoded" / f"{seed_tag}.png",
        "meta": d / "meta" / f"{seed_tag}.json",
    }


# ---------------------------------------------------------------------------
# Tracker → numpy
# ---------------------------------------------------------------------------

def tracker_to_arrays(tracker) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_t, eps_t) as float16 numpy arrays.

    tracker.trajectories : (num_steps+1, 1, C, H, W)
    tracker.velocities   : (num_steps,   1, C, H, W)
    """
    z = tracker.trajectories.cpu().float().numpy()
    e = tracker.velocities.cpu().float().numpy()
    # squeeze the batch dim (always 1 here).
    z = z.squeeze(1)
    e = e.squeeze(1)
    return z.astype(np.float16), e.astype(np.float16)


def write_cell(paths: dict[str, Path], *, z: np.ndarray, eps: np.ndarray,
               image: torch.Tensor, meta: dict) -> None:
    paths["z_t"].parent.mkdir(parents=True, exist_ok=True)
    paths["eps_t"].parent.mkdir(parents=True, exist_ok=True)
    paths["png"].parent.mkdir(parents=True, exist_ok=True)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["z_t"], z)
    np.save(paths["eps_t"], eps)
    write_decoded_image(image, paths["png"])
    paths["meta"].write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Per-condition samplers
# ---------------------------------------------------------------------------

def _common_kwargs(init, models, scheduler, embeddings, cfg) -> dict:
    return dict(
        init_latents=init, models=models, scheduler=scheduler,
        seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        height=cfg["height"], width=cfg["width"],
        euler_init_noise_sigma=cfg["euler_init_noise_sigma"],
        device=cfg["device"], dtype=cfg["dtype"],
    )


@torch.no_grad()
def run_cell(condition: str, lambda_value: float | None, *,
             init, models, scheduler, embeddings, cfg,
             adapter_name: str | None):
    """Dispatch to the right sampler for this condition."""
    kw = _common_kwargs(init, models, scheduler, embeddings, cfg)
    if condition == "mono":
        return run_cfg(
            seq_cond=embeddings["seq_j"], pool_cond=embeddings["pool_j"], **kw,
        )
    if condition == "solo_a":
        return run_cfg(
            seq_cond=embeddings["seq_a"], pool_cond=embeddings["pool_a"], **kw,
        )
    if condition == "solo_b":
        return run_cfg(
            seq_cond=embeddings["seq_b"], pool_cond=embeddings["pool_b"], **kw,
        )
    if condition == "poe_no_lora":
        return run_cfg_poe(
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            **kw,
        )
    if condition == "poe_plus_lora":
        if adapter_name is None:
            raise RuntimeError("poe_plus_lora requested but no LoRA was attached")
        return run_lora_residual_inject(
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
            lambda_value=float(lambda_value),
            lora_adapter_name=adapter_name,
            record_delta_at_steps=None,
            correction_max_rel_norm=None,
            **kw,
        )
    raise ValueError(f"unknown condition: {condition}")


# ---------------------------------------------------------------------------
# LoRA attach
# ---------------------------------------------------------------------------

def attach_lora(unet, checkpoint_path: Path) -> tuple[str, dict]:
    """Attach a LoRA adapter to ``unet`` and load state from the checkpoint.
    Returns (adapter_name, ckpt_config_dict)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    rank = int(cfg_dict.get("lora", {}).get("rank", 8))
    alpha = int(cfg_dict.get("lora", {}).get("alpha", rank))
    target_modules = tuple(
        cfg_dict.get("lora", {}).get(
            "target_modules", ("attn2.to_q", "attn2.to_k", "attn2.to_v"),
        )
    )
    adapter_name = str(cfg_dict.get("lora", {}).get("adapter_name", "lora"))
    log.info("LoRA ckpt: rank=%d alpha=%d targets=%s adapter=%s",
             rank, alpha, target_modules, adapter_name)
    from peft import LoraConfig
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
             len(ckpt["lora_state"]), checkpoint_path)
    return adapter_name, cfg_dict


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair-slug", default=PAIR_SLUG_DEFAULT)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--conditions",
                    default="mono,poe_no_lora,poe_plus_lora",
                    help=f"comma list from {VALID_CONDITIONS}")
    ap.add_argument("--lambdas", default="0.00,0.50,1.00")
    ap.add_argument("--lora-checkpoint", default=None,
                    help="required if poe_plus_lora is in --conditions")
    ap.add_argument("--prompt-a", default=PROMPT_A_DEFAULT)
    ap.add_argument("--prompt-b", default=PROMPT_B_DEFAULT)
    ap.add_argument("--joint-prompt", default=JOINT_PROMPT_DEFAULT)
    ap.add_argument("--out-root",
                    default=str(REPO_ROOT / "outputs" / "manifold_cache"))
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32",
                             "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None,
                    help="training_cache root (init-latent source)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-sample cells whose z_t.npy already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-cell plan and exit (no model load)")
    return ap.parse_args(argv)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _plan(seeds: list[int], conditions: list[str],
          lambdas: list[float]) -> list[tuple[int, str, float | None]]:
    plan: list[tuple[int, str, float | None]] = []
    for seed in seeds:
        for cond in conditions:
            if cond == "poe_plus_lora":
                for lam in lambdas:
                    plan.append((seed, cond, lam))
            else:
                plan.append((seed, cond, None))
    return plan


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("MANIFOLD_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    seeds = _parse_int_list(args.seeds)
    conditions = _parse_str_list(args.conditions)
    lambdas = _parse_float_list(args.lambdas)
    for c in conditions:
        if c not in VALID_CONDITIONS:
            raise SystemExit(f"unknown condition: {c}; valid: {VALID_CONDITIONS}")
    needs_lora = "poe_plus_lora" in conditions
    if needs_lora and not args.lora_checkpoint:
        raise SystemExit("poe_plus_lora requires --lora-checkpoint")

    out_root = ensure_dir(Path(args.out_root) / args.pair_slug)
    plan = _plan(seeds, conditions, lambdas)

    # Identify which cells need running.
    to_run: list[tuple[int, str, float | None]] = []
    to_skip: list[tuple[int, str, float | None]] = []
    for (s, cond, lam) in plan:
        paths = cell_paths(out_root, cond, lam, s)
        if paths["z_t"].is_file() and not args.overwrite:
            to_skip.append((s, cond, lam))
        else:
            to_run.append((s, cond, lam))

    log.info("plan: %d cells total, %d to run, %d already cached",
             len(plan), len(to_run), len(to_skip))
    for cell in to_run[:5]:
        log.info("  will run: %s", cell)
    if len(to_run) > 5:
        log.info("  ... (%d more)", len(to_run) - 5)

    if args.dry_run:
        log.info("dry-run mode — no model load. Exiting.")
        return 0
    if not to_run:
        log.info("nothing to do.")
        return 0

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    log.info("device=%s dtype=%s", device, dtype)

    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)

    # Encode prompts once.
    class _PromptShim:
        prompt_a = args.prompt_a
        prompt_b = args.prompt_b
        joint_prompt = args.joint_prompt
    class _CfgShim:
        cell = _PromptShim()
    embeddings = encode_all_prompts(_CfgShim(), models, device, dtype)

    # Attach LoRA only if needed; keep adapter name for the sampler.
    adapter_name: str | None = None
    ckpt_config: dict = {}
    if needs_lora:
        adapter_name, ckpt_config = attach_lora(
            models["unet"], Path(args.lora_checkpoint),
        )

    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    cfg_sampler = dict(
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        height=int(args.height), width=int(args.width),
        euler_init_noise_sigma=float(args.euler_sigma),
        device=device, dtype=dtype,
    )

    # Iterate seeds × conditions. Reload init per seed (cheap).
    for (i, (seed, condition, lam)) in enumerate(to_run, start=1):
        cell = CellPath.from_root(args.pair_slug, int(seed), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(args.euler_sigma),
        )
        paths = cell_paths(out_root, condition, lam, seed)
        log.info("[%d/%d] seed=%d condition=%s λ=%s → %s",
                 i, len(to_run), seed, condition,
                 f"{lam:.2f}" if lam is not None else "—",
                 paths["dir"].relative_to(out_root.parent))

        models["unet"].eval()
        out = run_cell(
            condition, lam,
            init=init, models=models, scheduler=scheduler,
            embeddings=embeddings, cfg=cfg_sampler,
            adapter_name=adapter_name,
        )
        z, eps = tracker_to_arrays(out.tracker)
        meta = {
            "pair_slug": args.pair_slug,
            "seed": int(seed),
            "condition": condition,
            "lambda_value": float(lam) if lam is not None else None,
            "guidance_scale": float(args.guidance_scale),
            "num_inference_steps": int(args.num_inference_steps),
            "height": int(args.height), "width": int(args.width),
            "euler_init_noise_sigma": float(args.euler_sigma),
            "prompt_a": args.prompt_a, "prompt_b": args.prompt_b,
            "joint_prompt": args.joint_prompt,
            "lora_checkpoint": (
                str(args.lora_checkpoint)
                if condition == "poe_plus_lora" else None
            ),
            "lora_config": ckpt_config.get("lora", {}) if condition == "poe_plus_lora" else {},
            "z_t_shape": list(z.shape),
            "eps_t_shape": list(eps.shape),
        }
        write_cell(paths, z=z, eps=eps, image=out.image.cpu(), meta=meta)
        del out

    log.info("done. wrote %d cells to %s", len(to_run), out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
