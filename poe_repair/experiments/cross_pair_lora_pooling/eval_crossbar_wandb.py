"""Mid-training crossbar eval — sample in_in + out_in from the latest
checkpoint under a Plan 16 within-group run and push the results to W&B.

Designed to run from a second shell while training is still in progress.
It resolves the most-recent ``lora_step_*.pt`` under ``<run_dir>/checkpoints/``,
samples one PNG per (quadrant, pair, seed) cell using the same renderer
the inline sampler uses (`run_lora_residual_inject`), composes per-quadrant
contact sheets, and logs everything to W&B under a dedicated
``eval_crossbar/*`` namespace.

Attaching to the live training run
----------------------------------
Pass ``--wandb-run-id <ID>`` and ``--wandb-project <NAME>`` to resume the
existing run (``resume='allow'``). To avoid colliding with the training
writer, all metrics use a custom step axis ``eval_crossbar/ckpt_step``
declared via ``wandb.define_metric``, so the panels render even if the
internal ``_step`` is shared with the training process.

Outputs on disk
---------------
``<run_dir>/eval_crossbar/step_<NNNNNN>/<quadrant>/<pair>/seed_<S>/lora.png``
``<run_dir>/eval_crossbar/step_<NNNNNN>/<quadrant>__contact_sheet.png``
``<run_dir>/eval_crossbar/step_<NNNNNN>/cells.jsonl``
``<run_dir>/eval_crossbar/step_<NNNNNN>/manifest.json``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import torch

from poe_repair.experiments.cross_pair_lora_pooling import _inline_sampling
from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair.experiments.cross_pair_lora_pooling.pair_prompts import (
    load_pair_prompts,
)
from poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar import (
    QUADRANTS,
    _attach_lora_from_ckpt,
    _build_cell_plan,
)
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import (
    load_seed_pool,
)
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT


log = logging.getLogger(__name__)


_CKPT_STEP_RE = re.compile(r"lora_step_(\d+)\.pt$")


def _resolve_latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints directory at {ckpt_dir}")
    candidates = sorted(
        ckpt_dir.glob("lora_step_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no lora_step_*.pt under {ckpt_dir}")
    latest = candidates[-1]
    m = _CKPT_STEP_RE.search(latest.name)
    step = int(m.group(1)) if m else 0
    return latest, step


def _encode_prompts_for(pair_slug: str, prompts_obj, models, device, dtype):
    pp = prompts_obj[pair_slug]

    class _PromptShim:
        prompt_a = pp.prompt_a
        prompt_b = pp.prompt_b
        joint_prompt = pp.joint_prompt

    class _CfgShim:
        cell = _PromptShim()

    return encode_all_prompts(_CfgShim(), models, device, dtype)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eval_crossbar_wandb")
    ap.add_argument("--run-dir", required=True,
                    help="root of the trained pooled-LoRA run "
                         "(e.g. outputs/cross_pair_lora_pooling/within_group/g6/main)")
    ap.add_argument("--pair-pool", required=True)
    ap.add_argument("--seed-pool-path", required=True)
    ap.add_argument("--pair-prompts", required=True)
    ap.add_argument("--quadrants", default="in_in,out_in",
                    help="comma-separated subset of: in_in,in_out,out_in,out_out")
    ap.add_argument("--out-in-train-seeds", type=int, default=12,
                    help="for out_in: number of train-pool seeds per held pair")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--lambda", dest="lambda_value", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--thumb", type=int, default=256,
                    help="contact-sheet thumbnail size (px)")
    # W&B attach
    ap.add_argument("--wandb-run-id", default=None,
                    help="resume into this W&B run id (resume='allow')")
    ap.add_argument("--wandb-project", default="poe-repair-cross-pair")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default="online",
                    choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-namespace", default="eval_crossbar",
                    help="key prefix under which images/sheets are logged")
    return ap


def _init_wandb(args, run_dir: Path):
    if args.wandb_mode == "disabled":
        log.info("W&B disabled by --wandb-mode disabled")
        return None
    try:
        import wandb
    except Exception as exc:
        log.warning("wandb unavailable (%s) — proceeding without W&B", exc)
        return None
    if args.wandb_mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        dir=str(run_dir),
        reinit=True,
    )
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    # Custom step axis so our writes don't fight the live training process.
    step_metric = f"{args.wandb_namespace}/ckpt_step"
    wandb.define_metric(step_metric)
    wandb.define_metric(f"{args.wandb_namespace}/*", step_metric=step_metric)
    return run


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_PAIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argparser().parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    ckpt_path, ckpt_step = _resolve_latest_checkpoint(run_dir)
    log.info("latest checkpoint: %s (step=%d)", ckpt_path, ckpt_step)

    pair_pool = load_pair_pool(args.pair_pool)
    seed_pool = load_seed_pool(args.seed_pool_path)
    prompts = load_pair_prompts(args.pair_prompts, pair_pool=pair_pool)

    quadrants = [q.strip() for q in args.quadrants.split(",") if q.strip()]
    for q in quadrants:
        if q not in QUADRANTS:
            raise ValueError(f"unknown quadrant {q!r}; choose from {QUADRANTS}")
    plan = _build_cell_plan(
        pair_pool, seed_pool,
        quadrants=quadrants,
        out_in_train_seeds=int(args.out_in_train_seeds),
    )
    log.info("crossbar plan: %d cells over quadrants=%s", len(plan), quadrants)
    if not plan:
        log.warning("empty plan; nothing to sample")
        return 0

    eval_root = ensure_dir(
        run_dir / "eval_crossbar" / f"step_{ckpt_step:06d}"
    )
    log.info("eval outputs → %s", eval_root)

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    cfg_dict, adapter_name = _attach_lora_from_ckpt(models["unet"], ckpt_path)

    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT

    unique_pairs = sorted({pair for _, pair, _ in plan})
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]] = {}
    for pair in unique_pairs:
        embeddings_by_pair[pair] = _encode_prompts_for(
            pair, prompts, models, device, dtype,
        )
        log.info("encoded prompts for %s", pair)

    sampler_ctx = _inline_sampling.build_context(
        plan=plan,
        cache_root=cache_root,
        embeddings_by_pair=embeddings_by_pair,
        device=device, dtype=dtype,
        height=int(args.height), width=int(args.width),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        euler_init_noise_sigma=float(args.euler_sigma),
    )

    wb_run = _init_wandb(args, run_dir)
    ns = args.wandb_namespace
    step_metric_key = f"{ns}/ckpt_step"

    t_start = time.time()
    rendered = _inline_sampling.sample_all_cells(
        unet=models["unet"], models=models, scheduler=scheduler,
        ctx=sampler_ctx, out_dir=eval_root,
        lambda_value=float(args.lambda_value),
        lora_adapter_name=adapter_name,
    )
    elapsed = time.time() - t_start
    log.info("sampled %d cells in %.0f s (%.1f s/cell)",
             len(rendered), elapsed,
             elapsed / max(1, len(rendered)))

    cells_path = eval_root / "cells.jsonl"
    if cells_path.exists():
        cells_path.unlink()
    manifest: list[dict] = []
    with cells_path.open("a") as f:
        for cell, png, _where_applied in rendered:
            row = {
                "quadrant": cell.quadrant,
                "pair_slug": cell.pair_slug,
                "seed": int(cell.seed),
                "png_path": str(png),
                "ckpt_step": int(ckpt_step),
                "checkpoint": str(ckpt_path),
            }
            manifest.append(row)
            f.write(json.dumps(row) + "\n")

    if wb_run is not None:
        import wandb
        for cell, png, _where_applied in rendered:
            key = (f"{ns}/{cell.quadrant}/{cell.pair_slug}"
                   f"/seed_{cell.seed:02d}")
            wb_run.log({
                key: wandb.Image(str(png)),
                step_metric_key: int(ckpt_step),
            })

    # Per-quadrant contact sheets.
    rendered_by_q: dict[str, list] = {}
    for cell, png, _where_applied in rendered:
        rendered_by_q.setdefault(cell.quadrant, []).append((cell, png))
    for q, cells in rendered_by_q.items():
        sheet = _inline_sampling.compose_contact_sheet(
            rendered=cells,
            title=f"{q} — ckpt step {ckpt_step}",
            thumb=int(args.thumb),
        )
        sheet_path = eval_root / f"{q}__contact_sheet.png"
        sheet.save(sheet_path)
        log.info("contact sheet (%s) → %s", q, sheet_path)
        if wb_run is not None:
            import wandb
            wb_run.log({
                f"{ns}/{q}__contact_sheet": wandb.Image(str(sheet_path)),
                step_metric_key: int(ckpt_step),
            })

    write_json(eval_root / "manifest.json", {
        "checkpoint": str(ckpt_path),
        "ckpt_step": int(ckpt_step),
        "n_cells_planned": len(plan),
        "n_cells_sampled": len(rendered),
        "quadrants": quadrants,
        "out_in_train_seeds": int(args.out_in_train_seeds),
        "lambda_value": float(args.lambda_value),
        "guidance_scale": float(args.guidance_scale),
        "num_inference_steps": int(args.num_inference_steps),
        "wandb_run_id": args.wandb_run_id,
        "wandb_project": args.wandb_project,
        "wandb_namespace": ns,
    })

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception as exc:
            log.warning("wandb finish failed: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
