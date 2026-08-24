"""Cross-pair pooled-LoRA trainer (Plan 15, Task B).

Trains one rank-8 LoRA on the concatenation of cached steps across
``pair_pool.train × seed_pool.train_pool`` (5 pairs × 8 seeds = 40
cells by default). Per-step embedding lookup via
``multi_pair_trainer.train_epoch_multi_pair``.

Mirrors ``cross_seed_lora_pooling.train_pooled`` for everything that
isn't pair-axis-specific (LoRA attach, checkpointing, W&B, resume,
sizing). See plan-15 §B2 for the differences spelled out.

Usage::

    python -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \\
        --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \\
        --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \\
        --pair-prompts outputs/cross_pair_lora_pooling/pair_prompts.yaml \\
        --total-epochs 2400 --epoch-size 50 \\
        --output-root outputs/cross_pair_lora_pooling/all_groups \\
        --run-id main
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

import torch

from poe_repair import paths
from poe_repair.experiments.compose_scorer_validation.scorer import (
    _Embedders,
    score_output,
)
from poe_repair.experiments.cross_pair_lora_pooling import _inline_sampling
from poe_repair.experiments.cross_pair_lora_pooling.multi_pair_trainer import (
    train_epoch_multi_pair,
)
from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair.experiments.cross_pair_lora_pooling.pair_prompts import (
    load_pair_prompts,
)
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import (
    load_seed_pool,
)
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.config import RunConfig
from poe_repair.experiments.one_pair_one_seed.main import (
    REPO_ROOT,
    WandBLogger,
    encode_all_prompts,
)
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, resolve_cells


log = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _load_anchors_by_pair(
    anchors_root: Path = paths.resolve(paths.COMPOSE_SCORER_VALIDATION) / "anchors",
) -> dict[str, dict[str, Path]]:
    """Load the three anchor paths per pair: a_alone, b_alone, joint.

    Returns {pair_slug: {"a_alone": Path, "b_alone": Path, "joint": Path}}.
    """
    anchors = {}
    for pair_dir in anchors_root.iterdir():
        if not pair_dir.is_dir():
            continue
        pair_slug = pair_dir.name
        anchor_a = pair_dir / "anchor_a_alone.png"
        anchor_b = pair_dir / "anchor_b_alone.png"
        anchor_j = pair_dir / "anchor_joint.png"
        if anchor_a.exists() and anchor_b.exists() and anchor_j.exists():
            anchors[pair_slug] = {
                "a_alone": anchor_a,
                "b_alone": anchor_b,
                "joint": anchor_j,
            }
    return anchors


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train_pooled_cross_pair")
    ap.add_argument("--pair-pool", required=True,
                    help="path to outputs/cross_pair_lora_pooling/pair_pool.yaml")
    ap.add_argument("--seed-pool-path", required=True,
                    help="path to outputs/cross_pair_lora_pooling/seed_pool.yaml")
    ap.add_argument("--pair-prompts", required=True,
                    help="path to outputs/cross_pair_lora_pooling/pair_prompts.yaml")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--total-epochs", type=int, default=2400)
    ap.add_argument("--epoch-size", type=int, default=50)
    ap.add_argument("--train-batch-size", type=int, default=1,
                    help="K entries per optimizer step; sampled from a single "
                         "pair so the 3K-wide forward shares one embedding tuple")
    ap.add_argument("--probe-every-epochs", type=int, default=10**9,
                    help="skip the legacy λ-sweep probe")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--wandb-mode", default="disabled",
                    choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-project", default="poe-repair-cross-pair")
    ap.add_argument("--output-root",
                    default=str(paths.resolve(paths.GROUP_POOL_CONFIGS)
                                / "all_groups"))
    ap.add_argument("--run-id", default="auto")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="run 1 epoch then exit")
    ap.add_argument("--log-every-epochs", type=int, default=100)
    ap.add_argument("--ckpt-every-epochs", type=int, default=200)
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--resume-from", default=None,
                    help="path to a previous lora_step_*.pt checkpoint")
    ap.add_argument("--resume-wandb-id", default=None)
    # Inline sampling (per-epoch images + contact sheet) ------------------
    ap.add_argument("--sample-every-epochs", type=int, default=0,
                    help="render PNGs every N epochs (0 disables)")
    ap.add_argument("--sample-cells-per-train-pair", type=int, default=1,
                    help="in_in cells per train pair to sample per render "
                         "(seed = seed_pool.train_pool[0])")
    ap.add_argument("--sample-cells-per-heldout-pair", type=int, default=1,
                    help="out_out cells per heldout pair to sample per render "
                         "(seed = seed_pool.held_out[0])")
    ap.add_argument("--sample-num-inference-steps", type=int, default=20)
    ap.add_argument("--sample-height", type=int, default=1024)
    ap.add_argument("--sample-width", type=int, default=1024)
    ap.add_argument("--sample-thumb", type=int, default=256)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_PAIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_argparser()
    args = parser.parse_args(argv)

    pair_pool = load_pair_pool(args.pair_pool)
    seed_pool = load_seed_pool(args.seed_pool_path)
    prompts = load_pair_prompts(args.pair_prompts, pair_pool=pair_pool)
    log.info("pair_pool.train: %s", list(pair_pool.train))
    log.info("seed_pool.train_pool: %s", list(seed_pool.train_pool))

    # --- RunConfig ---------------------------------------------------------
    cfg = RunConfig()
    cfg.cell.pair_slug = "all_groups"              # display-only; per-step varies
    cfg.cell.seed = int(seed_pool.train_pool[0])   # cosmetic
    cfg.cell.split = "heldout"
    cfg.lora.rank = int(args.lora_rank)
    cfg.lora.alpha = int(args.lora_alpha)
    cfg.optim.lr = float(args.lr)
    cfg.schedule.total_epochs = int(args.total_epochs)
    cfg.schedule.epoch_size = int(args.epoch_size)
    cfg.schedule.train_batch_size = int(args.train_batch_size)
    cfg.probe.every_epochs = int(args.probe_every_epochs)
    cfg.sampler.guidance_scale = float(args.guidance_scale)
    cfg.sampler.num_inference_steps = int(args.num_inference_steps)
    cfg.sampler.height = int(args.height)
    cfg.sampler.width = int(args.width)
    cfg.sampler.euler_init_noise_sigma = float(args.euler_sigma)
    cfg.wandb.project = args.wandb_project
    cfg.wandb.mode = args.wandb_mode
    cfg.model_id = args.model_id
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.dtype = args.dtype
    cfg.seed = int(args.torch_seed)
    cfg.git_commit = _git_commit()
    cfg.skip_scoring = True

    if args.run_id == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.run_id = (
            f"cross_pair_pooled__r{cfg.lora.rank}__lr{cfg.optim.lr:.0e}"
            f"__ep{cfg.schedule.total_epochs}__{ts}"
        )
    else:
        cfg.run_id = args.run_id
    cfg.run_dir = str(Path(args.output_root) / cfg.run_id)

    run_dir = ensure_dir(Path(cfg.run_dir))
    checkpoints_root = ensure_dir(run_dir / "checkpoints")
    write_json(run_dir / "config.json", cfg.to_dict())
    pair_pool.persist_alongside(run_dir)
    seed_pool.persist_alongside(run_dir)
    write_json(run_dir / "pair_prompts.json", {
        slug: {
            "prompt_a": pp.prompt_a, "prompt_b": pp.prompt_b,
            "joint_prompt": pp.joint_prompt,
        }
        for slug, pp in prompts.items()
    })
    log.info("run_dir=%s", run_dir)

    # --- Models, scheduler, LoRA -------------------------------------------
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = infer_device(cfg.device)
    dtype = infer_dtype(cfg.dtype, device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    if args.xformers:
        try:
            models["unet"].enable_xformers_memory_efficient_attention()
            log.info("xformers memory-efficient attention enabled")
        except Exception as exc:
            log.warning("xformers enable failed (%s) — falling back", exc)
    scheduler = load_ddim_scheduler(cfg.model_id)
    attach_info = lora_trainer.attach_lora(models["unet"], cfg)
    log.info("LoRA attached: n_matched=%d trainable_params=%d",
             attach_info["n_matched"], attach_info["trainable_params"])
    write_json(run_dir / "lora_attach.json", {
        "adapter_name": attach_info["adapter_name"],
        "n_matched": attach_info["n_matched"],
        "trainable_params": attach_info["trainable_params"],
        "total_params": attach_info["total_params"],
    })

    # --- Build the multi-pair dataset (tag each step with source_pair) -----
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    dataset_by_pair: dict[str, list[lora_trainer.CachedStep]] = {}
    cells_meta: list[dict] = []
    total_steps = 0
    for pair in pair_pool.train:
        cells = resolve_cells(pair, list(seed_pool.train_pool),
                              cache_root=cache_root)
        pair_steps: list[lora_trainer.CachedStep] = []
        for cell in cells:
            steps = lora_trainer.load_cached_steps(
                cell, guidance_scale=cfg.sampler.guidance_scale,
            )
            for s in steps:
                s.source_pair = pair
            pair_steps.extend(steps)
            cells_meta.append({
                "pair": pair, "seed": int(cell.seed),
                "n_steps": len(steps), "root": str(cell.root),
            })
        dataset_by_pair[pair] = pair_steps
        total_steps += len(pair_steps)
    log.info("pooled dataset: %d cached steps from %d cells across %d pairs",
             total_steps, len(cells_meta), len(pair_pool.train))
    write_json(run_dir / "dataset_meta.json", {
        "n_cells": len(cells_meta),
        "n_steps_total": total_steps,
        "n_steps_per_pair": {p: len(v) for p, v in dataset_by_pair.items()},
        "cells": cells_meta,
    })

    # --- Encode one set of (A, B, J, ∅) embeddings per pair ---------------
    # Train pairs feed the optimizer; heldout pairs are needed by inline
    # sampling's out_out cells. Encoding both keeps a single lookup table.
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]] = {}
    for pair in pair_pool.all_pairs():
        pp = prompts[pair]
        # encode_all_prompts pulls from cfg.cell.prompt_a/b/joint, so we
        # set those transiently before each call.
        cfg.cell.pair_slug = pair
        cfg.cell.prompt_a = pp.prompt_a
        cfg.cell.prompt_b = pp.prompt_b
        cfg.cell.joint_prompt = pp.joint_prompt
        embeddings_by_pair[pair] = encode_all_prompts(cfg, models, device, dtype)
        log.info("encoded prompts for %s", pair)
    cfg.cell.pair_slug = "all_groups"

    # --- Optimizer, scaler, RNG -------------------------------------------
    logger = WandBLogger(
        cfg, run_dir=run_dir,
        resume_id=args.resume_wandb_id,
        resume_mode="must",
    )
    optimizer = lora_trainer.make_optimizer(models["unet"], cfg)
    grad_scaler = lora_trainer.make_grad_scaler(
        enabled=(dtype == torch.float16 and torch.cuda.is_available()),
    )
    state = lora_trainer.TrainerState()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.seed))
    train_dtype = dtype if dtype != torch.float16 else torch.float16

    # --- Resume (optional) -------------------------------------------------
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        log.info("resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        lora_trainer.load_lora_state(models["unet"], ckpt["lora_state"])
        state.epoch = int(ckpt.get("epoch", 0))
        state.optimizer_step = int(ckpt.get("step", 0))
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                log.info("loaded optimizer state from checkpoint")
            except Exception as exc:
                log.warning("could not load optimizer state (%s)", exc)
        if "scaler_state" in ckpt and grad_scaler is not None:
            try:
                grad_scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as exc:
                log.warning("could not load grad scaler state: %s", exc)
        log.info("resumed at epoch=%d optimizer_step=%d",
                 state.epoch, state.optimizer_step)

    # --- Inline-sampling context (optional) -------------------------------
    sampler_ctx = None
    samples_root = ensure_dir(run_dir / "samples" / "per_epoch")
    lora_adapter_name = str(attach_info.get("adapter_name", "lora"))
    sample_every = max(0, int(args.sample_every_epochs))
    if sample_every > 0:
        in_seed = int(seed_pool.train_pool[0])
        out_seed = int(seed_pool.held_out[0])
        plan: list[tuple[str, str, int]] = []
        n_in = max(0, int(args.sample_cells_per_train_pair))
        n_out = max(0, int(args.sample_cells_per_heldout_pair))
        # Cap by available seeds.
        in_seeds = list(seed_pool.train_pool[:n_in])
        out_seeds = list(seed_pool.held_out[:n_out])
        for pair in pair_pool.train:
            for s in in_seeds:
                plan.append(("in_in", pair, int(s)))
        for pair in pair_pool.heldout:
            for s in out_seeds:
                plan.append(("out_out", pair, int(s)))
        log.info("inline sampling: every %d epochs over %d cells "
                 "(in_in=%d, out_out=%d), %d DDIM steps",
                 sample_every, len(plan),
                 len(pair_pool.train) * len(in_seeds),
                 len(pair_pool.heldout) * len(out_seeds),
                 int(args.sample_num_inference_steps))
        sampler_ctx = _inline_sampling.build_context(
            plan=plan, cache_root=cache_root,
            embeddings_by_pair=embeddings_by_pair,
            device=device, dtype=dtype,
            height=int(args.sample_height), width=int(args.sample_width),
            num_inference_steps=int(args.sample_num_inference_steps),
            guidance_scale=float(cfg.sampler.guidance_scale),
            euler_init_noise_sigma=float(cfg.sampler.euler_init_noise_sigma),
        )
        # Δ̄_t across the whole training pool: the fixed reference direction
        # for direction-cosine / fraction-of-distance-reached (plan 02).
        # Static w.r.t. LoRA weights, so built once here, not per eval step.
        pool_mean_delta = _inline_sampling.build_pool_mean_cache(
            train_pairs=list(pair_pool.train),
            train_seeds=list(seed_pool.train_pool),
            cache_root=cache_root,
        )
        record_delta_at_steps = list(range(int(args.sample_num_inference_steps)))
        # Load anchors and compose-scorer embedders (plan 02): one anchor set
        # per pair, used to score each rendered output as compose vs blend live.
        anchors_by_pair = _load_anchors_by_pair()
        embedders = _Embedders(device=device)

    def _run_inline_sample(tag: str) -> None:
        if sampler_ctx is None:
            return
        sub = samples_root / f"epoch_{state.epoch:04d}_step_{state.optimizer_step:06d}"
        rendered = _inline_sampling.sample_all_cells(
            unet=models["unet"], models=models, scheduler=scheduler,
            ctx=sampler_ctx, out_dir=sub,
            lambda_value=1.0, lora_adapter_name=lora_adapter_name,
            record_delta_at_steps=record_delta_at_steps,
        )
        direction_payload: dict[str, float] = {}
        all_cosines, all_alphas, all_compose_rates = [], [], []
        for cell, png, where_applied in rendered:
            key = f"samples/{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
            # 3-column comparison: Mono (target) | PoE (default) | LoRA (this step).
            # Mono/PoE are the fixed cache references; only LoRA evolves over training.
            split = "train" if cell.quadrant == "in_in" else "heldout"
            cache_cell = cache_root / split / cell.pair_slug / f"seed_{cell.seed}"
            mono_p, poe_p = cache_cell / "mono.png", cache_cell / "poe.png"
            if mono_p.exists() and poe_p.exists():
                trip = _inline_sampling.compose_triptych(
                    mono_path=mono_p, poe_path=poe_p, lora_path=png,
                    title=f"{cell.pair_slug}  seed {cell.seed:02d}",
                    lora_label=f"LoRA @ step {state.optimizer_step}",
                    thumb=int(args.sample_thumb),
                )
                trip_path = sub / f"{cell.quadrant}__{cell.pair_slug}__seed{cell.seed:02d}__cmp.png"
                trip.save(trip_path)
                logger.log_image(key, trip_path, step=state.optimizer_step)
            else:
                logger.log_image(key, png, step=state.optimizer_step)
            # Direction axis (plan 02): per-cell cosine to the pool-mean Δ̄_t
            # and the fraction-of-distance-reached, logged as separate curves
            # so a floor compose-rate can be split delivery-null vs no-transfer.
            metrics = _inline_sampling.direction_metrics(where_applied, pool_mean_delta)
            cos_key = f"eval/direction_cosine/{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
            frac_key = f"eval/frac_distance_reached/{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
            direction_payload[cos_key] = metrics["direction_cosine"]
            direction_payload[frac_key] = metrics["frac_distance_reached"]
            if metrics["direction_cosine"] == metrics["direction_cosine"]:  # not nan
                all_cosines.append(metrics["direction_cosine"])
                all_alphas.append(metrics["frac_distance_reached"])
            # Compose-rate scoring (plan 02): score this output against the pair's
            # 3 anchors, log as a separate curve so delivery-null vs no-transfer
            # can be distinguished.
            if cell.pair_slug in anchors_by_pair:
                try:
                    scores = score_output(
                        Path(png),
                        anchors_by_pair[cell.pair_slug],
                        embedders=embedders,
                        margin=0.0,
                    )
                    # Use the "dino" space label (COMPOSE / BLEND); average across spaces.
                    dino_label = scores["dino"].label
                    compose_score = 1.0 if dino_label == "compose" else 0.0
                    compose_key = f"eval/compose_rate/{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
                    direction_payload[compose_key] = compose_score
                    all_compose_rates.append(compose_score)
                except Exception as e:
                    log.warning(
                        "compose-score failed for %s / seed %d: %s",
                        cell.pair_slug, cell.seed, e,
                    )
        if all_cosines:
            direction_payload["eval/direction_cosine/mean"] = (
                sum(all_cosines) / len(all_cosines)
            )
            direction_payload["eval/frac_distance_reached/mean"] = (
                sum(all_alphas) / len(all_alphas)
            )
        if all_compose_rates:
            direction_payload["eval/compose_rate/mean"] = (
                sum(all_compose_rates) / len(all_compose_rates)
            )
        if direction_payload:
            logger.log(direction_payload, step=state.optimizer_step)
        sheet = _inline_sampling.compose_contact_sheet(
            rendered=[(cell, png) for cell, png, _ in rendered],
            title=f"{tag} — epoch {state.epoch} / step {state.optimizer_step}",
            thumb=int(args.sample_thumb),
        )
        sheet_path = sub / "contact_sheet.png"
        sheet.save(sheet_path)
        logger.log_image("samples/contact_sheet", sheet_path,
                         step=state.optimizer_step)

    # --- Dry run -----------------------------------------------------------
    if args.dry_run:
        log.info("--dry-run: running 1 epoch then exiting")
        train_epoch_multi_pair(
            unet=models["unet"], scheduler=scheduler,
            optimizer=optimizer, dataset_by_pair=dataset_by_pair,
            embeddings_by_pair=embeddings_by_pair,
            cfg=cfg, state=state, device=device, train_dtype=train_dtype,
            rng=rng, grad_scaler=grad_scaler,
            logger_callback=lambda payload: logger.log(
                payload, step=state.optimizer_step,
            ),
        )
        _dump_checkpoint(models["unet"], cfg, state, checkpoints_root,
                         optimizer=optimizer, grad_scaler=grad_scaler)
        if sampler_ctx is not None:
            _run_inline_sample("dryrun")
        logger.finish()
        return 0

    # --- Main loop ---------------------------------------------------------
    import time as _time
    log_every = max(1, int(args.log_every_epochs))
    ckpt_every = max(1, int(args.ckpt_every_epochs))
    total_epochs = int(cfg.schedule.total_epochs)
    remaining_epochs = max(0, total_epochs - state.epoch)
    log.info("training plan: target=%d epochs, starting at epoch=%d (remaining=%d)",
             total_epochs, state.epoch, remaining_epochs)
    t_start = _time.time()
    try:
        for _ in range(remaining_epochs):
            t_epoch = _time.time()
            ok = train_epoch_multi_pair(
                unet=models["unet"], scheduler=scheduler,
                optimizer=optimizer, dataset_by_pair=dataset_by_pair,
                embeddings_by_pair=embeddings_by_pair,
                cfg=cfg, state=state, device=device, train_dtype=train_dtype,
                rng=rng, grad_scaler=grad_scaler,
                logger_callback=lambda payload: logger.log(
                    payload, step=state.optimizer_step,
                ),
            )
            if state.epoch % log_every == 0 or state.epoch == 1:
                bl = state.bucket_loss_running or {}
                elapsed = _time.time() - t_start
                remaining = (
                    elapsed / max(1, state.epoch)
                    * (int(cfg.schedule.total_epochs) - state.epoch)
                )
                log.info(
                    "epoch=%d/%d step=%d loss(early/commit/late)=%.4f/%.4f/%.4f "
                    "ep_t=%.1fs elapsed=%.0fs eta=%.0fs",
                    state.epoch, int(cfg.schedule.total_epochs),
                    state.optimizer_step,
                    bl.get("early", float("nan")),
                    bl.get("commit", float("nan")),
                    bl.get("late", float("nan")),
                    _time.time() - t_epoch, elapsed, remaining,
                )
            if state.epoch % ckpt_every == 0:
                _dump_checkpoint(models["unet"], cfg, state, checkpoints_root,
                                 optimizer=optimizer, grad_scaler=grad_scaler)
            if sample_every > 0 and state.epoch % sample_every == 0:
                _run_inline_sample("inline")
            if not ok:
                log.warning("kill criterion: %s", state.aborted_reason)
                write_json(run_dir / "verdict.json", {
                    "verdict": "aborted",
                    "reason": state.aborted_reason,
                    "epoch": state.epoch,
                    "optimizer_step": state.optimizer_step,
                })
                break
    finally:
        _dump_checkpoint(models["unet"], cfg, state, checkpoints_root,
                         optimizer=optimizer, grad_scaler=grad_scaler)
        logger.finish()

    write_json(run_dir / "verdict.json", {
        "verdict": "ok" if state.aborted_reason is None else "aborted",
        "reason": state.aborted_reason,
        "epoch": state.epoch,
        "optimizer_step": state.optimizer_step,
    })
    return 0


def _dump_checkpoint(
    unet, cfg, state, checkpoints_root: Path,
    *, optimizer=None, grad_scaler=None,
) -> Path:
    p = checkpoints_root / f"lora_step_{state.optimizer_step:06d}.pt"
    payload: dict = {
        "lora_state": lora_trainer.lora_state_dict(unet),
        "step": int(state.optimizer_step),
        "epoch": int(state.epoch),
        "config": cfg.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if grad_scaler is not None:
        try:
            payload["scaler_state"] = grad_scaler.state_dict()
        except Exception:
            pass
    torch.save(payload, p)
    (checkpoints_root / "latest.json").write_text(json.dumps({
        "path": str(p),
        "step": int(state.optimizer_step),
        "epoch": int(state.epoch),
    }, indent=2))
    return p


if __name__ == "__main__":
    raise SystemExit(main())
