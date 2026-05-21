"""Pooled-LoRA trainer for the cross-seed pooling experiment.

Trains one LoRA on a concatenation of cached steps from K seeds in the
train pool. Architecture and optimizer match the seed-42 reference run;
only the data pool (and therefore epoch_size scaling) changes.

Per-seed mixing comes for free from the trainer's uniform random sampling
over the dataset list — no scheduler changes needed.

Usage::

    python -m poe_repair.experiments.cross_seed_lora_pooling.train_pooled \
        --k 4 --total-epochs 200 \
        --output-root outputs/cross_seed_lora_pooling/task_b_learning_curve

For the k=1 case, pass ``--single-seed-pick 5`` to override which one
train-pool seed is used (defaults to the first).
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

from poe_repair.experiments.cross_seed_lora_pooling import _inline_sampling
from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool
from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.config import RunConfig
from poe_repair.experiments.lora.main import (
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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train_pooled")
    ap.add_argument("--k", type=int, required=True,
                    help="number of train-pool seeds to use (subset = first k)")
    ap.add_argument("--single-seed-pick", type=int, default=None,
                    help="for k=1: which train-pool seed to use "
                         "(default: train_pool[0])")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--total-epochs", type=int, default=200)
    ap.add_argument("--epoch-size", type=int, default=50)
    ap.add_argument("--train-batch-size", type=int, default=1)
    ap.add_argument("--probe-every-epochs", type=int, default=10**9,
                    help="skip the legacy λ-sweep probe; set lower if you "
                         "want intermediate samples during training")
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
    ap.add_argument("--wandb-project", default="poe-repair-cross-seed")
    ap.add_argument("--output-root",
                    default=str(REPO_ROOT / "outputs" / "cross_seed_lora_pooling"
                                / "task_b_learning_curve"))
    ap.add_argument("--run-id", default="auto")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--seed-pool-path", default=None,
                    help="override path to seed_pool.yaml (default: "
                         "outputs/cross_seed_lora_pooling/seed_pool.yaml)")
    ap.add_argument("--torch-seed", type=int, default=42,
                    help="seed for the trainer's RNG (sampling order). "
                         "Not the cell seed.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-every-epochs", type=int, default=100,
                    help="emit one INFO line every N epochs (default 100)")
    ap.add_argument("--ckpt-every-epochs", type=int, default=200,
                    help="dump intermediate checkpoint every N epochs (default 200)")
    ap.add_argument("--xformers", action="store_true",
                    help="enable xformers memory-efficient attention on the UNet")
    ap.add_argument("--resume-from", default=None,
                    help="path to a previous lora_step_*.pt checkpoint. The "
                         "trainer continues from its (epoch, step) until "
                         "--total-epochs is reached.")
    ap.add_argument("--resume-wandb-id", default=None,
                    help="wandb run id to resume (e.g. 'pueuo7bl'). When set, "
                         "wandb.init uses resume='must' to extend the same run.")
    ap.add_argument("--sample-every-epochs", type=int, default=0,
                    help="render one PNG per seed every N epochs and log to "
                         "wandb. 0 disables inline sampling.")
    ap.add_argument("--sample-seeds", default="all",
                    help="comma-separated seeds; 'all' = train_pool ∪ held_out.")
    ap.add_argument("--sample-num-inference-steps", type=int, default=20,
                    help="DDIM steps for inline samples (final HQ pass is "
                         "rendered post-hoc by render_per_epoch.py).")
    ap.add_argument("--sample-height", type=int, default=1024)
    ap.add_argument("--sample-width", type=int, default=1024)
    ap.add_argument("--sample-thumb", type=int, default=320,
                    help="contact-sheet thumbnail edge in pixels.")
    return ap


def _resolve_pooled_seeds(args, pool) -> list[int]:
    if args.k == 1:
        s = (args.single_seed_pick
             if args.single_seed_pick is not None else pool.train_pool[0])
        if s not in pool.train_pool:
            raise RuntimeError(
                f"--single-seed-pick {s} not in train_pool {list(pool.train_pool)}"
            )
        return [int(s)]
    return list(pool.subset(args.k))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_SEED_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_argparser()
    args = parser.parse_args(argv)
    pool = load_seed_pool(args.seed_pool_path)
    seeds_to_train = _resolve_pooled_seeds(args, pool)

    # Build a RunConfig that mirrors the seed-42 reference config; the
    # cell.seed field becomes meaningless under pooling, so we write the
    # pooled identifier into run_id instead.
    cfg = RunConfig()
    cfg.cell.pair_slug = pool.pair_slug
    cfg.cell.seed = int(seeds_to_train[0])     # purely cosmetic for downstream code
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
    cfg.skip_scoring = True                     # we sample outside the trainer

    if args.run_id == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.run_id = (
            f"lora_pooled_k{args.k:02d}__r{cfg.lora.rank}"
            f"__lr{cfg.optim.lr:.0e}__ep{cfg.schedule.total_epochs}__{ts}"
        )
        if args.k == 1:
            cfg.run_id = (
                f"lora_pooled_k01_pick{seeds_to_train[0]}"
                f"__r{cfg.lora.rank}__lr{cfg.optim.lr:.0e}"
                f"__ep{cfg.schedule.total_epochs}__{ts}"
            )
    else:
        cfg.run_id = args.run_id
    cfg.run_dir = str(Path(args.output_root) / cfg.run_id)

    run_dir = ensure_dir(Path(cfg.run_dir))
    checkpoints_root = ensure_dir(run_dir / "checkpoints")
    write_json(run_dir / "config.json", cfg.to_dict())
    pool.persist_alongside(run_dir)
    write_json(run_dir / "pooled_meta.json", {
        "k": int(args.k),
        "seeds_used": seeds_to_train,
        "single_seed_pick": args.single_seed_pick,
        "train_pool_full": list(pool.train_pool),
        "held_out": list(pool.held_out),
    })
    log.info("run_dir=%s", run_dir)
    log.info("pooling k=%d seeds=%s", args.k, seeds_to_train)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = infer_device(cfg.device)
    dtype = infer_dtype(cfg.dtype, device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    if args.xformers:
        try:
            models["unet"].enable_xformers_memory_efficient_attention()
            log.info("xformers memory-efficient attention enabled on UNet")
        except Exception as exc:
            log.warning("xformers enable failed (%s) — using default attention", exc)
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

    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    cells = resolve_cells(pool.pair_slug, seeds_to_train, cache_root=cache_root)
    dataset = lora_trainer.load_cached_steps_pooled(
        cells, guidance_scale=cfg.sampler.guidance_scale,
    )
    log.info("pooled dataset: %d cached steps from %d cells",
             len(dataset), len(cells))
    write_json(run_dir / "dataset_meta.json", {
        "n_cells": len(cells),
        "n_steps_total": len(dataset),
        "n_steps_per_cell": [
            sum(1 for e in dataset if e.source_seed == int(c.seed))
            for c in cells
        ],
        "cell_seeds": [int(c.seed) for c in cells],
    })

    embeddings = encode_all_prompts(cfg, models, device, dtype)

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

    # ---- resume from a previous checkpoint ------------------------------
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
                log.warning(
                    "could not load optimizer state (%s) — restarting Adam "
                    "moments cold.", exc,
                )
        else:
            log.warning(
                "checkpoint has no optimizer_state — Adam moments restart "
                "cold (one-shot artifact of the v0 checkpoint format).",
            )
        if "scaler_state" in ckpt and grad_scaler is not None:
            try:
                grad_scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as exc:
                log.warning("could not load grad scaler state: %s", exc)
        log.info("resumed at epoch=%d optimizer_step=%d",
                 state.epoch, state.optimizer_step)

    # ---- inline sampling setup ------------------------------------------
    sampler_ctx = None
    if int(args.sample_every_epochs) > 0:
        if args.sample_seeds == "all":
            sample_seeds_list = list(
                dict.fromkeys(list(pool.train_pool) + list(pool.held_out))
            )
        else:
            sample_seeds_list = [
                int(x) for x in args.sample_seeds.split(",") if x.strip()
            ]
        log.info("inline sampling: every %d epochs, seeds=%s, %d DDIM steps",
                 int(args.sample_every_epochs), sample_seeds_list,
                 int(args.sample_num_inference_steps))
        sampler_ctx = _inline_sampling.build_context(
            seeds=sample_seeds_list,
            pair_slug=pool.pair_slug,
            cache_root=cache_root,
            embeddings=embeddings,
            device=device,
            dtype=dtype,
            height=int(args.sample_height),
            width=int(args.sample_width),
            num_inference_steps=int(args.sample_num_inference_steps),
            guidance_scale=float(cfg.sampler.guidance_scale),
            euler_init_noise_sigma=float(cfg.sampler.euler_init_noise_sigma),
            train_pool=set(int(s) for s in pool.train_pool),
            held_out=set(int(s) for s in pool.held_out),
        )

    samples_root = ensure_dir(run_dir / "samples" / "per_epoch")
    lora_adapter_name = str(attach_info.get("adapter_name", "lora"))

    def _run_inline_sample(tag: str) -> None:
        """Sample all seeds at the current LoRA state, save to disk, log to wandb."""
        if sampler_ctx is None:
            return
        sub = samples_root / f"epoch_{state.epoch:04d}_step_{state.optimizer_step:06d}"
        pngs = _inline_sampling.sample_all_seeds(
            unet=models["unet"], models=models, scheduler=scheduler,
            ctx=sampler_ctx, out_dir=sub,
            lambda_value=1.0, lora_adapter_name=lora_adapter_name,
        )
        for s, png in pngs.items():
            logger.log_image(f"samples/seed_{int(s):02d}", png,
                             step=state.optimizer_step)
        sheet = _inline_sampling.compose_contact_sheet(
            pngs_by_seed=pngs,
            seeds=sampler_ctx.seeds,
            train_pool=sampler_ctx.train_pool,
            held_out=sampler_ctx.held_out,
            title=f"{tag} — epoch {state.epoch} / step {state.optimizer_step}",
            thumb=int(args.sample_thumb),
        )
        sheet_path = sub / "contact_sheet.png"
        sheet.save(sheet_path)
        logger.log_image("samples/contact_sheet", sheet_path,
                         step=state.optimizer_step)

    if args.dry_run:
        log.info("--dry-run: running 1 epoch then exiting.")
        lora_trainer.train_epoch(
            unet=models["unet"], scheduler=scheduler,
            optimizer=optimizer, dataset=dataset,
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
            cfg=cfg, state=state, device=device, train_dtype=train_dtype,
            rng=rng, grad_scaler=grad_scaler,
            logger_callback=lambda payload: logger.log(
                payload, step=state.optimizer_step,
            ),
        )
        _dump_checkpoint(models["unet"], cfg, state, checkpoints_root,
                         optimizer=optimizer, grad_scaler=grad_scaler)
        logger.finish()
        return 0

    import time as _time
    log_every = max(1, int(args.log_every_epochs))
    ckpt_every = max(1, int(args.ckpt_every_epochs))
    sample_every = max(0, int(args.sample_every_epochs))
    total_epochs = int(cfg.schedule.total_epochs)
    remaining_epochs = max(0, total_epochs - state.epoch)
    log.info("training plan: target=%d epochs, starting at epoch=%d "
             "(remaining=%d)", total_epochs, state.epoch, remaining_epochs)
    if sampler_ctx is not None and args.resume_from is not None:
        log.info("anchor sample at resume (epoch=%d)", state.epoch)
        _run_inline_sample("anchor")
    t_start = _time.time()
    try:
        for _ in range(remaining_epochs):
            t_epoch = _time.time()
            ok = lora_trainer.train_epoch(
                unet=models["unet"], scheduler=scheduler,
                optimizer=optimizer, dataset=dataset,
                seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
                seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
                seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
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
