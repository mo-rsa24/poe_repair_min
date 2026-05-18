"""M5 entrypoint: orchestrates LoRA training + periodic inference probes.

Usage::

    python -m poe_repair.experiments.m5_lora_sdxl \
        --pair a_cat__x__a_dog --seed 42 --split heldout \
        --total-epochs 200 --probe-every-epochs 50 \
        --lr 1e-4 --lora-rank 8

Add ``--dry-run`` for a fast sanity sweep (build dataset, attach LoRA,
run one probe at epoch 0, exit).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from poe_repair.experiments.m5_lora_sdxl import figures as m5_figures
from poe_repair.experiments.m5_lora_sdxl import probe as m5_probe
from poe_repair.experiments.m5_lora_sdxl import trainer as m5_trainer
from poe_repair.experiments.m5_lora_sdxl.config import (
    RunConfig,
    derive_run_id,
    run_dir_for,
)
from poe_repair.training_cache import CellPath
from poe_repair.runtime import (
    encode_prompt_sdxl,
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)


log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_lambda_grid(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def _parse_int_list(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _parse_int_pair(s: str) -> tuple[int, int]:
    a, b = (p.strip() for p in s.split(","))
    return (int(a), int(b))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="m5_lora_sdxl",
        description="LoRA on SDXL UNet for guided-residual prediction (Phase 2).",
    )
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="heldout", choices=("heldout", "train"))

    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--total-epochs", type=int, default=200)
    ap.add_argument("--epoch-size", type=int, default=50)
    ap.add_argument("--train-batch-size", type=int, default=1,
                    help="cached steps batched per optimizer step. >1 cuts "
                         "wall-clock at the cost of VRAM (B*3 UNet branches).")
    ap.add_argument("--probe-every-epochs", type=int, default=50)
    ap.add_argument("--lambda-grid", type=_parse_lambda_grid,
                    default=(0.0, 0.25, 0.5, 0.75, 1.0))
    ap.add_argument("--commit-window", type=_parse_int_pair, default=(5, 25))
    ap.add_argument("--where-applied-steps", type=_parse_int_list,
                    default=(7, 15, 22))

    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)

    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))

    ap.add_argument("--wandb-project", default="poe-repair-m5-lora")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default="online",
                    choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-tags", default="m5,phase2")

    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--run-id", default="auto",
                    help="auto = derive from timestamp; or any string")
    ap.add_argument("--dry-run", action="store_true",
                    help="attach LoRA, run a single probe at epoch 0, exit.")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="skip GroundingDINO + VQAScore (fast wiring smoke).")
    ap.add_argument("--resume-from", default=None,
                    help="path to a lora_step_NNNNNN.pt checkpoint to resume "
                         "from. Restores LoRA weights + epoch/step counters; "
                         "optimizer moments restart fresh.")

    return ap


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig()
    cfg.cell.pair_slug = args.pair
    cfg.cell.seed = int(args.seed)
    cfg.cell.split = args.split
    cfg.cell.prompt_a = args.prompt_a
    cfg.cell.prompt_b = args.prompt_b
    cfg.cell.joint_prompt = args.joint_prompt

    cfg.lora.rank = int(args.lora_rank)
    cfg.lora.alpha = int(args.lora_alpha)
    cfg.lora.dropout = float(args.lora_dropout)

    cfg.optim.lr = float(args.lr)
    cfg.optim.weight_decay = float(args.weight_decay)
    cfg.optim.grad_clip = float(args.grad_clip)

    cfg.schedule.total_epochs = int(args.total_epochs)
    cfg.schedule.epoch_size = int(args.epoch_size)
    cfg.schedule.train_batch_size = int(args.train_batch_size)

    cfg.probe.every_epochs = int(args.probe_every_epochs)
    cfg.probe.lambda_grid = tuple(args.lambda_grid)
    cfg.probe.commit_window = tuple(args.commit_window)
    cfg.probe.where_applied_steps = tuple(args.where_applied_steps)

    cfg.sampler.guidance_scale = float(args.guidance_scale)
    cfg.sampler.num_inference_steps = int(args.num_inference_steps)
    cfg.sampler.height = int(args.height)
    cfg.sampler.width = int(args.width)
    cfg.sampler.euler_init_noise_sigma = float(args.euler_sigma)

    cfg.wandb.project = args.wandb_project
    cfg.wandb.entity = args.wandb_entity
    cfg.wandb.mode = args.wandb_mode
    cfg.wandb.tags = tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip())

    cfg.model_id = args.model_id
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.dtype = args.dtype
    cfg.seed = int(args.seed)
    cfg.dry_run = bool(args.dry_run)
    cfg.git_commit = _git_commit()
    cfg.skip_scoring = bool(args.skip_scoring)
    cfg.resume_from = args.resume_from

    if args.run_id == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.run_id = derive_run_id(cfg, timestamp=ts)
    else:
        cfg.run_id = args.run_id

    cfg.run_dir = str(run_dir_for(cfg, output_root=Path(args.output_root)))
    return cfg


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------


class WandBLogger:
    """Thin shim around wandb. Falls back to a no-op if mode=disabled or wandb
    is unavailable.
    """

    def __init__(self, cfg: RunConfig, run_dir: Path):
        self.run = None
        self.cfg = cfg
        self.run_dir = run_dir
        self.history_path = run_dir / "history.json"
        self._history: list[dict[str, Any]] = []
        if cfg.wandb.mode == "disabled":
            log.info("W&B disabled by config.")
            return
        try:
            import wandb
        except Exception as exc:
            log.warning("wandb unavailable (%s) — logging to history.json only.", exc)
            return
        if cfg.wandb.mode == "offline":
            os.environ.setdefault("WANDB_MODE", "offline")
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_id,
            group=cfg.cell.pair_slug,
            tags=list(cfg.wandb.tags) + [cfg.cell.pair_slug, f"seed{cfg.cell.seed}"],
            mode=cfg.wandb.mode,
            dir=str(run_dir),
            config=cfg.to_dict(),
            reinit=True,
        )

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is not None:
            try:
                self.run.log(payload, step=step)
            except Exception as exc:
                log.warning("wandb.log failed: %s", exc)
        # Always mirror to history.json for offline analysis.
        rec = dict(payload)
        if step is not None:
            rec["_step"] = int(step)
        self._history.append(rec)

    def log_image(self, key: str, path: Path, *, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            import wandb
            self.run.log({key: wandb.Image(str(path))}, step=step)
        except Exception as exc:
            log.warning("wandb.log_image failed for %s: %s", key, exc)

    def log_table(self, key: str, columns: list[str], rows: list[list], *,
                  step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            import wandb
            self.run.log({key: wandb.Table(columns=columns, data=rows)}, step=step)
        except Exception as exc:
            log.warning("wandb.log_table failed for %s: %s", key, exc)

    def log_artifact(self, path: Path, *, name: str, kind: str = "model",
                     aliases: list[str] | None = None) -> None:
        if self.run is None:
            return
        try:
            import wandb
            artifact = wandb.Artifact(name=name, type=kind)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=aliases or [])
        except Exception as exc:
            log.warning("wandb.log_artifact failed for %s: %s", name, exc)

    def finish(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self._history, indent=2))
        except Exception as exc:
            log.warning("history.json dump failed: %s", exc)
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def encode_all_prompts(cfg: RunConfig, models: dict, device, dtype):
    """Encode (A, B, J, ∅) prompts once at run start. Embeddings stay frozen."""
    sa, pa = encode_prompt_sdxl(cfg.cell.prompt_a, models=models, device=device, dtype=dtype)
    sb, pb = encode_prompt_sdxl(cfg.cell.prompt_b, models=models, device=device, dtype=dtype)
    sj, pj = encode_prompt_sdxl(cfg.cell.joint_prompt, models=models, device=device, dtype=dtype)
    se, pe = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
    return {
        "seq_a": sa, "pool_a": pa,
        "seq_b": sb, "pool_b": pb,
        "seq_j": sj, "pool_j": pj,
        "seq_e": se, "pool_e": pe,
    }


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _do_probe(
    *,
    models,
    scheduler,
    init_latents,
    embeddings,
    cfg,
    state,
    probes_root,
    figures_root,
    checkpoints_root,
    logger: WandBLogger,
    device,
    dtype,
) -> None:
    unet = models["unet"]
    result = m5_probe.run_probe(
        models=models, scheduler=scheduler,
        init_latents=init_latents, embeddings=embeddings,
        cfg=cfg, epoch=state.epoch, optimizer_step=state.optimizer_step,
        probes_root=probes_root, device=device, dtype=dtype,
        skip_scoring=cfg.skip_scoring,
    )

    # Scalars per λ.
    for r in result.results:
        logger.log(
            {
                f"probe/vqa_min/lambda_{r.lam:.2f}": r.metrics.vqa_min,
                f"probe/vqa_mean/lambda_{r.lam:.2f}": r.metrics.vqa_mean,
                f"probe/regime_both_distinct/lambda_{r.lam:.2f}": int(
                    r.metrics.regime == "both_distinct"
                ),
                f"probe/delta_norm_sum/lambda_{r.lam:.2f}": float(
                    sum(r.delta_norm_per_step)
                ),
            },
            step=state.optimizer_step,
        )

    # Tabular.
    logger.log_table(
        "probe/table",
        columns=["epoch", "optimizer_step", "lambda", "vqa_min", "vqa_mean", "regime"],
        rows=[
            [state.epoch, state.optimizer_step, r.lam,
             r.metrics.vqa_min, r.metrics.vqa_mean, r.metrics.regime]
            for r in result.results
        ],
        step=state.optimizer_step,
    )

    # Figures.
    ensure_dir(figures_root)
    curve_path = figures_root / "curve_vqa_vs_lambda.png"
    strip_path = figures_root / f"thumbnails_epoch_{state.epoch:04d}.png"
    grid_path = figures_root / "cumulative_grid.png"
    try:
        m5_figures.render_curve_vqa_vs_lambda(probes_root, curve_path)
        logger.log_image("probe/curve_vqa_vs_lambda", curve_path, step=state.optimizer_step)
    except Exception as exc:
        log.warning("curve render failed: %s", exc)
    try:
        m5_figures.render_thumbnail_strip(probes_root, state.epoch, strip_path)
        logger.log_image("probe/thumbnails_strip", strip_path, step=state.optimizer_step)
    except Exception as exc:
        log.warning("thumbnail render failed: %s", exc)
    try:
        m5_figures.render_cumulative_grid(probes_root, grid_path)
        logger.log_image("probe/cumulative_grid", grid_path, step=state.optimizer_step)
    except Exception as exc:
        log.warning("cumulative grid render failed: %s", exc)

    # Where-applied overlay at the top λ.
    if result.results:
        top_lam = max((r.lam for r in result.results), default=0.0)
        wa_path = figures_root / f"where_applied_epoch_{state.epoch:04d}.png"
        try:
            m5_figures.render_where_applied(
                cfg=cfg, probes_root=probes_root, epoch=state.epoch,
                lam=top_lam, output_path=wa_path,
                image_size=(cfg.sampler.height, cfg.sampler.width),
            )
            logger.log_image("probe/where_applied", wa_path, step=state.optimizer_step)
        except Exception as exc:
            log.warning("where-applied render failed: %s", exc)

    # Checkpoint + artifact.
    ckpt_path = checkpoints_root / f"lora_step_{state.optimizer_step:06d}.pt"
    ensure_dir(checkpoints_root)
    torch.save(
        {
            "lora_state": m5_trainer.lora_state_dict(unet),
            "step": int(state.optimizer_step),
            "epoch": int(state.epoch),
            "config": cfg.to_dict(),
            "probe_summary": [
                {
                    "lambda": r.lam,
                    "vqa_min": r.metrics.vqa_min,
                    "vqa_mean": r.metrics.vqa_mean,
                    "regime": r.metrics.regime,
                }
                for r in result.results
            ],
        },
        ckpt_path,
    )
    logger.log_artifact(
        ckpt_path,
        name=f"lora_state__epoch_{state.epoch:04d}",
        kind="model",
        aliases=["latest"],
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("M5_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = build_config(args)

    run_dir = Path(cfg.run_dir)
    ensure_dir(run_dir)
    probes_root = ensure_dir(run_dir / "probes")
    figures_root = ensure_dir(run_dir / "figures")
    checkpoints_root = ensure_dir(run_dir / "checkpoints")
    write_json(run_dir / "config.json", cfg.to_dict())

    # Repro: torch seed.
    torch.manual_seed(cfg.seed)

    # Models.
    device = infer_device(cfg.device)
    dtype = infer_dtype(cfg.dtype, device)
    log.info("device=%s dtype=%s run_dir=%s", device, dtype, run_dir)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(cfg.model_id)

    # LoRA attach.
    attach_info = m5_trainer.attach_lora(models["unet"], cfg)
    log.info(
        "LoRA attached: n_matched=%d trainable_params=%d total_params=%d",
        attach_info["n_matched"], attach_info["trainable_params"],
        attach_info["total_params"],
    )
    write_json(run_dir / "lora_attach.json", {
        "adapter_name": attach_info["adapter_name"],
        "n_matched": attach_info["n_matched"],
        "trainable_params": attach_info["trainable_params"],
        "total_params": attach_info["total_params"],
        "first_20_matched_modules": attach_info["matched_modules"][:20],
    })

    # Dataset.
    cell = CellPath.from_root(
        cfg.cell.pair_slug, cfg.cell.seed, split=cfg.cell.split,
    )
    dataset = m5_trainer.load_cached_steps(
        cell, guidance_scale=cfg.sampler.guidance_scale,
    )
    log.info("dataset: %d cached steps from %s", len(dataset), cell.root)

    # Prompts + pinned init latent.
    embeddings = encode_all_prompts(cfg, models, device, dtype)
    init_latents = m5_probe.load_pinned_init_latents(
        cell, device=device, dtype=dtype,
        euler_init_noise_sigma=cfg.sampler.euler_init_noise_sigma,
    )

    # Logger.
    logger = WandBLogger(cfg, run_dir=run_dir)

    # Optimizer + GradScaler (fp16 backward path needs it; see trainer.py).
    optimizer = m5_trainer.make_optimizer(models["unet"], cfg)
    grad_scaler = m5_trainer.make_grad_scaler(
        enabled=(dtype == torch.float16 and torch.cuda.is_available()),
    )
    state = m5_trainer.TrainerState()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.seed))

    train_dtype = dtype if dtype != torch.float16 else torch.float16

    # --- Resume from checkpoint (optional) ---------------------------------
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        log.info("resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m5_trainer.load_lora_state(models["unet"], ckpt["lora_state"])
        state.optimizer_step = int(ckpt.get("step", 0))
        state.epoch = int(ckpt.get("epoch", 0))
        log.info(
            "resumed: epoch=%d step=%d (loaded %d LoRA params)",
            state.epoch, state.optimizer_step, len(ckpt["lora_state"]),
        )
        write_json(run_dir / "resumed_from.json", {
            "ckpt_path": str(ckpt_path),
            "resumed_epoch": state.epoch,
            "resumed_step": state.optimizer_step,
            "n_lora_params": len(ckpt["lora_state"]),
        })

    # --- Probe at start (canary; reflects resumed state if --resume-from). --
    log.info("running probe at epoch=%d step=%d", state.epoch, state.optimizer_step)
    _do_probe(
        models=models, scheduler=scheduler, init_latents=init_latents,
        embeddings=embeddings, cfg=cfg, state=state,
        probes_root=probes_root, figures_root=figures_root,
        checkpoints_root=checkpoints_root, logger=logger,
        device=device, dtype=dtype,
    )

    if cfg.dry_run:
        log.info("--dry-run set; exiting after epoch-0 probe.")
        logger.finish()
        return 0

    # --- Training loop ------------------------------------------------------
    try:
        for _ in range(int(cfg.schedule.total_epochs)):
            ok = m5_trainer.train_epoch(
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
            if not ok:
                log.warning(
                    "training aborted by kill criterion (%s); writing verdict.",
                    state.aborted_reason,
                )
                write_json(run_dir / "verdict.md", {
                    "verdict": "aborted",
                    "reason": state.aborted_reason,
                    "epoch": state.epoch,
                    "optimizer_step": state.optimizer_step,
                })
                break
            if state.epoch % int(cfg.probe.every_epochs) == 0:
                _do_probe(
                    models=models, scheduler=scheduler,
                    init_latents=init_latents, embeddings=embeddings,
                    cfg=cfg, state=state, probes_root=probes_root,
                    figures_root=figures_root,
                    checkpoints_root=checkpoints_root, logger=logger,
                    device=device, dtype=dtype,
                )
    finally:
        # --- Final probe (only if the last in-loop probe wasn't already
        # at the current epoch — avoid re-probing the same state) ----------
        last_probe_epoch = (
            state.epoch // int(cfg.probe.every_epochs)
        ) * int(cfg.probe.every_epochs)
        if state.epoch != last_probe_epoch:
            try:
                _do_probe(
                    models=models, scheduler=scheduler,
                    init_latents=init_latents, embeddings=embeddings,
                    cfg=cfg, state=state, probes_root=probes_root,
                    figures_root=figures_root,
                    checkpoints_root=checkpoints_root, logger=logger,
                    device=device, dtype=dtype,
                )
            except Exception as exc:
                log.warning("final probe failed: %s", exc)
        logger.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
