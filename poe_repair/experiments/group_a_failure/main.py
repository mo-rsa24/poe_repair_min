"""Group A entrypoint: orchestrates external-corrector training + periodic
inference probes.

Usage::

    python -m poe_repair.experiments.group_a_failure \\
        --technique latent_cnn \\
        --pair a_cat__x__a_dog --seed 42 --split heldout \\
        --total-epochs 600 --probe-every-epochs 50 --lr 1e-4

Add ``--dry-run`` for a fast sanity sweep (build dataset, build corrector,
run one probe at epoch 0, exit).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from poe_repair.experiments.group_a_failure import figures as ga_figures
from poe_repair.experiments.group_a_failure import probe as ga_probe
from poe_repair.experiments.group_a_failure import trainer as ga_trainer
from poe_repair.experiments.group_a_failure.config import (
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
        prog="group_a_failure",
        description="External score-space corrector training (Stage 3, Group A).",
    )
    ap.add_argument("--technique",
                    choices=("latent_cnn", "latent_unet", "frozen_feature_mlp"),
                    default="latent_cnn")
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="heldout", choices=("heldout", "train"))

    # Architecture knobs (per technique; unused fields are ignored)
    ap.add_argument("--body-channels", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=5)
    ap.add_argument("--base-channels", type=int, default=96)
    ap.add_argument("--blocks-per-level", type=int, default=2)
    ap.add_argument("--head-channels", type=int, default=256)
    ap.add_argument("--head-blocks", type=int, default=3)
    ap.add_argument("--cond-dim", type=int, default=256)

    # Optim / schedule
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--total-epochs", type=int, default=600)
    ap.add_argument("--epoch-size", type=int, default=50)
    ap.add_argument("--train-batch-size", type=int, default=4)

    ap.add_argument("--t-sampler", choices=("uniform", "sigma_weighted", "commit_window"),
                    default="sigma_weighted")
    ap.add_argument("--t-sampler-floor", type=float, default=0.01)

    # Probe
    ap.add_argument("--probe-every-epochs", type=int, default=50)
    ap.add_argument("--lambda-grid", type=_parse_lambda_grid,
                    default=(0.0, 0.25, 0.5, 0.75, 1.0))
    ap.add_argument("--commit-window", type=_parse_int_pair, default=(5, 25))
    ap.add_argument("--where-applied-steps", type=_parse_int_list,
                    default=(7, 15, 22))

    # Sampler
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)

    ap.add_argument("--correction-max-rel-norm", type=float, default=None,
                    help="If set, cap ||r̂_t|| ≤ this fraction × ||ε̃_PoE||.")

    # SDXL / device
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))

    # W&B
    ap.add_argument("--wandb-project", default="poe-repair-group-a")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default="online",
                    choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-tags", default="group_a_failure")

    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--run-id", default="auto",
                    help="auto = derive from timestamp; or any string")
    ap.add_argument("--dry-run", action="store_true",
                    help="build corrector, run a single probe at epoch 0, exit.")
    ap.add_argument("--skip-scoring", action="store_true", default=True,
                    help="skip GroundingDINO + VQAScore (default true for Group A).")
    ap.add_argument("--enable-scoring", dest="skip_scoring", action="store_false",
                    help="turn scoring back on if you want the quantitative panels.")
    ap.add_argument("--resume-from", default=None,
                    help="path to a student_step_NNNNNN.pt checkpoint to resume from.")
    ap.add_argument("--resume-latest", action="store_true",
                    help="auto-discover the latest checkpoint for this "
                         "(technique, pair, seed) under --output-root and "
                         "resume into the *same* run dir. Wins over --resume-from.")
    ap.add_argument("--cache-root", default=None,
                    help="override the training-cache root (default reads "
                         "POE_REPAIR_TRAINING_CACHE env, then "
                         "/datasets/mmolefe/poe_repair_min/outputs/training_cache).")

    return ap


def _discover_latest_checkpoint(
    *,
    output_root: Path,
    technique: str,
    pair_slug: str,
    seed: int,
) -> tuple[Path, Path] | None:
    """Scan ``output_root/group_a_failure/<technique>/<pair>/seed_<N>/*/`` for runs
    that contain a checkpoint. Returns ``(run_dir, ckpt_path)`` for the
    most recent run, or None if nothing found.
    """
    base = output_root / "group_a_failure" / technique / pair_slug / f"seed_{seed}"
    if not base.exists():
        return None
    candidates: list[tuple[float, Path, Path]] = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        ck_dir = run_dir / "checkpoints"
        if not ck_dir.exists():
            continue
        # Prefer the symlink if present, else the highest numbered ckpt.
        ck: Path | None = None
        latest = ck_dir / "student_step_latest.pt"
        if latest.exists():
            ck = latest.resolve() if latest.is_symlink() else latest
        else:
            steps = sorted(ck_dir.glob("student_step_*.pt"))
            steps = [p for p in steps if p.name != "student_step_latest.pt"]
            if steps:
                ck = steps[-1]
        if ck is not None and ck.exists():
            candidates.append((ck.stat().st_mtime, run_dir, ck))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, run_dir, ck = candidates[0]
    return run_dir, ck


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

    cfg.technique.name = args.technique
    cfg.technique.body_channels = int(args.body_channels)
    cfg.technique.n_blocks = int(args.n_blocks)
    cfg.technique.base_channels = int(args.base_channels)
    cfg.technique.blocks_per_level = int(args.blocks_per_level)
    cfg.technique.head_channels = int(args.head_channels)
    cfg.technique.head_blocks = int(args.head_blocks)
    cfg.technique.cond_dim = int(args.cond_dim)

    cfg.optim.lr = float(args.lr)
    cfg.optim.weight_decay = float(args.weight_decay)
    cfg.optim.grad_clip = float(args.grad_clip)

    cfg.schedule.total_epochs = int(args.total_epochs)
    cfg.schedule.epoch_size = int(args.epoch_size)
    cfg.schedule.train_batch_size = int(args.train_batch_size)
    cfg.schedule.t_sampler = args.t_sampler
    cfg.schedule.t_sampler_floor = float(args.t_sampler_floor)

    cfg.probe.every_epochs = int(args.probe_every_epochs)
    cfg.probe.lambda_grid = tuple(args.lambda_grid)
    cfg.probe.commit_window = tuple(args.commit_window)
    cfg.probe.where_applied_steps = tuple(args.where_applied_steps)

    cfg.sampler.guidance_scale = float(args.guidance_scale)
    cfg.sampler.num_inference_steps = int(args.num_inference_steps)
    cfg.sampler.height = int(args.height)
    cfg.sampler.width = int(args.width)
    cfg.sampler.euler_init_noise_sigma = float(args.euler_sigma)

    cfg.correction_max_rel_norm = (
        None if args.correction_max_rel_norm is None
        else float(args.correction_max_rel_norm)
    )

    cfg.wandb.project = args.wandb_project
    cfg.wandb.entity = args.wandb_entity
    cfg.wandb.mode = args.wandb_mode
    cfg.wandb.tags = tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip())

    cfg.model_id = args.model_id
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.dtype = args.dtype
    cfg.seed = int(args.seed)
    cfg.dry_run = bool(args.dry_run)
    cfg.skip_scoring = bool(args.skip_scoring)
    cfg.resume_from = args.resume_from
    cfg.git_commit = _git_commit()

    output_root = Path(args.output_root)

    # --resume-latest: discover the most recent run+ckpt for this cell
    # and continue into the *same* run dir. Overrides --run-id / --resume-from.
    resume_run_dir: Path | None = None
    if getattr(args, "resume_latest", False):
        discovered = _discover_latest_checkpoint(
            output_root=output_root,
            technique=cfg.technique.name,
            pair_slug=cfg.cell.pair_slug,
            seed=cfg.cell.seed,
        )
        if discovered is None:
            raise FileNotFoundError(
                f"--resume-latest: no prior checkpoint for "
                f"{cfg.technique.name}/{cfg.cell.pair_slug}/seed_{cfg.cell.seed} "
                f"under {output_root}"
            )
        resume_run_dir, ckpt_path = discovered
        cfg.resume_from = str(ckpt_path)
        cfg.run_id = resume_run_dir.name
        cfg.run_dir = str(resume_run_dir)
    elif args.run_id == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.run_id = derive_run_id(cfg, timestamp=ts)
        cfg.run_dir = str(run_dir_for(cfg, output_root=output_root))
    else:
        cfg.run_id = args.run_id
        cfg.run_dir = str(run_dir_for(cfg, output_root=output_root))

    return cfg


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------


class WandBLogger:
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
            tags=list(cfg.wandb.tags) + [
                cfg.cell.pair_slug, f"seed{cfg.cell.seed}", cfg.technique.name,
            ],
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
# Embeddings
# ---------------------------------------------------------------------------


def encode_all_prompts(cfg: RunConfig, models: dict, device, dtype):
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
# Probe orchestration
# ---------------------------------------------------------------------------


def _do_probe(
    *,
    models,
    scheduler,
    init_latents,
    embeddings,
    corrector,
    optimizer,
    cfg,
    state,
    probes_root,
    figures_root,
    checkpoints_root,
    logger: WandBLogger,
    device,
    dtype,
) -> None:
    result = ga_probe.run_probe(
        models=models, scheduler=scheduler,
        init_latents=init_latents, embeddings=embeddings,
        corrector=corrector, cfg=cfg, epoch=state.epoch,
        optimizer_step=state.optimizer_step,
        probes_root=probes_root, device=device, dtype=dtype,
        skip_scoring=cfg.skip_scoring,
    )

    # Scalars per λ.
    for r in result.results:
        logger.log(
            {
                f"probe/r_hat_norm_sum/lambda_{r.lam:.2f}": float(
                    sum(r.delta_norm_per_step)
                ),
                f"probe/vqa_min/lambda_{r.lam:.2f}": r.metrics.vqa_min,
                f"probe/regime_both_distinct/lambda_{r.lam:.2f}": int(
                    r.metrics.regime == "both_distinct"
                ),
            },
            step=state.optimizer_step,
        )

    logger.log_table(
        "probe/table",
        columns=["epoch", "optimizer_step", "lambda", "r_hat_norm_sum",
                 "vqa_min", "regime"],
        rows=[
            [state.epoch, state.optimizer_step, r.lam,
             float(sum(r.delta_norm_per_step)),
             r.metrics.vqa_min, r.metrics.regime]
            for r in result.results
        ],
        step=state.optimizer_step,
    )

    # Figures.
    ensure_dir(figures_root)
    curve_path = figures_root / "curve_r_hat_norm_vs_lambda.png"
    strip_path = figures_root / f"thumbnails_epoch_{state.epoch:04d}.png"
    grid_path = figures_root / "cumulative_grid.png"
    try:
        ga_figures.render_curve_r_hat_norm(probes_root, curve_path)
        logger.log_image("probe/curve_r_hat_norm_vs_lambda", curve_path,
                         step=state.optimizer_step)
    except Exception as exc:
        log.warning("curve render failed: %s", exc)
    try:
        ga_figures.render_thumbnail_strip(probes_root, state.epoch, strip_path)
        logger.log_image("probe/thumbnails_strip", strip_path,
                         step=state.optimizer_step)
    except Exception as exc:
        log.warning("thumbnail render failed: %s", exc)
    try:
        ga_figures.render_cumulative_grid(
            probes_root, grid_path,
            technique_label=f"Group A — {cfg.technique.name}",
        )
        logger.log_image("probe/cumulative_grid", grid_path,
                         step=state.optimizer_step)
    except Exception as exc:
        log.warning("cumulative grid render failed: %s", exc)

    if result.results:
        top_lam = max((r.lam for r in result.results), default=0.0)
        wa_path = figures_root / f"where_applied_epoch_{state.epoch:04d}.png"
        try:
            ga_figures.render_where_applied(
                cfg=cfg, probes_root=probes_root, epoch=state.epoch,
                lam=top_lam, output_path=wa_path,
                image_size=(cfg.sampler.height, cfg.sampler.width),
            )
            logger.log_image("probe/where_applied", wa_path,
                             step=state.optimizer_step)
        except Exception as exc:
            log.warning("where-applied render failed: %s", exc)

    # Kill check: r̂_t collapse at top λ across consecutive probes. Skipped
    # at optimizer_step==0 because zero-init correctors output exactly 0 by
    # design — that's the canary, not a collapse.
    top = max(result.results, key=lambda r: r.lam, default=None)
    if top is not None and state.optimizer_step > 0:
        r_hat_sum = float(sum(top.delta_norm_per_step))
        if r_hat_sum < 0.5:
            state.r_collapse_streak += 1
            if state.r_collapse_streak >= cfg.kill.r_collapse_n_probes:
                state.aborted_reason = (
                    f"r̂_t collapsed (sum<0.5 for "
                    f"{state.r_collapse_streak} consecutive probes)"
                )
                log.warning("kill: %s", state.aborted_reason)
        else:
            state.r_collapse_streak = 0

    # Checkpoint.
    ckpt_path = checkpoints_root / f"student_step_{state.optimizer_step:06d}.pt"
    ensure_dir(checkpoints_root)
    payload = {
        "model_state": corrector.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(state.optimizer_step),
        "epoch": int(state.epoch),
        "config": cfg.to_dict(),
        "probe_summary": [
            {
                "lambda": r.lam,
                "r_hat_norm_sum": float(sum(r.delta_norm_per_step)),
                "vqa_min": r.metrics.vqa_min,
                "regime": r.metrics.regime,
            }
            for r in result.results
        ],
    }
    torch.save(payload, ckpt_path)
    latest = checkpoints_root / "student_step_latest.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)
    except Exception as exc:
        log.warning("symlink student_step_latest.pt failed: %s", exc)
    logger.log_artifact(
        ckpt_path,
        name=f"student_state__epoch_{state.epoch:04d}",
        kind="model",
        aliases=["latest"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("GROUP_A_LOG_LEVEL", "INFO"),
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

    torch.manual_seed(cfg.seed)

    device = infer_device(cfg.device)
    dtype = infer_dtype(cfg.dtype, device)
    log.info("device=%s dtype=%s run_dir=%s", device, dtype, run_dir)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(cfg.model_id)

    # Freeze SDXL — Group A never trains UNet params.
    for p in models["unet"].parameters():
        p.requires_grad_(False)

    # Build corrector.
    corrector = ga_trainer.build_corrector(
        cfg, unet=models["unet"], device=device, dtype=dtype,
    )
    summary = ga_trainer.attach_summary(corrector, cfg)
    log.info(
        "corrector: %s params=%d trainable=%d",
        summary["technique"], summary["num_parameters"], summary["num_trainable"],
    )
    write_json(run_dir / "attach.json", summary)

    # Dataset.
    from poe_repair.training_cache import DEFAULT_CACHE_ROOT
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    cell = CellPath.from_root(
        cfg.cell.pair_slug, cfg.cell.seed, split=cfg.cell.split,
        cache_root=cache_root,
    )
    dataset = ga_trainer.load_cached_steps(
        cell, guidance_scale=cfg.sampler.guidance_scale,
    )
    log.info("dataset: %d cached steps from %s", len(dataset), cell.root)

    # Embeddings + pinned init latent.
    embeddings = encode_all_prompts(cfg, models, device, dtype)
    init_latents = ga_probe.load_pinned_init_latents(
        cell, device=device, dtype=dtype,
        euler_init_noise_sigma=cfg.sampler.euler_init_noise_sigma,
    )

    # t-sampler.
    t_sampler = ga_trainer.TSampler.build(
        dataset,
        mode=cfg.schedule.t_sampler,
        commit_window=cfg.probe.commit_window,
        floor=cfg.schedule.t_sampler_floor,
        seed=cfg.seed,
    )

    # Optimizer + state.
    optimizer = ga_trainer.make_optimizer(corrector, cfg)
    state = ga_trainer.TrainerState()
    logger = WandBLogger(cfg, run_dir=run_dir)

    # Resume.
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        log.info("resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        corrector.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as exc:
                log.warning("optimizer state restore failed (%s) — restarting moments.", exc)
        state.optimizer_step = int(ckpt.get("step", 0))
        state.epoch = int(ckpt.get("epoch", 0))
        log.info("resumed: epoch=%d step=%d", state.epoch, state.optimizer_step)
        write_json(run_dir / "resumed_from.json", {
            "ckpt_path": str(ckpt_path),
            "resumed_epoch": state.epoch,
            "resumed_step": state.optimizer_step,
        })

    # Epoch-0 probe (canary; reflects resumed state if --resume-from).
    log.info("running probe at epoch=%d step=%d", state.epoch, state.optimizer_step)
    _do_probe(
        models=models, scheduler=scheduler, init_latents=init_latents,
        embeddings=embeddings, corrector=corrector, optimizer=optimizer,
        cfg=cfg, state=state,
        probes_root=probes_root, figures_root=figures_root,
        checkpoints_root=checkpoints_root, logger=logger,
        device=device, dtype=dtype,
    )

    if cfg.dry_run:
        log.info("--dry-run set; exiting after epoch-0 probe.")
        logger.finish()
        return 0

    # Training loop. ``total_epochs`` is the absolute target — resuming at
    # epoch=600 with --total-epochs 1800 does the next 1200 epochs.
    target_epochs = int(cfg.schedule.total_epochs)
    if state.epoch >= target_epochs:
        log.info(
            "current epoch %d already ≥ --total-epochs %d; running final probe only.",
            state.epoch, target_epochs,
        )
    try:
        while state.epoch < target_epochs:
            ga_trainer.train_epoch(
                net=corrector, optimizer=optimizer,
                dataset=dataset, sampler=t_sampler,
                seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
                cfg=cfg, state=state, device=device,
                logger_callback=lambda payload: logger.log(
                    payload, step=state.optimizer_step,
                ),
            )
            if state.aborted_reason:
                write_json(run_dir / "verdict.json", {
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
                    corrector=corrector, optimizer=optimizer,
                    cfg=cfg, state=state, probes_root=probes_root,
                    figures_root=figures_root,
                    checkpoints_root=checkpoints_root, logger=logger,
                    device=device, dtype=dtype,
                )
                if state.aborted_reason:
                    write_json(run_dir / "verdict.json", {
                        "verdict": "aborted",
                        "reason": state.aborted_reason,
                        "epoch": state.epoch,
                        "optimizer_step": state.optimizer_step,
                    })
                    break
    finally:
        # Final probe if we haven't already probed this epoch.
        last_probe_epoch = (
            state.epoch // int(cfg.probe.every_epochs)
        ) * int(cfg.probe.every_epochs)
        if state.epoch != last_probe_epoch and not state.aborted_reason:
            try:
                _do_probe(
                    models=models, scheduler=scheduler,
                    init_latents=init_latents, embeddings=embeddings,
                    corrector=corrector, optimizer=optimizer,
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
