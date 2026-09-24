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
        --pair-pool artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml \\
        --seed-pool-path artifacts/_shared/cross_pair_pool_configs/seed_pool.yaml \\
        --pair-prompts artifacts/_shared/cross_pair_pool_configs/pair_prompts.yaml \\
        --total-epochs 2400 --epoch-size 50 \\
        --output-root artifacts/rung4-scale/cross_pair/all_groups \\
        --run-id main
"""

from __future__ import annotations

import argparse
import math
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

# The frozen tracking-set manifest (plan 06, task 1.2), same path
# scripts/showcase/freeze_tracking_set.py writes to. Fixed so both A's and
# B's runs, and this trainer's own smoke run, read the same file.
SCOPE_TRACKING_SET_PATH = (
    REPO_ROOT / "artifacts/results/does-the-fix-reach-unseen-pairs"
    / "pooled_lora" / "tracking_set.json"
)


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
                    help="path to artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml")
    ap.add_argument("--seed-pool-path", required=True,
                    help="path to artifacts/_shared/cross_pair_pool_configs/seed_pool.yaml")
    ap.add_argument("--pair-prompts", required=True,
                    help="path to artifacts/_shared/cross_pair_pool_configs/pair_prompts.yaml")
    ap.add_argument("--branch-prompt-style", default="plain",
                    choices=("plain", "plurality", "connective"),
                    help="What the three branches are conditioned on. plain: the pair prompts "
                         "and the empty null, which is every run before this flag existed. "
                         "plurality: each concept prompt gains ', two animals' and the null "
                         "becomes 'two animals', so the composition divides by the "
                         "plurality-conditioned base rather than the unconditional one. "
                         "connective: 'a cat and' / 'a dog and' against a null of 'and'. "
                         "The three reach the same composed predictions at a fixed state, so a "
                         "difference between them is optimisation and inductive bias, never "
                         "capacity.")
    ap.add_argument("--render-teacher", action="store_true",
                    help="V6: train against the noise added to a render a person picked, "
                         "instead of against the joint-prompt prediction. The teacher becomes "
                         "the mono render judged by eye to show both concepts. Needs "
                         "scripts/showcase/build_render_teacher_cache.py run first.")
    ap.add_argument("--render-teacher-cache",
                    default="/datasets/mmolefe/poe_repair_min/artifacts/caches/render_teacher_cache",
                    help="where build_render_teacher_cache.py wrote its latents.")
    ap.add_argument("--adapt-null-only", action="store_true",
                    help="V3: the mirror of --freeze-null. Both concept branches come from the "
                         "cache and only the empty branch is adapted, so the one free tensor is "
                         "shared by every pair. Expected to fail; run for the record.")
    ap.add_argument("--no-dial", action="store_true",
                    help="V4: drop the guidance weight from both sides. The composition becomes "
                         "the raw (a + b - u) and the target the RAW cached joint prediction "
                         "rather than the guided one. With --freeze-null this is V1 up to the "
                         "constant w^2; on its own it is a different objective from V0.")
    ap.add_argument("--freeze-null", action="store_true",
                    help="V1 of the correction-loss variations: compose against the CACHED empty "
                         "branch instead of the adapted one, so the adapter can only move a and b. "
                         "The empty branch is re-read at -(w-1) during sampling, where a move the "
                         "loss cannot see is multiplied by 6.5, and this closes that direction.")
    ap.add_argument("--null-anchor", type=float, default=0.0,
                    help="Weight on ||eps_theta(null) - eps_frozen(null)||^2. The fit loss "
                         "pins only eps_1 + eps_2 - eps_null, while inference reads the null "
                         "branch again at -(w-1), so a perturbation the loss cannot see moves "
                         "the sample by 6.5x at w=7.5. Requires --branch-prompt-style plain, "
                         "because the cached eps_uncond is the frozen counterpart of the empty "
                         "null only. 0 disables.")
    ap.add_argument("--orth-weight", type=float, default=1.0,
                    help="multiplier on the part of the error that lies outside the plane the two "
                         "experts can already reach on their own. 1.0 is the plain mean-square "
                         "error and takes a short-circuit that skips the projection entirely.")
    ap.add_argument("--train-step-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="train only on cached steps whose index is in [LO, HI). The softness the "
                         "adapter adds is already in the running estimate by step 20, so the run "
                         "that fits only the early steps is the one that can be told apart from the "
                         "run that fits all fifty. Default: every step in the cell.")
    ap.add_argument("--cells", type=str, default=None,
                    help="JSON {pair_slug: [seeds]} naming exactly the cells to train on. The pool "
                         "this scope built is not a cross product: each pair keeps only the seeds a "
                         "person judged good, so pair_pool.train and seed_pool.train_pool cannot "
                         "express it. When given, both are ignored for the training set and the "
                         "pairs come from this file's keys.")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lora-rank-self", type=int, default=0,
                    help="rank for the attn1 (self-attention) projections only, with --lora-targets "
                         "cross+self. 0 keeps them at --lora-rank. The cross-attention half uses about "
                         "four of its thirty-two directions and the self-attention half about fourteen, "
                         "so only this half has a reason to grow.")
    ap.add_argument("--lora-targets", default="cross", choices=("cross", "cross+self"),
                    help="cross: attn2 q/k/v only (every run before 2026-09-22). cross+self adds the "
                         "attn1 q/k/v, where look-alike subjects leak features into each other "
                         "(Dahary et al. 2024, 2403.16990, section 4.3).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-decay", default="none", choices=("none", "cosine"),
                    help="none: constant --lr (every run before 2026-09-22). cosine: fall from --lr at the "
                         "start (or resume) epoch to 0 at the last epoch, set once per epoch, so late "
                         "checkpoints stop wandering between renders.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="per-step EMA decay of the LoRA weights, saved as lora_state_ema in every checkpoint; "
                         "applied once per epoch as decay**epoch_size; 0 disables")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW decoupled weight decay on the LoRA factors. Every pooled run "
                         "before experiment D (scope 01 plan 15) trained at 0.0.")
    ap.add_argument("--total-epochs", type=int, default=2400)
    ap.add_argument("--epoch-size", type=int, default=50)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the UNet before training. Costs a few minutes of graph "
                         "capture once, then runs the same arithmetic faster. Off by default "
                         "because PEFT LoRA and compile do not always agree.")
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
    ap.add_argument("--compose", default="poe", choices=("poe", "superdiff"),
                   help="composition rule the step trains against: poe (original) or "
                        "superdiff (SuperDiff AND blend at --kappa; plan 08, scope 06)")
    ap.add_argument("--kappa", type=float, default=0.5,
                   help="SuperDiff blending weight when --compose superdiff")
    ap.add_argument("--loss-space", default="eps", choices=("eps", "x0"),
                   help="eps: unweighted noise-space MSE (every run before scope 01 plan 16); "
                        "x0: the same error weighted by (1 - abar_t)/abar_t clipped at "
                        "--loss-weight-cap, i.e. the MSE between the two clean estimates")
    ap.add_argument("--loss-weight-cap", type=float, default=22.0,
                   help="clip on the x0 weight; 22.0 is the factor at DDIM step 10 (t 781), so "
                        "steps 0 to 10 share one weight and later steps fall off as 1/SNR")
    ap.add_argument("--huber-delta", type=float, default=0.0,
                    help="Huber on the fit term with this elbow, in units of the noise-prediction "
                         "error; quadratic below it, linear above. 0 keeps the squared error, which "
                         "is every run before 2026-09-24.")
    ap.add_argument("--out-of-span-only", type=float, default=0.0,
                    help="1 trains on the part of the correction no re-weighting of the two "
                         "experts could supply, leaving the damping to be applied at sampling "
                         "time with --expert-weights. 0 is the whole correction, which is every "
                         "run before 2026-09-24.")
    ap.add_argument("--contrast-weight", type=float, default=0.0,
                    help="nu for the contrast term (scope 09, experiment 6). Charges the adapter "
                         "for the fraction by which the clean picture its composition is heading "
                         "for is narrower in grey levels than the frozen one, one-sided so adding "
                         "contrast is never charged. 0 is off, which is every run before this "
                         "flag existed. Read train/contrast_shrink to see what it is charging, "
                         "and read the renders by eye: a contrast term can be satisfied by adding "
                         "grain, and this repo's sharpness measure would reward that.")
    ap.add_argument("--energy-penalty", type=float, default=0.0,
                    help="beta on the control energy of the adapter's own correction (plan 19, "
                         "scope 01): loss = fit + beta * mean_k[w_k * mean(r_hat_k^2)], w_k the "
                         "Girsanov weight of the sample's timestep normalised to mean 1 on the "
                         "DDIM grid. 0.0 is the original trainer.")
    ap.add_argument("--commit-window", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="step-index window the 'commit' loss bucket and the kill criteria read "
                        "(default: config's (5, 25), tuned for 50-step cells; a 200-step cache "
                        "wants the same fraction, 20 100)")
    ap.add_argument("--kill-halve-after-steps", type=int, default=None,
                   help="override cfg.kill.commit_bucket_halve_after_steps; a value past the run length, e.g. 10**9, disables that rule (0 makes it fire at once)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run 1 epoch then exit")
    ap.add_argument("--log-every-epochs", type=int, default=100)
    ap.add_argument("--ckpt-every-epochs", type=int, default=200)
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute UNet activations in the backward pass: same arithmetic, about a third slower, "
                         "fits rank-32 training on a 24 GB card (plan 20 of scope 01)")
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
    ap.add_argument("--sample-train-pairs", default=None,
                    help="comma-separated subset of pair_pool.train to "
                         "sample/track (inline eval only — training still "
                         "uses the full pair_pool.train). Default: all.")
    ap.add_argument("--sample-heldout-pairs", default=None,
                    help="comma-separated subset of pair_pool.heldout to "
                         "sample/track. Default: all.")
    ap.add_argument("--sample-num-inference-steps", type=int, default=20)
    ap.add_argument("--sample-height", type=int, default=1024)
    ap.add_argument("--sample-width", type=int, default=1024)
    ap.add_argument("--sample-thumb", type=int, default=256)
    return ap


def _branch_prompts(prompt_a: str, prompt_b: str, style: str) -> tuple[str, str, str]:
    """The two concept prompts and the null prompt for one branch-conditioning style.

    Strings are concatenated before tokenisation, which is the only well-defined way to
    combine prompts here: SDXL encodes a whole string to a fixed 77 positions in each of two
    tokenizers, so adding two encoded sequences would align token slot k of one prompt onto
    slot k of the other and sum padding embeddings that no string maps to.
    """
    if style == "plain":
        return prompt_a, prompt_b, ""
    if style == "plurality":
        return f"{prompt_a}, two animals", f"{prompt_b}, two animals", "two animals"
    if style == "connective":
        return f"{prompt_a} and", f"{prompt_b} and", "and"
    raise ValueError(f"unknown branch prompt style {style!r}")


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
    if args.cells:
        with open(args.cells) as fh:
            for _p in json.load(fh):
                if _p not in prompts:
                    raise SystemExit(
                        f"--cells names {_p!r} but the prompt registry has no entry for it; "
                        f"regenerate the prompts yaml from the same verdicts")
    log.info("pair_pool.train: %s", list(pair_pool.train))
    log.info("seed_pool.train_pool: %s", list(seed_pool.train_pool))

    # --- RunConfig ---------------------------------------------------------
    cfg = RunConfig()
    cfg.cell.pair_slug = "all_groups"              # display-only; per-step varies
    cfg.cell.seed = int(seed_pool.train_pool[0])   # cosmetic
    cfg.cell.split = "heldout"
    cfg.lora.rank = int(args.lora_rank)
    cfg.lora.alpha = int(args.lora_alpha)
    if args.lora_targets == "cross+self":
        cross = tuple(cfg.lora.target_modules)
        cfg.lora.target_modules = cross + tuple(t.replace("attn2.", "attn1.") for t in cross)
        if int(args.lora_rank_self) > 0:
            r_self = int(args.lora_rank_self)
            cfg.lora.rank_pattern = {t.replace("attn2.", "attn1."): r_self for t in cross}
            cfg.lora.alpha_pattern = dict(cfg.lora.rank_pattern)
    cfg.optim.lr = float(args.lr)
    cfg.optim.weight_decay = float(args.weight_decay)
    cfg.huber_delta = float(args.huber_delta)
    cfg.out_of_span_only = float(args.out_of_span_only)
    cfg.schedule.total_epochs = int(args.total_epochs)
    cfg.schedule.epoch_size = int(args.epoch_size)
    cfg.orth_weight = float(args.orth_weight)
    # Branch conditioning and the null anchor (the correction-loss variants, idea walk
    # designing-the-correction-loss). Read by _train_one_step via getattr; defaults unchanged.
    cfg.freeze_null = bool(args.freeze_null)
    cfg.adapt_null_only = bool(args.adapt_null_only)
    cfg.no_dial = bool(args.no_dial)
    cfg.render_teacher = bool(args.render_teacher)
    if cfg.freeze_null and cfg.adapt_null_only:
        raise SystemExit(
            "--freeze-null (V1) and --adapt-null-only (V3) are mirrors of each other: one "
            "adapts only the concept branches, the other only the empty branch. Setting both "
            "leaves nothing adapted, so the run would train a correction that cannot move.")
    cfg.branch_prompt_style = str(args.branch_prompt_style)
    if args.branch_prompt_style != "plain":
        log.warning(
            "branch prompt style %r: the cached eps_a_raw/eps_b_raw/eps_uncond were built for "
            "the plain prompts, so train/delta_hat_norm compares this run's correction against "
            "a frozen composition it no longer starts from. The fit loss and its target are "
            "unaffected. Read delta_target_norm and loss_fit instead.",
            args.branch_prompt_style)
    cfg.null_anchor = float(args.null_anchor)
    if cfg.null_anchor > 0.0 and cfg.branch_prompt_style != "plain":
        raise SystemExit(
            f"--null-anchor {cfg.null_anchor} with --branch-prompt-style "
            f"{cfg.branch_prompt_style}: the anchor pulls the adapted null toward the cached "
            f"eps_uncond, which is the frozen prediction for the empty prompt and not for this "
            f"style's null. Anchoring against the wrong reference is worse than not anchoring.")
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
    # Composition rule the trainer's step composes the three branches with. "poe" is the
    # original; "superdiff" (plan 08, scope 06) learns SuperDiff's residual at a fixed kappa
    # from a cache built by scripts/build_superdiff_cache.py. Read by
    # one_pair_one_seed.trainer._train_one_step via getattr, so the default is unchanged.
    cfg.compose = str(args.compose)
    cfg.kappa = float(args.kappa)
    # Loss space (plan 16, scope 01): read by _train_one_step via getattr, default unchanged.
    cfg.loss_space = str(args.loss_space)
    cfg.loss_weight_cap = float(args.loss_weight_cap)
    # Energy penalty (plan 19, scope 01): read by _train_one_step via getattr, default 0.0.
    cfg.energy_penalty = float(args.energy_penalty)
    cfg.contrast_weight = float(args.contrast_weight)
    cfg.gradient_checkpointing = bool(args.gradient_checkpointing)
    if args.commit_window is not None:
        cfg.probe.commit_window = (int(args.commit_window[0]), int(args.commit_window[1]))
    if args.kill_halve_after_steps is not None:
        cfg.kill.commit_bucket_halve_after_steps = int(args.kill_halve_after_steps)

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
    write_json(run_dir / "config.json", {
        **cfg.to_dict(), "compose": cfg.compose, "kappa": cfg.kappa,
        "ema_decay": float(args.ema_decay),
        "loss_space": cfg.loss_space, "loss_weight_cap": cfg.loss_weight_cap,
        "energy_penalty": cfg.energy_penalty,
        "contrast_weight": cfg.contrast_weight,
        "branch_prompt_style": cfg.branch_prompt_style,
        "null_anchor": cfg.null_anchor,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "freeze_null": cfg.freeze_null,
        "adapt_null_only": cfg.adapt_null_only,
        "no_dial": cfg.no_dial,
        "render_teacher": cfg.render_teacher,
        "render_teacher_cache": (args.render_teacher_cache if cfg.render_teacher else None),
        # Task 1.4 of the four instrument fixes: without these three a saved config cannot
        # identify its own objective, so a checkpoint cannot be attributed to a variation.
        "train_step_range": (list(args.train_step_range)
                             if args.train_step_range is not None else None),
        "orth_weight": float(getattr(cfg, "orth_weight", 1.0)),
        # There is no --exclude-cells flag: exclusion is expressed by which cells the
        # --cells file names, so the file itself is what identifies the training set.
        "cells_file": args.cells,
    })
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
    if getattr(args, "compile", False):
        # Captured after xformers and before gradient checkpointing, so the graph reflects the
        # attention path actually used. Off by default: PEFT LoRA and compile do not always agree.
        log.info("torch.compile: capturing the UNet graph, this takes a few minutes")
        models["unet"] = torch.compile(models["unet"])
    if args.gradient_checkpointing:
        models["unet"].enable_gradient_checkpointing()
        log.info("gradient checkpointing enabled on the UNet")
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
    explicit_cells = None
    if args.cells:
        with open(args.cells) as fh:
            explicit_cells = {k: [int(x) for x in v] for k, v in json.load(fh).items()}
        train_pairs = sorted(explicit_cells)
        log.info("explicit cell list: %d cells across %d pairs from %s",
                 sum(len(v) for v in explicit_cells.values()), len(train_pairs), args.cells)
    else:
        train_pairs = list(pair_pool.train)

    for pair in train_pairs:
        seeds = explicit_cells[pair] if explicit_cells else list(seed_pool.train_pool)
        # Task 1.5: name the split. CellPath.from_root searches ["heldout", "train"] when
        # split is None, so a cell present in both directories is taken from heldout/ and
        # trained on with no warning (known-failures poe-data-001). Naming it makes a cell
        # missing from train/ raise instead of falling back.
        cells = resolve_cells(pair, seeds, split="train", cache_root=cache_root)
        pair_steps: list[lora_trainer.CachedStep] = []
        for cell in cells:
            steps = lora_trainer.load_cached_steps(
                cell, guidance_scale=cfg.sampler.guidance_scale,
            )
            if args.train_step_range is not None:
                lo, hi = args.train_step_range
                steps = [st for st in steps if lo <= int(st.step_index) < hi]
                if not steps:
                    raise SystemExit(
                        f"--train-step-range {lo} {hi} selected no steps in {pair} seed "
                        f"{cell.seed}; a flag whose target group is empty is a silent no-op")
            for s in steps:
                s.source_pair = pair
            if cfg.render_teacher:
                # V6: attach the picked render's latent to every step of this cell. One
                # tensor per cell, shared by its steps; the noise is drawn per step later.
                _lat_path = (Path(args.render_teacher_cache) / pair
                             / f"seed_{int(cell.seed)}.pt")
                if not _lat_path.exists():
                    raise SystemExit(
                        f"--render-teacher but no latent at {_lat_path}. Run "
                        f"scripts/showcase/build_render_teacher_cache.py --cells {args.cells} "
                        f"first; a missing cell would otherwise shrink the corpus silently.")
                _lat = torch.load(_lat_path, map_location="cpu")["latent"]
                for s in steps:
                    s.render_latent = _lat
            pair_steps.extend(steps)
            cells_meta.append({
                "pair": pair, "seed": int(cell.seed), "split": cell.split,
                "n_steps": len(steps), "root": str(cell.root),
            })
        dataset_by_pair[pair] = pair_steps
        total_steps += len(pair_steps)
    _rng = ("all" if args.train_step_range is None
            else "[%d, %d) = %d per cell" % (args.train_step_range[0], args.train_step_range[1],
                                             args.train_step_range[1] - args.train_step_range[0]))
    log.info("pooled dataset: %d cached steps from %d cells across %d pairs; step range %s",
             total_steps, len(cells_meta), len(train_pairs), _rng)
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
        pa, pb, null_prompt = _branch_prompts(
            pp.prompt_a, pp.prompt_b, cfg.branch_prompt_style)
        cfg.cell.prompt_a = pa
        cfg.cell.prompt_b = pb
        cfg.cell.joint_prompt = pp.joint_prompt
        embeddings_by_pair[pair] = encode_all_prompts(
            cfg, models, device, dtype, null_prompt=null_prompt)
        log.info("encoded prompts for %s: a=%r b=%r null=%r", pair, pa, pb, null_prompt)
    cfg.cell.pair_slug = "all_groups"

    # --- Optimizer, scaler, RNG -------------------------------------------
    logger = WandBLogger(
        cfg, run_dir=run_dir,
        resume_id=args.resume_wandb_id,
        resume_mode="must",
    )
    # Frozen tracking-set manifest (plan 06): non-fatal if absent, so runs
    # that predate this plan (e.g. the original phase1_r8_100k) still work.
    tracking_set_path = SCOPE_TRACKING_SET_PATH
    if tracking_set_path.exists():
        tracking_set = json.loads(tracking_set_path.read_text())
        tracking_hash = tracking_set.get("manifest_hash", "unknown")
        log.info("tracking_set.json frozen: hash=%s (%d renders)",
                 tracking_hash, tracking_set.get("n_renders", -1))
        if logger.run is not None:
            logger.run.config.update({
                "tracking_set_hash": tracking_hash,
                "tracking_set_path": str(tracking_set_path),
            })
    else:
        log.warning("tracking_set.json not found at %s — eval/tracking/* "
                    "will still log, but no manifest hash in config",
                    tracking_set_path)
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
        # Sampling pairs can be a fixed subset of the training pool (plan 06:
        # one in-pair, one out-pair, so the live tracking pass stays cheap).
        # Training itself is untouched — it still uses the full pair_pool.train
        # via dataset_by_pair above, built before this block.
        sample_train_pairs = (
            [p.strip() for p in args.sample_train_pairs.split(",") if p.strip()]
            if args.sample_train_pairs else list(pair_pool.train)
        )
        sample_heldout_pairs = (
            [p.strip() for p in args.sample_heldout_pairs.split(",") if p.strip()]
            if args.sample_heldout_pairs else list(pair_pool.heldout)
        )
        for pair in sample_train_pairs:
            for s in in_seeds:
                plan.append(("in_in", pair, int(s)))
        for pair in sample_heldout_pairs:
            for s in out_seeds:
                plan.append(("out_out", pair, int(s)))
        log.info("inline sampling: every %d epochs over %d cells "
                 "(in_in=%d, out_out=%d), %d DDIM steps",
                 sample_every, len(plan),
                 len(sample_train_pairs) * len(in_seeds),
                 len(sample_heldout_pairs) * len(out_seeds),
                 int(args.sample_num_inference_steps))
        sampler_ctx = _inline_sampling.build_context(
            plan=plan, cache_root=cache_root,
            embeddings_by_pair=embeddings_by_pair,
            device=device, dtype=dtype,
            height=int(args.sample_height), width=int(args.sample_width),
            num_inference_steps=int(args.sample_num_inference_steps),
            guidance_scale=float(cfg.sampler.guidance_scale),
            freeze_null=bool(cfg.freeze_null),
            adapt_null_only=bool(cfg.adapt_null_only),
            euler_init_noise_sigma=float(cfg.sampler.euler_init_noise_sigma),
        )
        # Δ̄_t across the whole training pool: the fixed reference direction
        # for direction-cosine / fraction-of-distance-reached (plan 02).
        # Static w.r.t. LoRA weights, so built once here, not per eval step.
        pool_mean_delta = _inline_sampling.build_pool_mean_cache(
            train_pairs=(sorted(explicit_cells) if explicit_cells else list(pair_pool.train)),
            train_seeds=list(seed_pool.train_pool),
            cache_root=cache_root,
            cells_by_pair=explicit_cells,
        )
        record_delta_at_steps = list(range(int(args.sample_num_inference_steps)))
        # Tracking-set buckets/representative steps (plan 06) come straight
        # from cfg.probe, the same window train/loss_bucket/* already uses —
        # one definition of "commit", not a second one invented here.
        tracking_rep_steps = tuple(int(s) for s in cfg.probe.where_applied_steps)
        tracking_spectral_step = (
            tracking_rep_steps[-1] if tracking_rep_steps
            else _inline_sampling.TRACKING_SPECTRAL_STEP_DEFAULT
        )
        commit_lo, commit_hi = (int(x) for x in cfg.probe.commit_window)
        tracking_buckets = {
            "early": (0, commit_lo),
            "commit": (commit_lo, commit_hi),
            "late": (commit_hi, int(args.sample_num_inference_steps) - 1),
        }
        if int(args.sample_num_inference_steps) <= commit_hi:
            log.warning(
                "sample-num-inference-steps=%d does not cover cfg.probe."
                "commit_window's upper bound (%d): the 'late' tracking "
                "bucket and step %d will be empty",
                int(args.sample_num_inference_steps), commit_hi,
                tracking_rep_steps[-1] if tracking_rep_steps else -1,
            )
        # Load anchors and compose-scorer embedders (plan 02): one anchor set
        # per pair, used to score each rendered output as compose vs blend live.
        anchors_by_pair = _load_anchors_by_pair()
        embedders = _Embedders(device=device)

    def _run_inline_sample(tag: str) -> None:
        if sampler_ctx is None:
            return
        sub = samples_root / f"epoch_{state.epoch:04d}_step_{state.optimizer_step:06d}"
        all_cosines, all_alphas, all_compose_rates = [], [], []
        tracking_accum: dict[str, list[float]] = {}
        spectral_deltas: list[torch.Tensor] = []

        def _on_cell_done(cell, png, where_applied) -> None:
            # Streamed per cell (plan 06 fix): logs to W&B as each cell
            # renders, instead of silently rendering all cells first and
            # bursting everything to W&B at the end — the gap was long
            # enough at 152 cells x 50 DDIM steps to look like a hang.
            cell_payload: dict[str, float] = {}
            key = f"samples/{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
            # 3-column comparison: Mono (target) | PoE (default) | LoRA (this step).
            # Mono/PoE are the fixed cache references; only LoRA evolves over training.
            split = "train" if cell.quadrant == "in_in" else "heldout"
            cache_cell = cache_root / split / cell.pair_slug / f"seed_{cell.seed}"
            mono_p, poe_p = cache_cell / "mono.png", cache_cell / "poe.png"
            if mono_p.exists() and poe_p.exists():
                # 2x3: row 1 the adapter on the whole path, row 2 on over 0..T/2 and off after.
                # The half-path render sits beside the full one, written by the same sampling
                # pass. If it is missing the run is on an older sampler, so fall back to the
                # three-panel strip rather than dropping the cell from W&B.
                png_half = png.with_name(png.stem + "__half.png")
                if png_half.exists():
                    trip = _inline_sampling.compose_two_row_grid(
                        mono_path=mono_p, poe_path=poe_p,
                        lora_full_path=png, lora_half_path=png_half,
                        title=f"{cell.pair_slug}  seed {cell.seed:02d}",
                        lora_label=f"LoRA @ step {state.optimizer_step}",
                        thumb=int(args.sample_thumb),
                    )
                else:
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
            cell_payload[cos_key] = metrics["direction_cosine"]
            cell_payload[frac_key] = metrics["frac_distance_reached"]
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
                    cell_payload[compose_key] = compose_score
                    all_compose_rates.append(compose_score)
                except Exception as e:
                    log.warning(
                        "compose-score failed for %s / seed %d: %s",
                        cell.pair_slug, cell.seed, e,
                    )
            # Tracking-set additions (plan 06): learned-vs-actual cosine,
            # embedding drift, divergence-step profile — one read per cell,
            # keyed under eval/tracking/. Spectral share is cross-cell, built
            # from spectral_deltas after every cell is done. Buckets and
            # representative steps come from cfg.probe, so "commit" here
            # means the same window as train/loss_bucket/commit.
            tracking = _inline_sampling.tracking_reads(
                cell=cell, split=split, where_applied=where_applied,
                cache_root=cache_root, lora_path=Path(png),
                mono_path=mono_p, poe_path=poe_p, embedders=embedders,
                buckets=tracking_buckets, rep_steps=tracking_rep_steps,
            )
            for name, val in tracking.items():
                cell_key = (
                    f"eval/tracking/{name}/"
                    f"{cell.quadrant}/{cell.pair_slug}/seed_{cell.seed:02d}"
                )
                cell_payload[cell_key] = val
                tracking_accum.setdefault(name, []).append(val)
            spectral_step_rec = where_applied.get(tracking_spectral_step)
            if spectral_step_rec is not None:
                spectral_deltas.append(spectral_step_rec["delta_hat"])
            if cell_payload:
                logger.log(cell_payload, step=state.optimizer_step)

        rendered = _inline_sampling.sample_all_cells(
            unet=models["unet"], models=models, scheduler=scheduler,
            ctx=sampler_ctx, out_dir=sub,
            lambda_value=1.0, lora_adapter_name=lora_adapter_name,
            record_delta_at_steps=record_delta_at_steps,
            on_cell_done=_on_cell_done,
        )
        # Means only need the full accumulation, so they still land in one
        # payload after the last cell — everything per-cell already streamed.
        direction_payload: dict[str, float] = {}
        for name, vals in tracking_accum.items():
            if vals:
                direction_payload[f"eval/tracking/{name}/mean"] = (
                    sum(vals) / len(vals)
                )
        spec = _inline_sampling.spectral_share(spectral_deltas, top_k=3)
        for name, val in spec.items():
            direction_payload[f"eval/tracking/spectral_share/{name}"] = val
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

    # --- EMA of the LoRA weights (plan 20) ---------------------------------
    ema = None
    ema_epoch_decay = 0.0
    if float(args.ema_decay) > 0.0:
        ema = _ema_init(models["unet"])
        ema_epoch_decay = float(args.ema_decay) ** int(cfg.schedule.epoch_size)
        if args.resume_from is not None and "lora_state_ema" in ckpt:
            for k, t in ckpt["lora_state_ema"].items():
                if k in ema:
                    ema[k].copy_(t.to(device=ema[k].device, dtype=ema[k].dtype))
            log.info("resumed lora_state_ema (%d tensors)", len(ckpt["lora_state_ema"]))
        log.info("EMA on: per-step decay %.5f, per-epoch decay %.5f",
                 float(args.ema_decay), ema_epoch_decay)

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
                         optimizer=optimizer, grad_scaler=grad_scaler, ema=ema)
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
    start_epoch = int(state.epoch)
    try:
        for i_epoch in range(remaining_epochs):
            t_epoch = _time.time()
            if args.lr_decay == "cosine":
                lr_now = float(args.lr) * 0.5 * (1.0 + math.cos(math.pi * i_epoch / max(1, remaining_epochs)))
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
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
            if ema is not None:
                _ema_update(ema, models["unet"], ema_epoch_decay)
            epoch_t = _time.time() - t_epoch
            step_time_s = epoch_t / max(1, int(cfg.schedule.epoch_size))
            if state.epoch % log_every == 0 or state.epoch == 1:
                bl = state.bucket_loss_running or {}
                elapsed = _time.time() - t_start
                # Epochs actually run since t_start, not the absolute epoch
                # counter — on a resumed run those differ (e.g. resumed at
                # epoch 2000, only 20 run so far), and dividing by the
                # absolute counter understates the rate by however many
                # epochs the checkpoint already carried, wildly
                # underestimating ETA right after a resume.
                epochs_run = max(1, state.epoch - start_epoch)
                remaining = (
                    elapsed / epochs_run
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
                    epoch_t, elapsed, remaining,
                )
                # Stage timing (throughput.md): step time on its own curve so
                # a live run's rate is readable without doing epoch_t/epoch_size
                # arithmetic by hand every time.
                logger.log({"train/timing/step_time_s": step_time_s},
                           step=state.optimizer_step)
                norms = {"train/lora_weight_norm": _lora_norm(models["unet"])}
                if ema is not None:
                    norms["train/lora_weight_norm_ema"] = float(
                        torch.sqrt(sum((t.float() ** 2).sum() for t in ema.values())).item())
                logger.log(norms, step=state.optimizer_step)
            if state.epoch % ckpt_every == 0:
                t_ckpt = _time.time()
                _dump_checkpoint(models["unet"], cfg, state, checkpoints_root,
                                 optimizer=optimizer, grad_scaler=grad_scaler, ema=ema)
                ckpt_time_s = _time.time() - t_ckpt
                log.info("checkpoint saved: ckpt_t=%.1fs", ckpt_time_s)
                logger.log({"train/timing/checkpoint_save_s": ckpt_time_s},
                           step=state.optimizer_step)
            if sample_every > 0 and state.epoch % sample_every == 0:
                t_eval = _time.time()
                _run_inline_sample("inline")
                eval_time_s = _time.time() - t_eval
                log.info("eval/tracking pass done: eval_t=%.1fs (%d cells)",
                         eval_time_s, len(sampler_ctx.cells) if sampler_ctx else 0)
                logger.log({
                    "train/timing/eval_pass_s": eval_time_s,
                    "train/timing/eval_pass_s_per_cell": (
                        eval_time_s / len(sampler_ctx.cells)
                        if sampler_ctx and sampler_ctx.cells else float("nan")
                    ),
                }, step=state.optimizer_step)
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
                         optimizer=optimizer, grad_scaler=grad_scaler, ema=ema)
        logger.finish()

    write_json(run_dir / "verdict.json", {
        "verdict": "ok" if state.aborted_reason is None else "aborted",
        "reason": state.aborted_reason,
        "epoch": state.epoch,
        "optimizer_step": state.optimizer_step,
    })
    return 0


def _lora_params(unet) -> dict[str, torch.Tensor]:
    return {n: p for n, p in unet.named_parameters() if "lora_" in n}


def _lora_norm(unet) -> float:
    """Frobenius norm over every LoRA factor, the curve plan 20 reads against training step."""
    with torch.no_grad():
        return float(torch.sqrt(sum((p.float() ** 2).sum() for p in _lora_params(unet).values())).item())


def _ema_init(unet) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {n: p.detach().float().clone() for n, p in _lora_params(unet).items()}


def _ema_update(ema: dict[str, torch.Tensor], unet, decay: float) -> None:
    with torch.no_grad():
        for n, p in _lora_params(unet).items():
            ema[n].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def _dump_checkpoint(
    unet, cfg, state, checkpoints_root: Path,
    *, optimizer=None, grad_scaler=None, ema=None,
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
    if ema is not None:
        payload["lora_state_ema"] = {k: v.detach().to("cpu", torch.float32).clone() for k, v in ema.items()}
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
