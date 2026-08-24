"""CLI orchestrator for the LoRA-on-CFG-mask ablation.

Each ``(checkpoint, lambda)`` pair is a rendered cell. For each cell and
each composition mode, the same #1 schedule grammar is run via
``run_lora_residual_inject_masked``.

Outputs::

    outputs/conditioning_window_lora/<pair_slug>/seed_<n>/<mode>/
        epoch_<step>/lambda_<lam>/schedules/<id>/{image.png, summary.json}
        results/inspector_manifest.json   ← rollup across cells

The no-LoRA baseline lives at ``outputs/conditioning_window/...`` and is
read by the inspector for the side-by-side LoRA-off pane.

Examples::

    # Default: single cell (last checkpoint, λ=1.0).
    python -m poe_repair.experiments.cfg_window_with_lora

    # 3 lambdas × last checkpoint, with_prompt only.
    python -m poe_repair.experiments.cfg_window_with_lora \\
        --lambda-values 0.0,0.5,1.0 \\
        --modes with_prompt

    # 5 lambdas × 5 checkpoints sampled from the training timeline.
    python -m poe_repair.experiments.cfg_window_with_lora \\
        --lora-ckpts /path/to/lora_step_010000.pt,/path/to/lora_step_030000.pt,...\\
        --lambda-values 0.0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from poe_repair.experiments import _assert_env_ok
from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.cfg_window_without_lora import schedules as S
from poe_repair.experiments.cfg_window_with_lora import sweep as W
from poe_repair.experiments.cfg_window_with_lora.config import (
    COMPOSITION_MODES,
    RunConfig,
    epoch_for_ckpt,
)
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.methods._sampling import initial_latents_for_pair
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl, ensure_dir


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="conditioning_window_lora")
    ap.add_argument("--prompt", default="a cat and a dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--out-root", default=None,
                    help='Override the output root (default = outputs/).')
    ap.add_argument(
        "--lora-ckpts", default=None,
        help='Comma-separated list of lora_step_*.pt paths. Default = the '
             'last checkpoint in --lora-run-dir/checkpoints/.',
    )
    ap.add_argument(
        "--lambda-values", default="1.0",
        help='Comma-separated list of λ values in [0, 1]. Default = "1.0".',
    )
    ap.add_argument(
        "--lora-run-dir", default=None,
        help='LoRA run dir containing config.json. Default = '
             'outputs/lora/a_cat__x__a_dog/seed_42/results.',
    )
    ap.add_argument("--modes", default=",".join(COMPOSITION_MODES),
                    help=f'Comma-separated subset of {COMPOSITION_MODES}.')
    ap.add_argument("--schedules", default="standard",
                    help='"standard", "smoke", or comma-separated schedule ids.')
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def _resolve_schedules(arg: str) -> list[S.Schedule]:
    if arg == "smoke":
        return S.select(S.SMOKE_IDS)
    if arg == "standard":
        return list(S.STANDARD_SUITE)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    return S.select(names)


def _load_lora_config(run_dir: Path) -> LoRAConfig:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"LoRA run config not found at {cfg_path}. Pass --lora-run-dir."
        )
    raw = json.loads(cfg_path.read_text())
    lora_block = raw.get("lora")
    if not isinstance(lora_block, dict):
        raise ValueError(f"{cfg_path} has no 'lora' block")
    return LoRAConfig(
        rank=int(lora_block["rank"]),
        alpha=int(lora_block["alpha"]),
        dropout=float(lora_block.get("dropout", 0.0)),
        target_modules=tuple(lora_block["target_modules"]),
        init=str(lora_block.get("init", "gaussian")),
        adapter_name=str(lora_block.get("adapter_name", "lora")),
    )


def _attach_lora(unet: torch.nn.Module, lora_cfg: LoRAConfig) -> dict:
    fake_cfg = SimpleNamespace(lora=lora_cfg)
    return lora_trainer.attach_lora(unet, fake_cfg)


def _load_ckpt_into_attached_unet(unet: torch.nn.Module, ckpt_path: Path) -> int:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state")
    if state is None:
        raise KeyError(
            f"checkpoint {ckpt_path} has no 'lora_state' key "
            f"(found: {list(ckpt.keys())})"
        )
    lora_trainer.load_lora_state(unet, state)
    return len(state)


def _resolve_ckpts(arg: str | None, run_dir: Path) -> list[Path]:
    if arg:
        return [Path(p.strip()).expanduser().resolve()
                for p in arg.split(",") if p.strip()]
    ckpts_dir = run_dir / "checkpoints"
    pool = sorted(ckpts_dir.glob("lora_step_*.pt"),
                  key=lambda p: epoch_for_ckpt(p))
    if not pool:
        raise FileNotFoundError(
            f"no lora_step_*.pt found under {ckpts_dir}. Pass --lora-ckpts."
        )
    return [pool[-1]]


def _parse_lambdas(arg: str) -> list[float]:
    out: list[float] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"--lambda-values entry out of [0,1]: {v}")
        out.append(v)
    if not out:
        raise SystemExit("--lambda-values empty")
    return out


def main(argv: list[str] | None = None) -> int:
    _assert_env_ok()
    args = _build_argparser().parse_args(argv)

    cfg = RunConfig(
        prompt=args.prompt,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        pair_slug=args.pair_slug,
        seed=int(args.seed),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        height=int(args.height),
        width=int(args.width),
        model_id=args.model_id,
    )
    if args.out_root:
        cfg.output_root = Path(args.out_root).expanduser().resolve()
    if args.lora_run_dir:
        cfg.lora_run_dir = Path(args.lora_run_dir).expanduser().resolve()

    cfg.lora_ckpt_paths = tuple(_resolve_ckpts(args.lora_ckpts, cfg.lora_run_dir))
    cfg.lambda_values = tuple(_parse_lambdas(args.lambda_values))

    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in requested_modes:
        if m not in COMPOSITION_MODES:
            raise SystemExit(
                f"unknown composition mode {m!r}; expected subset of "
                f"{COMPOSITION_MODES}"
            )

    print(f"[conditioning_window_lora] LoRA run dir: {cfg.lora_run_dir}")
    print(f"[conditioning_window_lora] {len(cfg.lora_ckpt_paths)} checkpoint(s):")
    for p in cfg.lora_ckpt_paths:
        print(f"    {p.name}  ->  epoch={epoch_for_ckpt(p)}")
    print(f"[conditioning_window_lora] {len(cfg.lambda_values)} lambda(s): "
          f"{list(cfg.lambda_values)}")

    lora_cfg = _load_lora_config(cfg.lora_run_dir)

    ctx = make_ctx(
        model_id=cfg.model_id,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
    )
    attach_info = _attach_lora(ctx.models["unet"], lora_cfg)
    print(
        f"[conditioning_window_lora] LoRA attached: "
        f"n_matched={attach_info['n_matched']} "
        f"trainable_params={attach_info['trainable_params']}"
    )

    # Pair-cell x_T — shared with #1.
    cell = cell_for(
        cfg.prompt_a, cfg.prompt_b, cfg.seed,
        height=cfg.height, width=cfg.width,
    )
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )

    seq_a, pool_a = encode_prompt_sdxl(
        cfg.prompt_a, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_b, pool_b = encode_prompt_sdxl(
        cfg.prompt_b, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_e, pool_e = encode_prompt_sdxl(
        "", models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )

    common = W._common_kwargs(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_a=seq_a, pool_a=pool_a,
        seq_b=seq_b, pool_b=pool_b,
        seq_e=seq_e, pool_e=pool_e,
        cfg=cfg,
        euler_sigma=euler_sigma,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    chosen = _resolve_schedules(args.schedules)
    n_cells = len(cfg.lora_ckpt_paths) * len(cfg.lambda_values)
    print(
        f"[conditioning_window_lora] rendering {len(chosen)} schedules × "
        f"{len(requested_modes)} mode(s) × {n_cells} cell(s) at seed={cfg.seed}"
    )

    # Load the sanity checkpoint (last) once, run sanity once.
    sanity: dict | None = None
    if not args.skip_sanity and "with_prompt" in requested_modes:
        sanity_ckpt = cfg.lora_ckpt_paths[-1]
        _load_ckpt_into_attached_unet(ctx.models["unet"], sanity_ckpt)
        sanity_dir = cfg.mode_results_dir("with_prompt") / "sanity"
        sanity = W.run_sanity(
            common=common, cfg=cfg,
            sanity_dir=sanity_dir,
            lambda_value=1.0,
        )

    # Per-mode collection of cell records for the rollup manifest.
    by_mode: dict[str, list[dict]] = {m: [] for m in requested_modes}

    for ckpt_path in cfg.lora_ckpt_paths:
        epoch = epoch_for_ckpt(ckpt_path)
        n_loaded = _load_ckpt_into_attached_unet(
            ctx.models["unet"], ckpt_path,
        )
        print(
            f"[conditioning_window_lora] loaded {ckpt_path.name} "
            f"(epoch={epoch}, n_lora_params={n_loaded})"
        )
        for lam in cfg.lambda_values:
            for mode in requested_modes:
                schedules_dir = ensure_dir(
                    cfg.schedules_dir(mode, epoch, lam),
                )
                records = W.run_cell_sweep(
                    chosen, mode=mode, common=common,
                    schedules_dir=schedules_dir,
                    lambda_value=lam,
                    overwrite=args.overwrite,
                )
                by_mode[mode].append({
                    "epoch": epoch,
                    "lambda_value": float(lam),
                    "ckpt_path": str(ckpt_path),
                    "records": records,
                })

    for mode, cells in by_mode.items():
        W.build_mode_manifest(
            cfg=cfg, mode=mode, cells=cells,
            sanity=(sanity if mode == "with_prompt" else None),
        )

    print(f"[conditioning_window_lora] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
