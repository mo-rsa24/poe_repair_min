"""CLI orchestrator for the CFG conditioning-window ablation.

Three modes:

  --sanity-only           run the two equivalence checks, exit non-zero on fail
  --smoke                 render two schedules (fast wiring check)
  (default)               full STANDARD_SUITE + sanity + manifest + contact sheet

Outputs land under ``outputs/conditioning_window/<pair_slug>/seed_<n>/``::

    schedules/<schedule_id>/image.png
    schedules/<schedule_id>/summary.json
    results/sanity/{masked_all_on,run_cfg,masked_all_off,run_cfg_gs0}.png
    results/sanity/sanity.json
    results/inspector_manifest.json
    results/figures/contact_sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.experiments import _assert_env_ok
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.experiments.conditioning_window import figures as F
from poe_repair.experiments.conditioning_window import schedules as S
from poe_repair.experiments.conditioning_window import sweep as W
from poe_repair.experiments.conditioning_window.config import RunConfig
from poe_repair.methods._sampling import initial_latents_for_pair
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl, ensure_dir


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="conditioning_window")
    ap.add_argument("--prompt", default="a cat and a dog",
                    help='Joint prompt to encode for the conditional branch.')
    ap.add_argument("--prompt-a", default="a cat",
                    help='Subject A; only used to locate the shared x_T cell.')
    ap.add_argument("--prompt-b", default="a dog",
                    help='Subject B; only used to locate the shared x_T cell.')
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog",
                    help='Output directory slug under outputs/conditioning_window/.')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--out-root", default=None,
                    help='Override the output root (default = outputs/).')
    ap.add_argument("--schedules", default="standard",
                    help='"standard", "smoke", or a comma-separated list of schedule ids.')
    ap.add_argument("--sanity-only", action="store_true",
                    help='Only run the two equivalence checks; skip the sweep.')
    ap.add_argument("--smoke", action="store_true",
                    help='Render only the smoke subset (alias for --schedules smoke).')
    ap.add_argument("--skip-sanity", action="store_true",
                    help='Skip the sanity stage at the end of a normal run.')
    ap.add_argument("--skip-figures", action="store_true",
                    help='Skip rendering the contact-sheet figure.')
    ap.add_argument("--overwrite", action="store_true",
                    help='Re-render schedules even if image.png already exists.')
    return ap


def _resolve_schedules(arg: str, smoke: bool) -> list[S.Schedule]:
    if smoke or arg == "smoke":
        return S.select(S.SMOKE_IDS)
    if arg == "standard":
        return list(S.STANDARD_SUITE)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    return S.select(names)


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

    ctx = make_ctx(
        model_id=cfg.model_id,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
    )

    # Use the pair-cell x_T so the no-LoRA baseline shares its noise tensor
    # with the LoRA experiment — making the marginal-effect comparison valid.
    cell = cell_for(
        cfg.prompt_a, cfg.prompt_b, cfg.seed,
        height=cfg.height, width=cfg.width,
    )
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )

    seq_cond, pool_cond = encode_prompt_sdxl(
        cfg.prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_e, pool_e = encode_prompt_sdxl(
        "", models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    common = W._common_kwargs(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_cond=seq_cond, pool_cond=pool_cond,
        seq_e=seq_e, pool_e=pool_e,
        cfg=cfg,
        euler_sigma=euler_sigma,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    schedules_dir = ensure_dir(cfg.schedules_dir())
    results_dir = ensure_dir(cfg.results_dir())

    if args.sanity_only:
        sanity = W.run_sanity(
            common=common, cfg=cfg,
            schedules_dir=schedules_dir, results_dir=results_dir,
        )
        ok = bool(
            sanity["all_on_vs_run_cfg"]["pass"]
            and sanity["all_off_vs_uncond"]["pass"]
        )
        return 0 if ok else 1

    chosen = _resolve_schedules(args.schedules, smoke=args.smoke)
    print(f"[conditioning_window] rendering {len(chosen)} schedules at seed={cfg.seed}")
    records = W.run_sweep(
        chosen, common=common, schedules_dir=schedules_dir, overwrite=args.overwrite,
    )

    sanity: dict | None = None
    if not args.skip_sanity:
        sanity = W.run_sanity(
            common=common, cfg=cfg,
            schedules_dir=schedules_dir, results_dir=results_dir,
        )

    manifest_path = W.build_manifest(
        cfg=cfg, records=records, sanity=sanity, results_dir=results_dir,
    )

    if not args.skip_figures:
        fig_path = results_dir / "figures" / "contact_sheet.png"
        F.render_contact_sheet(manifest_path, fig_path)

    print(f"[conditioning_window] done. Inspect with:")
    print(f"  python scripts/lora_inspector.py --port 5050")
    print(f"  open http://127.0.0.1:5050/conditioning_window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
