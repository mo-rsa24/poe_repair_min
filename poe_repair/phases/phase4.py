"""Phase 4 — frozen vs online β-injection comparison.

Tests whether the residual r_t is approximately a fixed vector field along
the trajectory (frozen works) or state-dependent (only online works). At a
chosen β*, for each cell:

1. Reference probe: run online sampler at β=0 (i.e. plain PoE) with the J
   branch, capture r_t^{PoE-traj}.
2. Frozen sampler at β*: inject β · r_t^{PoE-traj} (precomputed, does not
   track the modified latent).
3. Online sampler at β*: r_t recomputed at the current x_t^β each step.

Output: 2 rows (pairs) × 2 cols (Frozen β*, Online β*).

Run:
    python -m poe_repair.phases.phase4
    python -m poe_repair.phases.phase4 --beta 0.5
    python -m poe_repair.phases.phase4 --pair-filter a_cat__x__a_dog --beta 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.diagnostics.residual import residual_trajectory
from poe_repair.figures._common import image_grid
from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_beta_inject_frozen,
    run_beta_inject_online,
    write_decoded_image,
)
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import (
    discover_pairs,
    encode_prompt_sdxl,
    ensure_dir,
)


DEFAULT_BETA = 0.5


def _filename(mode: str, beta: float) -> str:
    return f"{mode}_beta_{int(round(beta * 100)):03d}.png"


def _run_cell(cell, beta: float, ctx: MethodCtx) -> tuple[Path, Path]:
    """Generate frozen and online β PNGs for one cell. Idempotent per file."""
    cell_dir = ensure_dir(
        ctx.output_root / "phase4" / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    )
    frozen_path = cell_dir / _filename("frozen", beta)
    online_path = cell_dir / _filename("online", beta)
    need_frozen = not frozen_path.exists()
    need_online = not online_path.exists()
    if not (need_frozen or need_online):
        return frozen_path, online_path

    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    enc = lambda p: encode_prompt_sdxl(
        p, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc(cell.prompt_a)
    seq_b, pool_b = enc(cell.prompt_b)
    seq_j, pool_j = enc(
        joint_prompt(cell.prompt_a, cell.prompt_b, template=ctx.joint_template)
    )

    common = dict(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_a=seq_a, pool_a=pool_a,
        seq_b=seq_b, pool_b=pool_b,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )

    if need_frozen:
        # Probe: capture r_t along the clean PoE trajectory (β=0 online with J branch).
        probe = run_beta_inject_online(seq_j=seq_j, pool_j=pool_j, beta=0.0, **common)
        r_traj = residual_trajectory(
            probe.extras["eps_j_traj"], probe.extras["eps_poe_traj"]
        )
        frozen = run_beta_inject_frozen(r_traj=r_traj, beta=beta, **common)
        write_decoded_image(frozen.image, frozen_path)
        print(f"[phase4] frozen β={beta:.2f} {cell.pair_slug} seed={cell.seed} -> {frozen_path.name}")

    if need_online:
        online = run_beta_inject_online(
            seq_j=seq_j, pool_j=pool_j, beta=beta, **common
        )
        write_decoded_image(online.image, online_path)
        print(f"[phase4] online β={beta:.2f} {cell.pair_slug} seed={cell.seed} -> {online_path.name}")

    return frozen_path, online_path


def phase4(
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    beta: float | None = None,
    ctx: MethodCtx | None = None,
) -> Path:
    cfg = RunConfig()
    ctx = ctx or make_ctx()
    beta = beta if beta is not None else DEFAULT_BETA
    cells = discover_pairs(
        cfg.paths.pilot_dir, pair_filter=pair_filter, seed_filter=seed_filter
    )
    if not cells:
        raise RuntimeError(f"No pair-seed cells found at {cfg.paths.pilot_dir}.")

    rows: list[list[Path]] = []
    for cell in cells:
        frozen_path, online_path = _run_cell(cell, beta, ctx)
        rows.append([frozen_path, online_path])

    out = (
        cfg.paths.output_root
        / "figures"
        / f"phase4__frozen_vs_online_beta_{int(round(beta * 100)):03d}.png"
    )
    written = image_grid(
        rows,
        out,
        col_labels=[f"Frozen β = {beta:g}", f"Online β = {beta:g}"],
        row_labels=[
            f"{cell.regime}\n{cell.pair_slug}\nseed {cell.seed}" for cell in cells
        ],
        title=f"Phase 4 — frozen vs online β-injection (β = {beta:g})",
    )
    print(f"[phase4] wrote {written}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 4 — frozen vs online β-injection comparison."
    )
    ap.add_argument("--pair-filter", nargs="*", default=None)
    ap.add_argument("--seed-filter", nargs="*", type=int, default=None)
    ap.add_argument(
        "--beta",
        type=float,
        default=None,
        help=f"β value for the comparison (default {DEFAULT_BETA}).",
    )
    args = ap.parse_args()
    phase4(
        pair_filter=args.pair_filter,
        seed_filter=args.seed_filter,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
