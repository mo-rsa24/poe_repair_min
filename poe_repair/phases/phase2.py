"""Phase 2 — residual-norm trajectory along the PoE rollout.

For each (pair, seed): drive sampling with vanilla PoE while also evaluating
the UNet on the literal joint embedding e_J at each step. Compute

    r_t = ε̃_J(x_t^PoE, t) - ε̃_PoE(x_t^PoE, t)

then plot ||r_t||_2 and ||r_t|| / ||ε̃_J|| over t for both pairs on shared
axes. The contrast between cat × dog (collision) and butterfly × flower
(cooperative) is the headline of this phase.

Run:
    python -m poe_repair.phases.phase2
    python -m poe_repair.phases.phase2 --pair-filter a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.diagnostics.residual import (
    norm_trajectory,
    relative_norm_trajectory,
    residual_trajectory,
)
from poe_repair.figures._common import line_plot
from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_beta_inject_online,
)
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import discover_pairs, encode_prompt_sdxl


def _probe_cell(cell, ctx: MethodCtx):
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    enc = lambda p: encode_prompt_sdxl(
        p, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc(cell.prompt_a)
    seq_b, pool_b = enc(cell.prompt_b)
    seq_j, pool_j = enc(joint_prompt(cell.prompt_a, cell.prompt_b, template=ctx.joint_template))
    # beta=0 reduces the unified sampler to vanilla PoE (with the J probe).
    return run_beta_inject_online(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_a=seq_a, pool_a=pool_a,
        seq_b=seq_b, pool_b=pool_b,
        seq_j=seq_j, pool_j=pool_j,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        beta=0.0,
    )


def phase2(
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    ctx: MethodCtx | None = None,
) -> tuple[Path, Path]:
    cfg = RunConfig()
    ctx = ctx or make_ctx()
    cells = discover_pairs(
        cfg.paths.pilot_dir, pair_filter=pair_filter, seed_filter=seed_filter
    )
    if not cells:
        raise RuntimeError(f"No pair-seed cells found at {cfg.paths.pilot_dir}.")

    abs_series: dict[str, tuple[list[float], list[float]]] = {}
    rel_series: dict[str, tuple[list[float], list[float]]] = {}

    for cell in cells:
        out = _probe_cell(cell, ctx)
        eps_j = out.extras["eps_j_traj"]
        eps_poe = out.extras["eps_poe_traj"]
        ts = out.extras["timesteps"].tolist()
        r = residual_trajectory(eps_j, eps_poe)
        abs_n = norm_trajectory(r).tolist()
        rel_n = relative_norm_trajectory(r, eps_j).tolist()
        label = f"{cell.regime} — {cell.pair_slug}"
        abs_series[label] = (ts, abs_n)
        rel_series[label] = (ts, rel_n)
        print(f"[phase2] {cell.pair_slug} seed={cell.seed} max||r_t||={max(abs_n):.3e}")

    fig_dir = cfg.paths.output_root / "figures"
    abs_path = line_plot(
        abs_series,
        fig_dir / "phase2__residual_norm_abs.png",
        xlabel="DDIM timestep (high noise → low noise)",
        ylabel="||r_t||_2",
        title="Phase 2 — absolute residual norm along PoE trajectory",
        invert_x=True,
    )
    rel_path = line_plot(
        rel_series,
        fig_dir / "phase2__residual_norm_rel.png",
        xlabel="DDIM timestep (high noise → low noise)",
        ylabel="||r_t|| / ||ε̃_J||",
        title="Phase 2 — relative residual norm along PoE trajectory",
        invert_x=True,
    )
    print(f"[phase2] wrote {abs_path}")
    print(f"[phase2] wrote {rel_path}")
    return abs_path, rel_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 2 — residual norm along PoE trajectory."
    )
    ap.add_argument("--pair-filter", nargs="*", default=None)
    ap.add_argument("--seed-filter", nargs="*", type=int, default=None)
    args = ap.parse_args()
    phase2(pair_filter=args.pair_filter, seed_filter=args.seed_filter)


if __name__ == "__main__":
    main()
