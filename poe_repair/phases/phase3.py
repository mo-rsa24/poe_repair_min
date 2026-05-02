"""Phase 3 — online β-sweep along the (modified) PoE trajectory.

For each (pair, seed) and each β ∈ {0, 0.25, 0.5, 0.75, 1.0}, drive sampling
with the modified update

    ε̃_β = (1 - β)·ε̃_PoE + β·ε̃_J,

where the residual r_t = ε̃_J - ε̃_PoE is recomputed at the *current* latent
each step (online). By construction:

    β = 0  →  pure PoE
    β = 1  →  mono

Intermediate β tests whether moving along r_t causally repairs the PoE
failure. If cat × dog improves at some β* < 1, the residual is a useful
correction direction.

Output: a single grid figure (rows = pairs, columns = β values) plus per-cell
PNGs cached for reuse.

Run:
    python -m poe_repair.phases.phase3
    python -m poe_repair.phases.phase3 --pair-filter a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.figures._common import image_grid
from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_beta_inject_online,
    write_decoded_image,
)
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import (
    discover_pairs,
    encode_prompt_sdxl,
    ensure_dir,
)


BETAS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


def _beta_filename(beta: float) -> str:
    return f"beta_{int(round(beta * 100)):03d}.png"


def _run_one_beta(cell, beta: float, ctx: MethodCtx) -> Path:
    """Sample at this β and save the PNG. Idempotent — skip if it exists."""
    cell_dir = ensure_dir(
        ctx.output_root / "phase3" / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    )
    image_path = cell_dir / _beta_filename(beta)
    if image_path.exists():
        return image_path

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
    out = run_beta_inject_online(
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
        beta=beta,
    )
    write_decoded_image(out.image, image_path)
    print(f"[phase3] β={beta:.2f} {cell.pair_slug} seed={cell.seed} -> {image_path.name}")
    return image_path


def phase3(
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    betas: list[float] | None = None,
    ctx: MethodCtx | None = None,
) -> Path:
    cfg = RunConfig()
    ctx = ctx or make_ctx()
    betas = betas or BETAS
    cells = discover_pairs(
        cfg.paths.pilot_dir, pair_filter=pair_filter, seed_filter=seed_filter
    )
    if not cells:
        raise RuntimeError(f"No pair-seed cells found at {cfg.paths.pilot_dir}.")

    rows: list[list[Path]] = [
        [_run_one_beta(cell, beta, ctx) for beta in betas]
        for cell in cells
    ]

    out = cfg.paths.output_root / "figures" / "phase3__beta_sweep.png"
    written = image_grid(
        rows,
        out,
        col_labels=[f"β = {b:g}" for b in betas],
        row_labels=[
            f"{cell.regime}\n{cell.pair_slug}\nseed {cell.seed}" for cell in cells
        ],
        title="Phase 3 — online β-sweep (β=0: PoE, β=1: mono)",
    )
    print(f"[phase3] wrote {written}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — online β-sweep grid.")
    ap.add_argument("--pair-filter", nargs="*", default=None)
    ap.add_argument("--seed-filter", nargs="*", type=int, default=None)
    ap.add_argument(
        "--betas",
        nargs="*",
        type=float,
        default=None,
        help=f"β values to sweep (default {BETAS}).",
    )
    args = ap.parse_args()
    phase3(
        pair_filter=args.pair_filter,
        seed_filter=args.seed_filter,
        betas=args.betas,
    )


if __name__ == "__main__":
    main()
