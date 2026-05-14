"""E-residual-decomposition — does u_J already differ between seed 4 and 42 at step 0?

Premise (closed-form, no sampling). Each step-0 forward pass gives us
four guidance directions in epsilon space:

    eps_uncond,  eps_A,  eps_B,  eps_J

from which we form the implicit-classifier residuals (u-vectors):

    u_A = eps_A − eps_uncond     (how cat-alone pulls)
    u_B = eps_B − eps_uncond     (how dog-alone pulls)
    u_J = eps_J − eps_uncond     (how the joint pulls)
    r   = u_J − (u_A + u_B)      (the interaction we've been studying)

The hypothesis: seed 4's failure under Mono and seed 42's success under
Mono are determined at step 0 by *whether* u_J decomposes constructively
into u_A and u_B. We test this by least-squares fitting

    u_J ≈ c_A · u_A + c_B · u_B

(with a 2×2 Gram-matrix inverse — u_A and u_B are not orthogonal in
general). The perpendicular fraction ‖u_J − fit‖ / ‖u_J‖ measures the
genuine interaction term living outside the two subject directions.

Output is seven structurally-meaningful numbers per (seed, step):

    ‖u_A‖, ‖u_B‖, ‖u_J‖, cos(u_A,u_B), c_A^J, c_B^J, perp_frac

Plus the residual decomposition (c_A^r, c_B^r) for completeness.

This is a pure measurement. No sampling, no decode, no figures of
generated images. Two figures rendered:
  - geometry: u_A / u_B / u_J as arrows in their 2D plane (per seed×step)
  - coefficients: bar chart of (c_A^J, c_B^J, perp_frac) across seeds

Run:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY -m poe_repair.experiments.e_residual_decomposition \\
        --seeds 4 42 --steps 0 3 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for, slugify
from poe_repair.figures._common import save_fig
from poe_repair.methods._sampling import add_time_ids
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "e_residual_decomposition"


def _ls_decompose(
    target: torch.Tensor, basis_a: torch.Tensor, basis_b: torch.Tensor,
) -> tuple[float, float, float]:
    """Solve ``target ≈ c_a·basis_a + c_b·basis_b`` via Gram inverse.

    Returns ``(c_a, c_b, perp_frac)`` where ``perp_frac`` is
    ``‖target − fit‖ / ‖target‖``. All tensors are flattened internally.
    """
    a = basis_a.float().flatten()
    b = basis_b.float().flatten()
    y = target.float().flatten()
    aa = float(torch.dot(a, a).item())
    bb = float(torch.dot(b, b).item())
    ab = float(torch.dot(a, b).item())
    ya = float(torch.dot(y, a).item())
    yb = float(torch.dot(y, b).item())
    # Gram matrix [[aa, ab], [ab, bb]]; rhs [ya, yb]; closed-form inverse.
    det = aa * bb - ab * ab
    if abs(det) < 1e-20:
        # Degenerate basis; fall back to projection on whichever has more mass.
        ca = ya / aa if aa > 0 else 0.0
        cb = 0.0
        fit = ca * a
    else:
        ca = (bb * ya - ab * yb) / det
        cb = (aa * yb - ab * ya) / det
        fit = ca * a + cb * b
    y_norm = float(y.norm().item()) + 1e-20
    perp = float((y - fit).norm().item())
    return float(ca), float(cb), float(perp / y_norm)


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    fx = x.float().flatten()
    fy = y.float().flatten()
    n = float((fx.norm() * fy.norm()).item()) + 1e-20
    return float(torch.dot(fx, fy).item()) / n


@torch.no_grad()
def _step_forward(
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    scheduler, ctx: MethodCtx, height: int, width: int,
) -> dict[str, torch.Tensor]:
    """One 4-branch UNet forward at the given timestep. Returns the four
    raw eps tensors (no CFG mixing, no scaling)."""
    pe = torch.cat([seq_e, seq_a, seq_b, seq_j], dim=0)
    pool = torch.cat([pool_e, pool_a, pool_b, pool_j], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=4,
            device=ctx.device, dtype=ctx.dtype,
        ),
    }
    latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
    noise = ctx.models["unet"](
        latent_input, timestep,
        encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None,
    ).sample
    eps_e, eps_a, eps_b, eps_j = noise.chunk(4)
    return {"eps_e": eps_e, "eps_a": eps_a, "eps_b": eps_b, "eps_j": eps_j}


def _diagnostics(eps: dict[str, torch.Tensor]) -> dict[str, float]:
    u_a = eps["eps_a"] - eps["eps_e"]
    u_b = eps["eps_b"] - eps["eps_e"]
    u_j = eps["eps_j"] - eps["eps_e"]
    r = u_j - (u_a + u_b)
    cA_J, cB_J, perp_J = _ls_decompose(u_j, u_a, u_b)
    cA_r, cB_r, perp_r = _ls_decompose(r, u_a, u_b)
    return {
        "norm_uA": float(u_a.float().norm().item()),
        "norm_uB": float(u_b.float().norm().item()),
        "norm_uJ": float(u_j.float().norm().item()),
        "norm_r": float(r.float().norm().item()),
        "cos_uA_uB": _cosine(u_a, u_b),
        "cos_uA_uJ": _cosine(u_a, u_j),
        "cos_uB_uJ": _cosine(u_b, u_j),
        "cA_J": cA_J, "cB_J": cB_J, "perp_frac_J": perp_J,
        "cA_r": cA_r, "cB_r": cB_r, "perp_frac_r": perp_r,
    }


def _geometry_2d(u_a: torch.Tensor, u_b: torch.Tensor, u_j: torch.Tensor):
    """Project u_a, u_b, u_j onto the orthonormal basis of span(u_a, u_b)
    via QR. Returns three 2D vectors (numpy)."""
    fa = u_a.float().flatten()
    fb = u_b.float().flatten()
    fj = u_j.float().flatten()
    # Gram-Schmidt for an orthonormal (e1, e2) basis of span(fa, fb).
    e1 = fa / (fa.norm() + 1e-20)
    fb_perp = fb - torch.dot(fb, e1) * e1
    norm_fb_perp = fb_perp.norm()
    if norm_fb_perp < 1e-12:
        e2 = torch.zeros_like(e1)
    else:
        e2 = fb_perp / norm_fb_perp
    pa = (float(torch.dot(fa, e1).item()), float(torch.dot(fa, e2).item()))
    pb = (float(torch.dot(fb, e1).item()), float(torch.dot(fb, e2).item()))
    pj = (float(torch.dot(fj, e1).item()), float(torch.dot(fj, e2).item()))
    return np.array(pa), np.array(pb), np.array(pj)


def _draw_geometry_panel(
    ax, *, u_a, u_b, u_j, title: str,
):
    pa, pb, pj = _geometry_2d(u_a, u_b, u_j)
    arrows = [
        (pa, "tab:blue", "u_A (cat)"),
        (pb, "tab:red", "u_B (dog)"),
        (pj, "tab:purple", "u_J (joint)"),
    ]
    max_r = max(np.linalg.norm(v) for v, _, _ in arrows) * 1.15 + 1e-9
    for v, color, label in arrows:
        ax.annotate(
            "", xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        )
        ax.text(v[0] * 1.05, v[1] * 1.05, label, color=color, fontsize=8)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)


def _figure_geometry(
    eps_per_seed_step: dict[tuple[int, int], dict[str, torch.Tensor]],
    *, output_path: Path, title: str,
) -> Path:
    keys = sorted(eps_per_seed_step.keys())
    seeds = sorted({s for s, _ in keys})
    steps = sorted({k for _, k in keys})
    n_rows = len(seeds)
    n_cols = len(steps)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    for r, seed in enumerate(seeds):
        for c, step in enumerate(steps):
            eps = eps_per_seed_step[(seed, step)]
            u_a = eps["eps_a"] - eps["eps_e"]
            u_b = eps["eps_b"] - eps["eps_e"]
            u_j = eps["eps_j"] - eps["eps_e"]
            _draw_geometry_panel(
                axes[r][c], u_a=u_a, u_b=u_b, u_j=u_j,
                title=f"seed {seed}, step {step}",
            )
    fig.suptitle(title, fontsize=11)
    return save_fig(fig, output_path)


def _figure_coefficients(
    diag_per_seed_step: dict[tuple[int, int], dict[str, float]],
    *, output_path: Path, title: str,
) -> Path:
    keys = sorted(diag_per_seed_step.keys())
    seeds = sorted({s for s, _ in keys})
    steps = sorted({k for _, k in keys})
    n_steps = len(steps)
    fig, axes = plt.subplots(
        3, n_steps, figsize=(3.0 * n_steps, 7.5), squeeze=False, sharex=False,
    )
    metrics = [
        ("cA_J", "c_A^J  (cat coefficient of u_J)"),
        ("cB_J", "c_B^J  (dog coefficient of u_J)"),
        ("perp_frac_J", "perp_frac_J  (interaction outside (u_A,u_B))"),
    ]
    seed_colors = {s: f"C{i}" for i, s in enumerate(seeds)}
    width = 0.8 / max(1, len(seeds))
    for c, step in enumerate(steps):
        for r, (metric, ylabel) in enumerate(metrics):
            ax = axes[r][c]
            xs = np.arange(len(seeds))
            ys = [diag_per_seed_step[(s, step)][metric] for s in seeds]
            for i, (s, y) in enumerate(zip(seeds, ys)):
                ax.bar(
                    [xs[i]], [y], width=width * 4,
                    color=seed_colors[s], label=f"seed {s}" if (r == 0 and c == 0) else None,
                )
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            if r == 0:
                ax.set_title(f"step {step}", fontsize=10)
            ax.tick_params(axis="both", labelsize=7)
    if seeds:
        axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, output_path)


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[4, 42])
    ap.add_argument("--prompts", nargs=2, type=str, default=list(HEADLINE_PAIR),
                    metavar=("PROMPT_A", "PROMPT_B"))
    ap.add_argument("--steps", nargs="*", type=int, default=[0, 3, 5],
                    help="Step indices at which to do one 4-branch forward.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    prompt_a, prompt_b = args.prompts

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    summary_dir = ensure_dir(cfg.paths.output_root / EXP_NAME)
    slug = slugify(prompt_a, prompt_b)

    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timesteps = ctx.scheduler.timesteps
    selected_steps = sorted(set(int(s) for s in args.steps))
    for s in selected_steps:
        if s < 0 or s >= len(timesteps):
            raise ValueError(
                f"step {s} out of range; scheduler has {len(timesteps)} steps"
            )

    eps_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    diag_cache: dict[tuple[int, int], dict[str, float]] = {}
    summary_cells: list[dict] = []

    for seed in args.seeds:
        cell = cell_for(prompt_a, prompt_b, seed)
        emb = encode_pair(cell, ctx)
        seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="literal")
        init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
        latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)

        per_step: list[dict] = []
        # We have to step the latent forward for steps>0; the simplest
        # principled way is to run plain Mono CFG up to that step
        # *without* applying any correction, since we want the diagnostic
        # at the latent that vanilla Mono would have reached.
        cur_lat = latents.clone()
        for step_index, timestep in enumerate(timesteps):
            if step_index > max(selected_steps):
                break
            if step_index in selected_steps:
                eps = _step_forward(
                    latents=cur_lat, timestep=timestep,
                    seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                    seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                    seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                    seq_j=seq_j, pool_j=pool_j,
                    scheduler=ctx.scheduler, ctx=ctx,
                    height=cell.height, width=cell.width,
                )
                eps_cache[(seed, step_index)] = {
                    k: v.detach().cpu() for k, v in eps.items()
                }
                diag = _diagnostics(eps)
                diag_cache[(seed, step_index)] = diag
                per_step.append({"step": step_index, "diagnostics": diag})
                if args.verbose:
                    print(
                        f"  seed={seed:4d} step={step_index:2d}  "
                        f"|uA|={diag['norm_uA']:.2f} |uB|={diag['norm_uB']:.2f} "
                        f"|uJ|={diag['norm_uJ']:.2f}  "
                        f"cos(A,B)={diag['cos_uA_uB']:+.3f}  "
                        f"cA_J={diag['cA_J']:+.3f}  cB_J={diag['cB_J']:+.3f}  "
                        f"perp_J={diag['perp_frac_J']:.3f}"
                    )
            # Mono-CFG step to advance the latent.
            from poe_repair.runtime import (
                ddim_prev_from_x0_eps, guided_eps, tweedie_mean,
            )
            pe2 = torch.cat([seq_j, emb["seq_e"]], dim=0)
            pool2 = torch.cat([pool_j, emb["pool_e"]], dim=0)
            cond2 = {
                "text_embeds": pool2,
                "time_ids": add_time_ids(
                    height=cell.height, width=cell.width, batch_size=2,
                    device=ctx.device, dtype=ctx.dtype,
                ),
            }
            latent_input = ctx.scheduler.scale_model_input(
                cur_lat.repeat(2, 1, 1, 1), timestep,
            )
            with torch.no_grad():
                noise = ctx.models["unet"](
                    latent_input, timestep, encoder_hidden_states=pe2,
                    added_cond_kwargs=cond2, timestep_cond=None,
                ).sample
            eps_j_raw, eps_uncond = noise.chunk(2)
            eps_t = guided_eps(eps_j_raw, eps_uncond, ctx.guidance_scale)
            alpha_bar_t = ctx.scheduler.alphas_cumprod[int(timestep.item())].to(
                device=ctx.device, dtype=ctx.dtype,
            )
            x0 = tweedie_mean(cur_lat, alpha_bar_t, eps_t)
            cur_lat = ddim_prev_from_x0_eps(
                scheduler=ctx.scheduler, timestep=timestep,
                step_index=step_index, x0=x0, eps=eps_t,
            )

        summary_cells.append({
            "seed": seed, "pair": [prompt_a, prompt_b],
            "per_step": per_step,
        })

    geometry_path = _figure_geometry(
        eps_cache, output_path=fig_dir / f"geometry__{slug}.png",
        title=(
            f"Step-0 geometry — {prompt_a} × {prompt_b}\n"
            "u_A, u_B, u_J in their orthonormalised plane (Gram-Schmidt)."
        ),
    )
    coefficients_path = _figure_coefficients(
        diag_cache, output_path=fig_dir / f"coefficients__{slug}.png",
        title=(
            f"Decomposition coefficients — {prompt_a} × {prompt_b}\n"
            "u_J ≈ c_A·u_A + c_B·u_B  via 2×2 Gram inverse."
        ),
    )
    write_json(summary_dir / "summary.json", {
        "exp": EXP_NAME, "seeds": args.seeds, "pair": [prompt_a, prompt_b],
        "steps": selected_steps,
        "geometry_path": str(geometry_path),
        "coefficients_path": str(coefficients_path),
        "cells": summary_cells,
    })
    print(f"[resid-decomp] wrote {geometry_path}")
    print(f"[resid-decomp] wrote {coefficients_path}")


if __name__ == "__main__":
    main()
