"""Stage 1 — eleven-point λ-sweep + Mono reference panel.

For one held cell ``(prompt_a, prompt_b, seed)``:
  - Runs ``teacher_residual`` composer eleven times with constant
    ``λ ∈ {0.0, 0.1, …, 1.0}``. Each run shares the same ``x_T``,
    embeddings, scheduler, and CFG weight; only λ varies.
  - Persists per-step residuals ``Δ_t`` and the latent trajectory next
    to each PNG so downstream stages can compute residual norms,
    direction-stability matrices, PMI identity curves, and path-level
    distances without re-sampling.
  - Renders one extra Mono PNG via ``run_method("mono", …)`` for
    Figure 1's "reference" panel.

The output layout is:

    outputs/<exp>/pairs/<slug>/seed_<seed>/
        teacher_residual_const_lam000/...   # λ=0.0  (PoE anchor)
        teacher_residual_const_lam010/...
        …
        teacher_residual_const_lam100/...   # λ=1.0  (Mono anchor)
        mono/mono.png                       # reference panel

Idempotent: existing PNGs are reused unless ``overwrite=True``.
"""

from __future__ import annotations

from pathlib import Path

from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.experiments._eval_common import cell_for
from poe_repair.run import MethodCtx, run_method
from poe_repair.runtime import PairSeedCell


LAMBDA_GRID: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)


def lambda_method_name(lam: float) -> str:
    """Format method name for a constant-λ teacher-residual run."""
    return cmp_tr.method_name_for(
        lambda_schedule="constant",
        lambda_max=float(lam),
        correction_window=None,
    )


def run_sweep(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    exp_name: str,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    overwrite: bool = False,
    save_x0_estimates: bool = True,
    record_cross_attention: bool = False,
    attn_token_indices: dict | None = None,
    attn_resolution: int = 32,
) -> dict[float, Path]:
    """Run the λ-sweep and return ``{λ: image_path}``.

    ``save_x0_estimates=True`` adds ``eps_poe`` / ``eps_j`` per step to
    the residual ``.pt`` so Figure 6 can compute Tweedie x̂_0 panels at
    the peak step without re-running the UNet.

    ``record_cross_attention=True`` (with ``attn_token_indices`` mapping
    save-keys to ``{"branch_index": int, "token_index": int}``) records
    per-step token attention maps via ``_CrossAttnRecorder`` for App-B.
    Only enabled on λ=0 (PoE-anchor) to avoid disk bloat.
    """
    paths: dict[float, Path] = {}
    for lam in lambdas:
        method_name = lambda_method_name(lam)
        print(f"[veracity] sweep λ={lam:.2f}  →  {method_name}")
        # Only capture cross-attention on the PoE-anchor (λ=0) run; all
        # eleven λ runs share the same forward pass within a step, so
        # capturing once is enough for the App-B figure.
        capture_attn = (
            record_cross_attention
            and abs(float(lam)) < 1e-9
        )
        path = cmp_tr.run(
            cell, ctx,
            lambda_schedule="constant",
            lambda_max=float(lam),
            correction_window=None,
            save_residuals=True,
            save_x0_estimates=save_x0_estimates,
            save_trajectory=True,
            exp_name=exp_name,
            overwrite=overwrite,
            record_cross_attention=capture_attn,
            attn_token_indices=attn_token_indices if capture_attn else None,
            attn_resolution=attn_resolution,
        )
        paths[float(lam)] = path
    return paths


def run_reference(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    exp_name: str,
    overwrite: bool = False,
) -> Path:
    """Render the Figure 1 "reference" panel.

    Uses ``run_method("mono", …)`` from the cached baseline dispatcher,
    which writes a standalone Mono PNG decoded from the same ``x_T``.
    """
    return run_method("mono", cell, _proxy_ctx(ctx, exp_name), overwrite=overwrite)


def _proxy_ctx(ctx: MethodCtx, exp_name: str) -> MethodCtx:
    """Re-root the dispatcher's output under ``outputs/<exp_name>``.

    ``run_method`` writes to ``ctx.output_root/<name>/...``; we want it
    under ``ctx.output_root/<exp_name>/...``. We do this by handing
    ``run_method`` a ctx whose ``output_root`` already includes
    ``<exp_name>``, so its concatenation lands in the right place.
    """
    return MethodCtx(
        models=ctx.models,
        scheduler=ctx.scheduler,
        output_root=ctx.output_root / exp_name,
        device=ctx.device,
        dtype=ctx.dtype,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        joint_template=ctx.joint_template,
        synth=ctx.synth,
    )


def cell_dir_for(ctx: MethodCtx, exp_name: str, cell: PairSeedCell) -> Path:
    return (
        ctx.output_root / exp_name / "pairs"
        / cell.pair_slug / f"seed_{cell.seed}"
    )


def make_cell(prompt_a: str, prompt_b: str, seed: int) -> PairSeedCell:
    return cell_for(prompt_a, prompt_b, seed)
