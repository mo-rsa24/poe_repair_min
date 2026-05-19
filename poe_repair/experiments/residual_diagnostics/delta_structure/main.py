"""Δ_t structure-or-noise diagnostic — CLI orchestrator.

Three stages, each independently re-runnable:

    # Full pipeline (collect + analyze + figure) at N=4 seeds:
    python -m poe_repair.experiments.residual_diagnostics.delta_structure

    # Collect only (samples PoE trajectories for each seed; idempotent):
    python -m poe_repair.experiments.residual_diagnostics.delta_structure --stage collect

    # Re-run prongs and figure from cached tensors.pt:
    python -m poe_repair.experiments.residual_diagnostics.delta_structure --stage analyze

    # Push to N=8 (e.g. if Prong B was marginal):
    python -m poe_repair.experiments.residual_diagnostics.delta_structure --seeds 42,43,44,45,46,47,48,49

    # CFG sanity (single seed, unguided ε-space):
    python -m poe_repair.experiments.residual_diagnostics.delta_structure --cfg-sanity --seeds 42

Output layout (per the plan, §4):
    outputs/residual_diagnostics/delta_structure/
        meta.json
        tensors.pt
        results.json
        figures/
            fig_main.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps
from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.config import RunConfig
from poe_repair.diagnostics import delta_structure as DS
from poe_repair.experiments._eval_common import HEADLINE_PAIR, cell_for
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "residual_diagnostics/delta_structure"
DEFAULT_SEEDS = (42, 43, 44, 45)
LAMBDA_FOR_COLLECTION = 0.0   # PoE trajectory; saved Δ_t is independent of λ.


# ---------------------------------------------------------------------------
# Stage 1 — collect
# ---------------------------------------------------------------------------


def _reconstruct_guided_eps(payload: dict, guidance_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From the saved per-step .pt payload, reconstruct guided (eps_poe, eps_mono, delta)."""
    eps_a_raw = payload["eps_a_raw"].float()
    eps_b_raw = payload["eps_b_raw"].float()
    eps_j_raw = payload["eps_j_raw"].float()
    eps_uncond = payload["eps_uncond"].float()

    eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
    eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
    eps_j = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
    eps_poe_g = poe_eps(eps_a, eps_b, eps_uncond)
    eps_mono_g = eps_j
    delta = eps_mono_g - eps_poe_g
    return eps_poe_g.squeeze(0), eps_mono_g.squeeze(0), delta.squeeze(0)


def _reconstruct_unguided_eps(payload: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unguided-ε-space variant for the CFG sanity check (§Phase 3).

    Here Δ_t = ε_J^raw - (ε_A^raw + ε_B^raw - ε_∅) using only the raw
    conditional UNet outputs — no CFG amplification.
    """
    eps_a_raw = payload["eps_a_raw"].float()
    eps_b_raw = payload["eps_b_raw"].float()
    eps_j_raw = payload["eps_j_raw"].float()
    eps_uncond = payload["eps_uncond"].float()
    eps_poe_u = poe_eps(eps_a_raw, eps_b_raw, eps_uncond)
    eps_mono_u = eps_j_raw
    delta = eps_mono_u - eps_poe_u
    return eps_poe_u.squeeze(0), eps_mono_u.squeeze(0), delta.squeeze(0)


def _load_seed_tensors(
    residuals_dir: Path, guidance_scale: float, *, unguided: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Load all step_XXX.pt files in a residuals/ dir; stack into (T, C, H, W) tensors."""
    files = sorted(residuals_dir.glob("step_*.pt"))
    if not files:
        raise FileNotFoundError(f"no step_*.pt in {residuals_dir}")
    deltas, poes, monos, timesteps = [], [], [], []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        if unguided:
            ep, em, dt = _reconstruct_unguided_eps(payload)
        else:
            ep, em, dt = _reconstruct_guided_eps(payload, guidance_scale)
        deltas.append(dt)
        poes.append(ep)
        monos.append(em)
        timesteps.append(int(payload["timestep"]))
    return (
        torch.stack(deltas, dim=0),
        torch.stack(poes, dim=0),
        torch.stack(monos, dim=0),
        timesteps,
    )


def stage_collect(
    *,
    out_root: Path,
    seeds: tuple[int, ...],
    prompt_a: str,
    prompt_b: str,
    overwrite: bool,
    cfg_sanity: bool,
) -> Path:
    """Run teacher_residual for each seed, build the canonical tensors.pt."""
    ctx = make_ctx()
    cfg = RunConfig()

    print(f"[delta_structure] collecting {len(seeds)} seed(s): {list(seeds)}")
    delta_per_seed = []
    poe_per_seed = []
    mono_per_seed = []
    timesteps_ref: list[int] | None = None

    for seed in seeds:
        cell = cell_for(prompt_a, prompt_b, int(seed))
        print(f"[delta_structure]   seed {seed} — sampling (λ=0.0)")
        cmp_tr.run(
            cell, ctx,
            lambda_schedule="constant",
            lambda_max=float(LAMBDA_FOR_COLLECTION),
            correction_window=None,
            save_residuals=True,
            save_x0_estimates=False,
            save_trajectory=False,
            exp_name=EXP_NAME,
            overwrite=overwrite,
        )
        method_name = cmp_tr.method_name_for(
            lambda_schedule="constant",
            lambda_max=float(LAMBDA_FOR_COLLECTION),
            correction_window=None,
        )
        residuals_dir = (
            ctx.output_root / EXP_NAME / "pairs" / cell.pair_slug
            / f"seed_{seed}" / method_name / "residuals"
        )
        delta_s, poe_s, mono_s, timesteps = _load_seed_tensors(
            residuals_dir, ctx.guidance_scale, unguided=cfg_sanity,
        )
        delta_per_seed.append(delta_s)
        poe_per_seed.append(poe_s)
        mono_per_seed.append(mono_s)
        if timesteps_ref is None:
            timesteps_ref = timesteps
        elif timesteps_ref != timesteps:
            raise RuntimeError(
                f"seed {seed} timesteps differ from first seed; cannot stack"
            )

    delta = torch.stack(delta_per_seed, dim=0)   # (S, T, C, H, W)
    eps_poe_tensor = torch.stack(poe_per_seed, dim=0)
    eps_mono_tensor = torch.stack(mono_per_seed, dim=0)
    print(
        f"[delta_structure]   stacked: delta {tuple(delta.shape)}, "
        f"dtype on disk = float16"
    )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cfg.paths.output_root.parent,
        ).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "cell": f"{prompt_a}__x__{prompt_b}",
        "seeds": list(seeds),
        "num_inference_steps": ctx.num_inference_steps,
        "guidance_scale": float(ctx.guidance_scale),
        "scheduler": type(ctx.scheduler).__name__,
        "trajectory": "poe",
        "lambda_for_collection": float(LAMBDA_FOR_COLLECTION),
        "sampler_entrypoint": "poe_repair.methods._sampling.run_teacher_residual",
        "composer_entrypoint": "poe_repair.composers.teacher_residual.run",
        "sampler_commit": commit,
        "torch_dtype_save": "float16",
        "torch_dtype_compute": "float32",
        "cfg_sanity_mode": bool(cfg_sanity),
        "timesteps": list(map(int, timesteps_ref)),
    }
    write_json(out_root / "meta.json", meta)

    tensors_path = out_root / "tensors.pt"
    torch.save(
        {
            "delta": delta.to(torch.float16),
            "eps_poe": eps_poe_tensor.to(torch.float16),
            "eps_mono": eps_mono_tensor.to(torch.float16),
            "timesteps": torch.tensor(timesteps_ref, dtype=torch.int64),
            "seeds": torch.tensor(list(seeds), dtype=torch.int64),
        },
        tensors_path,
    )
    print(f"[delta_structure]   wrote {tensors_path} ({tensors_path.stat().st_size / 1e6:.1f} MB)")
    return tensors_path


# ---------------------------------------------------------------------------
# Stage 2 — analyze + figure
# ---------------------------------------------------------------------------


def stage_analyze(*, out_root: Path) -> dict:
    """Load tensors.pt, run the five prongs, write results.json, write figure."""
    tensors_path = out_root / "tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(
            f"missing {tensors_path}; run --stage collect first"
        )
    blob = torch.load(tensors_path, map_location="cpu", weights_only=False)
    delta = blob["delta"].float()
    eps_poe_tensor = blob["eps_poe"].float()
    eps_mono_tensor = blob["eps_mono"].float()
    print(
        f"[delta_structure] analyzing — delta {tuple(delta.shape)}, "
        f"|seeds|={delta.shape[0]}, T={delta.shape[1]}"
    )

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_bnull = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    a_out = DS.prong_a_time_compressibility(delta, ax=ax_a)
    b_out = DS.prong_b_cross_seed_alignment(
        delta, k=None, n_null=100, ax_matrix=ax_b, ax_null=ax_bnull,
    )
    c_out = DS.prong_c_energy_vs_time(delta, eps_poe=eps_poe_tensor, ax=ax_c)
    d_out = DS.prong_d_spatial_locus(delta, ax=ax_d)
    e_out = DS.prong_e_noise_floor(eps_poe_tensor, eps_mono_tensor, ax=ax_e)

    results = {
        "prong_a_time_compressibility": a_out,
        "prong_b_cross_seed_alignment": b_out,
        "prong_c_energy_vs_time": c_out,
        "prong_d_spatial_locus": d_out,
        "prong_e_noise_floor": e_out,
        "n_seeds_used": int(delta.shape[0]),
    }
    results["final_landing"] = DS.synthesize_landing(results)

    fig.suptitle(
        f"Δ_t structure-or-noise — final_landing = {results['final_landing']}",
        fontsize=12, fontweight="bold",
    )
    fig_dir = ensure_dir(out_root / "figures")
    fig_path = fig_dir / "fig_main.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[delta_structure]   wrote {fig_path}")

    write_json(out_root / "results.json", results)
    print(f"[delta_structure]   wrote {out_root / 'results.json'}")
    print(f"[delta_structure]   final_landing = {results['final_landing']}")
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair", type=str, default=None,
        help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (a cat × a dog).',
    )
    ap.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed list. Default: 42,43,44,45.",
    )
    ap.add_argument(
        "--stage", choices=("all", "collect", "analyze"), default="all",
        help="all = collect + analyze; collect = sample only; analyze = "
             "re-run prongs from cached tensors.pt.",
    )
    ap.add_argument(
        "--cfg-sanity", action="store_true",
        help="Phase 3 CFG sanity: compute Δ in unguided ε-space. Writes to a "
             "sibling output dir 'delta_structure_unguided' so it does not "
             "clobber the guided run.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.pair:
        a, _, b = args.pair.partition("|")
        if not a or not b:
            raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
        prompt_a, prompt_b = a, b
    else:
        prompt_a, prompt_b = HEADLINE_PAIR

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    cfg = RunConfig()
    exp_subdir = "residual_diagnostics/delta_structure"
    if args.cfg_sanity:
        exp_subdir = "residual_diagnostics/delta_structure_unguided"
    out_root = ensure_dir(cfg.paths.output_root / exp_subdir)

    if args.stage in ("all", "collect"):
        stage_collect(
            out_root=out_root,
            seeds=seeds,
            prompt_a=prompt_a, prompt_b=prompt_b,
            overwrite=args.overwrite,
            cfg_sanity=args.cfg_sanity,
        )
    if args.stage in ("all", "analyze"):
        stage_analyze(out_root=out_root)

    print(f"[delta_structure] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
