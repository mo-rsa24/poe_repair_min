"""Per-schedule sampling driver, sanity checker, and manifest builder.

Renders one PNG per schedule under
``<run_dir>/schedules/<schedule_id>/image.png`` and a small ``summary.json``
next to it. The inspector consumes the aggregated
``<run_dir>/results/inspector_manifest.json``.

The sanity stage runs two equivalence checks against the existing
``run_cfg`` sampler:

  1. ``run_cfg_masked(mask=all_on)`` ≡ ``run_cfg(...)`` to <1e-5 max abs
     latent delta.
  2. ``run_cfg_masked(mask=all_off)`` ≡ ``run_cfg(..., guidance_scale=0)``
     (where ``guided_eps`` collapses to ε_uncond) to the same tolerance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from poe_repair.experiments.conditioning_window.config import RunConfig
from poe_repair.experiments.conditioning_window.schedules import (
    Schedule,
    all_off,
    all_on,
    mask_to_str,
    num_on,
)
from poe_repair.methods._sampling import (
    run_cfg,
    run_cfg_masked,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir, write_json


SANITY_TOLERANCE = 1.0e-5


def _common_kwargs(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_cond: torch.Tensor,
    pool_cond: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    cfg: RunConfig,
    euler_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Shared kwargs across run_cfg / run_cfg_masked invocations."""
    return dict(
        init_latents=init_latents,
        models=models,
        scheduler=scheduler,
        seq_cond=seq_cond,
        pool_cond=pool_cond,
        seq_e=seq_e,
        pool_e=pool_e,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        height=cfg.height,
        width=cfg.width,
        euler_init_noise_sigma=euler_sigma,
        device=device,
        dtype=dtype,
    )


def render_schedule(
    schedule: Schedule,
    *,
    common: dict[str, Any],
    schedules_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one schedule; write image.png + summary.json. Returns manifest record."""
    out_dir = ensure_dir(schedules_dir / schedule.id)
    image_path = out_dir / "image.png"
    summary_path = out_dir / "summary.json"

    if image_path.exists() and not overwrite:
        record = json.loads(summary_path.read_text())
        return record

    t0 = time.time()
    out = run_cfg_masked(cfg_mask=list(schedule.mask), **common)
    write_decoded_image(out.image, image_path)
    elapsed = time.time() - t0

    record = {
        "id": schedule.id,
        "family": schedule.family,
        "mask": mask_to_str(schedule.mask),
        "num_on": num_on(schedule.mask),
        "image_path": str(image_path),
        "elapsed_s": float(elapsed),
        "sanity": schedule.family == "sanity",
    }
    write_json(summary_path, record)
    print(
        f"[conditioning_window] {schedule.id:<22} num_on={record['num_on']:>2}/"
        f"{len(schedule.mask)}  {elapsed:5.1f}s -> {image_path.name}"
    )
    return record


def run_sweep(
    schedules: Sequence[Schedule],
    *,
    common: dict[str, Any],
    schedules_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Render each schedule; return the manifest records in order."""
    records: list[dict[str, Any]] = []
    for sched in schedules:
        records.append(
            render_schedule(
                sched, common=common, schedules_dir=schedules_dir, overwrite=overwrite,
            )
        )
    return records


def run_sanity(
    *,
    common: dict[str, Any],
    cfg: RunConfig,
    schedules_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """All-on must equal run_cfg; all-off must equal run_cfg(guidance_scale=0).

    Compares latents (max abs delta over the full tensor) and asserts each
    delta is below ``SANITY_TOLERANCE``. Always re-renders (no cache).
    """
    n = cfg.num_inference_steps
    print(f"[conditioning_window] sanity: all_on vs run_cfg")
    out_masked_on = run_cfg_masked(cfg_mask=list(all_on(n)), **common)
    out_cfg = run_cfg(**common)
    delta_on = float((out_masked_on.latents - out_cfg.latents).abs().max().item())
    pass_on = delta_on <= SANITY_TOLERANCE

    print(f"[conditioning_window] sanity: all_off vs run_cfg(guidance_scale=0)")
    common_uncond = dict(common)
    common_uncond["guidance_scale"] = 0.0
    out_masked_off = run_cfg_masked(cfg_mask=list(all_off(n)), **common)
    out_uncond = run_cfg(**common_uncond)
    delta_off = float((out_masked_off.latents - out_uncond.latents).abs().max().item())
    pass_off = delta_off <= SANITY_TOLERANCE

    # Persist reference PNGs for visual confirmation if the user wants to
    # spot-check them.
    sanity_dir = ensure_dir(results_dir / "sanity")
    write_decoded_image(out_masked_on.image, sanity_dir / "masked_all_on.png")
    write_decoded_image(out_cfg.image,        sanity_dir / "run_cfg.png")
    write_decoded_image(out_masked_off.image, sanity_dir / "masked_all_off.png")
    write_decoded_image(out_uncond.image,     sanity_dir / "run_cfg_gs0.png")

    summary = {
        "tolerance": SANITY_TOLERANCE,
        "all_on_vs_run_cfg":  {"max_abs_delta": delta_on,  "pass": bool(pass_on)},
        "all_off_vs_uncond":  {"max_abs_delta": delta_off, "pass": bool(pass_off)},
    }
    write_json(sanity_dir / "sanity.json", summary)
    status = "PASS" if (pass_on and pass_off) else "FAIL"
    print(
        f"[conditioning_window] sanity {status}: "
        f"all_on Δ={delta_on:.2e}, all_off Δ={delta_off:.2e} "
        f"(tol={SANITY_TOLERANCE:.0e})"
    )
    return summary


def build_manifest(
    *,
    cfg: RunConfig,
    records: Sequence[dict[str, Any]],
    sanity: dict[str, Any] | None,
    results_dir: Path,
) -> Path:
    """Write the inspector-consumable JSON. Paths are repo-relative."""
    repo_root = Path(__file__).resolve().parents[3]
    schedules_payload: list[dict[str, Any]] = []
    for r in records:
        rec = dict(r)
        img_abs = Path(rec["image_path"])
        try:
            rec["image_path"] = str(img_abs.resolve().relative_to(repo_root))
        except ValueError:
            rec["image_path"] = str(img_abs)
        schedules_payload.append(rec)

    manifest = {
        "experiment": "conditioning_window",
        "prompt": cfg.prompt,
        "prompt_a": cfg.prompt_a,
        "prompt_b": cfg.prompt_b,
        "pair_slug": cfg.pair_slug,
        "seed": cfg.seed,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "model_id": cfg.model_id,
        "schedules": schedules_payload,
        "sanity": sanity,
    }
    path = ensure_dir(results_dir) / "inspector_manifest.json"
    write_json(path, manifest)
    print(f"[conditioning_window] manifest -> {path}")
    return path
