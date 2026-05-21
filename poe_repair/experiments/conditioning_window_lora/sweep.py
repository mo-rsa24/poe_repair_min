"""Per-cell sampling driver for the LoRA-on-CFG-mask ablation.

A "cell" is a ``(checkpoint, lambda)`` pair. For each cell × composition
mode × schedule we render one PNG and a per-step ‖δ_eps‖ vector.

Layout::

    <run_dir>/<mode>/epoch_<step>/lambda_<lam>/schedules/<id>/{image.png, summary.json}
    <run_dir>/<mode>/results/inspector_manifest.json   ← rollup across cells

Sanity check (with_prompt, all_on, λ=1.0, final ckpt): masked-with_prompt-all_on
must equal the unmasked ``run_lora_residual_inject`` to <1e-5 latent delta.
Re-run once per training session (not per cell).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from poe_repair.experiments.conditioning_window.schedules import (
    Schedule,
    all_on,
    mask_to_str,
    num_on,
)
from poe_repair.experiments.conditioning_window_lora.config import (
    COMPOSITION_MODES,
    RunConfig,
    epoch_tag,
    lambda_tag,
)
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    run_lora_residual_inject_masked,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir, write_json

SANITY_TOLERANCE = 1.0e-5


def _common_kwargs(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cfg: RunConfig,
    euler_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Shared kwargs across run_lora_residual_inject{,_masked}. ``lambda_value``
    is injected per-cell by ``render_schedule``."""
    return dict(
        init_latents=init_latents,
        models=models,
        scheduler=scheduler,
        seq_a=seq_a, pool_a=pool_a,
        seq_b=seq_b, pool_b=pool_b,
        seq_e=seq_e, pool_e=pool_e,
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
    mode: str,
    common: dict[str, Any],
    schedules_dir: Path,
    lambda_value: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one (schedule, mode) at the current cell's lambda; write
    image.png + summary.json. Returns the manifest record."""
    out_dir = ensure_dir(schedules_dir / schedule.id)
    image_path = out_dir / "image.png"
    summary_path = out_dir / "summary.json"

    if image_path.exists() and not overwrite:
        cached = json.loads(summary_path.read_text())
        if cached.get("image_path") != image_path.name:
            cached["image_path"] = image_path.name
            write_json(summary_path, cached)
        record = dict(cached)
        record["image_path"] = str(image_path)
        return record

    t0 = time.time()
    out = run_lora_residual_inject_masked(
        cfg_mask=list(schedule.mask),
        composition_mode=mode,
        lambda_value=lambda_value,
        **common,
    )
    write_decoded_image(out.image, image_path)
    elapsed = time.time() - t0

    on_disk_record = {
        "id": schedule.id,
        "family": schedule.family,
        "mask": mask_to_str(schedule.mask),
        "num_on": num_on(schedule.mask),
        "composition_mode": mode,
        "lambda_value": float(lambda_value),
        "image_path": image_path.name,
        "delta_norm_per_step": list(out.extras["delta_norm_per_step"]),
        "elapsed_s": float(elapsed),
        "sanity": schedule.family == "sanity",
    }
    write_json(summary_path, on_disk_record)
    record = dict(on_disk_record)
    record["image_path"] = str(image_path)
    print(
        f"[conditioning_window_lora/{mode}] {schedule.id:<22} "
        f"num_on={record['num_on']:>2}/{len(schedule.mask)}  "
        f"{elapsed:5.1f}s -> {image_path.name}"
    )
    return record


def run_cell_sweep(
    schedules: Sequence[Schedule],
    *,
    mode: str,
    common: dict[str, Any],
    schedules_dir: Path,
    lambda_value: float,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sched in schedules:
        records.append(
            render_schedule(
                sched, mode=mode, common=common,
                schedules_dir=schedules_dir, lambda_value=lambda_value,
                overwrite=overwrite,
            )
        )
    return records


def run_sanity(
    *,
    common: dict[str, Any],
    cfg: RunConfig,
    sanity_dir: Path,
    lambda_value: float = 1.0,
) -> dict[str, Any]:
    """``masked(all_on, with_prompt, λ)`` must equal ``run_lora_residual_inject(λ)``.

    Run once per ``main`` invocation, not per cell — it only validates the
    sampler grammar, not the checkpoint.
    """
    n = cfg.num_inference_steps
    print(f"[conditioning_window_lora] sanity: masked(all_on, with_prompt, "
          f"λ={lambda_value}) vs run_lora_residual_inject(λ={lambda_value})")
    out_masked = run_lora_residual_inject_masked(
        cfg_mask=list(all_on(n)),
        composition_mode="with_prompt",
        lambda_value=lambda_value,
        **common,
    )
    out_ref = run_lora_residual_inject(
        seq_j=common["seq_e"], pool_j=common["pool_e"],
        lambda_value=lambda_value,
        **common,
    )
    delta = float((out_masked.latents - out_ref.latents).abs().max().item())
    passed = delta <= SANITY_TOLERANCE

    ensure_dir(sanity_dir)
    write_decoded_image(out_masked.image, sanity_dir / "masked_all_on_with_prompt.png")
    write_decoded_image(out_ref.image,    sanity_dir / "run_lora_residual_inject.png")
    summary = {
        "tolerance": SANITY_TOLERANCE,
        "lambda_value": float(lambda_value),
        "all_on_with_prompt_vs_lora_inject": {
            "max_abs_delta": delta, "pass": bool(passed),
        },
    }
    write_json(sanity_dir / "sanity.json", summary)
    status = "PASS" if passed else "FAIL"
    print(
        f"[conditioning_window_lora] sanity {status}: "
        f"Δ={delta:.2e} (tol={SANITY_TOLERANCE:.0e})"
    )
    return summary


def build_mode_manifest(
    *,
    cfg: RunConfig,
    mode: str,
    cells: list[dict[str, Any]],
    sanity: dict[str, Any] | None,
) -> Path:
    """Aggregate per-cell records into a 3D rollup manifest.

    Schema::

        {
          "experiment": "conditioning_window_lora",
          "composition_mode": mode,
          "epochs": [int, ...],          # sorted asc, present in this mode
          "lambdas": ["0.00", ...],      # canonical tags, sorted asc
          "cells": {
            "<epoch>": {
              "<lambda_tag>": {
                 "ckpt_path": str,
                 "schedules": [ {id, family, mask, num_on, image_path,
                                 delta_norm_per_step, elapsed_s, sanity}, ... ],
              },
              ...
            }, ...
          },
          ...metadata...
        }
    """
    repo_root = Path(__file__).resolve().parents[3]

    def _relpath(p: str | Path) -> str:
        ap = Path(p)
        try:
            return str(ap.resolve().relative_to(repo_root))
        except ValueError:
            return str(ap)

    epochs_set = set()
    lambdas_set = set()
    cells_payload: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        epoch = int(cell["epoch"])
        lam = float(cell["lambda_value"])
        lam_tag = lambda_tag(lam)
        epochs_set.add(epoch)
        lambdas_set.add(lam_tag)
        schedules_payload = []
        for r in cell["records"]:
            rec = dict(r)
            rec["image_path"] = _relpath(rec["image_path"])
            schedules_payload.append(rec)
        cells_payload.setdefault(str(epoch), {})[lam_tag] = {
            "ckpt_path": _relpath(cell["ckpt_path"]),
            "lambda_value": lam,
            "schedules": schedules_payload,
        }

    manifest = {
        "experiment": "conditioning_window_lora",
        "composition_mode": mode,
        "prompt": cfg.prompt,
        "prompt_a": cfg.prompt_a,
        "prompt_b": cfg.prompt_b,
        "pair_slug": cfg.pair_slug,
        "seed": cfg.seed,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "model_id": cfg.model_id,
        "epochs": sorted(epochs_set),
        "lambdas": sorted(lambdas_set),
        "cells": cells_payload,
        "sanity": sanity,
    }
    out = ensure_dir(cfg.mode_results_dir(mode)) / "inspector_manifest.json"
    write_json(out, manifest)
    print(f"[conditioning_window_lora/{mode}] manifest -> {out}")
    return out


__all__ = [
    "COMPOSITION_MODES",
    "SANITY_TOLERANCE",
    "_common_kwargs",
    "render_schedule",
    "run_cell_sweep",
    "run_sanity",
    "build_mode_manifest",
]
