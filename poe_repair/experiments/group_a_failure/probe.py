"""Group A probe — λ-sweep inference + where-applied artifact dump.

Mirrors lora.probe in shape; calls ``run_external_corrector_inject``
with the trained corrector instead of toggling a LoRA adapter.

The probe is the only signal that catches rollout drift; training MSE on
cached tensors can fall while the deployed image stays a chimera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poe_repair.experiments.group_a_failure.config import RunConfig
from poe_repair.training_cache import CellPath, load_step_raw
from poe_repair.experiments.residual_diagnostics import metrics as v_metrics
from poe_repair.methods._sampling import (
    run_external_corrector_inject,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir, write_json


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinned init latent (reuses the LoRA trick)
# ---------------------------------------------------------------------------


def load_pinned_init_latents(
    cell: CellPath,
    *,
    device: torch.device,
    dtype: torch.dtype,
    euler_init_noise_sigma: float,
) -> torch.Tensor:
    """``step_000.pt['x_t']`` is the pre-scale latent at step 0; multiply
    back so the sampler's standard divide reconstructs the cache exactly.
    """
    raw = load_step_raw(cell.step_files()[0])
    x_t0 = raw["x_t"]
    return x_t0.to(device=device, dtype=dtype) * float(euler_init_noise_sigma)


# ---------------------------------------------------------------------------
# Scoring (optional; off by default for the qualitative MVP)
# ---------------------------------------------------------------------------


def _vqa_questions(cfg_cell) -> list[str]:
    a, b = cfg_cell.prompt_a, cfg_cell.prompt_b
    return [
        f"Is there {a} in the image?",
        f"Is there {b} in the image?",
        f"Is the {a.replace('a ', '').replace('an ', '')} clearly separate from the {b.replace('a ', '').replace('an ', '')}?",
    ]


@dataclass
class ProbeMetrics:
    lam: float
    vqa_yes: list[float]
    vqa_min: float
    vqa_mean: float
    detections: list[dict]
    regime: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda": self.lam,
            "vqa_yes": list(self.vqa_yes),
            "vqa_min": self.vqa_min,
            "vqa_mean": self.vqa_mean,
            "detections": [
                {**d, "box": list(d["box"])} for d in self.detections
            ],
            "regime": self.regime,
        }


def score_image(
    image_path: Path,
    cfg_cell,
    *,
    skip_scoring: bool = False,
) -> ProbeMetrics:
    if skip_scoring:
        return ProbeMetrics(
            lam=float("nan"), vqa_yes=[0.0, 0.0, 0.0],
            vqa_min=0.0, vqa_mean=0.0,
            detections=[], regime="skipped",
        )
    queries = (cfg_cell.prompt_a, cfg_cell.prompt_b)
    try:
        dets = v_metrics.detect_boxes(
            image_path, list(queries),
            box_threshold=0.35, text_threshold=0.25,
        )
        regime = v_metrics.classify_detection_regime(
            dets, queries=queries,
            threshold=0.35, iou_overlap_threshold=0.4,
        )
    except Exception as exc:
        log.warning("GroundingDINO failed for %s: %s", image_path, exc)
        dets, regime = [], "error"
    questions = _vqa_questions(cfg_cell)
    try:
        vqa = v_metrics.vqascore_yesno(image_path, questions)
    except Exception as exc:
        log.warning("VQAScore failed for %s: %s", image_path, exc)
        vqa = [0.0, 0.0, 0.0]
    return ProbeMetrics(
        lam=float("nan"),
        vqa_yes=[float(v) for v in vqa],
        vqa_min=float(min(vqa)),
        vqa_mean=float(sum(vqa) / max(1, len(vqa))),
        detections=dets,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Where-applied dump
# ---------------------------------------------------------------------------


def dump_delta_overlays(
    where_applied_cache: dict[int, dict[str, torch.Tensor]],
    out_dir: Path,
) -> dict[int, Path]:
    ensure_dir(out_dir)
    paths: dict[int, Path] = {}
    for step_index, payload in where_applied_cache.items():
        out = out_dir / f"step_{int(step_index):02d}.pt"
        torch.save(payload, out)
        paths[int(step_index)] = out
    return paths


# ---------------------------------------------------------------------------
# Probe driver
# ---------------------------------------------------------------------------


@dataclass
class LambdaResult:
    lam: float
    decoded_path: Path
    delta_overlays_dir: Path
    metrics: ProbeMetrics
    delta_norm_per_step: list[float]


@dataclass
class ProbeResult:
    epoch: int
    optimizer_step: int
    results: list[LambdaResult]


def run_probe(
    *,
    models: dict,
    scheduler,
    init_latents: torch.Tensor,
    embeddings: dict[str, torch.Tensor],
    corrector,
    cfg: RunConfig,
    epoch: int,
    optimizer_step: int,
    probes_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    skip_scoring: bool = True,
) -> ProbeResult:
    was_training = corrector.training
    corrector.eval()
    epoch_dir = ensure_dir(probes_root / f"epoch_{epoch:04d}")
    results: list[LambdaResult] = []

    for lam in cfg.probe.lambda_grid:
        lam_dir = ensure_dir(epoch_dir / f"lambda_{lam:.2f}")
        decoded_path = lam_dir / "decoded.png"

        out = run_external_corrector_inject(
            init_latents=init_latents,
            models=models, scheduler=scheduler,
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
            seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
            guidance_scale=cfg.sampler.guidance_scale,
            num_inference_steps=cfg.sampler.num_inference_steps,
            height=cfg.sampler.height, width=cfg.sampler.width,
            euler_init_noise_sigma=cfg.sampler.euler_init_noise_sigma,
            device=device, dtype=dtype,
            lambda_value=float(lam),
            corrector=corrector,
            record_delta_at_steps=list(cfg.probe.where_applied_steps),
            correction_max_rel_norm=cfg.correction_max_rel_norm,
        )

        write_decoded_image(out.image, decoded_path)
        overlays_dir = ensure_dir(lam_dir / "delta_overlays")
        dump_delta_overlays(out.extras["where_applied_cache"], overlays_dir)

        metrics = score_image(decoded_path, cfg.cell, skip_scoring=skip_scoring)
        metrics.lam = float(lam)

        write_json(
            lam_dir / "metrics.json",
            {
                "epoch": int(epoch),
                "optimizer_step": int(optimizer_step),
                **metrics.to_dict(),
                "delta_norm_per_step": list(out.extras["delta_norm_per_step"]),
                "lambda_value": float(lam),
            },
        )
        results.append(
            LambdaResult(
                lam=float(lam),
                decoded_path=decoded_path,
                delta_overlays_dir=overlays_dir,
                metrics=metrics,
                delta_norm_per_step=list(out.extras["delta_norm_per_step"]),
            )
        )
        log.info(
            "probe epoch=%d step=%d lambda=%.2f r_hat_norm_sum=%.2f regime=%s",
            epoch, optimizer_step, float(lam),
            float(sum(out.extras["delta_norm_per_step"])),
            metrics.regime,
        )

    write_json(
        epoch_dir / "summary.json",
        {
            "epoch": int(epoch),
            "optimizer_step": int(optimizer_step),
            "results": [
                {
                    "lambda": r.lam,
                    "vqa_min": r.metrics.vqa_min,
                    "vqa_mean": r.metrics.vqa_mean,
                    "regime": r.metrics.regime,
                    "r_hat_norm_sum": float(sum(r.delta_norm_per_step)),
                }
                for r in results
            ],
        },
    )

    if was_training:
        corrector.train()

    return ProbeResult(epoch=int(epoch), optimizer_step=int(optimizer_step), results=results)
