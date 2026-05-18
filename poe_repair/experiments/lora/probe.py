"""LoRA probe: λ-sweep inference + §4 scoring + where-applied artifact dump.

The probe is the only signal that catches rollout drift; train loss can
fall to zero while the deployed image stays a chimera. Run at the start
of training (epoch 0) and every ``cfg.probe.every_epochs`` epochs.

What's pinned across probes within a run:
  - initial latent z_T  (recovered from cached step_000's x_t)
  - DDIM step sequence  (50 steps from cache meta)
  - CFG, seed           (matches cache build settings)

Only the LoRA weights change across probes; only λ changes within a probe.
The λ=0 column is byte-identical across probes (canary).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poe_repair.experiments.lora.config import RunConfig
from poe_repair.training_cache import CellPath, load_step_raw
from poe_repair.experiments.residual_diagnostics import metrics as v_metrics
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir, write_json


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinned init latent
# ---------------------------------------------------------------------------


def load_pinned_init_latents(
    cell: CellPath,
    *,
    device: torch.device,
    dtype: torch.dtype,
    euler_init_noise_sigma: float,
) -> torch.Tensor:
    """The cache's ``step_000.pt['x_t']`` IS the pre-scale latent at step 0
    (i.e. ``init_latents / euler_init_noise_sigma``). We multiply back so the
    sampler's normal ``init_latents / sigma`` divide yields the cache's x_t
    byte-identically.
    """
    raw = load_step_raw(cell.step_files()[0])
    x_t0 = raw["x_t"]                     # (1, 4, H, W), float32 CPU
    init = x_t0.to(device=device, dtype=dtype) * float(euler_init_noise_sigma)
    return init


# ---------------------------------------------------------------------------
# Scoring (§4 protocol)
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
            image_path,
            list(queries),
            box_threshold=0.35,
            text_threshold=0.25,
        )
        regime = v_metrics.classify_detection_regime(
            dets,
            queries=queries,
            threshold=0.35,
            iou_overlap_threshold=0.4,
        )
    except Exception as exc:
        log.warning(
            "GroundingDINO failed for %s: %s — recording empty regime.",
            image_path, exc,
        )
        dets = []
        regime = "error"
    questions = _vqa_questions(cfg_cell)
    try:
        vqa = v_metrics.vqascore_yesno(image_path, questions)
    except Exception as exc:
        log.warning(
            "VQAScore failed for %s: %s — recording zeros and continuing.",
            image_path, exc,
        )
        vqa = [0.0, 0.0, 0.0]
    vqa_min = float(min(vqa))
    vqa_mean = float(sum(vqa) / max(1, len(vqa)))
    return ProbeMetrics(
        lam=float("nan"),                 # set by caller
        vqa_yes=[float(v) for v in vqa],
        vqa_min=vqa_min,
        vqa_mean=vqa_mean,
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
    """Persist the per-step ``{delta_hat, eps_poe, x_t, tweedie_x0,
    timestep}`` payloads for later overlay rendering. Returns a map
    ``step_index -> on-disk path``.
    """
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
    cfg: RunConfig,
    epoch: int,
    optimizer_step: int,
    probes_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    skip_scoring: bool = False,
) -> ProbeResult:
    """Run the full λ-sweep at this checkpoint. ``embeddings`` carries
    ``seq_{a,b,j,e}`` and matching ``pool_{a,b,j,e}`` tensors. ``models``
    must include both ``"unet"`` (LoRA-attached) and ``"vae"`` for decoding.
    """
    unet = models["unet"]
    was_training = unet.training
    unet.eval()

    epoch_dir = ensure_dir(probes_root / f"epoch_{epoch:04d}")
    results: list[LambdaResult] = []

    for lam in cfg.probe.lambda_grid:
        lam_dir = ensure_dir(epoch_dir / f"lambda_{lam:.2f}")
        decoded_path = lam_dir / "decoded.png"

        out = run_lora_residual_inject(
            init_latents=init_latents,
            models=models,
            scheduler=scheduler,
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
            lora_adapter_name=cfg.lora.adapter_name,
            record_delta_at_steps=list(cfg.probe.where_applied_steps),
            correction_max_rel_norm=None,
        )

        # decode_latents result is in [0, 1] (see sdipc_utils); write_decoded_image expects same.
        decoded_tensor = out.image
        write_decoded_image(decoded_tensor, decoded_path)

        # Where-applied overlays
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
            "probe epoch=%d step=%d lambda=%.2f vqa_min=%.3f regime=%s",
            epoch, optimizer_step, float(lam),
            metrics.vqa_min, metrics.regime,
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
                }
                for r in results
            ],
        },
    )

    if was_training:
        unet.train()

    return ProbeResult(
        epoch=int(epoch),
        optimizer_step=int(optimizer_step),
        results=results,
    )
