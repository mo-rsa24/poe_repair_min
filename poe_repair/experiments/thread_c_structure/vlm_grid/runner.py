"""Run the §7c VLM-projection grid.

Per the consolidated plan §7c:

  * Inject α · Δ_t at exactly step t (other steps vanilla PoE).
  * Decode final z_0 (never an intermediate / Tweedie x̂_0).
  * Use a deterministic-tail completion — already the default for our DDIM
    scheduler.
  * Two-axis grader:
      x = co-occurrence VQA score (LLaVA yes-probability on the
          "separate animals" question).
      y = separation confidence = max P(box_A) · max P(box_B).
  * Optional N_reruns reseeded x_T per (seed, t, α) for 95% ellipses.
  * Calibration sweep on the strongest commit-window timestep at
    α ∈ {0, 0.25, 0.5, 0.75, 1.0} → flag non-monotonicity.
  * Route tag (cross-attention A vs B) defaults to "ambiguous" unless
    explicitly wired; the figure honours that by drawing hollow arrowheads.

Costs are reported in the result JSON: at the typical setting
(6 panels × 3 seeds × 3 α-points × 1 rerun = 54 sampler runs per pair)
this is meaningful GPU time. The runner is idempotent at the (seed, t, α,
rerun) image-path level so partial reruns reuse cached images.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from poe_repair.experiments._eval_common import slugify
from poe_repair.experiments.thread_c_structure.d4a.overrides import (
    OverrideBuilder, build_overrides_for_seed,
)
from poe_repair.experiments.thread_c_structure.loader import CellPath
from poe_repair.methods._sampling import run_delta_override, write_decoded_image


DEFAULT_PANEL_STEPS = (49, 39, 29, 19, 9, 1)
DEFAULT_ALPHAS = (0.0, 0.5, 1.0)
DEFAULT_CALIBRATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_COMMIT_CALIBRATION_STEP = 15
DEFAULT_VQA_QUESTION = (
    "Does this image show one {a} and one {b} as separate animals?"
)


@dataclass
class VlmGridSample:
    seed: int
    step_index: int
    timestep: int
    alpha: float
    rerun: int
    image_path: str
    cooccurrence_score: float
    separation_confidence: float
    detection_regime: str
    detection_conf_a: float
    detection_conf_b: float
    route_tag: str = "ambiguous"        # "a", "b", or "ambiguous"


@dataclass
class VlmGridResult:
    pair_slug: str
    prompt_a: str
    prompt_b: str
    panel_steps: list[int]
    alphas: list[float]
    seeds: list[int]
    n_reruns: int
    calibration: dict
    samples: list[VlmGridSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair_slug": self.pair_slug,
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "panel_steps": self.panel_steps,
            "alphas": self.alphas,
            "seeds": self.seeds,
            "n_reruns": self.n_reruns,
            "calibration": self.calibration,
            "samples": [asdict(s) for s in self.samples],
        }


# ---------------------------------------------------------------------------
# Sampler call
# ---------------------------------------------------------------------------


def _build_single_step_override(
    builder: OverrideBuilder, seed: int, step_index: int, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (delta_override, alpha_per_step) for single-step injection."""
    deltas = builder.deltas_by_seed[seed]                # (T, B, C, H, W)
    T = int(deltas.shape[0])
    override = torch.zeros_like(deltas)
    if 0 <= step_index < T:
        override[step_index] = deltas[step_index]
    alpha_per_step = torch.zeros(T, dtype=torch.float32)
    if 0 <= step_index < T:
        alpha_per_step[step_index] = float(alpha)
    return override, alpha_per_step


def _resolve_init_latents(
    cell, ctx, *, rerun: int, seed: int,
) -> tuple[torch.Tensor, float]:
    """For rerun=0, use the cached x_T (matches D4-A). For rerun>0, draw
    fresh standard-normal latents with a deterministic generator."""
    from poe_repair.composers._helpers import init_latents_for_cell
    if rerun == 0:
        return init_latents_for_cell(cell, ctx)
    unet = ctx.models["unet"]
    in_channels = int(getattr(unet.config, "in_channels", 4))
    latent_h = int(cell.height // 8)
    latent_w = int(cell.width // 8)
    cpu_dtype = ctx.dtype if ctx.dtype != torch.bfloat16 else torch.float32
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 10_000 + int(rerun))
    latents = torch.randn(
        1, in_channels, latent_h, latent_w,
        device="cpu", dtype=cpu_dtype, generator=g,
    )
    return latents.to(device=ctx.device, dtype=ctx.dtype), 1.0


def _run_single_injection(
    ctx, cell, builder: OverrideBuilder,
    *, seed: int, step_index: int, alpha: float, rerun: int,
    image_path: Path, overwrite: bool,
) -> None:
    if image_path.exists() and not overwrite:
        return
    from poe_repair.composers._helpers import encode_pair
    init_latents, euler_sigma = _resolve_init_latents(
        cell, ctx, rerun=rerun, seed=seed,
    )
    emb = encode_pair(cell, ctx)
    delta_override, alpha_per_step = _build_single_step_override(
        builder, seed=seed, step_index=step_index, alpha=alpha,
    )
    out = run_delta_override(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        delta_override=delta_override.to(dtype=ctx.dtype),
        alpha_per_step=alpha_per_step,
        window=(step_index, step_index + 1),
    )
    write_decoded_image(out.image, image_path)


# ---------------------------------------------------------------------------
# Grader (two axes)
# ---------------------------------------------------------------------------


def _grade_two_axis(
    image_path: Path, prompt_a: str, prompt_b: str,
    *, vqa_template: str = DEFAULT_VQA_QUESTION,
    box_threshold: float = 0.25, text_threshold: float = 0.25,
) -> tuple[float, float, str, float, float]:
    """Return (co-occurrence, separation, regime, conf_a, conf_b)."""
    from poe_repair.experiments.veracity.metrics import (
        classify_detection_regime, detect_boxes, vqascore_yesno,
    )
    a, b = prompt_a.strip().lower(), prompt_b.strip().lower()
    dets = detect_boxes(
        image_path, [a, b],
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    regime = classify_detection_regime(dets, queries=(a, b), threshold=box_threshold)
    conf_a = max([d["confidence"] for d in dets if d.get("label") == a], default=0.0)
    conf_b = max([d["confidence"] for d in dets if d.get("label") == b], default=0.0)
    separation = float(conf_a * conf_b)
    question = vqa_template.format(a=a, b=b)
    cooccur = float(vqascore_yesno(image_path, [question])[0])
    return cooccur, separation, regime, float(conf_a), float(conf_b)


def _flatness_flag(values: list[float], min_span: float = 0.05) -> bool:
    """Cheap monotonicity check: report True iff values are strictly
    increasing within ``min_span`` tolerance."""
    if len(values) < 2:
        return True
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    # Allow tiny dips (numerical noise) but require overall span ≥ min_span.
    overall = values[-1] - values[0]
    monotone_ish = all(d >= -min_span for d in diffs)
    return monotone_ish and overall >= min_span


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_vlm_grid(
    *,
    cells: list[CellPath],
    prompt_a: str,
    prompt_b: str,
    out_dir: Path,
    ctx=None,
    panel_steps: tuple[int, ...] = DEFAULT_PANEL_STEPS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    n_reruns: int = 1,
    calibration_step: int | None = DEFAULT_COMMIT_CALIBRATION_STEP,
    calibration_alphas: tuple[float, ...] = DEFAULT_CALIBRATION_ALPHAS,
    overwrite: bool = False,
) -> VlmGridResult:
    """Run the §7c grid for a single pair.

    Returns a populated ``VlmGridResult``; also writes the result JSON at
    ``out_dir / vlm_grid.json``. Per-image PNGs land under
    ``out_dir / seed_<n> / step_<t> / alpha_<a>_rerun_<r>.png``.
    """
    from poe_repair.experiments._eval_common import cell_for
    builder = build_overrides_for_seed(cells)
    pair_slug = slugify(prompt_a, prompt_b)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ctx is None:
        from poe_repair.run import make_ctx
        ctx = make_ctx()

    samples: list[VlmGridSample] = []

    # ----- Calibration sweep (one seed, one timestep, full α-set) ---------
    calibration_payload: dict = {
        "step_index": calibration_step,
        "alphas": list(calibration_alphas),
        "axis_x_values": [],
        "axis_y_values": [],
        "x_monotone": None,
        "y_monotone": None,
        "warning": None,
    }
    if calibration_step is not None:
        calib_seed = builder.seeds[0]
        calib_cell = cell_for(prompt_a, prompt_b, calib_seed)
        calib_dir = out_dir / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        xs: list[float] = []
        ys: list[float] = []
        for alpha in calibration_alphas:
            image_path = calib_dir / f"alpha_{int(round(alpha * 100)):03d}.png"
            _run_single_injection(
                ctx, calib_cell, builder,
                seed=calib_seed, step_index=int(calibration_step),
                alpha=float(alpha), rerun=0,
                image_path=image_path, overwrite=overwrite,
            )
            try:
                cooccur, sep, *_ = _grade_two_axis(
                    image_path, prompt_a, prompt_b,
                )
            except Exception as exc:  # noqa: BLE001
                cooccur, sep = float("nan"), float("nan")
                calibration_payload["warning"] = (
                    f"grader unavailable: {type(exc).__name__}: {exc}"
                )
            xs.append(cooccur); ys.append(sep)
        calibration_payload["axis_x_values"] = xs
        calibration_payload["axis_y_values"] = ys
        calibration_payload["x_monotone"] = _flatness_flag(xs)
        calibration_payload["y_monotone"] = _flatness_flag(ys)

    # ----- Main grid -----------------------------------------------------
    for seed in builder.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cell = cell_for(prompt_a, prompt_b, seed)
        for step_idx in panel_steps:
            if step_idx not in builder.step_indices:
                continue
            step_dir = seed_dir / f"step_{step_idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            ts = builder.timesteps[builder.step_indices.index(step_idx)]
            for alpha in alphas:
                for rerun in range(int(n_reruns)):
                    image_path = (
                        step_dir
                        / f"alpha_{int(round(alpha * 100)):03d}_rerun_{rerun:02d}.png"
                    )
                    _run_single_injection(
                        ctx, cell, builder,
                        seed=int(seed), step_index=int(step_idx),
                        alpha=float(alpha), rerun=int(rerun),
                        image_path=image_path, overwrite=overwrite,
                    )
                    try:
                        cooccur, sep, regime, ca, cb = _grade_two_axis(
                            image_path, prompt_a, prompt_b,
                        )
                    except Exception:  # noqa: BLE001
                        cooccur, sep, regime, ca, cb = (
                            float("nan"), float("nan"), "unavailable", 0.0, 0.0,
                        )
                    samples.append(VlmGridSample(
                        seed=int(seed),
                        step_index=int(step_idx),
                        timestep=int(ts),
                        alpha=float(alpha),
                        rerun=int(rerun),
                        image_path=str(image_path),
                        cooccurrence_score=cooccur,
                        separation_confidence=sep,
                        detection_regime=regime,
                        detection_conf_a=ca,
                        detection_conf_b=cb,
                    ))

    result = VlmGridResult(
        pair_slug=pair_slug,
        prompt_a=prompt_a, prompt_b=prompt_b,
        panel_steps=list(panel_steps),
        alphas=list(alphas),
        seeds=builder.seeds,
        n_reruns=int(n_reruns),
        calibration=calibration_payload,
        samples=samples,
    )
    (out_dir / "vlm_grid.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
