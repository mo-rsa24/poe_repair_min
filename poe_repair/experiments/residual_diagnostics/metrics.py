"""Stage 2 — distance tables, residual stats, PMI identity curve.

Pure-functions module. No model loading except CLIP for the image-image
cosine distance. Reads the per-run artefacts laid down by ``sweep.py``
and writes three consolidated JSONs into ``outputs/<exp>/metrics/``:

  - ``distances.json``      — d_PoE(λ), d_Mono(λ) for latent-L2 and CLIP.
  - ``residual_stats.json`` — per-step ‖Δ_t‖, direction-stability matrix,
                              per-λ total injected residual norm.
  - ``pmi_identity.json``   — per-step relative residual of the four-eps
                              identity, gathered across all 11 runs (they
                              must coincide because the four raw eps don't
                              depend on λ).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from poe_repair.experiments.residual_diagnostics.sweep import LAMBDA_GRID, lambda_method_name


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def run_dir_for(seed_dir: Path, lam: float) -> Path:
    return seed_dir / lambda_method_name(lam)


def summary_json_path(run_dir: Path) -> Path:
    name = run_dir.name
    return run_dir / f"summary_{name}.json"


def image_path_for(run_dir: Path) -> Path:
    name = run_dir.name
    return run_dir / f"{name}.png"


def trajectory_path_for(run_dir: Path) -> Path:
    return run_dir / "latent_trajectory.pt"


def residuals_dir_for(run_dir: Path) -> Path:
    return run_dir / "residuals"


# ---------------------------------------------------------------------------
# Latent-L2 distance
# ---------------------------------------------------------------------------


def load_final_latent(run_dir: Path) -> torch.Tensor:
    """Return the final ``x_0`` latent (from the saved trajectory)."""
    payload = torch.load(trajectory_path_for(run_dir), map_location="cpu")
    traj = payload["trajectories"]  # (num_steps + 1, batch, C, H, W)
    return traj[-1].float()


def latent_l2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Per-element-RMS L2 distance between two latent tensors."""
    diff = (x.float() - y.float()).flatten()
    return float(diff.norm().item() / math.sqrt(max(1, diff.numel())))


# ---------------------------------------------------------------------------
# CLIP image-image cosine distance
# ---------------------------------------------------------------------------


@dataclass
class _CLIPCache:
    model: object
    processor: object
    device: torch.device


_clip_cache: _CLIPCache | None = None


def _get_clip(device: torch.device | None = None) -> _CLIPCache:
    global _clip_cache
    if _clip_cache is not None:
        return _clip_cache
    from transformers import CLIPModel, CLIPProcessor

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_cache = _CLIPCache(model=model, processor=processor, device=dev)
    return _clip_cache


@torch.no_grad()
def clip_image_embed(image_paths: Sequence[Path], *, device: torch.device | None = None) -> torch.Tensor:
    """Return ``(N, D)`` L2-normalised CLIP image features."""
    clip = _get_clip(device)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = clip.processor(images=images, return_tensors="pt")
    feats = clip.model.get_image_features(
        pixel_values=inputs["pixel_values"].to(clip.device),
    )
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    return feats.detach().cpu().float()


def clip_cosine_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """``1 − cos(a, b)`` between two L2-normalised feature vectors."""
    cos = float((emb_a.flatten() * emb_b.flatten()).sum().item())
    return 1.0 - cos


@torch.no_grad()
def clip_text_embed(
    texts: Sequence[str], *, device: torch.device | None = None,
) -> torch.Tensor:
    """Return ``(N, D)`` L2-normalised CLIP text features."""
    clip = _get_clip(device)
    inputs = clip.processor(
        text=list(texts), return_tensors="pt", padding=True, truncation=True,
    )
    feats = clip.model.get_text_features(
        input_ids=inputs["input_ids"].to(clip.device),
        attention_mask=inputs["attention_mask"].to(clip.device),
    )
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    return feats.detach().cpu().float()


def clip_image_text_similarities(
    image_paths: Sequence[Path],
    texts: Sequence[str],
    *,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """For each text, return the per-image CLIP image-text cosine similarity.

    Useful for tracing concept presence across a λ-sweep — e.g., scoring
    each λ-image against ``"a cat"``, ``"a dog"``, and ``"a cat and a dog"``.
    """
    img_emb = clip_image_embed(image_paths, device=device)  # (N_img, D)
    txt_emb = clip_text_embed(texts, device=device)         # (N_txt, D)
    # cosine since both are L2-normalised
    sim = (img_emb @ txt_emb.t()).cpu().float()             # (N_img, N_txt)
    return {
        str(text): [float(sim[i, j].item()) for i in range(img_emb.shape[0])]
        for j, text in enumerate(texts)
    }


# ---------------------------------------------------------------------------
# Residual norm + direction stability
# ---------------------------------------------------------------------------


def load_summary(run_dir: Path) -> dict:
    return json.loads(summary_json_path(run_dir).read_text())


def per_step_residual_norms(run_dir: Path) -> list[float]:
    """Return ``‖Δ_t‖`` per step from the run's summary JSON."""
    return list(load_summary(run_dir)["delta_norm_per_step"])


def lambda_per_step(run_dir: Path) -> list[float]:
    return list(load_summary(run_dir)["lambda_per_step"])


def pmi_identity_per_step(run_dir: Path) -> list[float]:
    return list(load_summary(run_dir)["pmi_identity_residual_per_step"])


def load_residual_tensor(run_dir: Path, step_index: int) -> torch.Tensor:
    payload = torch.load(
        residuals_dir_for(run_dir) / f"step_{step_index:03d}.pt",
        map_location="cpu",
    )
    return payload["delta"].float()


def direction_stability_matrix(run_dir: Path) -> torch.Tensor:
    """Return T × T cosine similarity matrix between per-step ``Δ_t``."""
    res_dir = residuals_dir_for(run_dir)
    files = sorted(res_dir.glob("step_*.pt"))
    flats: list[torch.Tensor] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        flat = payload["delta"].float().flatten()
        flat = flat / (flat.norm() + 1e-12)
        flats.append(flat)
    if not flats:
        return torch.zeros(0, 0)
    mat = torch.stack(flats, dim=0)  # (T, D)
    return (mat @ mat.t()).float()


# ---------------------------------------------------------------------------
# Aggregate / write JSONs
# ---------------------------------------------------------------------------


def compute_distance_table(
    run_dirs_by_lambda: dict[float, Path],
    *,
    device: torch.device | None = None,
) -> dict:
    """Compute d_PoE(λ) and d_Mono(λ) under latent-L2 and CLIP cosine."""
    lambdas = sorted(run_dirs_by_lambda.keys())
    # Anchors: λ=0 ↔ PoE; λ=1 ↔ Mono.
    poe_lam = lambdas[0]
    mono_lam = lambdas[-1]
    if abs(poe_lam) > 1e-9 or abs(mono_lam - 1.0) > 1e-9:
        raise ValueError(
            f"expected λ-grid to span [0.0, 1.0]; got {lambdas}"
        )

    # Latent-L2.
    latents = {lam: load_final_latent(run_dirs_by_lambda[lam]) for lam in lambdas}
    d_poe_lat: dict[float, float] = {}
    d_mono_lat: dict[float, float] = {}
    for lam in lambdas:
        d_poe_lat[lam] = latent_l2(latents[lam], latents[poe_lam])
        d_mono_lat[lam] = latent_l2(latents[lam], latents[mono_lam])

    # CLIP image cosine.
    image_paths = [image_path_for(run_dirs_by_lambda[lam]) for lam in lambdas]
    embeds = clip_image_embed(image_paths, device=device)
    poe_emb = embeds[0]
    mono_emb = embeds[-1]
    d_poe_clip: dict[float, float] = {}
    d_mono_clip: dict[float, float] = {}
    for i, lam in enumerate(lambdas):
        d_poe_clip[lam] = clip_cosine_distance(embeds[i], poe_emb)
        d_mono_clip[lam] = clip_cosine_distance(embeds[i], mono_emb)

    return {
        "lambdas": lambdas,
        "anchor_poe_lambda": poe_lam,
        "anchor_mono_lambda": mono_lam,
        "latent_l2": {
            "d_poe": [d_poe_lat[lam] for lam in lambdas],
            "d_mono": [d_mono_lat[lam] for lam in lambdas],
        },
        "clip_image_cosine": {
            "d_poe": [d_poe_clip[lam] for lam in lambdas],
            "d_mono": [d_mono_clip[lam] for lam in lambdas],
        },
    }


def compute_residual_stats(run_dirs_by_lambda: dict[float, Path]) -> dict:
    """Aggregate per-step residual norms + per-λ total injected norm.

    Direction-stability matrix is computed from the λ=0.0 run (the
    residual tensor is λ-independent because Δ is computed before
    scaling, so any run gives the same matrix).
    """
    lambdas = sorted(run_dirs_by_lambda.keys())
    anchor = run_dirs_by_lambda[lambdas[0]]

    delta_norms = per_step_residual_norms(anchor)
    sum_delta = float(sum(delta_norms))

    total_injected: dict[float, float] = {}
    for lam in lambdas:
        # Σ_t λ_t · ‖Δ_t‖. With constant schedule λ_t == λ for all t.
        total_injected[lam] = float(lam) * sum_delta

    stab = direction_stability_matrix(anchor)
    return {
        "lambdas": lambdas,
        "anchor_run": str(anchor),
        "delta_norm_per_step": delta_norms,
        "sum_delta_norm": sum_delta,
        "total_injected_per_lambda": [total_injected[lam] for lam in lambdas],
        "direction_stability_matrix": stab.tolist(),
        "num_steps": stab.shape[0],
    }


def compute_pmi_identity(run_dirs_by_lambda: dict[float, Path]) -> dict:
    """Gather the PMI self-consistency curve across all 11 runs."""
    lambdas = sorted(run_dirs_by_lambda.keys())
    curves = {lam: pmi_identity_per_step(run_dirs_by_lambda[lam]) for lam in lambdas}
    flat = [v for curve in curves.values() for v in curve]
    return {
        "lambdas": lambdas,
        "per_lambda_curve": {f"{lam:.2f}": curves[lam] for lam in lambdas},
        "max_relative_error": float(max(flat)) if flat else 0.0,
        "mean_relative_error": float(sum(flat) / len(flat)) if flat else 0.0,
    }


# ---------------------------------------------------------------------------
# Latent-trajectory distance per step
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# x̂_0-prediction stability (Fig 2 bottom panel)
# ---------------------------------------------------------------------------


def _tweedie_x0_latent(
    x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: float,
) -> torch.Tensor:
    """Tweedie-mean x̂_0 in *latent* space (no VAE decode)."""
    sa = float(alpha_bar) ** 0.5
    so = float(1.0 - alpha_bar) ** 0.5
    return (x_t - so * eps) / sa


def compute_x0_stability(
    run_dir: Path,
    *,
    eps_key: str = "eps_j",
    scheduler=None,
) -> dict:
    """Compute per-step ``1 − corr(x̂_0(t), x̂_0(T))`` along a trajectory.

    ``run_dir`` is a single λ run dir (e.g., the Mono-anchor λ=1 dir; the
    eps used is selected by ``eps_key`` ∈ {"eps_j", "eps_poe"}; the
    caller picks which trajectory to characterise).

    The correlation is computed in *latent* space (cheap; avoids 50 VAE
    decodes). The terminal step uses the highest-index step file as the
    reference. ``scheduler`` is required to look up ``alphas_cumprod[t]``.
    """
    res_dir = residuals_dir_for(run_dir)
    files = sorted(res_dir.glob("step_*.pt"))
    if not files:
        return {"step_indices": [], "stability": [], "anchor_step": None}
    if scheduler is None:
        raise ValueError(
            "compute_x0_stability requires a scheduler (for alphas_cumprod)"
        )

    x0s: list[torch.Tensor] = []
    step_indices: list[int] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        if eps_key not in payload:
            raise KeyError(
                f"{f} missing {eps_key!r}; ensure save_x0_estimates=True at sweep time"
            )
        timestep = int(payload["timestep"])
        alpha_bar = float(
            scheduler.alphas_cumprod[timestep].to(dtype=torch.float64).item()
        )
        x_t = payload["x_t"].float()
        eps = payload[eps_key].float()
        x0 = _tweedie_x0_latent(x_t, eps, alpha_bar)
        x0s.append(x0.flatten())
        step_indices.append(int(payload["step_index"]))

    # Anchor at the largest step_index (smallest t — most denoised).
    anchor_idx = int(max(range(len(step_indices)), key=lambda i: step_indices[i]))
    anchor = x0s[anchor_idx]
    anchor_centered = anchor - anchor.mean()
    anchor_norm = anchor_centered.norm()

    stability: list[float] = []
    for x0 in x0s:
        c = x0 - x0.mean()
        denom = (c.norm() * anchor_norm).item()
        if denom < 1e-12:
            corr = 0.0
        else:
            corr = float((c @ anchor_centered).item() / denom)
        stability.append(1.0 - corr)

    return {
        "step_indices": step_indices,
        "stability": stability,
        "anchor_step": int(step_indices[anchor_idx]),
        "eps_key": eps_key,
        "run_dir": str(run_dir),
    }


def basin_commit_window(
    stability_curve: list[float],
    *,
    threshold: float = 0.05,
) -> tuple[int, int]:
    """Return (commit_start, commit_end) — the indices where stability first
    crosses below ``threshold`` and remains there.

    Reads as: ``commit_start`` is the earliest step at which x̂_0 has
    "locked in" to the terminal prediction. ``commit_end`` is the last
    such step (typically the terminal step).
    """
    if not stability_curve:
        return (0, 0)
    below = [i for i, v in enumerate(stability_curve) if v <= threshold]
    if not below:
        # Never crosses threshold; commit "happens" only at the final step.
        return (len(stability_curve) - 1, len(stability_curve) - 1)
    return (below[0], below[-1])


# ---------------------------------------------------------------------------
# Per-step residual norms recomputed from cached residual .pt files
# ---------------------------------------------------------------------------


def per_step_residual_norms_from_residuals(run_dir: Path) -> list[float]:
    """Per-step ``‖Δ_t‖`` recomputed directly from residual ``.pt`` files.

    Used for control runs where summary JSONs may not exist or to avoid
    the (small) drift between summary cache and on-disk residual norms.
    """
    res_dir = residuals_dir_for(run_dir)
    files = sorted(res_dir.glob("step_*.pt"))
    norms: list[float] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        norms.append(float(payload["delta"].float().norm().item()))
    return norms


def image_space_residual_trajectory(
    run_dir: Path,
    *,
    ctx,
    reduce: str = "mean",
) -> dict:
    """For each cached residual step, decode ``x̂_0(ε̃_J)`` and ``x̂_0(ε̃_PoE)``
    via VAE, take per-pixel L2 norm of the difference, and aggregate to a
    scalar per step.

    ``reduce`` ∈ {"mean", "max", "fro"} controls the aggregation.

    Returns ``{"step_indices": [...], "image_space_residual": [...],
    "reduce": reduce, "run_dir": str}``.
    """
    from poe_repair.runtime import decode_latents
    from poe_repair._sdxl.metrics import tweedie_mean

    res_dir = residuals_dir_for(run_dir)
    files = sorted(res_dir.glob("step_*.pt"))
    if not files:
        return {"step_indices": [], "image_space_residual": [],
                "reduce": reduce, "run_dir": str(run_dir)}

    out_steps: list[int] = []
    out_vals: list[float] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        if "eps_poe" not in payload or "eps_j" not in payload:
            continue
        timestep = int(payload["timestep"])
        alpha_bar_t = ctx.scheduler.alphas_cumprod[timestep].to(
            device=ctx.device, dtype=ctx.dtype,
        )
        x_t = payload["x_t"].float().to(ctx.device, ctx.dtype)
        eps_poe = payload["eps_poe"].float().to(ctx.device, ctx.dtype)
        eps_j = payload["eps_j"].float().to(ctx.device, ctx.dtype)
        x0_poe = tweedie_mean(x_t, alpha_bar_t, eps_poe)
        x0_j = tweedie_mean(x_t, alpha_bar_t, eps_j)
        img_poe = decode_latents(ctx.models, x0_poe).cpu().float()  # (1,3,H,W) in [0,1]
        img_j = decode_latents(ctx.models, x0_j).cpu().float()
        diff = (img_j - img_poe).squeeze(0)  # (3, H, W)
        per_pixel_norm = diff.norm(dim=0)    # (H, W)
        if reduce == "mean":
            val = float(per_pixel_norm.mean().item())
        elif reduce == "max":
            val = float(per_pixel_norm.max().item())
        elif reduce == "fro":
            val = float(diff.norm().item())
        else:
            raise ValueError(f"unknown reduce={reduce!r}")
        out_steps.append(int(payload["step_index"]))
        out_vals.append(val)
    return {
        "step_indices": out_steps,
        "image_space_residual": out_vals,
        "reduce": reduce,
        "run_dir": str(run_dir),
    }



# ---------------------------------------------------------------------------
# Cross-attention entropy (App-B)
# ---------------------------------------------------------------------------


def attention_entropy(attn_map: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Shannon entropy of an attention map (treated as a probability over
    spatial positions). Higher = more diffuse; lower = more peaked.

    ``attn_map`` is a 2D tensor (H, W); values must be non-negative.
    """
    p = attn_map.float().flatten()
    p = p / (p.sum() + eps)
    plog = torch.where(p > eps, p * p.log(), torch.zeros_like(p))
    return float(-plog.sum().item())


def compute_attn_entropy_from_dir(
    attn_dir: Path,
    *,
    token_keys: list[str],
    branch_keys: list[str],
) -> dict:
    """Walk ``attn_dir`` for files matching
    ``step_{idx:03d}_token_{tok}_branch_{br}.pt`` and compute per-step,
    per-token, per-branch attention entropy.
    """
    out: dict[str, dict[str, list[float]]] = {
        br: {tok: [] for tok in token_keys} for br in branch_keys
    }
    step_indices: list[int] = []
    files = sorted(attn_dir.glob("step_*_token_*_branch_*.pt"))
    if not files:
        return {"step_indices": [], "per_branch_per_token": out}

    by_step: dict[int, dict[tuple[str, str], torch.Tensor]] = {}
    for f in files:
        # Parse step idx, token, branch from name.
        stem = f.stem  # step_XXX_token_T_branch_B
        parts = stem.split("_")
        try:
            idx = int(parts[1])
            tok = parts[3]
            br = parts[5]
        except (IndexError, ValueError):
            continue
        attn = torch.load(f, map_location="cpu")
        if isinstance(attn, dict) and "map" in attn:
            attn = attn["map"]
        by_step.setdefault(idx, {})[(tok, br)] = attn

    step_indices = sorted(by_step.keys())
    for idx in step_indices:
        bucket = by_step[idx]
        for br in branch_keys:
            for tok in token_keys:
                a = bucket.get((tok, br))
                if a is None:
                    out[br][tok].append(float("nan"))
                else:
                    out[br][tok].append(attention_entropy(a))

    return {
        "step_indices": step_indices,
        "per_branch_per_token": out,
        "attn_dir": str(attn_dir),
    }


# ---------------------------------------------------------------------------
# Latent-trajectory distance per step (existing)
# ---------------------------------------------------------------------------


def trajectory_distance_per_step(
    run_dirs_by_lambda: dict[float, Path],
) -> dict:
    """For every λ, return ``‖x_t(λ) − x_t(λ=1)‖`` per step.

    Final entry is ``x_0``. We compute distance per saved trajectory
    entry (T+1 entries), normalised by element count.
    """
    lambdas = sorted(run_dirs_by_lambda.keys())
    mono_traj = torch.load(
        trajectory_path_for(run_dirs_by_lambda[lambdas[-1]]),
        map_location="cpu",
    )["trajectories"].float()  # (T+1, B, C, H, W)

    per_lambda: dict[float, list[float]] = {}
    for lam in lambdas:
        traj = torch.load(
            trajectory_path_for(run_dirs_by_lambda[lam]),
            map_location="cpu",
        )["trajectories"].float()
        if traj.shape != mono_traj.shape:
            raise ValueError(
                f"trajectory shape mismatch at λ={lam}: {traj.shape} vs {mono_traj.shape}"
            )
        diff = (traj - mono_traj).flatten(start_dim=1)
        n_per_step = diff.shape[1]
        per_step = (diff.norm(dim=1) / math.sqrt(max(1, n_per_step))).tolist()
        per_lambda[lam] = per_step

    return {
        "lambdas": lambdas,
        "num_steps_plus_one": int(mono_traj.shape[0]),
        "per_lambda_distance_to_mono": {
            f"{lam:.2f}": per_lambda[lam] for lam in lambdas
        },
    }


# ---------------------------------------------------------------------------
# Open-vocabulary detection (App-B', Fig 4, App-E) — GroundingDINO-Tiny
# ---------------------------------------------------------------------------


class _DetectorCache:
    """Lazy-loaded GroundingDINO-Tiny via HuggingFace Transformers.

    Loaded on first ``detect_boxes()`` call. Cached at module scope so
    subsequent calls reuse the model. Raises ``RuntimeError`` with a
    pip-install hint if Transformers can't find the model.
    """

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device: torch.device | None = None


_DETECTOR: _DetectorCache | None = None


def _get_detector(device: torch.device | None = None) -> _DetectorCache:
    global _DETECTOR
    if _DETECTOR is not None and _DETECTOR.model is not None:
        return _DETECTOR
    try:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GroundingDINO requires `transformers>=4.40`. "
            "Install with `pip install transformers`."
        ) from exc

    cache = _DetectorCache()
    cache.device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_id = "IDEA-Research/grounding-dino-tiny"
    cache.processor = AutoProcessor.from_pretrained(model_id)
    cache.model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id
    ).to(cache.device).eval()
    _DETECTOR = cache
    return cache


def detect_boxes(
    image_path: Path,
    text_queries: list[str],
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    device: torch.device | None = None,
) -> list[dict]:
    """Run GroundingDINO-Tiny on ``image_path`` once per query in
    ``text_queries`` and return a flat list of detections:

        [{"box": (x0, y0, x1, y1), "confidence": float, "label": str}, ...]

    Boxes are in image-pixel coordinates (matching the image's H, W).
    ``label`` is set to the *original query string* (lower-cased), not
    GroundingDINO's matched-span text — because HF's processor returns
    the full joint-query span for each detection, which breaks
    per-concept downstream matching.

    Per-query separate forward passes (~+50ms each on GPU; cheap):
    cleaner label semantics with no need for downstream substring
    matching against ambiguous returned spans.
    """
    cache = _get_detector(device=device)
    image = Image.open(image_path).convert("RGB")
    target_sizes = torch.tensor([image.size[::-1]], device=cache.device)

    # HF API used `box_threshold` up through 4.44; later renamed to
    # `threshold`. Probe both at call time.
    import inspect
    post = cache.processor.post_process_grounded_object_detection
    accepts_box_threshold = "box_threshold" in inspect.signature(post).parameters

    out: list[dict] = []
    for query in text_queries:
        q = query.strip().lower()
        # Single-phrase prompt — period-terminated per HF GroundingDINO
        # convention.
        text = q if q.endswith(".") else q + "."
        inputs = cache.processor(
            images=image, text=text, return_tensors="pt",
        ).to(cache.device)
        with torch.no_grad():
            outputs = cache.model(**inputs)
        kwargs = dict(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
            text_threshold=text_threshold,
        )
        if accepts_box_threshold:
            kwargs["box_threshold"] = box_threshold
        else:
            kwargs["threshold"] = box_threshold
        results = post(**kwargs)[0]
        for score, box in zip(results["scores"], results["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            out.append({
                "box": (x0, y0, x1, y1),
                "confidence": float(score.item()),
                "label": q,
            })
    return out


def box_iou(
    b1: tuple[float, float, float, float],
    b2: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = b1
    bx0, by0, bx1, by1 = b2
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def classify_detection_regime(
    detections: list[dict],
    *,
    queries: tuple[str, ...],
    threshold: float = 0.35,
    iou_overlap_threshold: float = 0.4,
) -> str:
    """Classify the failure regime from a list of detections against two queries.

    Returns one of:
        "both_distinct"     — both queries detected, boxes don't overlap much
        "both_overlapping"  — both queries detected but heavily overlapping
                              (chimera-typical: blended in same region)
        "single"            — exactly one of the two queries detected
        "none"              — neither query detected
    """
    if len(queries) != 2:
        raise ValueError("classify_detection_regime expects exactly two queries")
    q1, q2 = queries[0].strip().lower(), queries[1].strip().lower()

    def _best(label: str):
        cands = [
            d for d in detections
            if d.get("label", "").strip().lower() == label
            and d.get("confidence", 0.0) >= threshold
        ]
        if not cands:
            return None
        return max(cands, key=lambda d: d["confidence"])

    d1 = _best(q1)
    d2 = _best(q2)
    if d1 is None and d2 is None:
        return "none"
    if d1 is None or d2 is None:
        return "single"
    iou = box_iou(d1["box"], d2["box"])
    if iou >= iou_overlap_threshold:
        return "both_overlapping"
    return "both_distinct"


# ---------------------------------------------------------------------------
# VQAScore (App-E) — LLaVA-1.5-7b, yes/no grounded scoring
# ---------------------------------------------------------------------------


class _VQACache:
    """Lazy-loaded LLaVA-1.5 via HuggingFace Transformers.

    The first ``vqascore_yesno()`` call downloads and loads the model
    (~14GB on disk in fp16). Subsequent calls reuse it.
    """

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device: torch.device | None = None
        self.yes_token_id: int | None = None
        self.no_token_id: int | None = None


_VQA: _VQACache | None = None


def _get_vqa(device: torch.device | None = None) -> _VQACache:
    global _VQA
    if _VQA is not None and _VQA.model is not None:
        return _VQA
    try:
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LLaVA-1.5 requires `transformers>=4.40`. "
            "Install with `pip install transformers`."
        ) from exc

    cache = _VQACache()
    cache.device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_id = "llava-hf/llava-1.5-7b-hf"
    # Try fast tokenizer first; fall back to slow if the installed
    # `tokenizers` crate can't deserialise the model's tokenizer.json
    # ("data did not match any variant of untagged enum ModelWrapper").
    try:
        cache.processor = AutoProcessor.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        if "ModelWrapper" in str(exc) or "tokenizer" in str(exc).lower():
            cache.processor = AutoProcessor.from_pretrained(
                model_id, use_fast=False,
            )
        else:
            raise
    dtype = torch.float16 if cache.device.type == "cuda" else torch.float32
    cache.model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(cache.device).eval()

    # Yes/No token ids for log-prob scoring.
    tok = cache.processor.tokenizer
    # Use the leading-space variants since they appear in mid-sentence
    # generation context after the "ASSISTANT:" prefix.
    yes_ids = tok.encode(" Yes", add_special_tokens=False)
    no_ids = tok.encode(" No", add_special_tokens=False)
    cache.yes_token_id = yes_ids[0] if yes_ids else tok.encode("Yes")[0]
    cache.no_token_id = no_ids[0] if no_ids else tok.encode("No")[0]
    _VQA = cache
    return cache


def vqascore_yesno(
    image_path: Path,
    questions: list[str],
    *,
    device: torch.device | None = None,
) -> list[float]:
    """LLaVA-1.5 Yes-probability per question, grounded on the image.

    For each question we form a single-turn LLaVA prompt:

        USER: <image>\n{question}\nASSISTANT:

    do one forward pass, and read the logit at the next-token position
    for the "Yes" and "No" tokens, returning
    ``softmax([logit_yes, logit_no])[0]`` — the Bayes-renormalised
    yes-probability.

    Returns a list aligned with ``questions``.
    """
    cache = _get_vqa(device=device)
    image = Image.open(image_path).convert("RGB")

    out: list[float] = []
    for q in questions:
        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = cache.processor(
            images=image, text=prompt, return_tensors="pt",
        ).to(cache.device)
        with torch.no_grad():
            outputs = cache.model(**inputs)
        # logits: (1, seq_len, vocab) — pick the last position to predict
        # the next token (the first ASSISTANT response token).
        last_logits = outputs.logits[0, -1, :]
        yes_l = last_logits[cache.yes_token_id]
        no_l = last_logits[cache.no_token_id]
        # Renormalised softmax over only {Yes, No}.
        pair = torch.stack([yes_l, no_l])
        p_yes = float(torch.softmax(pair.float(), dim=0)[0].item())
        out.append(p_yes)
    return out
