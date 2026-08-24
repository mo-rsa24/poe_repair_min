"""Stage 1 — compute Tweedie x̂_0 snapshots + CLIP scores.

For one veracity run (one cell, one λ value) and a list of step indices,
loads the saved per-step ``x_t`` + ``eps_poe`` + ``eps_j`` from disk,
reconstructs ``ε_t`` for the chosen λ, applies Tweedie's formula, decodes
the resulting clean-image guess through SDXL's VAE, and scores it against
a list of CLIP text targets.

All forward passes are ``@no_grad`` — this is purely diagnostic. The
only model evaluations are: VAE decode (once per snapshot) + CLIP
encoder forwards (once per snapshot, once per text target at startup).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from poe_repair.experiments.residual_between_mono_and_poe.metrics import _get_clip
from poe_repair.runtime import decode_latents, tweedie_mean


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_SNAPSHOT_STEPS: tuple[int, ...] = (
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49,
)
DEFAULT_TEXT_TARGETS: tuple[str, ...] = (
    "a cat",
    "a dog",
    "a cat and a dog",
)
DEFAULT_LAMBDAS: tuple[tuple[float, str], ...] = (
    (0.0, "PoE"),
    (0.6, "λ=0.6 (basin transition)"),
    (1.0, "Mono"),
)


# ---------------------------------------------------------------------------
# CLIP text-target embedding cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def cache_text_embeddings(text_targets: Sequence[str], *, device=None) -> torch.Tensor:
    """Return ``(N, D)`` L2-normalised CLIP text features for the targets."""
    clip = _get_clip(device)
    inputs = clip.processor(
        text=list(text_targets), return_tensors="pt", padding=True, truncation=True,
    )
    feats = clip.model.get_text_features(
        input_ids=inputs["input_ids"].to(clip.device),
        attention_mask=inputs["attention_mask"].to(clip.device),
    )
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    return feats.detach().cpu().float()


# ---------------------------------------------------------------------------
# CLIP image scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def clip_score_image(
    decoded_image: torch.Tensor,
    text_embeds: torch.Tensor,
    *,
    device=None,
) -> list[float]:
    """Score one decoded image against ``(N, D)`` text embeddings.

    ``decoded_image`` is a ``[1, 3, H, W]`` or ``[3, H, W]`` tensor in
    ``[0, 1]``. CLIP's processor handles the actual normalisation /
    resize.
    """
    clip = _get_clip(device)
    arr = decoded_image.detach().float().clamp(0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    arr_u8 = (arr * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    pil = Image.fromarray(arr_u8)

    inputs = clip.processor(images=[pil], return_tensors="pt")
    img_feats = clip.model.get_image_features(
        pixel_values=inputs["pixel_values"].to(clip.device),
    )
    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
    img_feats = img_feats.detach().cpu().float()       # (1, D)

    sims = (img_feats * text_embeds).sum(dim=-1)        # (N,)
    return [float(s.item()) for s in sims]


# ---------------------------------------------------------------------------
# Tweedie reconstruction at a chosen λ
# ---------------------------------------------------------------------------


def eps_for_lambda(*, eps_poe: torch.Tensor, eps_j: torch.Tensor, lam: float) -> torch.Tensor:
    """``ε̃_t = ε̃_PoE + λ · (ε̃_J − ε̃_PoE)``."""
    if lam == 0.0:
        return eps_poe
    if lam == 1.0:
        return eps_j
    return eps_poe + float(lam) * (eps_j - eps_poe)


# ---------------------------------------------------------------------------
# Main per-trajectory analysis
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySnapshots:
    label: str                         # human-readable
    lam: float                         # λ value (mainly bookkeeping)
    run_dir: Path                      # source veracity run on disk
    step_indices: list[int]            # which steps were sampled
    timesteps: list[int]               # the scheduler timestep at each
    decoded_paths: list[Path]          # per-snapshot PNG paths
    clip_scores: list[list[float]]     # [snapshot][text_target] cosine
    final_image_path: Path | None      # full final x_0 (decoded once)


def _residual_payload(run_dir: Path, step_index: int) -> dict:
    p = run_dir / "residuals" / f"step_{step_index:03d}.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def _trajectory_payload(run_dir: Path) -> dict:
    return torch.load(
        run_dir / "latent_trajectory.pt", map_location="cpu", weights_only=False,
    )


@torch.no_grad()
def analyse_trajectory(
    *,
    run_dir: Path,
    label: str,
    lam: float,
    snapshot_steps: Sequence[int],
    text_embeds: torch.Tensor,
    text_targets: Sequence[str],
    ctx,
    out_dir: Path,
) -> TrajectorySnapshots:
    """Compute snapshots + CLIP scores for one trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_paths: list[Path] = []
    timesteps: list[int] = []
    clip_scores: list[list[float]] = []

    print(f"[idea5a]   {label} (λ={lam}) — {len(snapshot_steps)} snapshots")
    for step_index in snapshot_steps:
        payload = _residual_payload(run_dir, step_index)
        x_t = payload["x_t"].float()
        eps_poe = payload["eps_poe"].float()
        eps_j = payload["eps_j"].float()
        timestep = int(payload["timestep"])
        timesteps.append(timestep)

        eps_t = eps_for_lambda(eps_poe=eps_poe, eps_j=eps_j, lam=lam)

        # Tweedie x̂_0 in latent space.
        alpha_bar_t = ctx.scheduler.alphas_cumprod[timestep].to(
            device=ctx.device, dtype=torch.float32,
        )
        x_hat_0 = tweedie_mean(
            x_t.to(ctx.device), alpha_bar_t, eps_t.to(ctx.device),
        )

        # VAE decode.
        decoded = decode_latents(ctx.models, x_hat_0).cpu()    # [1, 3, H, W] in [0, 1]

        # Save PNG.
        from poe_repair.methods._sampling import write_decoded_image
        png_path = out_dir / f"snapshot_step_{step_index:03d}.png"
        write_decoded_image(decoded, png_path)
        decoded_paths.append(png_path)

        # CLIP scores.
        scores = clip_score_image(decoded, text_embeds, device=ctx.device)
        clip_scores.append(scores)
        print(
            f"[idea5a]     step {step_index:3d} (t={timestep:4d})  "
            + "  ".join(
                f"{tgt[:14]}={s:+.3f}" for tgt, s in zip(text_targets, scores)
            )
        )

    # Decode and save the actual final x_0 from the saved trajectory.
    traj = _trajectory_payload(run_dir)["trajectories"].float()    # (T+1, B, C, H, W)
    final_latent = traj[-1].to(ctx.device, dtype=torch.float32)
    final_decoded = decode_latents(ctx.models, final_latent).cpu()
    final_path = out_dir / "final_x0.png"
    from poe_repair.methods._sampling import write_decoded_image as wdi
    wdi(final_decoded, final_path)

    return TrajectorySnapshots(
        label=label,
        lam=float(lam),
        run_dir=run_dir,
        step_indices=list(snapshot_steps),
        timesteps=timesteps,
        decoded_paths=decoded_paths,
        clip_scores=clip_scores,
        final_image_path=final_path,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def serialise_snapshots(
    snapshots_by_label: dict[str, TrajectorySnapshots],
    text_targets: Sequence[str],
) -> dict:
    """Build a JSON-serialisable record of the diagnostic run."""
    out: dict = {
        "text_targets": list(text_targets),
        "trajectories": {},
    }
    for label, snap in snapshots_by_label.items():
        out["trajectories"][label] = {
            "label": snap.label,
            "lam": snap.lam,
            "run_dir": str(snap.run_dir),
            "step_indices": snap.step_indices,
            "timesteps": snap.timesteps,
            "decoded_paths": [str(p) for p in snap.decoded_paths],
            "final_image_path": (
                None if snap.final_image_path is None
                else str(snap.final_image_path)
            ),
            "clip_scores": snap.clip_scores,
        }
    return out


def write_clip_scores_json(
    snapshots_by_label: dict[str, TrajectorySnapshots],
    text_targets: Sequence[str],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        serialise_snapshots(snapshots_by_label, text_targets), indent=2,
    ))
    return out_path
