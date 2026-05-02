"""Pilot-data loading helpers (initial latents from saved trajectories + VAE decode).

Vendored and pruned from the upstream `scripts/sdxl_sdipc_utils.py`. Only the
functions reached by the four samplers are kept:

- ``decode_latents_to_tensor`` — VAE decode helper used by ``runtime.decode_latents``.
- ``load_shared_init_latents`` — reconstructs the same x_T from a saved
  ``trajectory_flat_poe.npy`` so reruns align with the reference PoE pilot.
- ``canonicalize_grid_asset_payload`` — finds + validates trajectory paths;
  pruned from the original to skip MDS projection (we don't render projections).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

BASE_CONDITIONS = ("prompt_a", "prompt_b", "monolithic", "poe")

CANONICAL_DECODED_PATHS = {
    "prompt_a": "solo_a.png",
    "prompt_b": "solo_b.png",
    "monolithic": "monolithic.png",
    "poe": "poe.png",
}
CANONICAL_FLAT_PATHS = {
    "prompt_a": "grid_assets/trajectory_flat_prompt_a.npy",
    "prompt_b": "grid_assets/trajectory_flat_prompt_b.npy",
    "monolithic": "grid_assets/trajectory_flat_monolithic.npy",
    "poe": "grid_assets/trajectory_flat_poe.npy",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_decoded_paths(pair_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    decoded_paths = dict(payload.get("decoded_image_paths", {}) or {})
    for cond in BASE_CONDITIONS:
        rel = Path(CANONICAL_DECODED_PATHS[cond])
        if (pair_dir / rel).exists():
            decoded_paths[cond] = str(rel)
    return decoded_paths


def _resolve_flat_paths(
    pair_dir: Path,
    payload: dict[str, Any],
    *,
    require_base_flats: bool,
) -> dict[str, str]:
    flat_paths = dict(payload.get("trajectory_flat_paths", {}) or {})
    resolved: dict[str, str] = {}

    for cond in BASE_CONDITIONS:
        canonical_rel = Path(CANONICAL_FLAT_PATHS[cond])
        if (pair_dir / canonical_rel).exists():
            resolved[cond] = str(canonical_rel)
            continue

        rel = flat_paths.get(cond)
        if rel and (pair_dir / rel).exists():
            resolved[cond] = rel
            continue

        if require_base_flats:
            raise FileNotFoundError(
                f"Missing base flat trajectory for '{cond}' in {pair_dir}"
            )

    return resolved


def canonicalize_grid_asset_payload(
    *,
    pair_dir: Path,
    require_base_flats: bool = True,
) -> dict[str, Any]:
    asset_path = pair_dir / "grid_assets.json"
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing grid asset: {asset_path}")

    payload = _load_json(asset_path)
    decoded_paths = _resolve_decoded_paths(pair_dir, payload)
    flat_paths = _resolve_flat_paths(pair_dir, payload, require_base_flats=require_base_flats)

    payload["decoded_image_paths"] = decoded_paths
    payload["trajectory_flat_paths"] = flat_paths
    return payload


def load_shared_init_latents(
    *,
    pair_dir: Path,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    reference_condition: str = "poe",
) -> torch.Tensor:
    """Recover the exact shared x_T used by the base PoE pilot run.

    The stored trajectory flats begin from the DDIM-space latent after dividing
    by Euler's init_noise_sigma. Multiply by that sigma to reconstruct the
    original Euler-scaled x_T.
    """
    payload = canonicalize_grid_asset_payload(pair_dir=pair_dir, require_base_flats=True)
    flat_paths = payload.get("trajectory_flat_paths") or {}
    rel = flat_paths.get(reference_condition)
    if not rel:
        raise KeyError(
            f"Missing trajectory flat path for reference condition "
            f"'{reference_condition}' in {pair_dir / 'grid_assets.json'}"
        )

    arr = np.load(pair_dir / rel).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] % 4 != 0:
        raise ValueError(f"Expected flat SDXL trajectory array for {pair_dir / rel}, got {arr.shape}")
    first = arr[0]
    spatial = first.shape[0] // 4
    side = int(round(spatial ** 0.5))
    if side * side * 4 != first.shape[0]:
        raise ValueError(f"Cannot reshape initial latent from {pair_dir / rel} with shape {arr.shape}")
    latents = first.reshape(1, 4, side, side) * float(euler_init_noise_sigma)
    return torch.from_numpy(latents).to(device=device, dtype=dtype)


@torch.no_grad()
def decode_latents_to_tensor(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    images = vae.decode(
        latents / vae.config.scaling_factor + shift_factor,
        return_dict=False,
    )[0]
    return ((images / 2 + 0.5).clamp(0, 1)).float()
