"""Δ_t loader on the training-cache layout.

Each cell lives at::

    outputs/training_cache/{split}/{pair_slug}/seed_{N}/
        meta.json
        residuals/step_{000..049}.pt    # {x_t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond}

We compute, per step::

    ε̃_PoE = gs · (eps_a_raw + eps_b_raw − 2·eps_uncond) + eps_uncond
    ε̃_J   = gs · (eps_j_raw − eps_uncond) + eps_uncond
    Δ_t   = ε̃_J − ε̃_PoE = gs · (eps_j_raw − eps_a_raw − eps_b_raw + eps_uncond)

The closed form means Δ_t is recoverable exactly from the cached raw eps
without re-running the UNet. Two Mono-free candidate bases used by D1-C are
also exposed::

    cand_j_minus_null = ε̃_J − eps_uncond  = gs · (eps_j_raw − eps_uncond)
    cand_a_minus_b    = ε̃_A − ε̃_B        = gs · (eps_a_raw − eps_b_raw)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "training_cache"


@dataclass(frozen=True)
class CellPath:
    split: str
    pair_slug: str
    seed: int
    root: Path

    @classmethod
    def from_root(
        cls,
        pair_slug: str,
        seed: int,
        *,
        split: str | None = None,
        cache_root: Path = DEFAULT_CACHE_ROOT,
    ) -> "CellPath":
        """Locate a cell. If ``split`` is None, search heldout then train."""
        candidates = [split] if split else ["heldout", "train"]
        for sp in candidates:
            root = cache_root / sp / pair_slug / f"seed_{seed}"
            if (root / "meta.json").exists():
                return cls(split=sp, pair_slug=pair_slug, seed=seed, root=root)
        searched = ", ".join(
            str(cache_root / sp / pair_slug / f"seed_{seed}")
            for sp in candidates
        )
        raise FileNotFoundError(
            f"no training-cache cell for pair={pair_slug} seed={seed}; "
            f"searched: {searched}"
        )

    @property
    def meta(self) -> dict:
        return json.loads((self.root / "meta.json").read_text())

    @property
    def residuals_dir(self) -> Path:
        return self.root / "residuals"

    def step_files(self) -> list[Path]:
        return sorted(self.residuals_dir.glob("step_*.pt"))

    def num_steps(self) -> int:
        return len(self.step_files())


def load_step_raw(step_path: Path) -> dict:
    """Load one step file. Cast eps tensors to float32 for math."""
    payload = torch.load(step_path, map_location="cpu", weights_only=False)
    out = {
        "step_index": int(payload["step_index"]),
        "timestep": int(payload["timestep"]),
    }
    for key in ("x_t", "eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond"):
        out[key] = payload[key].float()
    return out


def delta_t_from_raw(
    eps_a_raw: torch.Tensor,
    eps_b_raw: torch.Tensor,
    eps_j_raw: torch.Tensor,
    eps_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Δ_t = ε̃_J − ε̃_PoE, closed form."""
    return guidance_scale * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond)


def candidate_j_minus_null(
    eps_j_raw: torch.Tensor,
    eps_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """ε̃_J − ε̃_∅  (Mono-free; uses only joint + uncond)."""
    return guidance_scale * (eps_j_raw - eps_uncond)


def candidate_a_minus_b(
    eps_a_raw: torch.Tensor,
    eps_b_raw: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """ε̃_A − ε̃_B  (Mono-free; uses only single-prompt conditionals)."""
    return guidance_scale * (eps_a_raw - eps_b_raw)


def iter_cell_deltas(
    cell: CellPath,
    *,
    candidates: bool = False,
) -> Iterator[dict]:
    """Yield per-step tensors. Each dict has keys ``step_index``,
    ``timestep``, ``delta``; with ``candidates=True`` also ``cand_j_null`` and
    ``cand_a_minus_b``. Tensors are float32 on CPU, shape (1, C, H, W)."""
    gs = float(cell.meta["guidance_scale"])
    for f in cell.step_files():
        raw = load_step_raw(f)
        delta = delta_t_from_raw(
            raw["eps_a_raw"], raw["eps_b_raw"],
            raw["eps_j_raw"], raw["eps_uncond"],
            gs,
        )
        out = {
            "step_index": raw["step_index"],
            "timestep": raw["timestep"],
            "delta": delta,
        }
        if candidates:
            out["cand_j_null"] = candidate_j_minus_null(
                raw["eps_j_raw"], raw["eps_uncond"], gs,
            )
            out["cand_a_minus_b"] = candidate_a_minus_b(
                raw["eps_a_raw"], raw["eps_b_raw"], gs,
            )
        yield out


def stack_deltas(cell: CellPath) -> tuple[torch.Tensor, list[int], list[int]]:
    """Return ``(deltas, step_indices, timesteps)`` where ``deltas`` is
    (T, D_flat) float32 — one row per step, latent flattened.

    D for SDXL at 1024x1024 is 4·128·128 = 65536. Memory: 50 · 65536 · 4B ≈ 13 MB.
    """
    rows: list[torch.Tensor] = []
    step_indices: list[int] = []
    timesteps: list[int] = []
    for entry in iter_cell_deltas(cell):
        rows.append(entry["delta"].flatten())
        step_indices.append(entry["step_index"])
        timesteps.append(entry["timestep"])
    if not rows:
        raise RuntimeError(f"no step files found under {cell.residuals_dir}")
    return torch.stack(rows, dim=0), step_indices, timesteps


def delta_shape(cell: CellPath) -> tuple[int, int, int, int]:
    """Return (B, C, H, W) of Δ_t for spatial-heatmap rendering."""
    first = next(iter_cell_deltas(cell))
    return tuple(first["delta"].shape)  # type: ignore[return-value]
