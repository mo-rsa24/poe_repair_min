"""Runtime helpers for the minimal PoE-repair project.

Re-exports SDXL helpers from `poe_repair._sdxl.*` and provides a small
filesystem-based pair discovery walking ``data/pilot/seed_<n>/<slug>/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poe_repair._sdxl.runtime import (  # noqa: F401  (re-export)
    LatentTrajectoryCollector,
    PairContext,
    decode_latents,
    encode_prompt_sdxl,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    load_shared_latents,
)
from poe_repair._sdxl.metrics import (  # noqa: F401  (re-export)
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)


@dataclass
class PairSeedCell:
    """A single pair-seed cell located on disk."""

    pair_dir: Path
    pair_slug: str
    prompt_a: str
    prompt_b: str
    seed: int
    regime: str
    height: int
    width: int
    grid_assets: dict[str, Any]


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path | str, data: Any) -> Path:
    """Pretty-print JSON, handling numpy/torch/Path objects."""
    from dataclasses import asdict, is_dataclass

    p = Path(path)
    ensure_dir(p.parent)

    def _convert(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return _convert(asdict(obj))
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    p.write_text(json.dumps(_convert(data), indent=2, sort_keys=False))
    return p


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_grid_assets(pair_dir: Path) -> dict[str, Any] | None:
    grid_assets_path = pair_dir / "grid_assets.json"
    if not grid_assets_path.exists():
        return None
    return json.loads(grid_assets_path.read_text())


def _load_pairs_module() -> tuple[list[dict[str, str]], list[int]]:
    """Import pairs.py from the project root."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pairs  # type: ignore[import-not-found]

    return list(pairs.PAIRS), list(pairs.SEEDS)


def discover_pairs(
    pilot_dir: str | Path,
    *,
    pair_filter: Iterable[str] | None = None,
    seed_filter: Iterable[int] | None = None,
) -> list[PairSeedCell]:
    """Walk pilot_dir/seed_<n>/<slug>/ for each (pair, seed) in pairs.PAIRS × pairs.SEEDS."""
    root = Path(pilot_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Pilot dir does not exist: {root}")

    declared_pairs, declared_seeds = _load_pairs_module()
    pair_set = {p.lower() for p in pair_filter} if pair_filter else None
    seed_set = {int(s) for s in seed_filter} if seed_filter else None

    cells: list[PairSeedCell] = []
    for pair_spec in declared_pairs:
        slug = pair_spec["slug"]
        if pair_set is not None and slug.lower() not in pair_set:
            continue
        for seed in declared_seeds:
            if seed_set is not None and seed not in seed_set:
                continue
            pair_dir = root / f"seed_{seed}" / slug
            ga = _load_grid_assets(pair_dir)
            if ga is None:
                # Pilot cell missing grid_assets.json — skip silently.
                continue
            cfg = ga.get("config") or {}
            cells.append(
                PairSeedCell(
                    pair_dir=pair_dir,
                    pair_slug=slug,
                    prompt_a=pair_spec["prompt_a"],
                    prompt_b=pair_spec["prompt_b"],
                    seed=int(seed),
                    regime=pair_spec.get("regime", "unknown"),
                    height=_coerce_int(cfg.get("height"), 1024),
                    width=_coerce_int(cfg.get("width"), 1024),
                    grid_assets=ga,
                )
            )
    return cells
