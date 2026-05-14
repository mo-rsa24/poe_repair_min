"""Cache-backed dataset for Method 2 distillation.

Reads ``outputs/training_cache/<split>/<pair>/<seed>/`` produced by
``scripts/build_training_cache.py``. Each sample is one (cell, step) pair
yielding everything the trainer needs:

  - ``x_t``: the noisy latent at this step (shape (4, H, W); leading
    batch dim squeezed off for stack-friendliness).
  - ``timestep``: scalar diffusion timestep at this step.
  - ``step_index``: index into [0, num_inference_steps).
  - ``eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond``: the four raw UNet
    outputs at this x_t (each shape (4, H, W)).
  - ``seq_a, pool_a, ...``: per-cell prompt embeddings (constant across
    steps within a cell; cached in RAM after first load).

The dataset does NOT compute the distillation target. The trainer does
that — typically::

    target = guidance_scale * (eps_j_raw + eps_uncond - eps_a_raw - eps_b_raw)

so that the same cache can serve different methods (Method 2a's residual
target, Method 2b's Δ_t = ε_J − ε_PoE, etc.).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Cell index
# ---------------------------------------------------------------------------


@dataclass
class CellInfo:
    cell_dir: Path
    pair_slug: str
    seed: int
    split: str
    num_steps: int


def _discover_cells(
    split_dir: Path,
    *,
    pair_filter: list[str] | None,
    seed_filter: list[int] | None,
) -> list[CellInfo]:
    pair_set = set(pair_filter) if pair_filter else None
    seed_set = set(seed_filter) if seed_filter else None
    cells: list[CellInfo] = []
    for pair_dir in sorted(split_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        if pair_set is not None and pair_dir.name not in pair_set:
            continue
        for seed_dir in sorted(pair_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if seed_set is not None and seed not in seed_set:
                continue
            meta_path = seed_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            num_steps = int(meta.get("num_inference_steps", 50))
            cells.append(CellInfo(
                cell_dir=seed_dir, pair_slug=pair_dir.name, seed=seed,
                split=split_dir.name, num_steps=num_steps,
            ))
    return cells


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


_STEP_KEYS = ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")
_EMB_KEYS = (
    "seq_a", "pool_a",
    "seq_b", "pool_b",
    "seq_j", "pool_j",
    "seq_uncond", "pool_uncond",
)


def _squeeze_leading(t: torch.Tensor) -> torch.Tensor:
    """Drop a leading singleton dim so default collate can stack."""
    if t.dim() >= 1 and t.size(0) == 1:
        return t.squeeze(0)
    return t


class TrainingCacheDataset(Dataset):
    """Lazy-loading dataset over (cell, step) pairs from the cache.

    Step files are loaded fresh from disk per ``__getitem__`` (small —
    a few MB per file in fp16). Embeddings are loaded once per cell and
    kept in RAM (a few MB per cell × ~30 cells = ~100 MB).

    Pass to :class:`torch.utils.data.DataLoader` for batching, or use
    :func:`iter_minibatches_from_cache` for a manual loop without
    DataLoader overhead.
    """

    def __init__(
        self,
        cache_root: Path | str,
        split: str = "train",
        *,
        pair_filter: list[str] | None = None,
        seed_filter: list[int] | None = None,
        step_filter: list[int] | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.out_dtype = out_dtype
        split_dir = self.cache_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"split dir not found: {split_dir}")

        self._cells = _discover_cells(
            split_dir, pair_filter=pair_filter, seed_filter=seed_filter,
        )
        if not self._cells:
            raise RuntimeError(
                f"No cells discovered under {split_dir} "
                f"(pair_filter={pair_filter} seed_filter={seed_filter})"
            )

        step_set = set(step_filter) if step_filter else None
        self._samples: list[tuple[int, int]] = []
        for ci, cell in enumerate(self._cells):
            for s in range(cell.num_steps):
                if step_set is not None and s not in step_set:
                    continue
                if (cell.cell_dir / "residuals" / f"step_{s:03d}.pt").exists():
                    self._samples.append((ci, s))

        if not self._samples:
            raise RuntimeError(
                f"No (cell, step) samples found under {split_dir}"
            )

        self._embeddings_cache: dict[int, dict[str, torch.Tensor]] = {}

    # ----- introspection ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def num_cells(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> list[CellInfo]:
        return list(self._cells)

    @property
    def num_inference_steps(self) -> int:
        steps = {c.num_steps for c in self._cells}
        if len(steps) != 1:
            raise RuntimeError(
                f"Cells have inconsistent num_inference_steps: {steps}"
            )
        return steps.pop()

    # ----- data access -----------------------------------------------------

    def _load_embeddings(self, cell_idx: int) -> dict[str, torch.Tensor]:
        cached = self._embeddings_cache.get(cell_idx)
        if cached is not None:
            return cached
        emb_path = self._cells[cell_idx].cell_dir / "embeddings.pt"
        emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        out: dict[str, torch.Tensor] = {}
        for k in _EMB_KEYS:
            if k not in emb:
                raise KeyError(
                    f"embeddings.pt at {emb_path} missing key {k!r}"
                )
            out[k] = _squeeze_leading(emb[k]).to(self.out_dtype)
        self._embeddings_cache[cell_idx] = out
        return out

    def __getitem__(self, idx: int) -> dict:
        cell_idx, step_idx = self._samples[idx]
        cell = self._cells[cell_idx]
        emb = self._load_embeddings(cell_idx)
        step_path = cell.cell_dir / "residuals" / f"step_{step_idx:03d}.pt"
        step = torch.load(step_path, map_location="cpu", weights_only=False)

        sample: dict = {
            "x_t": _squeeze_leading(step["x_t"]).to(self.out_dtype),
            "timestep": int(step["timestep"]),
            "step_index": int(step_idx),
            "num_inference_steps": cell.num_steps,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
        }
        for k in _STEP_KEYS:
            sample[k] = _squeeze_leading(step[k]).to(self.out_dtype)
        sample.update(emb)
        return sample


# ---------------------------------------------------------------------------
# Manual minibatch iterator (no DataLoader overhead)
# ---------------------------------------------------------------------------


def iter_minibatches_from_cache(
    dataset: TrainingCacheDataset,
    *,
    batch_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
    device: torch.device | None = None,
) -> Iterator[dict]:
    """Infinite iterator: yields stacked (B, ...) batches.

    Iterates over a shuffled permutation of ``range(len(dataset))``;
    when exhausted, reshuffles and starts over. With ``batch_size > 1``
    the tensor fields are stacked; scalar fields are returned as lists.
    """
    rng = random.Random(seed)
    n = len(dataset)
    if n == 0:
        raise RuntimeError("dataset is empty")

    while True:
        order = list(range(n))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n, batch_size):
            chunk = order[start: start + batch_size]
            samples = [dataset[i] for i in chunk]
            batch: dict = {}
            for k in samples[0].keys():
                vals = [s[k] for s in samples]
                if isinstance(vals[0], torch.Tensor):
                    stacked = torch.stack(vals, dim=0)
                    if device is not None:
                        stacked = stacked.to(device, non_blocking=True)
                    batch[k] = stacked
                else:
                    batch[k] = vals
            yield batch


# ---------------------------------------------------------------------------
# Targets — convenience helpers (not stored in the cache, computed on the fly)
# ---------------------------------------------------------------------------


def pmi_target(batch: dict, guidance_scale: float) -> torch.Tensor:
    """Method 2a target: w · (ε_J + ε_∅ − ε_A − ε_B).

    Equivalent in score space to the guided residual r_t = ε_J_guided -
    ε_PoE_guided (modulo the guidance algebra). Lives in eps space.
    """
    return float(guidance_scale) * (
        batch["eps_j_raw"] + batch["eps_uncond"]
        - batch["eps_a_raw"] - batch["eps_b_raw"]
    )


def step_weight(
    step_index: torch.Tensor | int,
    *,
    num_inference_steps: int = 50,
    early_window_end: int = 25,
    late_weight: float = 0.25,
) -> torch.Tensor:
    """Front-load loss weighting per the diagnostic finding that the
    correction lives in steps 0–25.

    ``early_window_end`` steps get weight 1.0; later steps get
    ``late_weight``. Returns a tensor of shape matching ``step_index``.
    """
    if isinstance(step_index, int):
        step_index = torch.tensor(step_index)
    early = step_index < int(early_window_end)
    return torch.where(early, torch.ones_like(step_index, dtype=torch.float32),
                       torch.full_like(step_index, float(late_weight),
                                       dtype=torch.float32))
