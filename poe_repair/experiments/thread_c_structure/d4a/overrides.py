"""Build the four per-condition Δ override tensors from the training cache.

For each held-out seed s in the cache, the four substitution conditions are:

  oracle       — Δ_t^{(s)}                          (the seed's own cached Δ_t)
  shared_mean  — mean_{s' ≠ s} Δ_t^{(s')}           (leave-one-out average)
  shuffle      — Δ_t^{(s'')}  for one chosen s''    (a different seed's full Δ)
  zero         — zeros_like(Δ_t)                    (= plain PoE per step)

Each override is a tensor of shape ``(T, B, C, H, W)``; the runner feeds it
straight into ``run_delta_override``.

The shuffle pairing is asymmetric with N = 3: we pin it deterministically
(``shuffle[s] = next(s) in sorted seeds``) and record the pairing in the
output JSON, per the plan §7b D4-A caveat ("state the chosen pairing
explicitly in the caption").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from poe_repair.experiments.thread_c_structure.loader import (
    CellPath, iter_cell_deltas,
)


class Condition(str, Enum):
    ORACLE = "oracle"
    SHARED_MEAN = "shared_mean"
    SHUFFLE = "shuffle"
    ZERO = "zero"


@dataclass
class OverrideBuilder:
    seeds: list[int]
    cells_by_seed: dict[int, CellPath]
    # seed -> (T, B, C, H, W) cached Δ_t tensor.
    deltas_by_seed: dict[int, torch.Tensor]
    # seed -> the seed used for SHUFFLE substitution.
    shuffle_pairing: dict[int, int]
    step_indices: list[int]
    timesteps: list[int]

    def override_for(self, seed: int, cond: Condition) -> torch.Tensor:
        if cond is Condition.ORACLE:
            return self.deltas_by_seed[seed].clone()
        if cond is Condition.ZERO:
            return torch.zeros_like(self.deltas_by_seed[seed])
        if cond is Condition.SHUFFLE:
            return self.deltas_by_seed[self.shuffle_pairing[seed]].clone()
        if cond is Condition.SHARED_MEAN:
            others = [self.deltas_by_seed[s] for s in self.seeds if s != seed]
            if not others:
                raise ValueError(
                    "shared_mean requires ≥ 2 seeds to leave one out"
                )
            return torch.stack(others, dim=0).mean(dim=0)
        raise ValueError(f"unknown condition {cond!r}")


def _stack_deltas_per_step(cell: CellPath) -> tuple[
    torch.Tensor, list[int], list[int]
]:
    """Stack Δ_t along step axis, keeping the (B, C, H, W) shape per step."""
    rows: list[torch.Tensor] = []
    step_indices: list[int] = []
    timesteps: list[int] = []
    for entry in iter_cell_deltas(cell):
        rows.append(entry["delta"])
        step_indices.append(entry["step_index"])
        timesteps.append(entry["timestep"])
    if not rows:
        raise RuntimeError(f"no step files for {cell.root}")
    return torch.stack(rows, dim=0), step_indices, timesteps


def build_overrides_for_seed(cells: list[CellPath]) -> OverrideBuilder:
    """Load Δ_t per seed and prepare the four substitution conditions."""
    if len(cells) < 2:
        raise ValueError("D4-A needs ≥ 2 seeds (shared_mean and shuffle ill-defined otherwise)")
    seeds = sorted({c.seed for c in cells})
    cells_by_seed = {c.seed: c for c in cells}
    deltas: dict[int, torch.Tensor] = {}
    ref_step_indices: list[int] | None = None
    ref_timesteps: list[int] | None = None
    for seed in seeds:
        stack, step_indices, timesteps = _stack_deltas_per_step(cells_by_seed[seed])
        deltas[seed] = stack
        if ref_step_indices is None:
            ref_step_indices = step_indices
            ref_timesteps = timesteps
        elif step_indices != ref_step_indices:
            raise ValueError(
                f"step-index mismatch across seeds: seed {seed} has {step_indices}"
            )
    # Pin shuffle pairing deterministically: seed s ↔ next seed in sorted order.
    shuffle_pairing = {seeds[i]: seeds[(i + 1) % len(seeds)] for i in range(len(seeds))}
    assert ref_step_indices is not None and ref_timesteps is not None
    return OverrideBuilder(
        seeds=seeds,
        cells_by_seed=cells_by_seed,
        deltas_by_seed=deltas,
        shuffle_pairing=shuffle_pairing,
        step_indices=ref_step_indices,
        timesteps=ref_timesteps,
    )
