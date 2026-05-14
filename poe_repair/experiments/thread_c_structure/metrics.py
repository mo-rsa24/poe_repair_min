"""D-series numerics. All CPU, all pure functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from poe_repair.experiments.thread_c_structure.loader import (
    CellPath, iter_cell_deltas, stack_deltas,
)


_EPS = 1e-12


def _row_normalise(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = mat.norm(dim=1, keepdim=True).clamp_min(eps)
    return mat / norms


# ---------------------------------------------------------------------------
# D1-A — consecutive-step cosine of Δ_t
# ---------------------------------------------------------------------------


@dataclass
class ConsecutiveCosine:
    step_indices: list[int]      # length T-1 (cosines are between t and t+1)
    timesteps: list[int]         # length T-1, the timestep at t
    cosines: list[float]

    def mean_over(self, lo: int, hi: int) -> float:
        """Mean cosine over step_index ∈ [lo, hi] inclusive."""
        vals = [c for s, c in zip(self.step_indices, self.cosines) if lo <= s <= hi]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))


def consecutive_cosine(cell: CellPath) -> ConsecutiveCosine:
    deltas, step_indices, timesteps = stack_deltas(cell)
    norm = _row_normalise(deltas)
    cos = (norm[:-1] * norm[1:]).sum(dim=1)
    return ConsecutiveCosine(
        step_indices=step_indices[:-1],
        timesteps=timesteps[:-1],
        cosines=cos.tolist(),
    )


# ---------------------------------------------------------------------------
# D1-B — SVD energy of the stacked Δ_t matrix
# ---------------------------------------------------------------------------


@dataclass
class SvdEnergy:
    singular_values: list[float]        # length min(T, D); descending
    variance_share: list[float]         # cumulative-corrected per-component fraction
    cumulative_topk: dict[int, float]   # {k: top-k variance share} for k ∈ {1, 2, 3, 5, 10}


def svd_energy(cell: CellPath, *, top_ks: tuple[int, ...] = (1, 2, 3, 5, 10)) -> SvdEnergy:
    """Compute singular values of the (T × D) Δ_t matrix and the fraction of
    variance (sum of squared singular values) captured by the top-k components.

    Note: we do NOT mean-centre. The directional structure of Δ_t includes its
    bias direction — that's intentional. "Top-3 captures 80%" reads naturally
    against the un-centred energy.
    """
    deltas, _, _ = stack_deltas(cell)
    # Use SVD on (T, D) directly. T=50, D≈65k → svd_lowrank is cheap and exact.
    svals = torch.linalg.svdvals(deltas)  # length min(T, D)
    energy = (svals ** 2)
    total = energy.sum().clamp_min(1e-30)
    share = (energy / total).tolist()
    cumshare = torch.cumsum(energy / total, dim=0).tolist()
    cumulative_topk = {
        k: float(cumshare[k - 1]) if k - 1 < len(cumshare) else float(cumshare[-1])
        for k in top_ks
    }
    return SvdEnergy(
        singular_values=svals.tolist(),
        variance_share=share,
        cumulative_topk=cumulative_topk,
    )


# ---------------------------------------------------------------------------
# D1-C — alignment with Mono-free candidate bases
# ---------------------------------------------------------------------------


@dataclass
class BasisAlignment:
    step_indices: list[int]
    timesteps: list[int]
    cos_delta_vs_j_null: list[float]
    cos_delta_vs_a_minus_b: list[float]

    def max_window_cos(self, lo: int, hi: int) -> dict[str, float]:
        """Max-over-curves of mean-over-window cosine, restricted to step_index ∈ [lo, hi]."""
        def mean_in_window(curve: list[float]) -> float:
            vals = [c for s, c in zip(self.step_indices, curve) if lo <= s <= hi]
            return float(sum(vals) / len(vals)) if vals else float("nan")
        return {
            "j_minus_null": mean_in_window(self.cos_delta_vs_j_null),
            "a_minus_b": mean_in_window(self.cos_delta_vs_a_minus_b),
        }


def basis_alignment(cell: CellPath) -> BasisAlignment:
    step_indices: list[int] = []
    timesteps: list[int] = []
    cos_jn: list[float] = []
    cos_ab: list[float] = []
    for entry in iter_cell_deltas(cell, candidates=True):
        d = entry["delta"].flatten()
        n_jn = entry["cand_j_null"].flatten()
        n_ab = entry["cand_a_minus_b"].flatten()
        d_unit = d / d.norm().clamp_min(1e-12)
        jn_unit = n_jn / n_jn.norm().clamp_min(1e-12)
        ab_unit = n_ab / n_ab.norm().clamp_min(1e-12)
        step_indices.append(entry["step_index"])
        timesteps.append(entry["timestep"])
        cos_jn.append(float((d_unit * jn_unit).sum()))
        cos_ab.append(float((d_unit * ab_unit).sum()))
    return BasisAlignment(
        step_indices=step_indices,
        timesteps=timesteps,
        cos_delta_vs_j_null=cos_jn,
        cos_delta_vs_a_minus_b=cos_ab,
    )


# ---------------------------------------------------------------------------
# D2 — spatial L2-norm heatmap at chosen timesteps
# ---------------------------------------------------------------------------


@dataclass
class SpatialPanel:
    step_index: int
    timestep: int
    heatmap: torch.Tensor   # (H, W) float32 — L2 norm over the 4 latent channels


def spatial_heatmaps(
    cell: CellPath,
    *,
    step_indices: tuple[int, ...] = (5, 15, 25, 35),
) -> list[SpatialPanel]:
    """For each requested step index, return the per-pixel L2 norm of Δ_t
    across latent channels. Latent grid is typically (H=128, W=128) for SDXL.
    """
    wanted = set(int(s) for s in step_indices)
    out: list[SpatialPanel] = []
    for entry in iter_cell_deltas(cell):
        if entry["step_index"] not in wanted:
            continue
        delta = entry["delta"]
        if delta.ndim == 4:
            delta = delta[0]                 # (C, H, W)
        heat = delta.norm(dim=0)             # (H, W)
        out.append(SpatialPanel(
            step_index=entry["step_index"],
            timestep=entry["timestep"],
            heatmap=heat.float(),
        ))
    out.sort(key=lambda p: p.step_index)
    return out


# ---------------------------------------------------------------------------
# D3 — cross-seed cosine of Δ_t
# ---------------------------------------------------------------------------


@dataclass
class CrossSeedCosine:
    seeds: list[int]
    step_indices: list[int]
    timesteps: list[int]
    # pair_label -> list of cosines per step
    per_pair_cosines: dict[str, list[float]]
    # mean across pairs, per step
    mean_across_pairs: list[float]

    def mean_over_window(self, lo: int, hi: int) -> float:
        vals = [
            c for s, c in zip(self.step_indices, self.mean_across_pairs) if lo <= s <= hi
        ]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))


def cross_seed_cosine(cells: list[CellPath]) -> CrossSeedCosine:
    if len(cells) < 2:
        raise ValueError("D3 needs at least 2 seeds")
    stacked: dict[int, torch.Tensor] = {}    # seed -> (T, D_flat)
    step_idx_ref: list[int] | None = None
    ts_ref: list[int] | None = None
    for cell in cells:
        mat, step_indices, timesteps = stack_deltas(cell)
        if step_idx_ref is None:
            step_idx_ref = step_indices
            ts_ref = timesteps
        elif step_indices != step_idx_ref:
            raise ValueError(
                f"step-index mismatch across seeds: {cell.seed} has {step_indices}"
            )
        stacked[cell.seed] = _row_normalise(mat)

    seeds = sorted(stacked.keys())
    per_pair: dict[str, list[float]] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            sa, sb = seeds[i], seeds[j]
            cos = (stacked[sa] * stacked[sb]).sum(dim=1)
            per_pair[f"seed{sa}_vs_seed{sb}"] = cos.tolist()

    # Mean across pairs per step.
    arr = torch.tensor(list(per_pair.values()))    # (n_pairs, T)
    mean_across = arr.mean(dim=0).tolist() if arr.numel() else []
    assert step_idx_ref is not None and ts_ref is not None
    return CrossSeedCosine(
        seeds=seeds,
        step_indices=step_idx_ref,
        timesteps=ts_ref,
        per_pair_cosines=per_pair,
        mean_across_pairs=mean_across,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def consecutive_to_dict(c: ConsecutiveCosine) -> dict:
    return {
        "step_indices": c.step_indices,
        "timesteps": c.timesteps,
        "cosines": c.cosines,
    }


def svd_to_dict(s: SvdEnergy) -> dict:
    return {
        "singular_values": s.singular_values,
        "variance_share": s.variance_share,
        "cumulative_topk": {str(k): v for k, v in s.cumulative_topk.items()},
    }


def basis_to_dict(b: BasisAlignment) -> dict:
    return {
        "step_indices": b.step_indices,
        "timesteps": b.timesteps,
        "cos_delta_vs_j_minus_null": b.cos_delta_vs_j_null,
        "cos_delta_vs_a_minus_b": b.cos_delta_vs_a_minus_b,
    }


def cross_seed_to_dict(x: CrossSeedCosine) -> dict:
    return {
        "seeds": x.seeds,
        "step_indices": x.step_indices,
        "timesteps": x.timesteps,
        "per_pair_cosines": x.per_pair_cosines,
        "mean_across_pairs": x.mean_across_pairs,
    }


# ---------------------------------------------------------------------------
# Shared loaders for D4 / §7c — stack Δ_t once per cell
# ---------------------------------------------------------------------------


def _load_all_seeds(cells: list[CellPath]) -> tuple[
    dict[int, torch.Tensor], list[int], list[int]
]:
    """Stack Δ_t per seed; verify step-index alignment. Returns
    ``({seed: (T, D)}, step_indices, timesteps)``."""
    if len(cells) < 2:
        raise ValueError("D4 / PCA grid needs ≥ 2 seeds")
    stacked: dict[int, torch.Tensor] = {}
    step_idx_ref: list[int] | None = None
    ts_ref: list[int] | None = None
    for cell in cells:
        mat, step_indices, timesteps = stack_deltas(cell)
        if step_idx_ref is None:
            step_idx_ref = step_indices
            ts_ref = timesteps
        elif step_indices != step_idx_ref:
            raise ValueError(
                f"step-index mismatch across seeds: {cell.seed} has {step_indices}"
            )
        stacked[cell.seed] = mat
    assert step_idx_ref is not None and ts_ref is not None
    return stacked, step_idx_ref, ts_ref


# ---------------------------------------------------------------------------
# D4-B — per-seed norm + alignment with leave-one-out mean (with null band)
# ---------------------------------------------------------------------------


@dataclass
class DirectionMagnitudeSplit:
    seeds: list[int]
    step_indices: list[int]
    timesteps: list[int]
    # seed -> per-step ||Δ_t||
    norms_per_seed: dict[int, list[float]]
    # seed -> per-step cos(Δ_t^{(s)}, mean_{s' ≠ s} Δ_t^{(s')})
    cos_vs_loo_mean_per_seed: dict[int, list[float]]
    # Permutation-null band (median, p2.5, p97.5) per step.
    null_median: list[float]
    null_lo: list[float]
    null_hi: list[float]

    def mean_cos_in_window(self, lo: int, hi: int) -> dict[int, float]:
        """Per-seed mean of cos(seed, loo-mean) over step_index ∈ [lo, hi]."""
        out: dict[int, float] = {}
        for seed, curve in self.cos_vs_loo_mean_per_seed.items():
            vals = [
                c for s, c in zip(self.step_indices, curve) if lo <= s <= hi
            ]
            out[int(seed)] = (
                float(sum(vals) / len(vals)) if vals else float("nan")
            )
        return out


def direction_magnitude_split(
    cells: list[CellPath],
    *,
    n_permutations: int = 32,
    rng_seed: int = 0,
) -> DirectionMagnitudeSplit:
    """For each seed, compute ‖Δ_t‖ per step and cos(Δ_t, mean_{s' ≠ s} Δ_t^{(s')}).

    Null band: at each step, draw ``n_permutations`` cosine values between
    independently shuffled copies of ``Δ_t`` of two random seeds. This is a
    coordinate-shuffle null — controls for the high-dimensional ambient
    space (very high D means random vectors have near-zero cosine).
    """
    stacked, step_indices, timesteps = _load_all_seeds(cells)
    seeds = sorted(stacked.keys())

    norms_per_seed: dict[int, list[float]] = {
        s: stacked[s].norm(dim=1).tolist() for s in seeds
    }

    cos_vs_loo: dict[int, list[float]] = {}
    for s in seeds:
        others = [stacked[o] for o in seeds if o != s]
        loo_mean = torch.stack(others, dim=0).mean(dim=0)            # (T, D)
        loo_unit = loo_mean / loo_mean.norm(dim=1, keepdim=True).clamp_min(_EPS)
        own = stacked[s]
        own_unit = own / own.norm(dim=1, keepdim=True).clamp_min(_EPS)
        cos_vs_loo[int(s)] = (own_unit * loo_unit).sum(dim=1).tolist()

    g = torch.Generator().manual_seed(int(rng_seed))
    T, D = stacked[seeds[0]].shape
    null_samples: list[list[float]] = [[] for _ in range(T)]
    for _ in range(int(n_permutations)):
        # Pick two seeds at random with replacement, shuffle each row.
        for t in range(T):
            i_a = int(torch.randint(0, len(seeds), (1,), generator=g).item())
            i_b = int(torch.randint(0, len(seeds), (1,), generator=g).item())
            a = stacked[seeds[i_a]][t]
            b = stacked[seeds[i_b]][t]
            perm_a = a[torch.randperm(D, generator=g)]
            perm_b = b[torch.randperm(D, generator=g)]
            a_u = perm_a / perm_a.norm().clamp_min(_EPS)
            b_u = perm_b / perm_b.norm().clamp_min(_EPS)
            null_samples[t].append(float((a_u * b_u).sum()))

    null_median: list[float] = []
    null_lo: list[float] = []
    null_hi: list[float] = []
    for samp in null_samples:
        if not samp:
            null_median.append(0.0); null_lo.append(0.0); null_hi.append(0.0)
            continue
        ts = torch.tensor(samp, dtype=torch.float32).sort().values
        null_median.append(float(ts.median()))
        null_lo.append(float(ts[max(0, int(0.025 * len(ts)) - 1)]))
        null_hi.append(float(ts[min(len(ts) - 1, int(0.975 * len(ts)))]))

    return DirectionMagnitudeSplit(
        seeds=seeds,
        step_indices=step_indices,
        timesteps=timesteps,
        norms_per_seed=norms_per_seed,
        cos_vs_loo_mean_per_seed=cos_vs_loo,
        null_median=null_median,
        null_lo=null_lo,
        null_hi=null_hi,
    )


def direction_magnitude_to_dict(d: DirectionMagnitudeSplit) -> dict:
    return {
        "seeds": d.seeds,
        "step_indices": d.step_indices,
        "timesteps": d.timesteps,
        "norms_per_seed": {str(k): v for k, v in d.norms_per_seed.items()},
        "cos_vs_loo_mean_per_seed": {
            str(k): v for k, v in d.cos_vs_loo_mean_per_seed.items()
        },
        "null_median": d.null_median,
        "null_p025": d.null_lo,
        "null_p975": d.null_hi,
    }


# ---------------------------------------------------------------------------
# D4-C — cluster-ordered cosine grid (per timestep, N×N)
# ---------------------------------------------------------------------------


def _average_linkage_order(dist: torch.Tensor) -> list[int]:
    """Tiny average-linkage agglomerative clustering returning a leaf order.

    ``dist`` is an N×N symmetric matrix; entries on the diagonal are
    ignored. This is a self-contained implementation (no scipy dependency)
    because the diagnostic only needs an ordering, not a dendrogram.
    """
    n = int(dist.shape[0])
    clusters: list[list[int]] = [[i] for i in range(n)]
    # Working distance matrix between clusters.
    dmat = dist.clone()
    dmat.fill_diagonal_(float("inf"))
    while len(clusters) > 1:
        # Find the closest pair.
        flat = dmat.view(-1).argmin().item()
        i, j = divmod(int(flat), dmat.shape[0])
        if i > j:
            i, j = j, i
        merged = clusters[i] + clusters[j]
        # Update dmat: row i becomes average of i, j.
        new_row = (dmat[i] + dmat[j]) / 2.0
        new_row[i] = float("inf")
        new_row[j] = float("inf")
        dmat[i] = new_row
        dmat[:, i] = new_row
        # Remove column / row j.
        keep = [k for k in range(dmat.shape[0]) if k != j]
        dmat = dmat[keep][:, keep]
        clusters[i] = merged
        del clusters[j]
    return clusters[0]


@dataclass
class ClusterOrderedPanels:
    seeds_original: list[int]
    seeds_ordered: list[int]
    step_indices: list[int]
    timesteps: list[int]
    # Per timestep: N×N cosine matrix in seeds_ordered order.
    panels: list[torch.Tensor]


def cluster_ordered_cosine_panels(
    cells: list[CellPath],
    *,
    step_indices_to_render: tuple[int, ...] = (5, 15, 25, 35, 45),
    ordering_step_index: int | None = 15,
) -> ClusterOrderedPanels:
    """Compute per-timestep N×N pairwise cosine matrices for D4-C.

    The seed ordering is computed from the cosine matrix at
    ``ordering_step_index`` (default 15, mid-commit) via average-linkage
    on ``1 - cosine`` distance, then reused across all rendered panels so
    cluster membership is visually trackable.
    """
    stacked, step_indices, timesteps = _load_all_seeds(cells)
    seeds = sorted(stacked.keys())

    wanted = list(step_indices_to_render)
    panels_raw: dict[int, torch.Tensor] = {}
    for step_idx in wanted:
        if step_idx not in step_indices:
            continue
        t_pos = step_indices.index(step_idx)
        # Build N×N cosine matrix at this step.
        rows = torch.stack([stacked[s][t_pos] for s in seeds], dim=0)
        unit = rows / rows.norm(dim=1, keepdim=True).clamp_min(_EPS)
        cos_mat = unit @ unit.T
        panels_raw[step_idx] = cos_mat

    # Compute ordering once.
    if ordering_step_index is not None and ordering_step_index in panels_raw:
        order_mat = panels_raw[ordering_step_index]
    else:
        # Fallback: ordering from mean across rendered panels.
        order_mat = torch.stack(list(panels_raw.values()), dim=0).mean(dim=0)
    dist = 1.0 - order_mat
    order = _average_linkage_order(dist)
    seeds_ordered = [seeds[i] for i in order]

    # Reorder each panel.
    panels_ordered: list[torch.Tensor] = []
    rendered_step_indices: list[int] = []
    rendered_timesteps: list[int] = []
    for step_idx in wanted:
        if step_idx not in panels_raw:
            continue
        mat = panels_raw[step_idx]
        mat = mat[order][:, order]
        panels_ordered.append(mat)
        rendered_step_indices.append(step_idx)
        rendered_timesteps.append(timesteps[step_indices.index(step_idx)])

    return ClusterOrderedPanels(
        seeds_original=seeds,
        seeds_ordered=seeds_ordered,
        step_indices=rendered_step_indices,
        timesteps=rendered_timesteps,
        panels=panels_ordered,
    )


def cluster_panels_to_dict(c: ClusterOrderedPanels) -> dict:
    return {
        "seeds_original": c.seeds_original,
        "seeds_ordered": c.seeds_ordered,
        "step_indices": c.step_indices,
        "timesteps": c.timesteps,
        "panels": [p.tolist() for p in c.panels],
    }


# ---------------------------------------------------------------------------
# D4-D — PCA with guards (uncentred top-1 share + angle to mean + centred vs uncentred angle)
# ---------------------------------------------------------------------------


@dataclass
class PcaWithGuards:
    step_indices: list[int]
    timesteps: list[int]
    # Per step:
    uncentred_top1_share: list[float]
    null_top1_median: list[float]
    null_top1_lo: list[float]
    null_top1_hi: list[float]
    angle_pc1_vs_mean_deg: list[float]
    angle_centred_vs_uncentred_pc1_deg: list[float]


def _principal_direction(mat: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return (unit PC1 vector, top-1 squared singular value)."""
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    pc1 = Vh[0]
    pc1 = pc1 / pc1.norm().clamp_min(_EPS)
    return pc1, float(S[0]) ** 2


def pca_with_guards(
    cells: list[CellPath],
    *,
    n_permutations: int = 32,
    rng_seed: int = 0,
) -> PcaWithGuards:
    """Per-step PCA structure check (uncentred vs centred), with permutation null.

    At each step:
      mat (N, D) stacks the seeds' Δ_t.
      - uncentred_top1_share: σ_1^2 / Σ σ_i^2 on ``mat`` directly.
      - null band: shuffle each row's entries independently; recompute.
      - angle_pc1_vs_mean_deg: arccos(|cos(PC1, mean_row)|).
      - angle_centred_vs_uncentred_pc1_deg: arccos(|cos(PC1_centred, PC1_uncentred)|).
    """
    stacked, step_indices, timesteps = _load_all_seeds(cells)
    seeds = sorted(stacked.keys())
    T, D = stacked[seeds[0]].shape

    g = torch.Generator().manual_seed(int(rng_seed))
    uncentred_top1: list[float] = []
    null_med: list[float] = []
    null_lo: list[float] = []
    null_hi: list[float] = []
    angle_pc1_mean: list[float] = []
    angle_c_vs_u: list[float] = []
    for t in range(T):
        mat = torch.stack([stacked[s][t] for s in seeds], dim=0)        # (N, D)
        svals = torch.linalg.svdvals(mat)
        energy = svals ** 2
        total = energy.sum().clamp_min(_EPS)
        top1 = float(energy[0] / total)
        uncentred_top1.append(top1)

        # Null: shuffle each row's entries independently.
        samples: list[float] = []
        for _ in range(int(n_permutations)):
            perm_rows = []
            for r in range(mat.shape[0]):
                perm_rows.append(mat[r][torch.randperm(D, generator=g)])
            perm_mat = torch.stack(perm_rows, dim=0)
            ssv = torch.linalg.svdvals(perm_mat)
            e = ssv ** 2
            samples.append(float(e[0] / e.sum().clamp_min(_EPS)))
        ts_s = torch.tensor(samples).sort().values
        null_med.append(float(ts_s.median()))
        null_lo.append(float(ts_s[max(0, int(0.025 * len(ts_s)) - 1)]))
        null_hi.append(float(ts_s[min(len(ts_s) - 1, int(0.975 * len(ts_s)))]))

        # PC1 (uncentred) vs mean row direction.
        pc1_u, _ = _principal_direction(mat)
        mean_row = mat.mean(dim=0)
        mean_unit = mean_row / mean_row.norm().clamp_min(_EPS)
        cos_pm = float(torch.clamp((pc1_u * mean_unit).sum().abs(), -1.0, 1.0))
        angle_pc1_mean.append(float(torch.acos(torch.tensor(cos_pm)) * 180.0 / 3.141592653589793))

        # PC1 (centred) vs PC1 (uncentred).
        centred = mat - mean_row.unsqueeze(0)
        if centred.shape[0] >= 2:
            pc1_c, _ = _principal_direction(centred)
            cos_cu = float(torch.clamp((pc1_c * pc1_u).sum().abs(), -1.0, 1.0))
            angle_c_vs_u.append(float(torch.acos(torch.tensor(cos_cu)) * 180.0 / 3.141592653589793))
        else:
            angle_c_vs_u.append(float("nan"))

    return PcaWithGuards(
        step_indices=step_indices,
        timesteps=timesteps,
        uncentred_top1_share=uncentred_top1,
        null_top1_median=null_med,
        null_top1_lo=null_lo,
        null_top1_hi=null_hi,
        angle_pc1_vs_mean_deg=angle_pc1_mean,
        angle_centred_vs_uncentred_pc1_deg=angle_c_vs_u,
    )


def pca_guards_to_dict(p: PcaWithGuards) -> dict:
    return {
        "step_indices": p.step_indices,
        "timesteps": p.timesteps,
        "uncentred_top1_share": p.uncentred_top1_share,
        "null_top1_median": p.null_top1_median,
        "null_top1_p025": p.null_top1_lo,
        "null_top1_p975": p.null_top1_hi,
        "angle_pc1_vs_mean_deg": p.angle_pc1_vs_mean_deg,
        "angle_centred_vs_uncentred_pc1_deg": p.angle_centred_vs_uncentred_pc1_deg,
    }


# ---------------------------------------------------------------------------
# §7c — PCA grid: per-step projection of seeds onto top-2 PCs
# ---------------------------------------------------------------------------


@dataclass
class PcaGrid:
    seeds: list[int]
    step_indices: list[int]
    timesteps: list[int]
    # step_index -> (seeds × 2) coordinates on PC1 / PC2 at that step.
    coords_per_step: dict[int, torch.Tensor]
    # step_index -> (top1_variance_share, top2_variance_share).
    variance_share_per_step: dict[int, tuple[float, float]]


def pca_projection_grid(
    cells: list[CellPath],
    *,
    step_indices_to_render: tuple[int, ...] = (49, 39, 29, 19, 9, 1),
) -> PcaGrid:
    """For each requested step, compute the top-2 principal components of the
    across-seed Δ_t matrix and project each seed onto them.

    The PCA is computed *uncentred* — D1-B's argument applies here too: the
    common-mean direction is part of the structure we want a shared model to
    pick up, not a nuisance to subtract.
    """
    stacked, step_indices, timesteps = _load_all_seeds(cells)
    seeds = sorted(stacked.keys())
    coords: dict[int, torch.Tensor] = {}
    var_share: dict[int, tuple[float, float]] = {}
    rendered_steps: list[int] = []
    rendered_ts: list[int] = []
    for step_idx in step_indices_to_render:
        if step_idx not in step_indices:
            continue
        t_pos = step_indices.index(step_idx)
        mat = torch.stack([stacked[s][t_pos] for s in seeds], dim=0)    # (N, D)
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        # Projection: rows of U · S give (N, k); take first two cols.
        proj = (U * S.unsqueeze(0))[:, :2]
        if proj.shape[1] < 2:
            pad = torch.zeros(proj.shape[0], 2 - proj.shape[1])
            proj = torch.cat([proj, pad], dim=1)
        coords[step_idx] = proj
        energy = (S ** 2)
        total = energy.sum().clamp_min(_EPS)
        share1 = float(energy[0] / total) if len(energy) >= 1 else 0.0
        share2 = float(energy[1] / total) if len(energy) >= 2 else 0.0
        var_share[step_idx] = (share1, share2)
        rendered_steps.append(step_idx)
        rendered_ts.append(timesteps[t_pos])

    return PcaGrid(
        seeds=seeds,
        step_indices=rendered_steps,
        timesteps=rendered_ts,
        coords_per_step=coords,
        variance_share_per_step=var_share,
    )


def pca_grid_to_dict(p: PcaGrid) -> dict:
    return {
        "seeds": p.seeds,
        "step_indices": p.step_indices,
        "timesteps": p.timesteps,
        "coords_per_step": {
            str(k): v.tolist() for k, v in p.coords_per_step.items()
        },
        "variance_share_per_step": {
            str(k): list(v) for k, v in p.variance_share_per_step.items()
        },
    }
