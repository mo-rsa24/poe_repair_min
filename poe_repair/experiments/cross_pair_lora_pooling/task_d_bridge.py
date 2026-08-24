"""Δ̄_t bridge across (pair, seed) — cross-pair version of Task D.

Consumes ``samples/cells.jsonl`` (produced by ``sample_crossbar
--record-eps``) and computes three cosines per (cell, step):

    cos_own         = cos(ε_LoRA_t, own Δ_t)
                       — does the LoRA match this cell's stored residual?
    cos_pair_mean   = cos(ε_LoRA_t, Δ̄_t^(P))
                       — does it match the pair-mean residual
                         (averaged over the pair's train-pool seeds)?
                       Defined only for cells whose pair has the train
                       seeds cached (typically pair_pool.train pairs).
    cos_seed_mean   = cos(ε_LoRA_t, Δ̄_t^(s))
                       — does it match the seed-mean residual
                         (averaged over pair_pool.train at the same seed)?

Outputs:
    {pooled_run}/task_d_cosines.csv
    {pooled_run}/task_d_cosines.png    (one panel per quadrant)

Reuses ``_cos`` / ``_flat`` from the cross-seed implementation rather
than re-deriving them.

Usage::

    python -m poe_repair.experiments.cross_pair_lora_pooling.task_d_bridge \\
        --pooled-run artifacts/rung4-scale/cross_pair/all_groups/main \\
        --pair-pool artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml \\
        --seed-pool-path artifacts/_shared/cross_pair_pool_configs/seed_pool.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import (
    load_pair_pool,
)
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import (
    load_seed_pool,
)
from poe_repair.experiments.held_out_seeds.task_d_bridge import (
    _cos,
    _flat,
    _norm,
)
from poe_repair.training_cache import (
    DEFAULT_CACHE_ROOT,
    CellPath,
    iter_cell_deltas,
    mean_delta_per_step,
    resolve_cells,
)


log = logging.getLogger(__name__)
QUADRANTS = ("in_in", "in_out", "out_in", "out_out")


def _cache_delta_by_step(cell: CellPath) -> dict[int, torch.Tensor]:
    return {int(e["step_index"]): e["delta"] for e in iter_cell_deltas(cell)}


def _build_pair_mean_cache(
    pair_pool, seed_pool, cache_root: Path,
) -> dict[str, dict[int, torch.Tensor]]:
    """Δ̄_t^(P) for each train pair = mean Δ_t across train-pool seeds."""
    out: dict[str, dict[int, torch.Tensor]] = {}
    for pair in pair_pool.train:
        try:
            cells = resolve_cells(pair, list(seed_pool.train_pool),
                                  cache_root=cache_root)
        except FileNotFoundError as exc:
            log.warning("pair-mean unavailable for %s: %s", pair, exc)
            continue
        deltas, step_indices, _ = mean_delta_per_step(cells)
        out[pair] = {int(si): t for si, t in zip(step_indices, deltas)}
        log.info("pair-mean cached for %s (%d steps)", pair, len(out[pair]))
    return out


def _build_seed_mean_cache(
    pair_pool, all_seeds: list[int], cache_root: Path,
) -> dict[int, dict[int, torch.Tensor]]:
    """Δ̄_t^(s) for each seed = mean Δ_t across pair_pool.train at that seed."""
    out: dict[int, dict[int, torch.Tensor]] = {}
    for seed in all_seeds:
        present_cells: list[CellPath] = []
        for pair in pair_pool.train:
            try:
                present_cells.append(CellPath.from_root(
                    pair, int(seed), cache_root=cache_root,
                ))
            except FileNotFoundError:
                pass
        if len(present_cells) < 2:
            log.warning("seed-mean for seed=%d: only %d/%d train pairs cached "
                        "— skipping", seed, len(present_cells), len(pair_pool.train))
            continue
        deltas, step_indices, _ = mean_delta_per_step(present_cells)
        out[int(seed)] = {int(si): t for si, t in zip(step_indices, deltas)}
        log.info("seed-mean cached for seed=%d across %d pairs (%d steps)",
                 seed, len(present_cells), len(out[int(seed)]))
    return out


def _load_cells(samples_root: Path) -> list[dict]:
    f = samples_root / "cells.jsonl"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found — run sample_crossbar first"
        )
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _aggregate_per_quadrant(rows: list[dict]) -> dict[str, dict]:
    """rows: per (cell, step) cosines. Reduce to per-quadrant mean/std vs t."""
    out: dict[str, dict] = {}
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["quadrant"]].append(r)
    for q, rs in by_q.items():
        by_t: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"own": [], "pair": [], "seed": []}
        )
        for r in rs:
            t = int(r["timestep"])
            for k, kk in (("cos_own", "own"),
                          ("cos_pair_mean", "pair"),
                          ("cos_seed_mean", "seed")):
                v = r[k]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                by_t[t][kk].append(float(v))
        ts = sorted(by_t)
        summary = {"t": ts}
        for kk in ("own", "pair", "seed"):
            means = [
                float(np.mean(by_t[t][kk])) if by_t[t][kk] else float("nan")
                for t in ts
            ]
            stds = [
                float(np.std(by_t[t][kk])) if by_t[t][kk] else float("nan")
                for t in ts
            ]
            summary[f"mean_{kk}"] = means
            summary[f"std_{kk}"] = stds
        out[q] = summary
    return out


def _plot_panels(summary: dict[str, dict], out_path: Path) -> None:
    quadrants_present = [q for q in QUADRANTS if q in summary]
    n = max(1, len(quadrants_present))
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0), squeeze=False)
    for ax, q in zip(axes[0], quadrants_present):
        s = summary[q]
        t = s["t"]
        for kk, label, marker in (
            ("own", "vs own Δ_t", "o"),
            ("pair", "vs pair-mean Δ_t", "s"),
            ("seed", "vs seed-mean Δ_t", "^"),
        ):
            ax.plot(t, s[f"mean_{kk}"], marker=marker, ms=3, label=label)
            mean = np.array(s[f"mean_{kk}"], dtype=float)
            std = np.array(s[f"std_{kk}"], dtype=float)
            ax.fill_between(t, mean - std, mean + std, alpha=0.15)
        ax.set_title(f"quadrant = {q}")
        ax.set_xlabel("timestep t")
        ax.set_ylabel("cos(ε_LoRA, ·)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_PAIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(prog="task_d_bridge_cross_pair")
    ap.add_argument("--pooled-run", required=True)
    ap.add_argument("--pair-pool", required=True)
    ap.add_argument("--seed-pool-path", required=True)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--cells", default=None,
                    help="optional override: path to cells.jsonl "
                         "(default: <pooled-run>/samples/cells.jsonl)")
    args = ap.parse_args(argv)

    pooled_run = Path(args.pooled_run)
    pair_pool = load_pair_pool(args.pair_pool)
    seed_pool = load_seed_pool(args.seed_pool_path)
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT

    cells_path = Path(args.cells) if args.cells else (
        pooled_run / "samples" / "cells.jsonl"
    )
    cells = _load_cells(cells_path.parent)
    cells_with_eps = [c for c in cells if c.get("eps_record_path")]
    log.info("cells total=%d with-eps=%d", len(cells), len(cells_with_eps))
    if not cells_with_eps:
        log.error("no eps records found — re-run sample_crossbar with --record-eps")
        return 2

    pair_mean = _build_pair_mean_cache(pair_pool, seed_pool, cache_root)
    all_seeds = sorted(set(list(seed_pool.train_pool) + list(seed_pool.held_out)))
    seed_mean = _build_seed_mean_cache(pair_pool, all_seeds, cache_root)

    rows: list[dict] = []
    for c in cells_with_eps:
        pair = c["pair_slug"]
        seed = int(c["seed"])
        eps_records = torch.load(
            c["eps_record_path"], map_location="cpu", weights_only=False,
        )
        try:
            cell = CellPath.from_root(pair, seed, cache_root=cache_root)
        except FileNotFoundError:
            log.warning("no cache for own-Δ at %s seed=%d; skipping cell",
                        pair, seed)
            continue
        own_delta = _cache_delta_by_step(cell)
        pair_mean_d = pair_mean.get(pair, {})
        seed_mean_d = seed_mean.get(seed, {})
        for rec in eps_records:
            si = int(rec["step_index"])
            t = int(rec["timestep"])
            v_lora = rec["delta_hat"]
            cos_own = _cos(v_lora, own_delta[si]) if si in own_delta else float("nan")
            cos_pair = (_cos(v_lora, pair_mean_d[si])
                        if si in pair_mean_d else float("nan"))
            cos_seed = (_cos(v_lora, seed_mean_d[si])
                        if si in seed_mean_d else float("nan"))
            rows.append({
                "quadrant": c["quadrant"],
                "pair_slug": pair,
                "seed": seed,
                "step_index": si,
                "timestep": t,
                "cos_own": cos_own,
                "cos_pair_mean": cos_pair,
                "cos_seed_mean": cos_seed,
                "v_lora_norm": _norm(v_lora),
            })

    csv_path = pooled_run / "task_d_cosines.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quadrant", "pair_slug", "seed", "step_index", "timestep",
                    "cos_own", "cos_pair_mean", "cos_seed_mean", "v_lora_norm"])
        for r in rows:
            w.writerow([
                r["quadrant"], r["pair_slug"], r["seed"],
                r["step_index"], r["timestep"],
                f"{r['cos_own']:.6f}",
                f"{r['cos_pair_mean']:.6f}",
                f"{r['cos_seed_mean']:.6f}",
                f"{r['v_lora_norm']:.6f}",
            ])
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    summary = _aggregate_per_quadrant(rows)
    png_path = pooled_run / "task_d_cosines.png"
    _plot_panels(summary, png_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
