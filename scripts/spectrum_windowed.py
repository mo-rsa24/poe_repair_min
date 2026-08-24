#!/usr/bin/env python
"""Is the correction low rank, split by timestep window, and do the
directions that carry the energy also carry the outcome.

Three questions, one script, run separately per timestep window because the
correction's shared component is known to differ sharply across the
trajectory (19.6% shared at steps 0-2, near zero after step 20, from the
delta_structure prong-E read). Pooling the whole trajectory into one SVD
would let whichever window dominates the vector norm decide the answer for
both, so this script fits a separate basis in each window instead.

Smallness (per window): stack the cached r_t vectors restricted to that
window's steps, take their singular values, and compare the cumulative
energy at k against the same measurement on a Gaussian of the same shape.

Sharedness (per window): fit the subspace on training pairs only, then
measure how much of the held-out pairs' energy it captures at the same k.

Outcome alignment (per window, held-out pairs only): project each held-out
(pair, seed)'s window-restricted correction onto each of the train-fitted
directions, ranked by singular value. For each direction, correlate the
fraction of that example's own window energy sitting along that direction
against whether that pair's real correction, injected at lambda=0.5,
composed (the oracle row in dose_curves.json). Lambda=0.5 is used rather
than the full-strength lambda=1.0 setting because at full strength the
oracle correction composes 30 of 32 times; a label that is 94% one class
carries almost no variance to correlate against. At lambda=0.5 the split is
24 no / 8 yes, the most informative point on the dose curve for this
question. A permutation
null (outcome labels shuffled, same correlation recomputed many times) marks
what correlation a direction carrying no real signal would show by chance.
Energy is normalized per example before correlating, specifically so a
direction cannot look predictive purely because compose-success examples
happen to have a larger correction norm overall.

Cache-only, no GPU.

Usage:
    python scripts/spectrum_windowed.py --pool outputs/animals_compose_transfer/pair_pool.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
DOSE_SCORES = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_curves.json")
KS = (1, 2, 4, 8, 16, 32, 64)
R_DIRECTIONS = 64
WINDOWS = {"early": (0, 20), "late": (20, 50)}
DOSE_SEEDS = (9, 10, 11, 12)


def collect_window(
    root: Path, wanted: list[str], max_pairs: int | None, max_seeds: int,
    stride: int, window: tuple[int, int], *, seeds: tuple[int, ...] | None = None,
    min_steps: int = 2,
) -> tuple[torch.Tensor, list[str], list[tuple[str, int]]]:
    """Stack r_t rows restricted to ``window`` steps. [N, D], pairs used, and
    a (pair, seed) tag per row so per-example aggregation is possible later.

    A cell whose full trajectory is shorter than the window's end is skipped
    rather than truncated, since a partial window would silently understate
    that example's energy relative to a full one.
    """
    a, b = window
    rows: list[torch.Tensor] = []
    used: list[str] = []
    tags: list[tuple[str, int]] = []
    for slug in wanted:
        if max_pairs and len(used) >= max_pairs:
            break
        got = 0
        for split in ("train", "heldout"):
            d = root / split / slug
            if not d.is_dir():
                continue
            for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
                if got >= max_seeds:
                    break
                seed = int(sd.name.split("_")[1])
                if seeds is not None and seed not in seeds:
                    continue
                if len(list((sd / "residuals").glob("step_*.pt"))) < min_steps:
                    continue
                c = load_cell(slug, seed, root=root)
                if c.n_steps < b:
                    continue
                flat = c.r_t()[a:b:stride].flatten(1)
                rows.append(flat)
                tags.extend([(slug, seed)] * flat.shape[0])
                got += 1
        if got:
            used.append(slug)
    if not rows:
        return torch.empty(0), [], []
    return torch.cat(rows, dim=0), used, tags


def energy_at_k(s: np.ndarray) -> dict[int, float]:
    e = s ** 2
    total = float(e.sum())
    return {k: float(e[:k].sum() / total) for k in KS if k <= len(e)}


def load_compose_labels(path: Path, lam: float) -> dict[tuple[str, int], int]:
    """(pair, seed) -> compose (0/1), oracle row at the given injection strength."""
    data = json.loads(path.read_text())
    out: dict[tuple[str, int], int] = {}
    for rec in data["scores"]:
        if rec["row"] == "oracle" and rec["lambda"] == lam:
            out[(rec["pair"], rec["seed"])] = int(rec["compose"])
    return out


def per_direction_correlation(
    held_flat: torch.Tensor, held_tags: list[tuple[str, int]],
    mean: torch.Tensor, Vh: torch.Tensor, labels: dict[tuple[str, int], int],
    *, n_perm: int, rng_seed: int,
) -> dict | None:
    """Per direction, correlate that example's own-energy fraction along it
    with the compose outcome, plus a permutation-null band.

    Returns None if fewer than 4 labeled examples or the labels have no
    variance (correlation is undefined and would print as a fluke).
    """
    hc = (held_flat - mean).numpy()
    r = min(R_DIRECTIONS, Vh.shape[0])
    basis = Vh[:r].numpy()
    proj_energy = (hc @ basis.T) ** 2          # [n_rows, r]
    row_total_energy = (hc ** 2).sum(axis=1)   # [n_rows]

    examples = sorted(set(held_tags))
    ex_index = {t: i for i, t in enumerate(examples)}
    agg_energy = np.zeros((len(examples), r))
    agg_total = np.zeros(len(examples))
    for row_i, tag in enumerate(held_tags):
        j = ex_index[tag]
        agg_energy[j] += proj_energy[row_i]
        agg_total[j] += row_total_energy[row_i]

    frac = agg_energy / np.clip(agg_total, 1e-12, None)[:, None]  # [n_ex, r]

    y_list, keep_idx = [], []
    for i, t in enumerate(examples):
        if t in labels:
            y_list.append(labels[t])
            keep_idx.append(i)
    if len(y_list) < 4 or len(set(y_list)) < 2:
        return None
    frac = frac[keep_idx]
    y = np.asarray(y_list, dtype=float)
    n = len(y)

    corr = np.array([stats.spearmanr(frac[:, k], y).statistic for k in range(r)])

    rng = np.random.default_rng(rng_seed)
    perm = np.empty((n_perm, r))
    for p in range(n_perm):
        yp = rng.permutation(y)
        perm[p] = [stats.spearmanr(frac[:, k], yp).statistic for k in range(r)]
    lo = np.nanpercentile(perm, 2.5, axis=0)
    hi = np.nanpercentile(perm, 97.5, axis=0)

    return {
        "n_examples": int(n),
        "n_positive": int(y.sum()),
        "correlation": np.nan_to_num(corr).tolist(),
        "null_p2_5": lo.tolist(),
        "null_p97_5": hi.tolist(),
    }


def run_window(
    root: Path, pool, name: str, window: tuple[int, int], *,
    max_pairs: int, max_train_seeds: int, stride: int,
    labels: dict[tuple[str, int], int], n_perm: int,
) -> dict:
    train, train_pairs, _ = collect_window(
        root, pool.train, max_pairs, max_train_seeds, stride, window,
    )
    if train.numel() == 0:
        print(f"[{name}] no training cells matched (need >= {window[1]} full steps)")
        return {"window": list(window), "error": "no_train_cells"}
    print(f"[{name}] train: {train.shape[0]} vectors x {train.shape[1]} dims "
          f"from {len(train_pairs)} pairs")

    mean = train.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(train - mean, full_matrices=False)
    s = S.numpy()
    ek = energy_at_k(s)

    g = torch.randn(train.shape, generator=torch.Generator().manual_seed(0))
    sg = torch.linalg.svdvals(g - g.mean(dim=0, keepdim=True)).numpy()
    ekg = energy_at_k(sg)

    print(f"  {'k':>4}  {'r_t':>8}  {'gaussian':>9}")
    for k in sorted(ek):
        print(f"  {k:>4}  {ek[k]:>7.1%}  {ekg[k]:>8.1%}")
    if train.shape[0] < 4 * max(k for k in ek):
        print(f"  WARNING: only {train.shape[0]} vectors for k up to "
              f"{max(ek)}; energy-at-k numbers are partly forced by N alone.")

    held_wanted = pool.heldout(roles=("transfer", "reference", "control"))
    overlap = sorted(set(held_wanted) & set(train_pairs))
    if overlap:
        held_wanted = [p for p in held_wanted if p not in train_pairs]

    held, held_pairs, held_tags = collect_window(
        root, held_wanted, None, len(DOSE_SEEDS), stride, window, seeds=DOSE_SEEDS,
    )

    result: dict = {
        "window": list(window),
        "train_pairs": train_pairs,
        "train_vectors": int(train.shape[0]),
        "dims": int(train.shape[1]),
        "energy_at_k": {str(k): v for k, v in ek.items()},
        "gaussian_floor_at_k": {str(k): v for k, v in ekg.items()},
    }

    if held.numel() > 0:
        hc = held - mean
        total = float((hc ** 2).sum())
        proj = {}
        for k in KS:
            if k > Vh.shape[0]:
                break
            basis = Vh[:k]
            captured = float(((hc @ basis.T) ** 2).sum())
            proj[k] = captured / max(total, 1e-12)
        print(f"  held-out energy captured by train subspace "
              f"({held.shape[0]} rows, {len(held_pairs)} pairs): "
              + ", ".join(f"k={k}:{v:.1%}" for k, v in proj.items()))
        result["heldout_pairs"] = held_pairs
        result["heldout_projection_at_k"] = {str(k): v for k, v in proj.items()}

        corr = per_direction_correlation(
            held, held_tags, mean, Vh, labels, n_perm=n_perm, rng_seed=0,
        )
        if corr is not None:
            n_sig = sum(
                1 for c, lo, hi in zip(corr["correlation"], corr["null_p2_5"], corr["null_p97_5"])
                if c < lo or c > hi
            )
            print(f"  outcome correlation: {corr['n_examples']} labeled examples "
                  f"({corr['n_positive']} compose), {n_sig}/{len(corr['correlation'])} "
                  f"directions clear the permutation-null band")
            result["outcome_correlation"] = corr
        else:
            print("  outcome correlation: skipped, too few labeled held-out examples")
    else:
        print("  no held-out rows available for this window")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="outputs/animals_compose_transfer/pair_pool.yaml")
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--max-train-seeds", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2, help="step subsampling within a window")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--dose-scores", type=Path, default=DOSE_SCORES)
    ap.add_argument("--label-lambda", type=float, default=0.5,
                    help="injection strength whose oracle compose label is correlated "
                         "against (0.5 is the most balanced point on the dose curve)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    pool = load_pool(args.pool)
    print(pool.summary())
    labels = load_compose_labels(args.dose_scores, args.label_lambda)
    n_pos = sum(labels.values())
    print(f"compose labels loaded: {len(labels)} (pair, seed) pairs at oracle, "
          f"lambda={args.label_lambda} ({n_pos} compose, {len(labels) - n_pos} no)")

    results = {}
    for name, window in WINDOWS.items():
        print(f"\n=== window {name} = steps {window[0]}:{window[1]} ===")
        results[name] = run_window(
            args.cache_root, pool, name, window,
            max_pairs=args.max_pairs, max_train_seeds=args.max_train_seeds,
            stride=args.stride, labels=labels, n_perm=args.n_perm,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "spectrum_windowed.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    if not args.no_figure:
        render_figure(results, args.out_dir / "spectrum_windowed.png")

    return 0


def render_figure(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(WINDOWS.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 9))
    if len(names) == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(names):
        res = results[name]
        ax_e, ax_c = axes[0, col], axes[1, col]
        a, b = res["window"]

        if "error" in res:
            ax_e.text(0.5, 0.5, res["error"], ha="center", va="center")
            ax_c.axis("off")
            continue

        ek = {int(k): v for k, v in res["energy_at_k"].items()}
        ekg = {int(k): v for k, v in res["gaussian_floor_at_k"].items()}
        ks = sorted(ek)
        ax_e.plot(ks, [ek[k] for k in ks], "o-", color="tab:blue", label="r_t, training pairs")
        ax_e.plot(ks, [ekg[k] for k in ks], "s--", color="0.6", label="same-shape Gaussian")
        if "heldout_projection_at_k" in res:
            proj = {int(k): v for k, v in res["heldout_projection_at_k"].items()}
            pk = sorted(proj)
            ax_e.plot(pk, [proj[k] for k in pk], "^-", color="tab:orange",
                      label="held-out pairs, train subspace")
        ax_e.set_xscale("log", base=2)
        ax_e.set_xlabel("k (number of directions)")
        ax_e.set_ylabel("fraction of energy captured")
        ax_e.set_ylim(0, 1)
        ax_e.set_title(f"steps {a} to {b}, energy at k")
        ax_e.legend(frameon=False, fontsize=8, loc="lower right")
        ax_e.grid(alpha=0.3)

        oc = res.get("outcome_correlation")
        if oc is None:
            ax_c.text(0.5, 0.5, "too few labeled\nheld-out examples", ha="center", va="center")
            ax_c.axis("off")
            continue
        rank = np.arange(1, len(oc["correlation"]) + 1)
        corr = np.array(oc["correlation"])
        lo = np.array(oc["null_p2_5"])
        hi = np.array(oc["null_p97_5"])
        ax_c.fill_between(rank, lo, hi, color="0.7", alpha=0.5, label="permutation null (95%)")
        ax_c.plot(rank, corr, "-", color="tab:red", linewidth=1.5, label="observed")
        ax_c.axhline(0, color="k", linewidth=0.6)
        ax_c.set_xlabel("direction rank (1 = largest singular value)")
        ax_c.set_ylabel("Spearman correlation with compose outcome")
        ax_c.set_ylim(-1, 1)
        ax_c.set_title(f"steps {a} to {b}, direction vs outcome "
                        f"(n={oc['n_examples']}, {oc['n_positive']} compose)")
        ax_c.legend(frameon=False, fontsize=8, loc="upper right")
        ax_c.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"figure: {out_path} (+ .pdf)")


if __name__ == "__main__":
    raise SystemExit(main())
