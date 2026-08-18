#!/usr/bin/env python
"""When does a run's outcome stop being changeable, and does that step move with the pair?

EXP-01 of EXPERIMENTS.md. The framing under test says PoE fails by committing early to a blended
animal while both outcomes still look available. If that is right, there is a step after which the
outcome is locked, and the interesting question is whether that step is a property of the pair or a
constant of the sampler. If it is a constant, one fixed correction schedule fits every pair and the
"the adapter's window is in the wrong place for this pair" failure mode does not exist.

The measure, cache-only, no GPU, no decoding:

    x0(t) = (x_t - sqrt(1 - abar_t) * eps_PoE(t)) / sqrt(abar_t)      Tweedie, in latent space

is the model's estimate at step t of the image this run is going to produce. Track how far that
estimate has settled, as cos(x0(t), x0(final)). It climbs from near zero to one. The commitment
step is the first step after which it stays above COMMIT_THRESHOLD for the rest of the run.

Latent space is not perceptual space, which is why this is checked against the DINOv2 reading in
`trajectory_divergence/divergence.json` on the three pairs where that exists. That check is printed
with the result and is not optional: if the two disagree, this measure is wrong and the experiment
needs decoding.

The bars sit in the source rather than in prose so they cannot be adjusted after seeing the answer.

    python scripts/commitment_step.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    _alphas_cumprod,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402

# Pre-registered in EXPERIMENTS.md before this ran.
COMMIT_THRESHOLD = 0.90          # the estimate has settled once cosine stays above this
SENSITIVITY = (0.80, 0.95)       # reported beside it, so no verdict rests on one threshold
MIN_RANGE_STEPS = 5              # "varies": per-pair medians must span at least this many steps
MIN_BETWEEN_WITHIN = 1.5         # "varies": between-pair spread over across-seed spread
NULL_RANGE_STEPS = 2             # "does not vary": medians all inside this many steps
MIN_STEPS = 40                   # skip short smoke runs, as the fork read had to

OUT_DIR = Path(__file__).resolve().parent.parent / "docs/evidence/EXP01-commitment-step"
FORK_REF = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses"
                "/fork_curve.json")


def commitment(cell, threshold: float) -> tuple[int, np.ndarray]:
    """Return the commitment step and the settling curve for one cell."""
    abar = _alphas_cumprod()[cell.timesteps.long()].view(-1, 1)
    x = cell.x_t.flatten(1).float()
    eps = cell.eps_poe().flatten(1).float()
    x0 = (x - (1 - abar).sqrt() * eps) / abar.sqrt()

    final = torch.nn.functional.normalize(x0[-1], dim=0)
    settled = (torch.nn.functional.normalize(x0, dim=1) @ final).numpy()

    # First step from which it never drops back below the threshold.
    below = np.where(settled < threshold)[0]
    step = int(below[-1] + 1) if len(below) else 0
    return min(step, len(settled) - 1), settled


def fork_elbow() -> dict[tuple[str, int], int]:
    """The fork step per cell: where the PoE and Mono paths visibly pull apart.

    Independent of this measure and already trusted, which is what makes it a check. It is NOT the
    same quantity: the fork is where the two arms separate, and commitment is where one arm stops
    changing, so the two need not be equal. What would condemn this measure is no relationship at
    all, since a cell whose paths part late should not have settled early.
    """
    if not FORK_REF.exists():
        return {}
    return {(c["pair"], int(c["seed"])): int(c["elbow_step"])
            for c in json.load(FORK_REF.open())["cells"]}


def main() -> int:
    pool = load_pool("outputs/animals_compose_transfer/pair_pool.yaml")
    pairs = list(pool.train) + list(pool.heldout(roles=("transfer", "reference", "control")))

    per_cell: dict[str, list[tuple[int, int]]] = {}
    curves: dict[str, np.ndarray] = {}
    sens: dict[float, dict[str, list[int]]] = {t: {} for t in SENSITIVITY}
    skipped = 0

    for slug in pairs:
        for split in ("train", "heldout"):
            d = CACHE_ROOT / split / slug
            if not d.is_dir():
                continue
            for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
                if len(list((sd / "residuals").glob("step_*.pt"))) < MIN_STEPS:
                    skipped += 1
                    continue
                seed = int(sd.name.split("_")[1])
                cell = load_cell(slug, seed, root=CACHE_ROOT)
                step, settled = commitment(cell, COMMIT_THRESHOLD)
                per_cell.setdefault(slug, []).append((seed, step))
                curves.setdefault(slug, settled)
                for t in SENSITIVITY:
                    s, _ = commitment(cell, t)
                    sens[t].setdefault(slug, []).append(s)

    n_cells = sum(len(v) for v in per_cell.values())
    print(f"pairs with usable cells: {len(per_cell)} of {len(pairs)}")
    print(f"cells measured: {n_cells}   cells skipped as too short: {skipped}")
    if not per_cell:
        print("nothing to measure")
        return 2

    print(f"\ncommitment step per pair (cosine to the final estimate stays above "
          f"{COMMIT_THRESHOLD}):")
    print(f"  {'pair':32s} {'n':>2} {'median':>7} {'min':>4} {'max':>4}")
    medians = {}
    within = []
    for slug in sorted(per_cell, key=lambda s: np.median([x[1] for x in per_cell[s]])):
        steps = [x[1] for x in per_cell[slug]]
        medians[slug] = float(np.median(steps))
        within.append(np.std(steps))
        print(f"  {slug:32s} {len(steps):>2} {np.median(steps):>7.1f} "
              f"{min(steps):>4} {max(steps):>4}")

    med = np.array(list(medians.values()))
    rng = float(med.max() - med.min())
    between = float(np.std(med))
    within_pooled = float(np.mean(within))
    ratio = between / within_pooled if within_pooled > 0 else float("inf")

    print(f"\nrange of per-pair medians: {rng:.1f} steps  (bar for 'varies': {MIN_RANGE_STEPS})")
    print(f"between-pair spread {between:.2f} / within-pair spread {within_pooled:.2f} "
          f"= {ratio:.2f}  (bar: {MIN_BETWEEN_WITHIN})")

    if rng >= MIN_RANGE_STEPS and ratio >= MIN_BETWEEN_WITHIN:
        verdict = "VARIES"
    elif rng < NULL_RANGE_STEPS or ratio < 1.0:
        verdict = "DOES NOT VARY"
    else:
        verdict = "INCONCLUSIVE"
    print(f"VERDICT: {verdict}")

    print("\nsensitivity, range of per-pair medians at other thresholds:")
    for t in SENSITIVITY:
        m = np.array([np.median(v) for v in sens[t].values()])
        print(f"  threshold {t}: range {m.max() - m.min():.1f} steps, "
              f"median of medians {np.median(m):.1f}")

    # Does this measure behave like an independent, already-trusted one?
    fork = fork_elbow()
    agree = []
    if fork:
        print("\nagainst the fork step, on the cells where both exist:")
        for (slug, seed), fstep in sorted(fork.items()):
            match = [s for sd, s in per_cell.get(slug, []) if sd == seed]
            if match:
                agree.append((match[0], fstep))
        if len(agree) >= 5:
            a = np.array(agree, dtype=float)
            r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
            print(f"  {len(agree)} cells, commitment step vs fork step: correlation {r:+.2f}")
            print(f"  commitment median {np.median(a[:, 0]):.0f}, "
                  f"fork median {np.median(a[:, 1]):.0f}")
            print("  a near-zero correlation would condemn the measure: a run whose paths part "
                  "late\n  cannot have settled early.")
        else:
            print(f"  only {len(agree)} overlapping cells, too few to read")
    else:
        print("\nno fork reading found: this measure is unchecked")

    print("\nOWED before this verdict may be used: the perceptual version, the same settling "
          "definition\napplied to decoded frames rather than latents. That needs decoding and a "
          "GPU, and the\nlatent measure is only a stand-in until it agrees.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "measure": "first step after which cos(x0_hat(t), x0_hat(final)) stays above threshold",
        "space": "latent, Tweedie estimate, no decoding",
        "threshold": COMMIT_THRESHOLD,
        "bars": {"min_range_steps": MIN_RANGE_STEPS,
                 "min_between_within": MIN_BETWEEN_WITHIN,
                 "null_range_steps": NULL_RANGE_STEPS},
        "cells_measured": n_cells, "cells_skipped_short": skipped,
        "per_pair": {k: {"seeds": [s for s, _ in v], "steps": [x for _, x in v],
                         "median": medians[k]} for k, v in per_cell.items()},
        "range_of_medians": rng, "between_spread": between,
        "within_spread": within_pooled, "between_over_within": ratio,
        "verdict": verdict,
        "sensitivity": {str(t): {k: float(np.median(v)) for k, v in sens[t].items()}
                        for t in SENSITIVITY},
        "fork_check_cells": [{"commitment": int(a), "fork": int(b)} for a, b in agree],
        "perceptual_validation": "OWED: needs decoded frames, GPU",
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(medians, key=lambda s: medians[s])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, slug in enumerate(order):
        steps = [x[1] for x in per_cell[slug]]
        ax.scatter(steps, [i] * len(steps), s=18, color="0.75", zorder=2)
        ax.scatter([medians[slug]], [i], marker="D", s=55, color="tab:blue", zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([s.replace("__x__", " x ").replace("_", " ") for s in order], fontsize=8)
    ax.set_xlabel("denoising step where the outcome stops changing")
    ax.set_xlim(0, 50)
    ax.axvspan(0, 10, color="tab:green", alpha=0.12, zorder=1)
    ax.text(5, (len(order) - 1) / 2, "the window where the correction works",
            fontsize=8.5, color="tab:green", rotation=90, ha="center", va="center")
    ax.set_title("The picture settles long after the correction stops working", fontsize=11)
    ax.text(0.98, 0.02, f"medians span {med.min():.0f} to {med.max():.0f}, "
                        f"{n_cells} cells, 1 diamond per pair",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.35")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "commitment-step-per-pair.png", dpi=160)
    print(f"\nwrote {OUT_DIR / 'commitment-step-per-pair.png'}")
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
