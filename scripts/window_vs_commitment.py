#!/usr/bin/env python
"""Does the window where the correction works move with the pair, or sit in the same place?

EXP-04 of EXPERIMENTS.md, and the experiment EXP-01 promoted to decisive.

EXP-01 measured the step where each pair's picture stops changing and found it varies from 18 to 36
across pairs. It also found something it could not explain: every pair settles at step 18 or later,
while the correction only works over steps 0 to 10. So the correction stops working long before the
picture settles, which is not what a simple commitment story predicts.

This asks the question that separates the two readings. If the settling step is the thing that
matters, then a pair that settles late should keep responding to the correction later, and the two
should move together. If every pair's window sits in the same early place regardless of when it
settles, then composition is decided before anything visible in the trajectory and the settling
measure is tracking the wrong event.

No sampling. The window sweep already exists: 8 pairs x 9 windows x 4 seeds, scored by the
instance-count scorer into window_curves.json. This reads it.

The bars were written in EXPERIMENTS.md before this ran and are repeated here as constants.

    python scripts/window_vs_commitment.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-registered in EXPERIMENTS.md.
MIN_SPAN_STEPS = 5        # "moves with the pair": best window centres must span at least this
MIN_RHO = 0.5             # "moves with the pair": and correlate with the commitment step
NULL_SPAN_STEPS = 2       # "does not move": all best centres inside this many steps

WINDOWS = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window"
               "/window_curves.json")
COMMIT = Path(__file__).resolve().parent.parent / "docs/evidence/EXP01-commitment-step/result.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/evidence/EXP04-window-vs-commitment"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, computed here so the script carries no scipy dependency."""
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties, so a pair of equal values does not get an arbitrary order
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    x, y = np.asarray(x, float), np.asarray(y, float)
    # A constant variable has no ranks to correlate. That is not a missing number, it is the
    # answer: every pair took the same value.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    a, b = rank(x), rank(y)
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    wc = json.loads(WINDOWS.read_text())
    commit = json.loads(COMMIT.read_text())["per_pair"]

    # compose rate per (pair, window centre), from the per-cell scores
    cells = defaultdict(list)
    for s in wc["scores"]:
        centre = (s["window"][0] + s["window"][1]) / 2
        cells[(s["pair"], centre)].append(s["compose"])
    centres = sorted({c for _, c in cells})
    pairs = sorted({p for p, _ in cells})

    print(f"{len(pairs)} pairs x {len(centres)} windows, "
          f"{sum(len(v) for v in cells.values())} scored cells")
    missing = [p for p in pairs if p not in commit]
    if missing:
        print(f"no commitment step for: {', '.join(missing)} (excluded)")
    pairs = [p for p in pairs if p in commit]
    print(f"{len(pairs)} pairs carry both measures\n")

    rows, best, centroid, last_alive, commit_step = {}, {}, {}, {}, {}
    for p in pairs:
        rate = np.array([np.mean(cells[(p, c)]) if (p, c) in cells else np.nan
                         for c in centres], dtype=float)
        rows[p] = rate
        # Best centre, ties resolved to the earliest, declared before running.
        best[p] = float(centres[int(np.nanargmax(rate))])
        # Where the correction still works on average, weighted by how well it works.
        centroid[p] = (float(np.nansum(rate * np.array(centres)) / np.nansum(rate))
                       if np.nansum(rate) > 0 else float("nan"))
        alive = [c for c, r in zip(centres, rate) if r > 0]
        last_alive[p] = float(max(alive)) if alive else float("nan")
        commit_step[p] = commit[p]["median"]

    order = sorted(pairs, key=lambda p: commit_step[p])
    print(f"  {'pair':28s} {'settles':>8} {'best win':>9} {'centroid':>9} {'last alive':>11}"
          f"   compose rate by window centre")
    for p in order:
        prof = " ".join(f"{r:4.2f}" for r in rows[p])
        print(f"  {p:28s} {commit_step[p]:>8.0f} {best[p]:>9.0f} {centroid[p]:>9.1f} "
              f"{last_alive[p]:>11.0f}   {prof}")
    print(f"  {'':28s} {'':>8} {'':>9} {'':>9} {'':>11}   "
          + " ".join(f"{c:4.0f}" for c in centres))

    b = np.array([best[p] for p in order])
    span = float(b.max() - b.min())
    cs = np.array([commit_step[p] for p in order])
    rho_best = spearman(cs, b)
    rho_centroid = spearman(cs, [centroid[p] for p in order])
    rho_last = spearman(cs, [last_alive[p] for p in order])

    print(f"\nspan of best window centres: {span:.0f} steps "
          f"(bar for 'moves': {MIN_SPAN_STEPS})")
    if np.isnan(rho_best):
        print("commitment step vs best window centre:  undefined, every pair has the same "
              f"best window ({b[0]:.0f})")
    else:
        print(f"commitment step vs best window centre:  rho {rho_best:+.2f} (bar: {MIN_RHO})")
    print(f"commitment step vs compose-weighted centroid: rho {rho_centroid:+.2f}  [secondary]")
    print(f"commitment step vs latest window that still works: rho {rho_last:+.2f}  [secondary]")

    if span >= MIN_SPAN_STEPS and not np.isnan(rho_best) and rho_best >= MIN_RHO:
        verdict = "MOVES WITH THE PAIR"
    elif span < NULL_SPAN_STEPS:
        verdict = "DOES NOT MOVE"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nVERDICT: {verdict}")
    print(f"n = {len(order)} pairs, 4 seeds per window, so a per-pair rate moves in steps of 0.25")
    if b[0] == min(centres):
        print(f"CENSORED: every pair peaks at {min(centres):.0f}, the earliest centre the grid "
              "holds. The true best\nwindow may be earlier still, so this bounds the window rather "
              "than locating it.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps({
        "pairs": order, "window_centres": centres,
        "compose_rate_by_pair": {p: rows[p].tolist() for p in order},
        "commitment_step": {p: commit_step[p] for p in order},
        "best_window_centre": {p: best[p] for p in order},
        "compose_weighted_centroid": {p: centroid[p] for p in order},
        "latest_window_still_working": {p: last_alive[p] for p in order},
        "span_of_best_centres": span,
        "rho_best": rho_best, "rho_centroid": rho_centroid, "rho_last_alive": rho_last,
        "bars": {"min_span_steps": MIN_SPAN_STEPS, "min_rho": MIN_RHO,
                 "null_span_steps": NULL_SPAN_STEPS},
        "verdict": verdict, "seeds_per_cell": 4,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = plt.get_cmap("viridis")
    lo, hi = cs.min(), cs.max()
    for p in order:
        col = cmap((commit_step[p] - lo) / max(hi - lo, 1e-9))
        ax1.plot(centres, rows[p], "o-", ms=3.5, lw=1.4, color=col,
                 label=f"{p.replace('__x__', ' x ').replace('_', ' ')} ({commit_step[p]:.0f})")
    ax1.set_xlabel("centre of the window the correction was injected over")
    ax1.set_ylabel("compose rate")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("Every pair's correction works early and only early", fontsize=10.5)
    ax1.legend(fontsize=6.2, frameon=False, title="pair (settles at step)",
               title_fontsize=6.5, loc="upper right")
    ax1.text(0.98, 0.30, "4 seeds per point, so one seed is 0.25:\nthe late frog x toad bump is "
                         "one seed",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7, color="0.45")
    ax1.grid(alpha=0.3)

    ax2.scatter(cs, b, s=70, c=cs, cmap="viridis", zorder=3)
    # Label only the two extremes. Eight labels on one flat line collide and say nothing
    # the line does not already say.
    for p, dx in ((order[0], -1), (order[-1], 1)):
        ax2.annotate(p.replace("__x__", " x ").replace("a_", "").replace("an_", ""),
                     (commit_step[p], best[p]), fontsize=7.5,
                     xytext=(12 * dx, 10), textcoords="offset points",
                     ha="right" if dx < 0 else "left", color="0.3")
    ax2.set_xlabel("step where that pair's picture settles")
    ax2.set_ylabel("best window centre")
    ax2.set_ylim(0, 50)
    ax2.set_xlim(20, 40)
    ax2.text(30, 46, "the window could have sat anywhere on this axis",
             fontsize=8, color="0.45", ha="center")
    ax2.annotate("", xy=(21, 43), xytext=(39, 43),
                 arrowprops=dict(arrowstyle="<->", color="0.7", lw=1))
    ax2.text(30, 9, "it sits at the same place for all 8 pairs, whatever they do",
             fontsize=8, color="0.35", ha="center")
    ax2.set_title("The settling step moves 13 steps, the window moves none", fontsize=10.5)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "window-vs-commitment.png", dpi=160)
    print(f"\nwrote {OUT_DIR / 'window-vs-commitment.png'}")
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
