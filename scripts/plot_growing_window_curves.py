#!/usr/bin/env python
"""How much of the trajectory, counted from one end, needs the correction on?

F4g/F4h's headline. F4a slides a fixed ten-step band across the run and asks
where it has to sit; these sweeps grow a window from one end instead, and ask
how much of the trajectory has to be inside it.

F4g corrects steps 0..c, plain PoE after: does extending the corrected start
past F4a's ten-step peak (compose rate 0.656 at window 0-10) raise the
compose rate any further, or does it already sit at the ceiling?

F4h is the converse: plain PoE for steps 0..c, correction from c to the end.
It asks whether the early steps are only where the correction happens to
work, or whether they are required, by testing whether a long correction at
the end can still rescue a run that started uncorrected.

Reads sampled images; it does not sample. Produce them with
scripts/mechanism_study/run_growing_window_sweep.sh.

Usage:
    python scripts/plot_growing_window_curves.py
    python scripts/plot_growing_window_curves.py --root outputs/interaction_term/window/pairs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term import window_grid as wg  # noqa: E402
from scripts.plot_dose_curves import require_validated_scorer  # noqa: E402

OUT_DIR = paths.resolve(paths.WINDOW)
DEFAULT_ROOTS = (paths.resolve(paths.WINDOW) / "pairs",)
SCORER_CONTRACT = paths.resolve(paths.COMPOSE_SCORER_VALIDATION) / "scorer_validated.json"
WIN_RE = re.compile(r"_w(\d+)-(\d+)")

# A window at step 20 of a 50-step run and a window at step 20 of a 20-step
# run are different moments in the trajectory, so they cannot share an axis.
MIN_STEPS = 40

INK = "#222222"
F4A_PEAK = 0.656  # window 0-10, from figures.md's F4a row. Plotted as a reference line.


def find_images(
    roots, *, min_steps: int = MIN_STEPS,
) -> tuple[dict[tuple[int, int], list[tuple[str, int, Path]]], list[dict]]:
    """Map (window start, end) -> [(pair, seed, image path)], plus what was skipped."""
    out: dict[tuple[int, int], list[tuple[str, int, Path]]] = defaultdict(list)
    skipped: list[dict] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_")[1])
                for run_dir in sorted(seed_dir.glob("teacher_residual_*_w*")):
                    m = WIN_RE.search(run_dir.name)
                    if not m:
                        continue
                    png = run_dir / f"{run_dir.name}.png"
                    if not png.exists():
                        continue
                    summary = run_dir / f"summary_{run_dir.name}.json"
                    if not summary.exists():
                        skipped.append({"path": str(run_dir), "reason": "no summary"})
                        continue
                    steps = int(json.loads(summary.read_text())
                                .get("num_inference_steps", 0))
                    if steps < min_steps:
                        skipped.append({"path": str(run_dir), "reason": "short run",
                                        "num_inference_steps": steps})
                        continue
                    out[(int(m.group(1)), int(m.group(2)))].append(
                        (pair_dir.name, seed, png)
                    )
    return out, skipped


def _curve(by_cutoff: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted(int(c) for c in by_cutoff)
    rates = np.asarray([by_cutoff[c]["rate"] for c in xs], dtype=float)
    ns = np.asarray([by_cutoff[c]["n"] for c in xs], dtype=float)
    return np.asarray(xs, dtype=float), rates, ns


def _style_axis(ax) -> None:
    ax.tick_params(labelsize=7, length=2.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family("serif")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#444444")
    ax.grid(alpha=0.3)


def _panel(ax, xs, rates, ns, *, title: str, xlabel: str) -> None:
    err = np.sqrt(np.maximum(rates * (1 - rates), 0) / np.maximum(ns, 1))
    ax.errorbar(xs, rates, yerr=err, fmt="o-", color="tab:green", lw=2, capsize=3,
                label="compose rate")
    ax.axhline(F4A_PEAK, color="#888888", ls="--", lw=1.0,
               label=f"F4a peak ({F4A_PEAK:.3f}, window 0-10)")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_xlabel(xlabel, fontsize=8, family="serif", color=INK)
    ax.set_ylabel("compose rate", fontsize=8, family="serif", color=INK)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=8.5, family="serif", color=INK, loc="left", pad=4)
    _style_axis(ax)


def _figure(by_direction: dict, out_dir: Path) -> None:
    """Diagnostic-only plot, not a paper figure.

    F4g and F4h are the picture grids (scripts/longer_correction_grid.py,
    scripts/later_start_grid.py), the same role F4a plays for the fixed-width
    sweep. This curve is this experiment's F4b-equivalent: it backs the
    grids' caption with a number, but it is not itself shipped to
    paper/iclr/figures, so it is written beside growing_window_curves.json
    instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    if by_direction["prefix"]:
        xs, rates, ns = _curve(by_direction["prefix"])
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        _panel(ax, xs, rates, ns,
               title="F4g backing curve: does more of the corrected start raise the ceiling?",
               xlabel="steps corrected from the start (0 to c, plain PoE after)")
        fig.tight_layout()
        out = out_dir / "growing_window_prefix_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"diagnostic figure: {out}")

    if by_direction["suffix"]:
        xs, rates, ns = _curve(by_direction["suffix"])
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        _panel(ax, xs, rates, ns,
               title="F4h backing curve: does correction after an uncorrected start still fix it?",
               xlabel="steps left uncorrected at the start (0 to c, correction after)")
        fig.tight_layout()
        out = out_dir / "growing_window_suffix_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"diagnostic figure: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS,
                    help=f"skip runs sampled at fewer steps (default {MIN_STEPS})")
    args = ap.parse_args()

    contract = require_validated_scorer(args.contract)
    print(f"scorer: {contract['method']} via {contract['detector']}\n")

    windows, skipped = find_images(args.roots or DEFAULT_ROOTS,
                                   min_steps=args.min_steps)
    if skipped:
        print(f"skipped {len(skipped)} run(s) that cannot share this axis:")
        for s in skipped:
            steps = s.get("num_inference_steps")
            detail = f"{steps} steps" if steps is not None else s["reason"]
            print(f"  {detail:<12} {s['path']}")
        print()
    if not windows:
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no windowed runs found. This script scores images, it does not "
            "sample. Produce them with:\n"
            "  bash scripts/mechanism_study/run_growing_window_sweep.sh\n"
            "Looked under:\n" + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    # window -> (direction, cutoff). Everything else found under the shared
    # output root (e.g. F4a's own fixed-width windows) is scored by that
    # figure's own script, not this one, so a window this grid does not
    # define is silently skipped here rather than mis-tagged.
    lookup: dict[tuple[int, int], tuple[str, int]] = {}
    for c, w in zip(wg.CUTOFFS, wg.prefix_windows()):
        lookup[w] = ("prefix", c)
    for c, w in zip(wg.CUTOFFS, wg.suffix_windows()):
        lookup[w] = ("suffix", c)

    by_direction: dict[str, dict[int, dict]] = {"prefix": {}, "suffix": {}}
    rows = []
    for window, items in sorted(windows.items()):
        tag = lookup.get(window)
        if tag is None:
            continue
        direction, cutoff = tag
        flags = []
        for pair, seed, png in items:
            n, _ = count_instances(png)
            flags.append(int(n >= 2))
            rows.append({"pair": pair, "seed": seed, "direction": direction,
                         "cutoff": cutoff, "window": list(window),
                         "n_instances": int(n), "compose": flags[-1]})
        rate = float(np.mean(flags))
        by_direction[direction][cutoff] = {
            "rate": rate, "n": len(flags), "window": list(window),
        }
        print(f"  {direction:<7} cutoff {cutoff:>3} "
              f"(window {window[0]:>2}-{window[1]:<3}): {rate:>4.0%}  (n={len(flags)})")

    for direction in ("prefix", "suffix"):
        missing = [c for c in wg.CUTOFFS if c not in by_direction[direction]]
        if missing:
            print(f"\n{direction}: {len(missing)} of {len(wg.CUTOFFS)} planned "
                  f"cutoffs have no images yet: {missing}")

    result = {
        "scorer": contract["method"],
        "cutoffs": list(wg.CUTOFFS),
        "f4a_peak": F4A_PEAK,
        "prefix": {str(k): v for k, v in by_direction["prefix"].items()},
        "suffix": {str(k): v for k, v in by_direction["suffix"].items()},
        "scores": rows,
        "skipped_runs": skipped,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "growing_window_curves.json").write_text(
        json.dumps(result, indent=2)
    )
    print(f"\nwrote {args.out_dir / 'growing_window_curves.json'}")

    if not args.no_figure:
        _figure(by_direction, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
