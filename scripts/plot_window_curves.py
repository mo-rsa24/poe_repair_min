#!/usr/bin/env python
"""When does injecting the correction actually help?

Plan 04's headline. Given a set of runs that inject r_t inside different step
windows, score each with the validated compose scorer and plot compose rate
against where the window sits. If the timing claim holds, the curve peaks in a
band rather than being flat: correcting early or late does less than correcting
in the middle.

Prints the peak band, meaning the contiguous run of windows whose compose rate
is within one standard error of the best.

Reads sampled images; it does not sample. Produce them with
scripts/interaction_term_window.py across a set of windows.

Usage:
    python scripts/plot_window_curves.py
    python scripts/plot_window_curves.py --root outputs/interaction_term/window/pairs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_dose_curves import require_validated_scorer  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window")
DEFAULT_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window/pairs"),
    Path("outputs/interaction_term/window/pairs"),
)
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
WIN_RE = re.compile(r"_w(\d+)-(\d+)")


def find_images(roots) -> dict[tuple[int, int], list[tuple[str, int, Path]]]:
    """Map (window start, end) -> [(pair, seed, image path)]."""
    out: dict[tuple[int, int], list[tuple[str, int, Path]]] = defaultdict(list)
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
                    if png.exists():
                        out[(int(m.group(1)), int(m.group(2)))].append(
                            (pair_dir.name, seed, png)
                        )
    return out


def peak_band(centres, rates, ns) -> tuple[float, float]:
    """Windows whose rate is within one standard error of the best."""
    rates = np.asarray(rates, dtype=float)
    best = rates.max()
    se = np.sqrt(np.maximum(rates * (1 - rates), 1e-12) / np.maximum(ns, 1))
    inside = rates >= best - se[int(np.argmax(rates))]
    c = np.asarray(centres, dtype=float)[inside]
    return float(c.min()), float(c.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    contract = require_validated_scorer(args.contract)
    print(f"scorer: {contract['method']} via {contract['detector']}\n")

    windows = find_images(args.roots or DEFAULT_ROOTS)
    if not windows:
        looked = args.roots or DEFAULT_ROOTS
        print(
            "no windowed runs found. This script scores images, it does not "
            "sample. Produce them with:\n"
            "  for W in 0,10 10,20 20,30 30,40 40,50; do \\\n"
            "    python scripts/interaction_term_window.py --pair <slug> "
            "--seed <n> --window $W; done\n"
            "Looked under:\n" + "\n".join(f"  {p}" for p in looked),
            file=sys.stderr,
        )
        return 2

    from poe_repair.experiments.compose_scorer.detection_scorer import count_instances

    centres, rates, ns, rows = [], [], [], []
    for (start, end), items in sorted(windows.items()):
        flags = []
        for pair, seed, png in items:
            n, _ = count_instances(png)
            flags.append(int(n >= 2))
            rows.append({"pair": pair, "seed": seed, "window": [start, end],
                         "n_instances": int(n), "compose": flags[-1]})
        rate = float(np.mean(flags))
        centres.append((start + end) / 2.0)
        rates.append(rate)
        ns.append(len(flags))
        print(f"  window {start:>3}-{end:<3} (centre {centres[-1]:5.1f}): "
              f"{rate:>4.0%}  (n={len(flags)})")

    result = {"scorer": contract["method"], "window_centres": centres,
              "compose_rate": rates, "n": ns, "scores": rows}

    if len(centres) > 1:
        lo, hi = peak_band(centres, rates, ns)
        best = centres[int(np.argmax(rates))]
        print(f"\npeak at window centre {best:.1f}, "
              f"band {lo:.1f} to {hi:.1f} (within one standard error)")
        result["peak_centre"] = best
        result["peak_band"] = [lo, hi]
    else:
        print("\nonly one window: no band to report")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "window_curves.json").write_text(json.dumps(result, indent=2))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        err = np.sqrt(np.maximum(np.array(rates) * (1 - np.array(rates)), 0)
                      / np.maximum(np.array(ns), 1))
        ax.errorbar(centres, rates, yerr=err, fmt="o-", color="tab:green",
                    lw=2, capsize=3)
        if len(centres) > 1:
            ax.axvspan(lo, hi, color="tab:green", alpha=0.15,
                       label="peak band (1 s.e.)")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("window centre (denoising step)")
        ax.set_ylabel("compose rate")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Compose rate vs when the correction is applied "
                     f"({len(centres)} windows)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "window_curves.png", dpi=150)
        print(f"figure: {args.out_dir / 'window_curves.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
