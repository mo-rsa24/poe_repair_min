#!/usr/bin/env python
"""Do a seed's correction curves predict whether the fix will work on it?

The reverse-engineering question. Some seeds compose under the correction and
others do not. Each seed's correction is 65536 numbers per step, and those are
useless for comparing seeds: two seeds' corrections are near-orthogonal by
construction, so every seed is equidistant from every other and any embedding
returns an information-free shape.

What is comparable is each seed's SUMMARY CURVES, which are short and whose
distances mean something:

    size      how large the correction is at each step, relative to the
              prediction it corrects (the F3 measure, imported not repeated)
    turn      how much each step's correction agrees in direction with the
              next one (the D1 measure)

That is 50 + 49 numbers per seed instead of 65536 per step. Two seeds whose
correction grows and turns alike really are alike in a way one can act on.

The label is whether that seed composed with the correction applied in the
earliest window, from the 12-seed sweep.

The honest guard. Twelve seeds against a hundred features will separate
perfectly by accident, so nothing here is fitted and scored on the same data:
the separability number is leave-one-out nearest neighbour, and it is read
against a null from shuffled labels rather than against 50%.

Cache-only, no GPU.

    python scripts/seed_curve_signature.py
    python scripts/seed_curve_signature.py --pair a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import load_cell  # noqa: E402
from scripts.snr_collapse import curve_for  # noqa: E402

LABELS = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/"
              "window_seeds/seed_breadth.json")
OUT_DIR = Path("outputs/interaction_term/seed_signature")


def features(pair: str, seed: int) -> dict[str, np.ndarray]:
    size = np.asarray(curve_for(pair, seed)[3], dtype=float)
    r = load_cell(pair, seed).r_t().float().flatten(1)
    turn = torch.nn.functional.cosine_similarity(r[:-1], r[1:], dim=1).numpy()
    return {"size": size, "turn": turn}


def loo_nearest_neighbour(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy. Nothing is fitted: the held-out point is
    classified by the nearest of the others, so a perfect fit on 12 points
    cannot inflate it."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ok = 0
    for i in range(len(y)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        ok += int(y[int(np.argmin(d))] == y[i])
    return ok / len(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    args = ap.parse_args()

    lab = json.loads(LABELS.read_text())
    if args.pair not in lab["composed"]:
        raise SystemExit(f"no labels for {args.pair} in {LABELS}")
    outcomes = {int(k): v for k, v in lab["composed"][args.pair].items()}

    seeds, feats, y = [], [], []
    for s in sorted(outcomes):
        try:
            feats.append(features(args.pair, s))
        except Exception as e:
            print(f"  seed {s}: no cached correction ({type(e).__name__}), "
                  f"skipped", file=sys.stderr)
            continue
        seeds.append(s)
        y.append(outcomes[s])
    y = np.array(y)
    if len(seeds) < 6 or y.std() == 0:
        raise SystemExit(f"only {len(seeds)} usable seeds, or all one class")
    print(f"{args.pair}: {len(seeds)} seeds with both a cached correction and "
          f"a label")
    print(f"  composed: {sorted(s for s, v in zip(seeds, y) if v)}")
    print(f"  failed:   {sorted(s for s, v in zip(seeds, y) if not v)}\n")

    blocks = {
        "size": np.stack([f["size"] for f in feats]),
        "turn": np.stack([f["turn"] for f in feats]),
    }
    blocks["both"] = np.hstack([blocks["size"], blocks["turn"]])

    rng = np.random.default_rng(0)
    results = {}
    print(f"{'features':>10} {'LOO 1-NN':>10} {'shuffled null':>16} {'verdict':>12}")
    for name, X in blocks.items():
        acc = loo_nearest_neighbour(X, y)
        null = np.array([loo_nearest_neighbour(X, rng.permutation(y))
                         for _ in range(400)])
        p = float((null >= acc).mean())
        verdict = "signal" if p < 0.05 else "none"
        results[name] = {"loo_accuracy": acc, "null_median": float(np.median(null)),
                         "null_p95": float(np.percentile(null, 95)),
                         "p_value": p, "verdict": verdict}
        print(f"{name:>10} {acc:>9.0%} {np.median(null):>10.0%} p95 {np.percentile(null,95):>3.0%}"
              f" {verdict:>12}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    for ax, key, ylab, title in (
        (axes[0], "size", "relative correction size", "a. how large"),
        (axes[1], "turn", "agreement with the next step", "b. how it turns"),
    ):
        for i, s in enumerate(seeds):
            c = "#1f77b4" if y[i] else "#d62728"
            ax.plot(blocks[key][i], color=c, lw=0.9, alpha=0.75)
        ax.set_xlabel("denoising step")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].text(0.02, 0.96, "composed", transform=axes[0].transAxes,
                 color="#1f77b4", fontsize=7, va="top")
    axes[0].text(0.02, 0.86, "did not", transform=axes[0].transAxes,
                 color="#d62728", fontsize=7, va="top")
    fig.tight_layout(w_pad=1.4)
    out = OUT_DIR / f"{args.pair}_curves_by_outcome.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    (OUT_DIR / f"{args.pair}_signature.json").write_text(json.dumps({
        "pair": args.pair, "seeds": seeds, "composed": y.tolist(),
        "label_source": str(LABELS),
        "features": {"size": "F3's measure via snr_collapse.curve_for, 50 steps",
                     "turn": "D1's measure, cosine between consecutive steps"},
        "separability": results,
        "guard": "leave-one-out 1-NN against a 400-permutation label null; "
                 "12 seeds against ~100 features would separate perfectly by "
                 "accident under any fitted classifier",
    }, indent=2))
    print(f"\nwrote {out}")
    print(f"      {OUT_DIR / f'{args.pair}_signature.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
