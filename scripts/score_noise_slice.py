#!/usr/bin/env python
"""Score the constructed slice of noise space and draw the map.

`scripts/noise_slice.py` builds a plane patch through three starting noises
and runs every point on it. This scores those runs and colours the grid, which
answers the question the slice was built for: do composing starts form
connected regions, or are they scattered?

Why a constructed plane and not an embedding: independent starting noises are
mutually orthogonal (measured cosine +0.003), so they are all equidistant and
any MDS or PCA of them returns a fixed shape carrying no information. Here the
two axes were chosen by us, so position on the map means something and every
cell is a real run.

Readings, from the sweep script's header, before anything was scored:
    connected patches  -> the outcome is a smooth function of where the run
                          starts, so an adapter could in principle anticipate
                          it from the state it is handed
    scattered          -> the outcome is chaotic in the starting noise; the
                          method still works per run, but no figure may claim
                          the adapter can predict which runs it will fix
    all one colour     -> the three corners sat on the same side of any
                          boundary; the map says nothing and needs rerunning
                          with corners of mixed outcome

    python scripts/score_noise_slice.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SLICE_DIR = Path("outputs/interaction_term/noise_slice")
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
OUT_DIR = Path("outputs/interaction_term/noise_slice")


def neighbour_agreement(M: np.ndarray) -> float:
    """Fraction of edge-adjacent cell pairs that share a verdict.

    The one number that separates patches from confetti. Under a random
    arrangement with the same number of composing cells it sits near
    p^2+(1-p)^2; the caller compares against a shuffled null rather than
    against that formula, so ties and edge effects are handled the same way in
    both.
    """
    same = tot = 0
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            for di, dj in ((0, 1), (1, 0)):
                a, b = i + di, j + dj
                if a < n and b < m:
                    tot += 1
                    same += int(M[i, j] == M[a, b])
    return same / tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=SLICE_DIR / "slice_manifest.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    contract = json.loads(SCORER_CONTRACT.read_text())
    if not contract.get("pass"):
        raise SystemExit("scorer is not marked validated: refusing")
    print(f"scorer: {contract['method']} via {contract['detector']}")
    print(f"slice: {man['pair']} corners {man['corners']} "
          f"grid {man['grid']}x{man['grid']} lambda {man['lambda']}\n")

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        count_instances,
    )
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    g = man["grid"]
    M = np.full((g, g), np.nan)
    counts = np.full((g, g), np.nan)
    for cell in man["cells"]:
        p = Path(cell["path"])
        if not p.exists():
            continue
        n, _ = count_instances(p, device=device)
        counts[cell["i"], cell["j"]] = n
        M[cell["i"], cell["j"]] = int(n >= 2)

    filled = int((~np.isnan(M)).sum())
    print(f"scored {filled} of {g*g} cells\n")
    if filled < g * g:
        print("  incomplete grid; the map is drawn with gaps", file=sys.stderr)

    print("composed (1) over the slice, row v increasing upward:")
    for i in range(g - 1, -1, -1):
        print("   " + " ".join("-" if np.isnan(x) else str(int(x))
                               for x in M[i]))

    rate = float(np.nanmean(M))
    print(f"\n  composing fraction {rate:.2f}")
    if rate in (0.0, 1.0):
        verdict = ("UNIFORM: every cell agrees, so the three corners sat on "
                   "the same side of any boundary. The map says nothing about "
                   "structure and needs rerunning with corners of mixed "
                   "outcome")
        agree = null_med = null_p95 = float("nan")
    else:
        agree = neighbour_agreement(M)
        rng = np.random.default_rng(0)
        flat = M[~np.isnan(M)]
        null = []
        for _ in range(2000):
            sh = M.copy()
            vals = rng.permutation(flat)
            sh[~np.isnan(sh)] = vals
            null.append(neighbour_agreement(sh))
        null_med = float(np.median(null))
        null_p95 = float(np.percentile(null, 95))
        print(f"  neighbouring cells that agree: {agree:.2f}")
        print(f"  shuffled null: median {null_med:.2f}, 95th {null_p95:.2f}")
        if agree > null_p95:
            verdict = ("PATCHES: neighbouring starts agree more than shuffling "
                       "them would, so the outcome is a smooth function of "
                       "where the run begins")
        else:
            verdict = ("SCATTERED: neighbouring starts agree no more than "
                       "chance, so the outcome is chaotic in the starting "
                       "noise. No figure may claim the fix is predictable "
                       "from the state it is handed")
    print(f"\n  {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="equal")
    for i in range(g):
        for j in range(g):
            if np.isnan(counts[i, j]):
                continue
            ax.text(j, i, str(int(counts[i, j])), ha="center", va="center",
                    fontsize=7,
                    color="white" if M[i, j] == 1 else "#555555")
    ticks = np.linspace(0, g - 1, min(g, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/(g-1):.2f}" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t/(g-1):.2f}" for t in ticks])
    ax.set_xlabel(f"toward seed {man['corners'][1]}")
    ax.set_ylabel(f"toward seed {man['corners'][2]}")
    ax.set_title(f"{man['pair'].replace('__x__',' x ')}, corner = seed "
                 f"{man['corners'][0]}", fontsize=8, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    out = OUT_DIR / "noise_slice_map.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    (OUT_DIR / "noise_slice_scores.json").write_text(json.dumps({
        "pair": man["pair"], "corners": man["corners"], "grid": g,
        "lambda": man["lambda"], "construction": man["construction"],
        "composed": [[None if np.isnan(x) else int(x) for x in row]
                     for row in M],
        "instance_counts": [[None if np.isnan(x) else int(x) for x in row]
                            for row in counts],
        "composing_fraction": rate,
        "neighbour_agreement": None if np.isnan(agree) else agree,
        "shuffled_null": {"median": None if np.isnan(null_med) else null_med,
                          "p95": None if np.isnan(null_p95) else null_p95},
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out} and noise_slice_scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
