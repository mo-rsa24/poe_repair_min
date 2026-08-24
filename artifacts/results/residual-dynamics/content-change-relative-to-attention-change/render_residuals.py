#!/usr/bin/env python
"""Show what the measure actually scores: the change, not the two maps.

Putting the adapter-off and adapter-on maps side by side asks the eye to see a
3 to 5% change, which it cannot. The measure does not look at the maps either.
It removes the best single rescale of off onto on, and scores what is left. So
this draws that leftover directly, for both maps, on a shared symmetric scale.

Three columns per step:
  raw difference, where it looks    on - off, dominated by the ~25% dimming
  what is left after rescaling      on - alpha*off, the term the claim is about
  the same, for what it paints      the one the claim says should be bigger

If the account is right, the third column carries more structure than the second.

Usage:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY artifacts/results/residual-dynamics/content-change-relative-to-attention-change/render_residuals.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from poe_repair import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPROBE = paths.resolve(paths.CONTENT_CHANGE_RELATIVE_TO_ATTENTION_CHANGE)
OUT = Path(__file__).resolve().parent / "follow-along"
DEFAULT_CELLS = ("a_cat__x__a_dog/seed_10", "a_frog__x__a_toad/seed_9")


def render(cell: str, out_dir: Path) -> list[Path]:
    cell_dir = REPROBE / cell
    manifest = json.loads((cell_dir / "value_probe_manifest.json").read_text())
    steps = manifest["steps"]
    blobs = {s: torch.load(cell_dir / f"step_{s:03d}_valuemaps.pt",
                           map_location="cpu", weights_only=False)
             for s in steps}
    tokens = sorted({k.split("_")[0] for k in blobs[steps[0]]
                     if k.endswith("_off_weight")})

    written = []
    for tok in tokens:
        fig, axes = plt.subplots(len(steps), 3, figsize=(7.6, 2.5 * len(steps)))
        axes = axes.reshape(len(steps), 3)
        for r, s in enumerate(steps):
            b = blobs[s]
            panels = []
            for kind in ("weight", "content"):
                off = b[f"{tok}_off_{kind}"].numpy()
                on = b[f"{tok}_on_{kind}"].numpy()
                alpha = b[f"{tok}_{kind}_change"]["alpha"]
                panels.append((on - off, on - alpha * off))
            # One scale across all three panels in the row, so the columns are
            # comparable to each other. Stated on the figure, not left implicit.
            raw_w, res_w = panels[0]
            _, res_c = panels[1]
            shown = [raw_w, res_w, res_c]
            lim = max(np.abs(x).max() for x in shown)
            titles = ["raw change, where it looks",
                      "after rescaling, where it looks",
                      "after rescaling, what it paints"]
            for c, (m, t) in enumerate(zip(shown, titles)):
                ax = axes[r, c]
                ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(t, fontsize=8.5)
                if c == 0:
                    ax.set_ylabel(f"step {s}", fontsize=9)
            wp = b[f"{tok}_weight_change"]["pattern"]
            cp = b[f"{tok}_content_change"]["pattern"]
            axes[r, 2].text(1.05, 0.5,
                            f"looks  {wp:.3f}\npaints {cp:.3f}\nratio  {cp / wp:.2f}\n"
                            f"scale ±{lim:.3f}",
                            transform=axes[r, 2].transAxes, fontsize=8.5,
                            va="center", family="monospace")

        pair, seed = cell.split("/")
        fig.suptitle(f"{pair.replace('_', ' ')}, {seed.replace('_', ' ')}, "
                     f"the word “{tok}”: what the adapter changed", fontsize=11)
        fig.tight_layout(rect=(0, 0, 0.85, 0.97))
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"residual__{pair}__{seed}__{tok}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=list(DEFAULT_CELLS))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for cell in args.cells:
        for p in render(cell, args.out):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
