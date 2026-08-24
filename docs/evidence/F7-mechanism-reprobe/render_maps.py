#!/usr/bin/env python
"""Turn the captured probe tensors into pictures you can look at.

The sweep stores four 32x32 maps per word per step: where the word looks and
what it paints, each with the adapter off and on. Those are tensors, so nothing
in the re-probe output is viewable. This renders them.

One file per cell and word. Rows are the three captured steps. Columns are the
four maps, off beside on, so the change is read left to right. The numbers under
each row are the ones the verdict is computed from, not re-derived here.

Usage:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY docs/evidence/F7-mechanism-reprobe/render_maps.py
    $PY docs/evidence/F7-mechanism-reprobe/render_maps.py --cells a_frog__x__a_toad/seed_9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from poe_repair import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPROBE = paths.resolve(paths.CONTENT_CHANGE_RELATIVE_TO_ATTENTION_CHANGE)
OUT = Path(__file__).resolve().parent / "follow-along"

# The example cell, the strongest pair, and the weakest one, so the contrast in
# the per-pair figure has faces to go with it.
DEFAULT_CELLS = (
    "a_cat__x__a_dog/seed_10",
    "a_frog__x__a_toad/seed_9",
    "a_leopard__x__a_jaguar/seed_9",
)

COLS = [
    ("off_weight", "where it looks: adapter off"),
    ("on_weight", "where it looks: adapter on"),
    ("off_content", "what it paints: adapter off"),
    ("on_content", "what it paints: adapter on"),
]


def render(cell: str, out_dir: Path) -> list[Path]:
    cell_dir = REPROBE / cell
    manifest = json.loads((cell_dir / "value_probe_manifest.json").read_text())
    steps = manifest["steps"]
    blobs = {s: torch.load(cell_dir / f"step_{s:03d}_valuemaps.pt",
                           map_location="cpu", weights_only=False)
             for s in steps}

    # The token names are the keys the capture used, e.g. cat_off_weight.
    tokens = sorted({k.split("_")[0] for k in blobs[steps[0]]
                     if k.endswith("_off_weight")})

    written = []
    for tok in tokens:
        fig, axes = plt.subplots(len(steps), 4, figsize=(9.2, 2.5 * len(steps)))
        axes = axes.reshape(len(steps), 4)
        for r, s in enumerate(steps):
            b = blobs[s]
            wp = b[f"{tok}_weight_change"]["pattern"]
            cp = b[f"{tok}_content_change"]["pattern"]
            for c, (suffix, title) in enumerate(COLS):
                ax = axes[r, c]
                m = b[f"{tok}_{suffix}"].numpy()
                # Each map on its own scale: the adapter dims the attention by
                # about 25%, and a shared scale would show that brightness drop
                # instead of the spatial pattern the claim is about.
                ax.imshow(m, cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8.5)
                if c == 0:
                    ax.set_ylabel(f"step {s}", fontsize=9)
            axes[r, 3].text(
                1.06, 0.5,
                f"pattern moved\nlooks  {wp:.3f}\npaints {cp:.3f}\nratio  {cp / wp:.2f}",
                transform=axes[r, 3].transAxes, fontsize=8.5, va="center",
                family="monospace")

        pair, seed = cell.split("/")
        fig.suptitle(f"{pair.replace('_', ' ')}, {seed.replace('_', ' ')}, "
                     f"the word “{tok}”", fontsize=11)
        fig.tight_layout(rect=(0, 0, 0.88, 0.97))
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{pair}__{seed}__{tok}.png"
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
