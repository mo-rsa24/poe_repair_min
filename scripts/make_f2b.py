#!/usr/bin/env python
"""F2-dose-response_b: F2's qualitative grid, with the detector's own boxes drawn on.

F2 (paper/iclr/figures/F2-dose-response.png) puts a number in the corner of
every panel: 1 or 2, filled if composed. This figure answers where that
number comes from. Every box GroundingDINO detected and the scorer kept
(confidence >= 0.30, spans at least 25% of the image's longer side, survives
NMS against the others) is drawn directly on the panel. "Compose" becomes
visibly two boxes; "blend" becomes visibly one. Nothing about the rule
changes: this is the same validated instance-count scorer F2, F8a, and every
other compose-rate number in the paper reads from, just with its intermediate
output made visible instead of collapsed into a count.

Same cell, same rows, same lambda grid as F2: cat x dog, seed 9, the pair's
own correction against three controls that each break one thing (wrong pair,
wrong seed, wrong step order). No curve panel here; this figure is the
qualitative grid alone.

Needs a GPU: every panel is scored fresh to get its kept boxes, which
dose_curves.json does not carry (only the resulting count and label do).

    python scripts/make_f2b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

PAIRS_ROOT = Path("outputs/interaction_term/dose/pairs")
FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "F2-dose-response_b"
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")

ROWS = ("oracle", "wrong_pair", "wrong_seed", "wrong_step")
ROW_LABEL = {"oracle": "real correction", "wrong_pair": "wrong pair",
             "wrong_seed": "wrong seed", "wrong_step": "wrong order"}
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Page geometry, in inches. Same column width as F2 so the two sit at the same
# scale beside each other; no curve band, so the freed height goes back into
# a small top margin instead of taller panels (panel size stays matched to F2).
FIG_W = 5.5
GUTTER = 0.92
PANEL = 0.92
TOP_MARGIN = 0.24      # holds one row of column titles, one per lambda
NOTE_BAND = 0.24
FIG_H = TOP_MARGIN + 4 * PANEL + NOTE_BAND

BOX_COLOR = "#e6b800"   # gold, legible over both light fur and dark backdrops


def run_dir(pair: str, seed: int, lam: float, row: str) -> Path:
    """Where the sampler wrote this cell. At lambda=0 nothing is injected, so all
    three control rows are the same image by construction and share one directory."""
    tag = f"teacher_residual_const_lam{int(round(lam * 100)):03d}"
    if lam > 0 and row != "oracle":
        tag = f"{tag}_{row}"
    return PAIRS_ROOT / pair / f"seed_{seed}" / tag


def panel_png(pair: str, seed: int, lam: float, row: str) -> Path:
    d = run_dir(pair, seed, lam, row)
    hits = sorted(d.glob("*.png"))
    if not hits:
        raise SystemExit(f"no image under {d}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--device", default=None,
                    help="torch device for the detector, e.g. cpu. Default: GPU "
                         "if visible.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    contract = json.loads(SCORER_CONTRACT.read_text())
    if not contract.get("pass"):
        raise SystemExit(f"scorer at {SCORER_CONTRACT} is not marked validated: "
                         "refusing to draw boxes from an uncertified instrument.")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from poe_repair.experiments.compose_scorer.detection_scorer import count_instances
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    panels = {}
    for row in ROWS:
        for lam in LAMS:
            png = panel_png(args.pair, args.seed, lam, row)
            n, boxes = count_instances(png, device=device)
            panels[(row, lam)] = (png, n, boxes)
            print(f"  {row:<10} lam={lam:.2f}  n={n}  {png}")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    x0 = GUTTER / FIG_W
    pw = PANEL / FIG_W
    ph = PANEL / FIG_H
    grid_bottom = NOTE_BAND / FIG_H
    row_bottoms = [grid_bottom + (3 - i) * ph for i in range(4)]   # row 0 on top

    for i, row in enumerate(ROWS):
        for j, lam in enumerate(LAMS):
            png, _, boxes = panels[(row, lam)]
            rect = [x0 + j * pw, row_bottoms[i], pw, ph]
            ax = fig.add_axes(rect)
            ax.imshow(plt.imread(png))
            for b in boxes:
                bx0, by0, bx1, by1 = b["box"]
                ax.add_patch(Rectangle(
                    (bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                    linewidth=1.1, edgecolor=BOX_COLOR, zorder=4))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color("#444444")
        fig.text(x0 - 0.012, row_bottoms[i] + ph / 2, ROW_LABEL[row],
                 ha="right", va="center", fontsize=8)

    for j, lam in enumerate(LAMS):
        fig.text(x0 + j * pw + pw / 2, row_bottoms[0] + ph + 0.05 / FIG_H,
                 f"$\\lambda$={lam:.2f}", ha="center", va="bottom", fontsize=8)

    # The shared lambda=0 column, disclosed rather than styled away, matching F2.
    fig.patches.append(FancyBboxPatch(
        (x0 + 0.004, grid_bottom + 0.004), pw - 0.008, 4 * ph - 0.008,
        boxstyle="round,pad=0.004", transform=fig.transFigure,
        facecolor="none", edgecolor="#1f77b4", linewidth=1.0, zorder=6))

    note_y = grid_bottom - NOTE_BAND / FIG_H * 0.55
    fig.text(x0, note_y, "$\\lambda$=0: same image, nothing injected",
             ha="left", va="center", fontsize=7, color="#1f77b4")
    fig.text(x0 + 5 * pw, note_y,
             "box = one kept detection; compose needs two",
             ha="right", va="center", fontsize=7, color="#333333")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}")
    print(f"      {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
