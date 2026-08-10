#!/usr/bin/env python
"""The dose strip: what rising lambda actually looks like.

Plan 03's Goal asks for the three-curve figure "with the five-image strip". The
curve says the compose rate rises; the strip says what that means on one cell,
so a reader can check the scorer against their own eyes.

Default is one row per arm, same pair and seed throughout:

    oracle      lambda 0 -> 1 with the pair's own r_t
    random      the same doses with a norm-matched random vector
    wrong_pair  the same doses with another pair's r_t

Reading down a column asks the question the experiment exists to answer: at
this dose, does it matter WHICH vector you injected? If the three rows look
alike, the controls have not been separated and the causal claim is not made.

Each panel is annotated with the scorer's own reading (instance count and
compose/blend), so agreement or disagreement between the picture and the
number is visible rather than assumed. A disagreement is Goal 1's
inconclusive arm and means fixing the instrument, not the threshold.

Reads sampled images; it does not sample.

    python scripts/dose_strip.py --pair a_cat__x__a_dog --seed 9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose")
DEFAULT_ROOTS = (
    Path("outputs/interaction_term/dose/pairs"),
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
)
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
ROWS = ("oracle", "random", "wrong_pair")
ROW_LABEL = {
    "oracle": "the pair's own r_t",
    "random": "norm-matched random",
    "wrong_pair": "another pair's r_t",
}


def find_row_images(roots, pair: str, seed: int) -> dict[str, dict[float, Path]]:
    """row -> {lambda: image} for one cell."""
    out: dict[str, dict[float, Path]] = {r: {} for r in ROWS}
    for root in roots:
        d = Path(root) / pair / f"seed_{seed}"
        if not d.is_dir():
            continue
        for run in sorted(d.glob("teacher_residual_const_lam*")):
            m = re.search(r"lam(\d{3})", run.name)
            if not m:
                continue
            png = run / f"{run.name}.png"
            if not png.exists():
                continue
            rm = re.search(r"lam\d{3}_(random|wrong_pair)$", run.name)
            row = rm.group(1) if rm else "oracle"
            lam = int(m.group(1)) / 100.0
            out[row][lam] = png
            if lam == 0.0 and row == "oracle":
                # Nothing is injected at lambda=0, so the oracle image IS the
                # control image there. Share it rather than showing a gap.
                for other in ("random", "wrong_pair"):
                    out[other].setdefault(0.0, png)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--rows", default=",".join(ROWS),
                    help="comma-separated subset of oracle,random,wrong_pair")
    ap.add_argument("--annotate-boxes", action="store_true",
                    help="draw the kept detector boxes with their confidences, so a "
                         "suspect sliver can be judged by eye before a floor is chosen")
    ap.add_argument("--device", default=None,
                    help="torch device for the detector; use cpu when the GPU is held "
                         "by someone else (slower, same boxes)")
    ap.add_argument("--no-score", action="store_true",
                    help="skip the scorer annotation (no GPU needed)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    wanted = [r for r in args.rows.split(",") if r in ROWS]
    imgs = find_row_images(args.roots or DEFAULT_ROOTS, args.pair, args.seed)
    have = {r: imgs[r] for r in wanted if imgs[r]}
    if not have:
        print(
            f"no dose images for {args.pair} seed {args.seed}. This script "
            "reads images, it does not sample.\n"
            "  bash scripts/mechanism_study/run_dose_sweep.sh",
            file=sys.stderr,
        )
        return 2

    lams = sorted({l for r in have for l in have[r]})
    print(f"{args.pair} seed {args.seed}")
    for r in wanted:
        got = sorted(have.get(r, {}))
        print(f"  {r:<12} {len(got)} doses: {got}")

    scores: dict[tuple[str, float], tuple[int, bool]] = {}
    if not args.no_score:
        contract = json.loads(SCORER_CONTRACT.read_text())
        if not contract.get("pass"):
            print("scorer not validated; refusing to annotate", file=sys.stderr)
            return 2
        from poe_repair.experiments.compose_scorer.detection_scorer import (
            count_instances,
        )
        import torch
        dev = torch.device(args.device) if args.device else None
        boxes: dict[tuple[str, float], list[dict]] = {}
        for r in have:
            for lam, png in sorted(have[r].items()):
                n, kept = count_instances(png, device=dev)
                scores[(r, lam)] = (int(n), n >= 2)
                boxes[(r, lam)] = kept

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    nrow, ncol = len(have), len(lams)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.3 * nrow),
                             squeeze=False)
    for ri, r in enumerate([x for x in wanted if x in have]):
        for ci, lam in enumerate(lams):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            png = have[r].get(lam)
            if png is None:
                ax.text(0.5, 0.5, "not sampled", ha="center", va="center",
                        fontsize=7, color="0.5")
                ax.set_facecolor("0.95")
                continue
            ax.imshow(Image.open(png))
            if args.annotate_boxes and (r, lam) in boxes:
                # Every box the scorer KEPT, with its confidence and pixel size, so
                # a limb counted as an animal is visible rather than inferred.
                w, h = Image.open(png).size
                for b in boxes[(r, lam)]:
                    x0, y0, x1, y1 = b["box"]
                    side = max(x1 - x0, y1 - y0)
                    ax.add_patch(plt.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.2,
                        edgecolor="yellow" if side >= 0.25 * max(w, h) else "magenta"))
                    ax.text(x0, max(y0 - 4, 6),
                            f"{b['confidence']:.2f} {int(side)}px",
                            fontsize=5, color="yellow" if side >= 0.25 * max(w, h) else "magenta")
            if (r, lam) in scores:
                n, composed = scores[(r, lam)]
                # Label by the validated RULE (>=2), not by the raw count.
                # The count over-reports: a limb can clear the 0.30 confidence
                # floor as its own instance. See
                # docs/evidence/dose-response/scorer-count-caveat.md
                ax.set_xlabel(("compose" if composed else "blend")
                              + (f"  (n={n})" if args.annotate_boxes else ""),
                              fontsize=8,
                              color="tab:green" if composed else "tab:red")
            if ri == 0:
                ax.set_title(f"lambda {lam:g}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(ROW_LABEL[r], fontsize=8)

    fig.suptitle(f"Rising dose, three arms: {args.pair.replace('__x__', ' x ')} "
                 f"seed {args.seed}", fontsize=11)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_boxes" if args.annotate_boxes else ""
    out = args.out_dir / f"dose_strip_{args.pair}_seed{args.seed}{suffix}.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigure: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
