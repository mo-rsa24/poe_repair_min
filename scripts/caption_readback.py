#!/usr/bin/env python
"""Does the picture stop reading as a blend and start reading as two animals?

The compose scorer counts boxes. This asks a different question of the same
pictures: shown a short list of descriptions, which one does the image match
best? If the account is right, a low dose should pick the blend description
("one creature that is part cat and part dog") and a high dose should pick the
two-animal description ("a cat and a dog"), with a crossover somewhere between.

The bank holds four kinds of description, built from the prompts the cell was
actually generated with rather than from its folder name:

  two      the two animals, separately, both present
  blend    one hybrid creature made of both
  a_only   just the first animal
  b_only   just the second animal

Several wordings per kind, averaged, so no single awkward phrase decides the
answer.

The two control rows from the sweep are scored the same way: `random` is a
push of the same size in a random direction, `wrong_pair` is another pair's
correction. A crossover that also happens under a random push is not evidence.

**What this cannot tell you.** The scoring is CLIP image-text similarity, and
this repo has already NULLED whole-image CLIP as a way to TELL a blend from a
composition (outputs/compose_scorer/scorer_validated.json). Anchoring to text
instead of to other images is a different use of the space and may work where
that read failed, but it may equally come back flat. A flat curve here is a
limitation of CLIP, NOT evidence that the pictures did not change: the
instance-count scorer is the validated read, and it says they did. Only a
crossover that separates from the control rows counts for anything.

Reads sampled images; it does not sample.

Usage:
    python scripts/caption_readback.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    cell_dir,
)
from scripts.manifold_slide import (  # noqa: E402
    DEFAULT_ROOTS,
    ORACLE,
    find_runs,
    image_of,
)

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")

TEMPLATES = {
    "two": ["a photo of {a} and {b}",
            "a photo of two animals, {a} and {b}",
            "a photo of {a} next to {b}"],
    "blend": ["a photo of a single hybrid creature that is part {na} and part {nb}",
              "a photo of one animal that is a mix of {na} and {nb}",
              "a photo of a chimera of {na} and {nb}"],
    "a_only": ["a photo of {a}"],
    "b_only": ["a photo of {b}"],
}
KINDS = tuple(TEMPLATES)

# Bars sit on the largest INTERIOR dose, not on lambda=1. At full dose the
# injection adds all of r_t back onto eps_PoE, which is eps_J exactly, so the
# lambda=1 picture IS the joint-prompt render (scripts/manifold_slide.py
# measures the difference at about 2 grey levels of 255). A readback that
# flips at lambda=1 has therefore only shown that the joint render reads as two
# animals, which was never in doubt. The interior doses are where the claim
# lives. Every control row must stay at or under MAX_CONTROL_FRACTION of the
# oracle's gain at that same dose.
MIN_TWO_RATE_END = 0.50
MIN_TWO_RATE_GAIN = 0.20
MAX_CONTROL_FRACTION = 0.50


def bare(noun: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", noun.strip())


def bank_for(a: str, b: str) -> dict[str, list[str]]:
    f = {"a": a, "b": b, "na": bare(a), "nb": bare(b)}
    return {k: [t.format(**f) for t in ts] for k, ts in TEMPLATES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    runs = find_runs(args.roots or DEFAULT_ROOTS)
    if not runs:
        print("no dose runs found", file=sys.stderr)
        return 2
    control_rows = sorted(r for r in runs if r != ORACLE)
    shared = set(runs[ORACLE])
    for r in control_rows:
        shared &= set(runs[r])

    # Prompts come from the cell's own meta.json, never from the folder name:
    # a slug cannot be trusted to reconstruct the string the model was given.
    cells, banks = [], {}
    for cell in sorted(shared):
        meta_p = cell_dir(cell[0], cell[1], root=args.cache_root) / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        a, b = meta["pair"]
        banks[cell] = bank_for(a, b)
        cells.append(cell)
    if not cells:
        print("no cell has both the control rows and a cached meta.json",
              file=sys.stderr)
        return 2

    print(f"{len(cells)} cells, rows: {ORACLE} + {', '.join(control_rows)}")
    ex = banks[cells[0]]
    print(f"caption bank for {cells[0][0]}:")
    for k in KINDS:
        for t in ex[k]:
            print(f"  {k:<8} {t}")
    print()

    from poe_repair.experiments.residual_between_mono_and_poe.metrics import (
        clip_image_embed,
        clip_text_embed,
    )

    # Text embeddings per cell, averaged within each kind.
    text_by_cell = {}
    for cell in cells:
        rows = []
        for k in KINDS:
            e = clip_text_embed(banks[cell][k])
            rows.append(e.mean(0) / e.mean(0).norm().clamp_min(1e-12))
        text_by_cell[cell] = torch.stack(rows)          # (4, D)

    want = [(row, cell, lam, p)
            for cell in cells
            for row in [ORACLE] + control_rows
            for lam, run_dir in sorted(runs[row][cell].items())
            if (p := image_of(run_dir)) is not None]
    feats = torch.cat([clip_image_embed([w[3] for w in want][i:i + 32])
                       for i in range(0, len(want), 32)])

    picks: dict[str, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
    margins: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_cell = []
    for (row, cell, lam, p), f in zip(want, feats):
        sims = (text_by_cell[cell] @ f).numpy()
        pick = KINDS[int(sims.argmax())]
        margin = float(sims[KINDS.index("two")] - sims[KINDS.index("blend")])
        picks[row][lam].append(pick)
        margins[row][lam].append(margin)
        per_cell.append({"row": row, "pair": cell[0], "seed": cell[1],
                         "lambda": lam, "pick": pick, "two_minus_blend": margin,
                         **{f"sim_{k}": float(s) for k, s in zip(KINDS, sims)}})

    # lambda=0 is shared: a zero-sized push has no direction, so the control
    # rows have no lambda=0 of their own.
    for row in control_rows:
        if 0.0 in picks[ORACLE] and 0.0 not in picks[row]:
            picks[row][0.0] = list(picks[ORACLE][0.0])
            margins[row][0.0] = list(margins[ORACLE][0.0])

    lams = sorted(picks[ORACLE])
    curves = {}
    for row in [ORACLE] + control_rows:
        curves[row] = {
            "lambdas": lams,
            "two_rate": [float(np.mean([q == "two" for q in picks[row][l]]))
                         for l in lams],
            "blend_rate": [float(np.mean([q == "blend" for q in picks[row][l]]))
                           for l in lams],
            "solo_rate": [float(np.mean([q in ("a_only", "b_only")
                                         for q in picks[row][l]])) for l in lams],
            "mean_two_minus_blend": [float(np.mean(margins[row][l])) for l in lams],
            "n": [len(picks[row][l]) for l in lams],
        }

    for label, key in (("picks the two-animal caption", "two_rate"),
                       ("picks the blend caption", "blend_rate"),
                       ("picks one animal alone", "solo_rate"),
                       ("two minus blend similarity", "mean_two_minus_blend")):
        print(f"{label}")
        print(f"  {'row':<12}" + "".join(f"{l:>9.2f}" for l in lams))
        for row in [ORACLE] + control_rows:
            fmt = "{:>9.3f}" if key == "mean_two_minus_blend" else "{:>8.0%} "
            print(f"  {row:<12}" + "".join(fmt.format(v) for v in curves[row][key]))
        print()

    o = curves[ORACLE]
    interior = [l for l in lams if 0.0 < l < 1.0]
    if not interior:
        print("no interior dose: lambda=1 reproduces the joint render exactly, "
              "so with only the endpoints there is nothing here to read.",
              file=sys.stderr)
        return 2
    top = max(interior)
    i = lams.index(top)
    base, end = o["two_rate"][0], o["two_rate"][i]
    gain = end - base
    crossed = end >= MIN_TWO_RATE_END and gain >= MIN_TWO_RATE_GAIN
    # Where the two-animal description overtakes the blend description.
    over = [l for l, m in zip(lams, o["mean_two_minus_blend"]) if m > 0]
    verdicts = {"scored_at_lambda": top, "two_rate_base": base,
                "two_rate_at_scored_lambda": end, "gain": gain,
                "crossover_lambda": (min(over) if over else None),
                "two_rate_at_lambda_1": o["two_rate"][-1],
                "crossed": bool(crossed)}
    print(f"lambda=1 is the joint render itself, so the bars are read at the "
          f"largest interior\ndose, lambda={top}. (For the record the "
          f"two-animal caption wins {o['two_rate'][-1]:.0%} at lambda 1.)\n")
    print(f"oracle: the two-animal caption wins {base:.0%} at lambda 0 and "
          f"{end:.0%} at lambda {top} ({gain:+.0%})")
    print(f"  the two-animal description overtakes the blend description at "
          f"lambda {min(over) if over else 'never'}")
    print(f"  bar: at lambda {top}, rate >= {MIN_TWO_RATE_END:.0%} and gain >= "
          f"{MIN_TWO_RATE_GAIN:.0%}   {'crossed' if crossed else 'NO crossover'}")
    controls_flat = True
    for row in control_rows:
        c_gain = curves[row]["two_rate"][i] - curves[row]["two_rate"][0]
        frac = c_gain / gain if abs(gain) > 1e-9 else float("nan")
        flat = not (frac > MAX_CONTROL_FRACTION)
        controls_flat &= bool(flat)
        verdicts[f"{row}_fraction_of_oracle_gain"] = float(frac)
        print(f"  {row:<12} gains {c_gain:+.0%} = {frac:.0%} of the oracle's "
              f"{gain:+.0%}   {'flat' if flat else 'ALSO CROSSES'}")
    print(f"  bar: every control <= {MAX_CONTROL_FRACTION:.0%} of the oracle's gain")

    if crossed and controls_flat:
        verdict = ("the readback crosses over from blend to two animals, and "
                   "only the real correction crosses it")
    elif crossed:
        verdict = ("the readback crosses over, but a control push crosses it "
                   "too, so the crossover is not specific to the correction")
    else:
        verdict = ("no crossover: CLIP's caption readback does not separate "
                   "blend from two animals on these pictures")
    print(f"\nreading: {verdict}")
    if not crossed:
        print("  this is a limit of the readback, not a null result about the "
              "pictures:\n  the validated instance-count scorer does separate "
              "them on the same images.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "caption_readback.json"
    out.write_text(json.dumps({
        "n_cells": len(cells), "kinds": list(KINDS), "templates": TEMPLATES,
        "bars": {"min_two_rate_end": MIN_TWO_RATE_END,
                 "min_two_rate_gain": MIN_TWO_RATE_GAIN,
                 "max_control_fraction": MAX_CONTROL_FRACTION},
        "curves": curves, "verdicts": verdicts, "verdict": verdict,
        "cells": per_cell,
    }, indent=2))
    print(f"wrote {out}")

    if not args.no_figure:
        _figure(curves, control_rows, verdict, len(cells),
                args.out_dir / "caption_readback.png")
    return 0


def _figure(curves, control_rows, verdict, n_cells, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    o = curves[ORACLE]
    ax.plot(o["lambdas"], o["two_rate"], "-o", color="tab:green", lw=2.2,
            label="two animals")
    ax.plot(o["lambdas"], o["blend_rate"], "-o", color="tab:red", lw=2.2,
            label="one blended creature")
    ax.plot(o["lambdas"], o["solo_rate"], "-o", color="0.6", lw=1.4,
            label="one animal alone")
    ax.set_xlabel("dose (lambda)")
    ax.set_ylabel("share of cells picking that description")
    ax.set_title("Which description the picture matches best", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    style = {ORACLE: ("tab:purple", "-", "real correction"),
             "random": ("tab:gray", "--", "same-sized random push"),
             "wrong_pair": ("tab:orange", "--", "another pair's correction")}
    for row in [ORACLE] + list(control_rows):
        col, ls, lab = style.get(row, ("tab:blue", "-", row))
        ax2.plot(curves[row]["lambdas"], curves[row]["two_rate"], ls, color=col,
                 marker="o", lw=2, label=lab)
    ax2.set_xlabel("dose (lambda)")
    ax2.set_ylabel("share picking the two-animal description")
    ax2.set_title("Against the controls", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{verdict}   ({n_cells} cells)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
