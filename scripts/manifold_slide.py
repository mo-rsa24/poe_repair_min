#!/usr/bin/env python
"""Does raising the dose slide the picture out of the blend region?

The compose rate is a yes/no read: it says whether an image tipped over into
two animals, not how far it moved. This places every dose's picture on an axis
between the two known endpoints and measures how far along it sits, so a
partial dose shows as a partial slide.

The axis, per cell, runs from the uncorrected Product-of-Experts render to the
joint-prompt render, using the CACHED poe.png and mono.png for that pair and
seed.

**The endpoints are identities, and the script proves it rather than hiding
it.** At lambda=1 the injection adds the whole of r_t = eps_J - eps_PoE back
onto eps_PoE, which is eps_J exactly, so the full-dose picture IS the mono
render, same seed and all. Measured here: about 1 grey level out of 255,
against 24 to 58 between mono and poe. Moving the anchors off the sweep does
not rescue the endpoint, because the sweep's endpoint and the anchor are the
same image. So "the oracle reaches 1.0" is arithmetic, not evidence, and the
bars below sit on the INTERIOR doses (0 < lambda < 1) where nothing forces the
answer, and on the comparison against the control rows at matched lambda.

Two control rows come from the sweep and are the reason the number means
anything:

  random       a push of the SAME SIZE in a random direction. This is the
               control the claim needs: it separates "the correction moves the
               sample toward the joint render" from "any perturbation of this
               magnitude moves the sample somewhere".
  wrong_pair   the correction computed for a DIFFERENT pair, injected here.
               A push that is structured but aimed wrong.

**What this cannot tell you.** In CLIP image space, this repo has already
NULLED the whole-image embedding as a way to TELL a blend from a composition
(outputs/compose_scorer/scorer_validated.json: one chimera and two separate
animals land in the same region). So a position on this axis is not a compose
score and must never be read as one. What survives that null is the RELATIVE
question asked here: does the oracle row travel along the axis further than the
same-sized random push does? That is a comparison between rows measured the
same way, which the null does not touch. Read the slide, not the location.

Reads sampled images; it does not sample.

Usage:
    python scripts/manifold_slide.py
    python scripts/manifold_slide.py --space latent
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

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
DEFAULT_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
    Path("outputs/interaction_term/dose/pairs"),
)
RUN_RE = re.compile(r"^teacher_residual_const_lam(\d{3})(?:_(\w+))?$")
ORACLE = "oracle"

# Bars on the INTERIOR doses only (0 < lambda < 1), because both endpoints are
# identities. The oracle's mean interior projection must clear MIN_ORACLE_MID,
# and every control row must stay at or below MAX_CONTROL_FRACTION of the
# oracle's interior travel. Both sit in the source so neither can be moved
# after the curves are on screen.
MIN_ORACLE_MID = 0.30
MAX_CONTROL_FRACTION = 0.50

# A lambda=1 picture further than this many grey levels from the cached mono
# render would mean the injection is NOT reproducing eps_J, and the whole dose
# axis would need re-deriving before any of this could be read.
MAX_ENDPOINT_DRIFT = 5.0


def find_runs(roots) -> dict[str, dict[tuple[str, int], dict[float, Path]]]:
    """row -> (pair, seed) -> lambda -> run directory."""
    out: dict[str, dict[tuple[str, int], dict[float, Path]]] = defaultdict(
        lambda: defaultdict(dict))
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_")[1])
                for run_dir in sorted(seed_dir.iterdir()):
                    m = RUN_RE.match(run_dir.name)
                    if not m or not run_dir.is_dir():
                        continue
                    row = m.group(2) or ORACLE
                    out[row][(pair_dir.name, seed)][int(m.group(1)) / 100.0] = run_dir
        if out:
            break
    return out


def clip_embed(paths: list[Path]) -> torch.Tensor:
    from poe_repair.experiments.residual_between_mono_and_poe.metrics import clip_image_embed

    chunks = [clip_image_embed(paths[i:i + 32]) for i in range(0, len(paths), 32)]
    return torch.cat(chunks)


def image_of(run_dir: Path) -> Path | None:
    p = run_dir / f"{run_dir.name}.png"
    return p if p.exists() else None


def endpoint_drift(oracle_runs, cells, cache_root: Path) -> float | None:
    """Mean |lambda=1 picture - cached mono render|, in grey levels of 255.

    Near zero is the expected answer, and finding it is the point: it confirms
    the full-dose injection really does reproduce eps_J, which is what makes
    the lambda=1 endpoint an identity rather than a result.
    """
    from PIL import Image

    diffs = []
    for cell in cells:
        run = oracle_runs.get(cell, {}).get(1.0)
        mono = cell_dir(cell[0], cell[1], root=cache_root) / "mono.png"
        img = image_of(run) if run else None
        if img is None or not mono.exists():
            continue
        a = np.asarray(Image.open(img).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(mono).convert("RGB"), dtype=np.float64)
        if a.shape == b.shape:
            diffs.append(float(np.abs(a - b).mean()))
    return float(np.mean(diffs)) if diffs else None


def latent_of(run_dir: Path) -> torch.Tensor | None:
    p = run_dir / "latent_trajectory.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu",
                      weights_only=True)["trajectories"][-1].float().flatten()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", choices=("clip", "latent"), default="clip",
                    help="clip: CLIP image embedding. latent: raw final latent.")
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    runs = find_runs(args.roots or DEFAULT_ROOTS)
    if not runs:
        looked = args.roots or DEFAULT_ROOTS
        print("no dose runs found under:\n"
              + "\n".join(f"  {p}" for p in looked), file=sys.stderr)
        return 2

    # Only cells the control rows also cover: a slide with nothing to compare
    # against is not evidence.
    control_rows = [r for r in runs if r != ORACLE]
    shared = set(runs[ORACLE])
    for r in control_rows:
        shared &= set(runs[r])
    if not shared:
        print("no cell has both the oracle row and a control row", file=sys.stderr)
        return 2

    # Anchors, from the cache rather than from the sweep.
    anchors = {}
    for pair, seed in sorted(shared):
        d = cell_dir(pair, seed, root=args.cache_root)
        if (d / "poe.png").exists() and (d / "mono.png").exists():
            anchors[(pair, seed)] = (d / "poe.png", d / "mono.png")
    dropped = len(shared) - len(anchors)
    cells = sorted(anchors)
    if not cells:
        print("no cell has cached poe.png and mono.png to build the axis from",
              file=sys.stderr)
        return 2

    print(f"space: {args.space}")
    print(f"{len(cells)} cells with the oracle row, every control row, and both "
          f"cached anchors")
    if dropped:
        print(f"  {dropped} cell(s) dropped for missing cached anchors")
    print(f"rows: {ORACLE} + {', '.join(sorted(control_rows))}")
    print("axis: cached poe.png (0) -> cached mono.png (1), independent of the "
          "sweep\n")

    # Gather every vector in one pass so CLIP loads once.
    want: list[tuple[str, tuple[str, int], float, Path]] = []
    for cell in cells:
        for row in [ORACLE] + control_rows:
            for lam, run_dir in sorted(runs[row][cell].items()):
                if args.space == "clip":
                    p = image_of(run_dir)
                    if p:
                        want.append((row, cell, lam, p))
                else:
                    want.append((row, cell, lam, run_dir))

    if args.space == "clip":
        anchor_paths = [p for c in cells for p in anchors[c]]
        emb = clip_embed(anchor_paths + [w[3] for w in want])
        anchor_vec = {c: (emb[2 * i], emb[2 * i + 1]) for i, c in enumerate(cells)}
        vecs = emb[2 * len(cells):]
    else:
        anchor_vec = {}
        for c in cells:
            d = cell_dir(c[0], c[1], root=args.cache_root)
            a = latent_of(runs[ORACLE][c].get(0.0, d))
            if a is None:
                continue
            anchor_vec[c] = None
        print("latent space cannot use the cached image anchors; "
              "falling back to the sweep's own lambda 0 and 1 endpoints.\n"
              "  note that lambda=1 then scores 1.0 by construction.",
              file=sys.stderr)
        for c in cells:
            lam_map = runs[ORACLE][c]
            if 0.0 in lam_map and 1.0 in lam_map:
                a, b = latent_of(lam_map[0.0]), latent_of(lam_map[1.0])
                if a is not None and b is not None:
                    anchor_vec[c] = (a, b)
        cells = [c for c in cells if anchor_vec.get(c) is not None]
        vecs = torch.stack([latent_of(w[3]) for w in want])

    rows_out = []
    by_row: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    off_by_row: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for (row, cell, lam, _), v in zip(want, vecs):
        pv = anchor_vec.get(cell)
        if pv is None:
            continue
        poe, mono = pv
        axis = mono - poe
        denom = float(axis.dot(axis))
        if denom < 1e-12:
            continue
        d = v - poe
        along = float(d.dot(axis)) / denom
        off = float((d - along * axis).norm() / max(d.norm().item(), 1e-12))
        by_row[row][lam].append(along)
        off_by_row[row][lam].append(off)
        rows_out.append({"row": row, "pair": cell[0], "seed": cell[1],
                         "lambda": lam, "projection": along,
                         "off_axis_fraction": off})

    if not rows_out:
        print("nothing projected", file=sys.stderr)
        return 2

    # lambda=0 is shared: the control rows have no lambda=0 of their own,
    # because a zero-sized push has no direction. Borrow the oracle's.
    zero = by_row[ORACLE].get(0.0)
    for row in control_rows:
        if zero and 0.0 not in by_row[row]:
            by_row[row][0.0] = list(zero)
            off_by_row[row][0.0] = list(off_by_row[ORACLE][0.0])

    print(f"{'row':<12}" + "".join(f"{l:>9.2f}" for l in
                                   sorted(by_row[ORACLE])) + "   (projection)")
    curves = {}
    for row in [ORACLE] + sorted(control_rows):
        lams = sorted(by_row[row])
        means = [float(np.mean(by_row[row][l])) for l in lams]
        curves[row] = {"lambdas": lams, "mean_projection": means,
                       "n": [len(by_row[row][l]) for l in lams],
                       "mean_off_axis": [float(np.mean(off_by_row[row][l]))
                                         for l in lams]}
        print(f"{row:<12}" + "".join(f"{m:>9.3f}" for m in means))
    print(f"{'':12}" + "".join(f"{l:>9.2f}" for l in sorted(by_row[ORACLE]))
          + "   (off-axis fraction)")
    for row in [ORACLE] + sorted(control_rows):
        print(f"{row:<12}" + "".join(f"{m:>9.1%}"
                                     for m in curves[row]["mean_off_axis"]))

    o = curves[ORACLE]
    verdicts: dict = {}

    # The endpoint is an identity. Measure it, so the claim is checked rather
    # than trusted, and so a broken injection shows up as a failed check.
    drift = endpoint_drift(runs[ORACLE], cells, args.cache_root)
    if drift is not None:
        verdicts["endpoint_drift_grey_levels"] = drift
        ok = drift <= MAX_ENDPOINT_DRIFT
        verdicts["endpoint_is_identity"] = bool(ok)
        print(f"\nendpoint check: |lambda=1 - cached mono| = {drift:.2f} grey "
              f"levels of 255")
        print(f"  at lambda=1 the injection adds all of r_t back, giving eps_J "
              f"exactly, so\n  the full-dose picture IS the mono render. "
              f"{'Confirmed' if ok else 'NOT CONFIRMED'} "
              f"(bar: <= {MAX_ENDPOINT_DRIFT}).")
        if not ok:
            print("  the dose axis does not do what it claims; fix that before "
                  "reading the curves.")
        print("  both endpoints are therefore arithmetic. Everything below is "
              "read off the\n  interior doses and the control rows.")

    interior = [l for l in o["lambdas"] if 0.0 < l < 1.0]
    if not interior:
        print("\nno interior dose: with only the endpoints there is nothing "
              "here that is not an identity.", file=sys.stderr)
        return 2

    def mid(row: str) -> float:
        c = curves[row]
        return float(np.mean([m for l, m in zip(c["lambdas"], c["mean_projection"])
                              if 0.0 < l < 1.0]))

    base = o["mean_projection"][0]
    o_mid = mid(ORACLE)
    monotone = all(b >= a - 1e-6 for a, b in
                   zip(o["mean_projection"], o["mean_projection"][1:]))
    verdicts.update({"interior_lambdas": interior, "oracle_mid": o_mid,
                     "oracle_monotone": monotone,
                     "oracle_clears_bar": bool(o_mid >= MIN_ORACLE_MID)})
    print(f"\ninterior doses {interior}")
    print(f"  oracle mean projection {o_mid:+.3f}   "
          f"monotone across all doses: {'yes' if monotone else 'NO'}")
    print(f"  bar: >= {MIN_ORACLE_MID}   "
          f"{'cleared' if o_mid >= MIN_ORACLE_MID else 'NOT cleared'}")

    travel = o_mid - base
    all_controls_flat = True
    for row in sorted(control_rows):
        c_travel = mid(row) - base
        frac = c_travel / travel if abs(travel) > 1e-9 else float("nan")
        flat = frac <= MAX_CONTROL_FRACTION
        all_controls_flat &= bool(flat)
        verdicts[f"{row}_fraction_of_oracle_travel"] = float(frac)
        print(f"  {row:<12} interior travel {c_travel:+.3f} = {frac:.0%} of the "
              f"oracle's {travel:+.3f}   {'flat' if flat else 'ALSO SLIDES'}")
    print(f"  bar: every control <= {MAX_CONTROL_FRACTION:.0%} of the oracle's "
          "interior travel")

    if o_mid >= MIN_ORACLE_MID and monotone and all_controls_flat:
        verdict = ("the picture slides with the dose, and only the real "
                   "correction slides it")
    elif o_mid >= MIN_ORACLE_MID and not all_controls_flat:
        verdict = ("the picture slides, but a control push slides it too: the "
                   "axis is not specific to the correction")
    elif o_mid < MIN_ORACLE_MID:
        verdict = "the correction does not move the picture along this axis"
    else:
        verdict = "the oracle slides but not monotonically in dose"
    print(f"\nreading: {verdict}")
    print("  read as a slide, not as a compose score: whole-image CLIP cannot "
          "tell a\n  blend from a composition, so only the row-to-row "
          "comparison is load-bearing,\n  and the lambda=0 and lambda=1 "
          "endpoints are arithmetic rather than evidence.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"manifold_slide_{args.space}.json"
    out.write_text(json.dumps({
        "space": args.space, "n_cells": len(cells),
        "axis": "cached poe.png -> cached mono.png" if args.space == "clip"
                else "sweep lambda 0 -> lambda 1 (endpoint is 1 by construction)",
        "bars": {"min_oracle_mid": MIN_ORACLE_MID,
                 "max_control_fraction": MAX_CONTROL_FRACTION,
                 "max_endpoint_drift": MAX_ENDPOINT_DRIFT},
        "endpoints_are_identities": (
            "lambda=0 is the PoE anchor and lambda=1 reproduces eps_J exactly, "
            "so both endpoints are arithmetic; the bars sit on the interior "
            "doses and on the control rows at matched lambda"),
        "curves": curves, "verdicts": verdicts, "verdict": verdict,
        "cells": rows_out,
    }, indent=2))
    print(f"wrote {out}")

    if not args.no_figure:
        _figure(curves, verdict, len(cells), args.space,
                args.out_dir / f"manifold_slide_{args.space}.png")
    return 0


def _figure(curves, verdict, n_cells, space, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {ORACLE: ("tab:purple", "-", "o", "real correction"),
             "random": ("tab:gray", "--", "s", "same-sized random push"),
             "wrong_pair": ("tab:orange", "--", "^", "another pair's correction")}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for row, c in curves.items():
        col, ls, mk, lab = style.get(row, ("tab:blue", "-", "o", row))
        ax.plot(c["lambdas"], c["mean_projection"], ls, color=col, marker=mk,
                lw=2, label=lab)
        ax2.plot(c["lambdas"], c["mean_off_axis"], ls, color=col, marker=mk,
                 lw=2, label=lab)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axhline(1, color="0.7", lw=0.8)
    ax.text(0.01, 0.0, " PoE (blend)", va="bottom", fontsize=8, color="0.4")
    ax.text(0.01, 1.0, " Mono (joint render)", va="top", fontsize=8, color="0.4")
    # The endpoints are arithmetic, so say so on the figure itself rather than
    # letting a reader take the 0-to-1 span as the result.
    for a in (ax, ax2):
        a.axvspan(0.0, 0.02, color="0.85", zorder=0)
        a.axvspan(0.98, 1.0, color="0.85", zorder=0)
    fig.text(0.5, 0.005, "shaded doses are identities: lambda=0 is the axis "
                         "anchor and lambda=1 reproduces eps_J exactly, so the "
                         "interior doses carry the result",
             ha="center", fontsize=8, color="0.35")
    ax.set_xlabel("dose (lambda)")
    ax.set_ylabel("projection onto PoE -> Mono axis")
    ax.set_title(f"How far the dose moves the picture ({space} space)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax2.set_xlabel("dose (lambda)")
    ax2.set_ylabel("fraction of motion off the axis")
    ax2.set_title("How much of the motion the axis fails to explain", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(f"{verdict}   ({n_cells} cells)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
