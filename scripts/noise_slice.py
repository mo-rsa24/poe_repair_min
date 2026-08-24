#!/usr/bin/env python
"""A map of noise space: are good seeds a region, or scattered?

The question. Some starting noises compose when the correction is applied and
others do not, and nothing so far says whether that is a smooth property of
where you start or an unpredictable one. If composing starts form connected
regions, the outcome is a smooth function of the starting state, which is the
property a network can learn to anticipate. If they are scattered
salt-and-pepper, no adapter can be expected to know in advance which runs it
will fix.

Why this is not an MDS plot, and cannot be. Two independent starting noises in
this model are 65536-dimensional Gaussians, and measured on the cache their
cosine is +0.003: essentially orthogonal. Any set of independent draws is
therefore mutually equidistant, so embedding the seeds and looking at the
picture returns a regular simplex whatever the outcomes are. Nothing is
learned from the shape because the shape is fixed by the dimension. The fix is
to stop embedding and start CONSTRUCTING: pick three starting noises, build
the spherical triangle they span, and sample a grid inside it. The axes are
then two numbers we chose ("how far toward B", "how far toward C") rather than
two directions an algorithm chose, and every point on the map is a real run.

Spherical interpolation throughout, so every constructed start has the same
magnitude as a real one; a plain weighted average would shrink it and quietly
change the noise level rather than the direction.

Readings declared before this runs:
    composing cells form connected patches
        -> the outcome is a smooth function of where the run starts, so
           "good seed" is a property of a neighbourhood and an adapter can in
           principle anticipate it from the state it is handed.
    composing cells are scattered with no contiguity
        -> the outcome is chaotic in the starting state. The method can still
           work, since it is applied per run, but no figure should claim the
           adapter could predict which runs it will fix, and D2b's smooth
           early regime does not extend to the OUTCOME.
    all cells compose or none do
        -> the slice was chosen badly (three starts on the same side of the
           boundary); rerun with corners of mixed outcome before reading
           anything into it.

    CUDA_VISIBLE_DEVICES=0 python scripts/noise_slice.py --pair a_cat__x__a_dog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers import teacher_residual as cmp_tr  # noqa: E402
from poe_repair.composers._helpers import init_latents_for_cell  # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

OUT_DIR = paths.resolve(paths.NOISE_SLICE)
# Corners chosen from the window grid's earliest-window column, where these
# three seeds of cat x dog did not all agree, so the slice has a boundary in
# it rather than being uniformly good or uniformly bad.
DEFAULT_CORNERS = (9, 11, 12)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    if t <= 0.0:
        return a.clone()
    if t >= 1.0:
        return b.clone()
    af, bf = a.flatten().double(), b.flatten().double()
    cos = float((af @ bf) / (af.norm() * bf.norm()))
    th = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    s = np.sin(th)
    return (float(np.sin((1 - t) * th) / s) * a
            + float(np.sin(t * th) / s) * b).to(a.dtype)


def plane_point(zs, u: float, v: float) -> torch.Tensor:
    """A point on the plane patch through the three starting noises.

    Chaining two slerps was the obvious construction and is wrong for a map:
    sliding all the way to the third corner collapses an entire row of the
    grid onto one point, which draws as a stripe of identical cells and reads
    as a result.

    So: take the first corner as the origin, the second as one direction, and
    the part of the third that is perpendicular to the first direction as the
    other. That gives two independent axes with nothing degenerate at the
    edges. The result is rescaled back to the magnitude of a real starting
    noise, because interior points of a plane through three near-orthogonal
    vectors are longer than the corners, and a longer start is a different
    noise level rather than a different direction.
    """
    a, b, c = (z.flatten().double() for z in zs)
    d1 = b - a
    d2 = c - a
    d2 = d2 - d1 * ((d2 @ d1) / (d1 @ d1))        # keep only what d1 misses
    p = a + float(u) * d1 + float(v) * d2
    p = p * (a.norm() / p.norm())
    return p.reshape(zs[0].shape).to(zs[0].dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--corners", type=int, nargs=3, default=list(DEFAULT_CORNERS))
    ap.add_argument("--grid", type=int, default=5, help="points per side")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.75,
                    help="dose. 0.75 is the largest interior dose; lambda=1 "
                         "reproduces the joint prediction by construction and "
                         "would make every cell compose for a trivial reason")
    ap.add_argument("--exp-name", default="interaction_term/noise_slice")
    args = ap.parse_args()

    ctx = make_ctx()
    cells = [cell_from_slug(args.pair, s) for s in args.corners]
    zs, sigmas = zip(*(init_latents_for_cell(c, ctx) for c in cells))
    assert max(sigmas) - min(sigmas) < 1e-6, "corners disagree on noise scale"

    us = np.linspace(0.0, 1.0, args.grid)
    vs = np.linspace(0.0, 1.0, args.grid)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for i, v in enumerate(vs):
        for j, u in enumerate(us):
            z = plane_point(zs, float(u), float(v))
            name = f"slice_u{j}_v{i}"
            out = cmp_tr.run(
                cells[0], ctx, lambda_schedule="constant", lambda_max=args.lam,
                save_residuals=False, save_trajectory=False,
                exp_name=args.exp_name, overwrite=False,
                method_name_override=name, init_latents_override=z,
            )
            rows.append({"u": float(u), "v": float(v), "i": i, "j": j,
                         "name": name, "path": str(out)})
            n += 1
            print(f"[{n}/{args.grid**2}] u={u:.2f} v={v:.2f} -> {out}")

    (OUT_DIR / "slice_manifest.json").write_text(json.dumps({
        "pair": args.pair, "corners": args.corners, "grid": args.grid,
        "lambda": args.lam,
        "construction": "plane patch: corner A as origin, direction to B as "
                        "one axis, the component of the direction to C "
                        "perpendicular to it as the other, rescaled to A's "
                        "magnitude so every start sits at the real noise "
                        "level. Chained slerps were rejected: they collapse "
                        "the far row onto a single point",
        "why_not_mds": "independent starting noises are mutually orthogonal "
                       "(measured cosine +0.003), so embedding them returns a "
                       "regular simplex carrying no information; the axes here "
                       "are constructed instead",
        "cells": rows,
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'slice_manifest.json'}")
    print("score with: python scripts/plot_dose_curves.py is NOT right here; "
          "use the scorer directly over the manifest paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
