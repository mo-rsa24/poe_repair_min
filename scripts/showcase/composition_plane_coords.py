#!/usr/bin/env python
"""Where the three branches and the joint prediction sit, in the plane PoE lives in.

The three cached predictions define a plane: the affine plane through eps_u spanned by
(eps_a - eps_u) and (eps_b - eps_u). Every prediction product-of-experts can produce lies in
it, for any weights at all:

    eps_u + alpha (eps_a - eps_u) + beta (eps_b - eps_u)

So a 2-D drawing of that plane loses nothing about the composition. The only thing that leaves
it is the joint prompt's prediction, and how far it leaves by is the part no re-weighting can
ever reach.

Two orthogonal shares are reported, and they are not the same quantity:

    share_2d   against this affine plane (the figure's plane: re-weight the two experts)
    share_3d   against span{a-u, b-u, u} (the filed finding's definition: also free the
               coefficient on the unconditional term, so a strictly larger reachable set and
               therefore a strictly smaller orthogonal share)

`report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md` reports share_3d.

Writes <out>/composition_plane.json: per seed, per step, the 2-D coordinates of every
participant with eps_u at the origin, plus both shares. No GPU, cache only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "showcase"))
from correction_span_common import load_step, num_steps, GUIDANCE, PAIR, SEEDS  # noqa: E402

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/composition_plane")


def coords_for(seed: int, step: int) -> dict:
    s = load_step(seed, step)
    u = s.eps_u.reshape(-1).double()
    da = s.eps_a.reshape(-1).double() - u          # a - u
    db = s.eps_b.reshape(-1).double() - u          # b - u
    dj = s.eps_j.reshape(-1).double() - u          # joint - u

    # orthonormal frame of the plane: e1 along (a-u), e2 the part of (b-u) that is not e1
    n1 = da.norm()
    e1 = da / n1.clamp_min(1e-12)
    p = db - (db @ e1) * e1
    n2 = p.norm()
    e2 = p / n2.clamp_min(1e-12)

    def xy(v):
        return [float(v @ e1), float(v @ e2)]

    A, B, J = xy(da), xy(db), xy(dj)
    # what PoE actually steps with, unguided and at the dial
    T  = [A[0] + B[0], A[1] + B[1]]
    Tw = [GUIDANCE * T[0], GUIDANCE * T[1]]
    Jw = [GUIDANCE * J[0], GUIDANCE * J[1]]

    # how far the joint prediction leaves the plane
    j_in = J[0] * e1 + J[1] * e2
    out_vec = dj - j_in
    out_norm = float(out_vec.norm())

    # the correction, and its two shares
    r_un = dj - da - db                      # unguided joint minus unguided composition
    r_in = r_un - out_vec                    # its component inside the plane
    share_2d = float((out_norm ** 2) / (r_un.norm() ** 2).clamp_min(1e-30))

    M = torch.stack([da, db, u], dim=1)      # the filed finding's 3-D span
    c = torch.linalg.lstsq(M, r_un.unsqueeze(1)).solution.squeeze(1)
    r3 = r_un - M @ c
    share_3d = float((r3.norm() ** 2) / (r_un.norm() ** 2).clamp_min(1e-30))

    return {
        "step": step, "timestep": s.timestep,
        "a": A, "b": B, "j": J, "T": T, "Tw": Tw, "Jw": Jw,
        "out_of_plane": out_norm,
        "norms": {"a_minus_u": float(n1), "b_minus_u": float(db.norm()),
                  "j_minus_u": float(dj.norm()), "u": float(u.norm()),
                  "r_unguided": float(r_un.norm()), "r_in_plane": float(r_in.norm())},
        "share_2d": share_2d, "share_3d": share_3d,
    }


def main() -> int:
    steps = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else [])]
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"pair": PAIR, "guidance": GUIDANCE, "seeds": {}}
    for seed in SEEDS:
        n = num_steps(seed)
        use = steps or list(range(n))
        rows = []
        t0 = time.time()
        for st in use:
            if st >= n:
                continue
            rows.append(coords_for(seed, st))
        out["seeds"][str(seed)] = rows
        print(f"[seed {seed}] {len(rows)} steps ({time.time()-t0:.1f}s)", flush=True)
    (OUT / "composition_plane.json").write_text(json.dumps(out))
    print(f"[done] -> {OUT/'composition_plane.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
