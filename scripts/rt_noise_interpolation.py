#!/usr/bin/env python
"""D2's graded control: how fast does the correction stop transferring?

D2 compares two runs from completely different starting noise and finds zero
direction agreement. That is one end of a scale nobody has measured. This
walks the scale: start from seed 9's noise, move a fraction of the way toward
seed 13's noise, run that, and measure how much the correction still agrees
with seed 9's.

Why it matters. A learned adapter has to output the correction as a function
of the current image. If agreement falls off smoothly as the starting point
moves, the correction is a continuous function of the state, which is the
property that makes it fittable. If it collapses the moment anything is
perturbed, the correction is chaotic in the state and a smooth network is
fighting the problem rather than fitting it.

Two anchors check the pipeline rather than being results: at fraction 0 the
agreement with seed 9's CACHED correction must be ~1.0 (the run reproduces the
cache), and at fraction 1 it must land near D2's +0.002.

Starting noises are interpolated on the sphere (slerp), so every run starts
from noise of the same magnitude; a straight average would shrink it and
change the noise level rather than the direction.

The correction per step comes from run_teacher_residual at lambda 0, which
walks the plain PoE path and saves eps_Mono - eps_PoE at every step. That is
the same quantity the training cache holds, computed by the same code.

    CUDA_VISIBLE_DEVICES=0 python scripts/rt_noise_interpolation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cache import load_cell  # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._sampling import run_teacher_residual  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

PAIR = "a_cat__x__a_dog"
SEED_FROM, SEED_TO = 9, 13
# Dense near zero: that is where a decay, if there is one, has to show.
FRACTIONS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
OUT_DIR = paths.resolve(paths.DIRECTION_WALL)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Great-circle interpolation, so the magnitude of the noise is preserved."""
    if t == 0.0:
        return a.clone()
    if t == 1.0:
        return b.clone()
    af, bf = a.flatten().double(), b.flatten().double()
    cos = float((af @ bf) / (af.norm() * bf.norm()))
    theta = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    s = np.sin(theta)
    w_a, w_b = np.sin((1 - t) * theta) / s, np.sin(t * theta) / s
    return (float(w_a) * a + float(w_b) * b).to(a.dtype)


def r_t_for(init_latents, euler_sigma, cell, ctx) -> torch.Tensor:
    """Walk the PoE path from this starting noise, return r_t per step [T, D]."""
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "residuals"
        run_teacher_residual(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"],
            seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_j=seq_j, pool_j=pool_j,
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=ctx.guidance_scale,
            num_inference_steps=ctx.num_inference_steps,
            height=cell.height, width=cell.width,
            euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype,
            lambda_schedule="constant", lambda_max=0.0,
            save_residuals_dir=res,
        )
        return torch.stack([
            torch.load(f, map_location="cpu", weights_only=True)["delta"].float().flatten()
            for f in sorted(res.glob("step_*.pt"))
        ])


def main() -> int:
    ctx = make_ctx()
    cell_a = cell_from_slug(PAIR, SEED_FROM)
    cell_b = cell_from_slug(PAIR, SEED_TO)
    z_a, sigma = init_latents_for_cell(cell_a, ctx)
    z_b, sigma_b = init_latents_for_cell(cell_b, ctx)
    assert abs(sigma - sigma_b) < 1e-6, "the two cells disagree on the noise scale"

    reference = load_cell(PAIR, SEED_FROM).r_t().float().flatten(1)
    target = load_cell(PAIR, SEED_TO).r_t().float().flatten(1)
    print(f"reference: cached {PAIR} seed {SEED_FROM}, {tuple(reference.shape)}")
    print(f"cached seed {SEED_FROM} vs seed {SEED_TO}: "
          f"{float(torch.nn.functional.cosine_similarity(reference, target, dim=1).median()):+.4f}"
          f"   (D2's number, the far end of this curve)\n")

    rows = []
    for t in FRACTIONS:
        z = slerp(z_a, z_b, t)
        noise_cos = float(torch.nn.functional.cosine_similarity(
            z.flatten().unsqueeze(0), z_a.flatten().unsqueeze(0), dim=1))
        r = r_t_for(z, sigma, cell_a, ctx)
        c = torch.nn.functional.cosine_similarity(r, reference, dim=1)
        rows.append({
            "fraction": t,
            "noise_cosine_with_start": noise_cos,
            "r_t_agreement_median": float(c.median()),
            "r_t_agreement_first3": float(c[:3].median()),
            "r_t_agreement_late": float(c[10:].median()),
        })
        print(f"fraction {t:<5} noise cos {noise_cos:+.3f}   "
              f"correction agreement: median {float(c.median()):+.4f}  "
              f"steps 0-2 {float(c[:3].median()):+.4f}  "
              f"steps 10-49 {float(c[10:].median()):+.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "noise_interpolation.json").write_text(json.dumps({
        "pair": PAIR, "from_seed": SEED_FROM, "to_seed": SEED_TO,
        "interpolation": "slerp on the starting noise",
        "measure": "cosine between this run's r_t and the cached seed "
                   f"{SEED_FROM} r_t, at matched steps",
        "anchor_check": "fraction 0 must agree ~1.0 with the cache; fraction 1 "
                        "must land near D2's cross-run number",
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'noise_interpolation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
