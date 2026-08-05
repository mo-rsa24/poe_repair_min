#!/usr/bin/env python
"""Inject the interaction term r_t back into PoE sampling at dose lambda.

A thin command line over ``poe_repair.composers.teacher_residual``, which
already implements

    eps_final = eps_PoE + lambda_t * (eps_Mono - eps_PoE)

This script adds no sampling logic. It exists so plan 03 can drive one
(pair, seed, lambda) cell from the shell.

The canary is the point of interest. At lambda=0 the sampler takes the
``eps_t = eps_poe`` branch directly (see _sampling.py), so injecting nothing
must change nothing. If it does, the injection path is perturbing the sampler
when it should be inert, and every dose number downstream is meaningless.

What the canary compares matters. It does *not* compare against
``run_cfg_poe``: that sampler batches three UNet branches where this one
batches four, and the same UNet returns different numbers per batch shape
(~2e-3 per step, compounding to ~0.6 over 50 steps). So the reference here is
the sampler's own saved ``eps_poe``, written before the injection branch runs,
compared against the eps actually stepped with, and scaled by ||r_t||.
See tests/test_interaction_term_canaries.py for the full argument.

Usage:
    python scripts/interaction_term_inject.py --pair a_cat__x__a_dog --seed 9 --lambda 0 --check-canary
    python scripts/interaction_term_inject.py --pair a_cat__x__a_dog --seed 9 --lambda 0.5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

# Running as `python scripts/foo.py` puts scripts/ on sys.path, not the repo
# root, so the package would not import. The plan's engagement instructions use
# that form, so make it work rather than requiring `-m` or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.methods._sampling import run_teacher_residual
from poe_repair.run import make_ctx


def check_canary(pair: str, seed: int, *, steps: int | None = None) -> int:
    """Assert lambda=0 steps with eps_PoE itself. Returns an exit code."""
    cell = cell_from_slug(pair, seed)
    ctx = make_ctx(num_inference_steps=steps) if steps else make_ctx()
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)

    with tempfile.TemporaryDirectory() as tmp:
        res_dir = Path(tmp) / "residuals"
        out = run_teacher_residual(
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
            save_residuals_dir=res_dir, save_x0_estimates=True,
        )
        worst = 0.0
        for f in sorted(res_dir.glob("step_*.pt")):
            sd = torch.load(f, map_location="cpu", weights_only=True)
            expected = sd["eps_poe"].float()
            stepped = out.tracker.velocities[int(sd["step_index"])].float()
            dn = sd["delta"].float().norm().item()
            worst = max(worst, (stepped - expected).norm().item() / max(dn, 1e-12))

    if any(lam != 0.0 for lam in out.extras["lambda_per_step"]):
        print("CANARY FAILED: lambda was nonzero at some step", file=sys.stderr)
        return 1
    if worst < 2e-4:
        print(
            f"canary ok, delta = {worst:.2e} of ||r_t|| (< 2e-4)"
            f"   pair={pair} seed={seed}"
        )
        return 0
    print(
        f"CANARY FAILED: at lambda=0 the sampler stepped with something that is "
        f"{worst:.4%} of ||r_t|| away from eps_PoE. The injection path is not "
        f"inert. Do not trust any dose result.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True, help="pair slug, e.g. a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0, help="dose")
    ap.add_argument("--schedule", default="constant",
                    choices=("constant", "linear_decay", "early_only"))
    ap.add_argument("--check-canary", action="store_true",
                    help="assert lambda=0 reproduces plain PoE, then exit")
    ap.add_argument("--steps", type=int, help="override inference steps (smoke runs)")
    ap.add_argument("--exp-name", default="interaction_term/dose")
    ap.add_argument("--save-residuals", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.check_canary:
        return check_canary(args.pair, args.seed, steps=args.steps)

    cell = cell_from_slug(args.pair, args.seed)
    ctx = make_ctx(num_inference_steps=args.steps) if args.steps else make_ctx()
    path = cmp_tr.run(
        cell, ctx,
        lambda_schedule=args.schedule,
        lambda_max=args.lam,
        save_residuals=args.save_residuals,
        save_trajectory=True,
        exp_name=args.exp_name,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
