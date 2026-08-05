#!/usr/bin/env python
"""Inject the interaction term r_t back into PoE sampling at dose lambda.

A thin command line over ``poe_repair.composers.teacher_residual``, which
already implements

    eps_final = eps_PoE + lambda_t * (eps_Mono - eps_PoE)

This script adds no sampling logic. It exists so plan 03 can drive one
(pair, seed, lambda) cell from the shell.

The canary is the point of interest. At lambda=0 the sampler takes the
``eps_t = eps_poe`` branch directly (see _sampling.py), so the result must be
*bit-exact* against plain PoE, not merely close. If it is not, the injection
path is perturbing the sampler when it should be inert, and every dose number
downstream is meaningless.

Usage:
    python scripts/interaction_term_inject.py --pair a_cat__x__a_dog --seed 9 --lambda 0 --check-canary
    python scripts/interaction_term_inject.py --pair a_cat__x__a_dog --seed 9 --lambda 0.5
"""

from __future__ import annotations

import argparse
import sys

import torch

from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import run_cfg_poe
from poe_repair.composers._helpers import encode_pair, init_latents_for_cell, get_joint_embeds
from poe_repair.methods._sampling import run_teacher_residual
from poe_repair.run import make_ctx


def check_canary(pair: str, seed: int, *, steps: int | None = None) -> int:
    """Assert lambda=0 reproduces plain PoE bit-exactly. Returns an exit code."""
    cell = cell_from_slug(pair, seed)
    ctx = make_ctx(num_inference_steps=steps) if steps else make_ctx()
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)

    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    poe = run_cfg_poe(**common)
    inj = run_teacher_residual(
        **common, seq_j=seq_j, pool_j=pool_j,
        lambda_schedule="constant", lambda_max=0.0,
    )

    delta = (inj.latents.float() - poe.latents.float()).abs().max().item()
    exact = torch.equal(inj.latents, poe.latents)
    if exact:
        print(f"canary ok, delta = 0 (bit-exact)   pair={pair} seed={seed}")
        return 0
    if delta < 1e-5:
        print(
            f"canary ok, delta < 1e-5 (max |diff| = {delta:.3e}) but NOT bit-exact "
            f"  pair={pair} seed={seed}",
            file=sys.stderr,
        )
        return 0
    print(
        f"CANARY FAILED: max |diff| = {delta:.3e} at lambda=0. The injection path "
        f"changes sampling when it should be inert. Do not trust any dose result.",
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
