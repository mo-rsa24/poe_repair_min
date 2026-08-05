#!/usr/bin/env python
"""Inject the interaction term only inside a step window.

A thin command line over ``poe_repair.composers.teacher_residual``, which
already takes ``correction_window=(start, end)`` and forces lambda to 0 outside
it. This script adds no sampling logic.

One trap worth naming, because getting it wrong makes the identity check
vacuous. In the sampler, ``correction_window=None`` means "lambda applies at
every step", which is the *opposite* of off. So here:

    --window off        =>  lambda_max = 0     (nothing injected anywhere)
    --window 10,25      =>  correction_window = (10, 25), lambda outside it 0
    --window all        =>  correction_window = None, lambda everywhere

``--window off --check-identity`` therefore asserts that injecting nothing
reproduces plain PoE bit-exactly.

Usage:
    python scripts/interaction_term_window.py --pair a_cat__x__a_dog --seed 9 --window off --check-identity
    python scripts/interaction_term_window.py --pair a_cat__x__a_dog --seed 9 --window 10,25
"""

from __future__ import annotations

import argparse
import sys

import torch

from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import run_cfg_poe, run_teacher_residual
from poe_repair.run import make_ctx


def parse_window(spec: str) -> tuple[tuple[int, int] | None, float]:
    """Return (correction_window, lambda_max) for a --window value."""
    s = spec.strip().lower()
    if s == "off":
        return None, 0.0            # inject nothing, anywhere
    if s == "all":
        return None, 1.0            # inject everywhere
    try:
        start, end = (int(x) for x in s.split(","))
    except ValueError:
        raise SystemExit(
            f"bad --window {spec!r}: expected 'off', 'all', or 'start,end'"
        )
    if start >= end:
        raise SystemExit(f"bad --window {spec!r}: start must be < end")
    return (start, end), 1.0


def check_identity(pair: str, seed: int, *, steps: int | None = None) -> int:
    """Assert window-off reproduces plain PoE bit-exactly. Returns exit code."""
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
    off = run_teacher_residual(
        **common, seq_j=seq_j, pool_j=pool_j,
        lambda_schedule="constant", lambda_max=0.0,
    )

    if torch.equal(off.latents, poe.latents):
        print(f"window off is byte-identical to PoE   pair={pair} seed={seed}")
        return 0
    delta = (off.latents.float() - poe.latents.float()).abs().max().item()
    print(
        f"IDENTITY FAILED: max |diff| = {delta:.3e} with the window off. "
        f"Windowed timing results cannot be trusted.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True, help="pair slug, e.g. a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--window", required=True,
                    help="'off', 'all', or 'start,end' step indices")
    ap.add_argument("--lambda", dest="lam", type=float,
                    help="dose inside the window (default 1.0)")
    ap.add_argument("--check-identity", action="store_true",
                    help="assert window-off reproduces plain PoE, then exit")
    ap.add_argument("--steps", type=int, help="override inference steps")
    ap.add_argument("--exp-name", default="interaction_term/window")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.check_identity:
        return check_identity(args.pair, args.seed, steps=args.steps)

    window, lam_default = parse_window(args.window)
    lam = args.lam if args.lam is not None else lam_default

    cell = cell_from_slug(args.pair, args.seed)
    ctx = make_ctx(num_inference_steps=args.steps) if args.steps else make_ctx()
    path = cmp_tr.run(
        cell, ctx,
        lambda_schedule="constant",
        lambda_max=lam,
        correction_window=window,
        save_trajectory=True,
        exp_name=args.exp_name,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
