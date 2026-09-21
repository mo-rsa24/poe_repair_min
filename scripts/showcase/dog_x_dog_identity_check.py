#!/usr/bin/env python
"""Decide what "Mono(A)" means for the dog x dog same-prompt check's identity gate.

Plan 02 (plans/01-showcase-the-trained-lora/plans/tests/02-the-dog-x-dog-same-prompt-check.md)
pre-registers "PoE(A,A) reduces to Mono(A) within fp16 drift" as the gate before anything else
in that plan is trusted. Tracing the formulas in poe_repair/methods/_sampling.py shows this
cannot hold literally: eps_PoE = eps_tilde_A + eps_tilde_B - eps_empty, so at A=B it applies the
full guidance term twice, while a single CFG branch (Mono, at any prompt) applies it once. The
gap is the guidance term itself, not floating-point noise.

This script settles which reference actually matches, empirically, rather than by algebra alone,
by comparing eps_PoE("a dog","a dog") against three Mono candidates at the FIRST denoising step
only (all three start from the same x_T, so step 0 isolates the formula difference from any
downstream trajectory drift):

    A. Mono(A) at the same guidance scale as each PoE branch (7.5) — the literal reading of the
       pre-registered claim.
    B. Mono(A) at double guidance (15.0) — what the algebra predicts will match exactly.
    C. Mono via the project's own canonical "mono" method (poe_repair/run.py run_method):
       single CFG branch on the literal joint prompt ("a dog and a dog" via joint_template),
       at the same guidance scale (7.5) — what compose-rate scoring actually treats as Mono
       everywhere else in this project.

Whichever candidate lands within fp16 drift (~1e-3, per this project's established fp16 drift
band) becomes the identity check plan 02 task 1.2 runs for real, across the full 50-step
trajectory and the held-out seeds.

Usage:
    python scripts/showcase/dog_x_dog_identity_check.py --seed 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.methods._sampling import add_time_ids, initial_latents_for_pair
from poe_repair.run import make_ctx
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl


def guided_eps(eps_cond: torch.Tensor, eps_uncond: torch.Tensor, w: float) -> torch.Tensor:
    return eps_uncond + w * (eps_cond - eps_uncond)


def poe_eps(eps_a: torch.Tensor, eps_b: torch.Tensor, eps_uncond: torch.Tensor) -> torch.Tensor:
    return eps_a + eps_b - eps_uncond


def _diff(name: str, candidate: torch.Tensor, reference: torch.Tensor, ref_norm: float) -> dict:
    d = (candidate.float() - reference.float())
    max_abs = float(d.abs().max().item())
    rel = float(d.norm().item()) / ref_norm
    return {"name": name, "max_abs_diff": max_abs, "rel_l2_diff": rel}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9, help="held-out seed (default: first of phase1_r8_100k's held_out pool)")
    ap.add_argument("--prompt", default="a dog")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--out", default=None, help="write the result JSON here (default: printed only)")
    args = ap.parse_args(argv)

    cfg = RunConfig()
    ctx = make_ctx(guidance_scale=args.guidance_scale)

    cell = PairSeedCell(
        pair_dir=cfg.paths.pilot_dir / f"seed_{args.seed}" / "dog_x_dog_identity_check",
        pair_slug="dog_x_dog_identity_check",
        prompt_a=args.prompt, prompt_b=args.prompt,
        seed=args.seed, regime="collision",
        height=1024, width=1024,
        grid_assets={},
    )
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)

    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timestep = ctx.scheduler.timesteps[0]

    joint = joint_prompt(args.prompt, args.prompt, template=ctx.joint_template)
    seq_a, pool_a = encode_prompt_sdxl(args.prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_j, pool_j = encode_prompt_sdxl(joint, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    # Batch the three distinct branches (A, joint, empty) into one UNet call.
    pe = torch.cat([seq_a, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_j, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(height=1024, width=1024, batch_size=3, device=ctx.device, dtype=ctx.dtype),
    }
    latent_input = ctx.scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
    with torch.no_grad():
        noise = ctx.models["unet"](
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
    eps_a_raw, eps_j_raw, eps_uncond_raw = noise.chunk(3)

    w = args.guidance_scale
    eps_tilde_a_w = guided_eps(eps_a_raw, eps_uncond_raw, w)
    eps_tilde_a_2w = guided_eps(eps_a_raw, eps_uncond_raw, 2.0 * w)
    eps_tilde_joint_w = guided_eps(eps_j_raw, eps_uncond_raw, w)

    eps_poe_aa = poe_eps(eps_tilde_a_w, eps_tilde_a_w, eps_uncond_raw)
    ref_norm = float(eps_poe_aa.float().norm().item())

    candidates = [
        _diff("mono_A_same_guidance (literal reading, w=%.1f)" % w, eps_tilde_a_w, eps_poe_aa, ref_norm),
        _diff("mono_A_double_guidance (w=%.1f)" % (2 * w), eps_tilde_a_2w, eps_poe_aa, ref_norm),
        _diff("mono_joint_prompt (project's canonical mono, w=%.1f)" % w, eps_tilde_joint_w, eps_poe_aa, ref_norm),
    ]

    result = {
        "seed": args.seed,
        "prompt": args.prompt,
        "joint_prompt": joint,
        "guidance_scale": w,
        "timestep": int(timestep.item()),
        "eps_poe_AA_norm": ref_norm,
        "candidates": candidates,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
