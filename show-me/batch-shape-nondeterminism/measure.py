#!/usr/bin/env python
"""Evidence for the batch-shape effect that made the lambda=0 canary fail.

Three measurements, each one a separate claim:

  A. One UNet call, identical inputs, batch 3 vs batch 4. Do the shared
     branches come back the same?
  B. The same batch shape run twice. Is the difference in A randomness, or
     is it the batch shape?
  C. Full sampling. Does a per-step difference of that size matter, or does
     it wash out?

Writes a JSON record and a two-panel figure.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import (
    add_time_ids,
    run_cfg_poe,
    run_teacher_residual,
)
from poe_repair.run import make_ctx

OUT = Path("show-me/batch-shape-nondeterminism")
PAIR, SEED, STEPS = "a_cat__x__a_dog", 9, 50


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell = cell_from_slug(PAIR, SEED)
    ctx = make_ctx(num_inference_steps=STEPS)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    unet = ctx.models["unet"]
    lat = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
    ctx.scheduler.set_timesteps(STEPS)
    t = ctx.scheduler.timesteps[0]

    @torch.no_grad()
    def one_call(seqs, pools, n):
        li = ctx.scheduler.scale_model_input(lat.repeat(n, 1, 1, 1), t)
        cond = {
            "text_embeds": torch.cat(pools, 0),
            "time_ids": add_time_ids(height=1024, width=1024, batch_size=n,
                                     device=ctx.device, dtype=ctx.dtype),
        }
        out = unet(li, t, encoder_hidden_states=torch.cat(seqs, 0),
                   added_cond_kwargs=cond, timestep_cond=None).sample
        return out.float().cpu()

    three = ([emb["seq_a"], emb["seq_b"], emb["seq_e"]],
             [emb["pool_a"], emb["pool_b"], emb["pool_e"]])
    four = ([emb["seq_a"], emb["seq_b"], seq_j, emb["seq_e"]],
            [emb["pool_a"], emb["pool_b"], pool_j, emb["pool_e"]])

    # A. batch 3 vs batch 4, comparing only the branches both runs share.
    n3 = one_call(*three, 3)
    gc.collect(); torch.cuda.empty_cache()
    n4 = one_call(*four, 4)
    a3, b3, u3 = n3.chunk(3)
    a4, b4, _, u4 = n4.chunk(4)
    cross = {
        name: {
            "max_abs_diff": float((x - y).abs().max()),
            "mean_abs_diff": float((x - y).abs().mean()),
            "bit_identical": bool(torch.equal(x, y)),
        }
        for name, (x, y) in
        {"eps_a": (a3, a4), "eps_b": (b3, b4), "eps_uncond": (u3, u4)}.items()
    }

    # B. same batch shape, run twice. Isolates batch shape from randomness.
    gc.collect(); torch.cuda.empty_cache()
    n3b = one_call(*three, 3)
    gc.collect(); torch.cuda.empty_cache()
    n4b = one_call(*four, 4)
    repeat = {
        "batch3_twice_bit_identical": bool(torch.equal(n3, n3b)),
        "batch3_twice_max_diff": float((n3 - n3b).abs().max()),
        "batch4_twice_bit_identical": bool(torch.equal(n4, n4b)),
        "batch4_twice_max_diff": float((n4 - n4b).abs().max()),
    }

    # C. does it compound? Full sampling, PoE (batch 3) vs injection at
    # lambda=0 (batch 4). Both should be "plain PoE".
    del n3, n4, n3b, n4b
    gc.collect(); torch.cuda.empty_cache()
    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=STEPS,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    poe = run_cfg_poe(**common)
    gc.collect(); torch.cuda.empty_cache()
    inj0 = run_teacher_residual(**common, seq_j=seq_j, pool_j=pool_j,
                                lambda_schedule="constant", lambda_max=0.0)
    gc.collect(); torch.cuda.empty_cache()
    inj0b = run_teacher_residual(**common, seq_j=seq_j, pool_j=pool_j,
                                 lambda_schedule="constant", lambda_max=0.0)

    tp, ti = poe.tracker.trajectories.float(), inj0.tracker.trajectories.float()
    n = min(tp.shape[0], ti.shape[0])
    drift = [float((tp[s] - ti[s]).abs().max()) for s in range(n)]

    compound = {
        "per_step_max_abs_diff": drift,
        "final_latent_max_abs_diff": float(
            (poe.latents.float() - inj0.latents.float()).abs().max()),
        "same_batch_shape_twice_bit_identical": bool(
            torch.equal(inj0.latents, inj0b.latents)),
    }

    record = {
        "pair": PAIR, "seed": SEED, "steps": STEPS,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(ctx.dtype),
        "A_one_call_batch3_vs_batch4": cross,
        "B_same_shape_repeated": repeat,
        "C_compounding_over_sampling": compound,
    }
    (OUT / "measurements.json").write_text(json.dumps(record, indent=2))

    print("A. one UNet call, identical inputs, batch 3 vs batch 4")
    for k, v in cross.items():
        print(f"   {k:<11} max|diff| = {v['max_abs_diff']:.3e}   "
              f"bit-identical: {v['bit_identical']}")
    print("\nB. same batch shape, run twice")
    print(f"   batch 3: bit-identical = {repeat['batch3_twice_bit_identical']}")
    print(f"   batch 4: bit-identical = {repeat['batch4_twice_bit_identical']}")
    print("\nC. compounding over 50 sampling steps")
    print(f"   step 0:  {drift[0]:.3e}")
    print(f"   step 10: {drift[10]:.3e}")
    print(f"   step 25: {drift[25]:.3e}")
    print(f"   final:   {compound['final_latent_max_abs_diff']:.3e}")
    print(f"   same shape twice, bit-identical: "
          f"{compound['same_batch_shape_twice_bit_identical']}")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    names = list(cross)
    vals = [cross[k]["max_abs_diff"] for k in names]
    bars = ax1.bar(range(len(names)), vals, color="tab:red", width=0.5)
    ax1.bar(range(len(names)), [0] * len(names), color="tab:blue", width=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("max abs difference")
    ax1.set_title("One call: batch 3 vs batch 4")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1e}",
                 ha="center", va="bottom", fontsize=9)
    ax1.text(0.5, 0.90,
             "same batch size, run twice: exactly 0",
             transform=ax1.transAxes, ha="center", fontsize=9, color="tab:blue")
    ax1.set_ylim(0, max(vals) * 1.30)

    ax2.plot(drift, color="tab:red", lw=2)
    ax2.axhline(0, color="tab:blue", lw=2, label="same batch shape twice")
    ax2.set_yscale("symlog", linthresh=1e-4)
    ax2.set_xlabel("denoising step")
    ax2.set_ylabel("max abs difference")
    first_nz = next((v for v in drift if v > 0), 0.0)
    ax2.set_title(f"It compounds: {first_nz:.0e} at step 1 to "
                  f"{compound['final_latent_max_abs_diff']:.2f} at the end")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.suptitle("Same UNet, same inputs: batch size changes the answer")
    fig.tight_layout()
    fig.savefig(OUT / "batch_shape_effect.png", dpi=150)
    print(f"\nfigure: {OUT / 'batch_shape_effect.png'}")


if __name__ == "__main__":
    main()
