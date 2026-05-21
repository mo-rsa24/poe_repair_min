"""Build a single training-cache cell for the LoRA trainer.

Writes one cell at::

    $POE_REPAIR_TRAINING_CACHE/<split>/<pair_slug>/seed_<N>/
        meta.json
        embeddings.pt           # seq/pool for A, B, J, ∅ + init_latents + euler_sigma
        residuals/step_NNN.pt   # x_t, t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond
        poe.png                 # decoded PoE-trajectory final latent (sanity)
        mono.png                # decoded Mono-trajectory final latent (sanity)

Usage::

    python -m scripts.build_training_cache \
        --prompt-a "a camel" --prompt-b "a desert landscape" \
        --joint-prompt "a camel and a desert landscape" \
        --seed 42 --split heldout

Idempotent: skips if ``residuals/step_{N-1:03d}.pt`` already exists, unless
``--overwrite`` is passed. The pair slug is derived via ``slugify`` from the
two prompts (e.g. "a camel" / "a desert landscape" →
``a_camel__x__a_desert_landscape``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from poe_repair.composers._helpers import (
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.methods._sampling import add_time_ids, write_decoded_image
from poe_repair.run import make_ctx
from poe_repair.runtime import (
    decode_latents,
    ddim_prev_from_x0_eps,
    encode_prompt_sdxl,
    ensure_dir,
    guided_eps,
    poe_eps,
    tweedie_mean,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT


@torch.no_grad()
def build_cell(
    *,
    prompt_a: str,
    prompt_b: str,
    joint_prompt_text: str,
    seed: int,
    split: str,
    out_root: Path,
    overwrite: bool,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
) -> Path:
    slug = slugify(prompt_a, prompt_b)
    cell_dir = out_root / split / slug / f"seed_{seed}"
    residuals_dir = cell_dir / "residuals"

    ctx = make_ctx(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    final_step = ctx.num_inference_steps - 1
    sentinel = residuals_dir / f"step_{final_step:03d}.pt"
    if sentinel.exists() and not overwrite:
        print(f"[cached] {cell_dir}")
        return cell_dir

    ensure_dir(residuals_dir)
    cell = cell_for(prompt_a, prompt_b, seed)

    emb = encode_pair(cell, ctx)
    seq_j, pool_j = encode_prompt_sdxl(
        joint_prompt_text, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)

    seq_a, pool_a = emb["seq_a"], emb["pool_a"]
    seq_b, pool_b = emb["seq_b"], emb["pool_b"]
    seq_e, pool_e = emb["seq_e"], emb["pool_e"]

    branch_order = ["a", "b", "j", "uncond"]
    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=cell.height, width=cell.width, batch_size=4,
            device=ctx.device, dtype=ctx.dtype,
        ),
    }

    save_dtype = torch.float16
    torch.save(
        {
            "seq_a": seq_a.detach().to(save_dtype).cpu(),
            "pool_a": pool_a.detach().to(save_dtype).cpu(),
            "seq_b": seq_b.detach().to(save_dtype).cpu(),
            "pool_b": pool_b.detach().to(save_dtype).cpu(),
            "seq_j": seq_j.detach().to(save_dtype).cpu(),
            "pool_j": pool_j.detach().to(save_dtype).cpu(),
            "seq_uncond": seq_e.detach().to(save_dtype).cpu(),
            "pool_uncond": pool_e.detach().to(save_dtype).cpu(),
            "init_latents": init_latents.detach().to(save_dtype).cpu(),
            "euler_init_noise_sigma": float(euler_sigma),
        },
        cell_dir / "embeddings.pt",
    )

    scheduler = ctx.scheduler
    scheduler.set_timesteps(ctx.num_inference_steps)
    latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
    unet = ctx.models["unet"]

    mono_latents = latents.clone()
    mono_pe = torch.cat([seq_j, seq_e], dim=0)
    mono_pool = torch.cat([pool_j, pool_e], dim=0)
    mono_cond = {
        "text_embeds": mono_pool,
        "time_ids": add_time_ids(
            height=cell.height, width=cell.width, batch_size=2,
            device=ctx.device, dtype=ctx.dtype,
        ),
    }

    timesteps_recorded: list[int] = []
    t0 = time.time()
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(
            latents.repeat(4, 1, 1, 1), timestep,
        )
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)

        eps_a_g = guided_eps(eps_a_raw, eps_uncond, ctx.guidance_scale)
        eps_b_g = guided_eps(eps_b_raw, eps_uncond, ctx.guidance_scale)
        eps_p = poe_eps(eps_a_g, eps_b_g, eps_uncond)

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=ctx.device, dtype=ctx.dtype,
        )
        x0_poe = tweedie_mean(latents, alpha_bar_t, eps_p)

        torch.save(
            {
                "x_t": latents.detach().to(save_dtype).cpu(),
                "timestep": int(timestep.item()),
                "step_index": int(step_index),
                "eps_a_raw": eps_a_raw.detach().to(save_dtype).cpu(),
                "eps_b_raw": eps_b_raw.detach().to(save_dtype).cpu(),
                "eps_j_raw": eps_j_raw.detach().to(save_dtype).cpu(),
                "eps_uncond": eps_uncond.detach().to(save_dtype).cpu(),
            },
            residuals_dir / f"step_{step_index:03d}.pt",
        )
        timesteps_recorded.append(int(timestep.item()))

        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0_poe, eps=eps_p,
        )

        mono_input = scheduler.scale_model_input(
            mono_latents.repeat(2, 1, 1, 1), timestep,
        )
        mono_noise = unet(
            mono_input, timestep, encoder_hidden_states=mono_pe,
            added_cond_kwargs=mono_cond, timestep_cond=None,
        ).sample
        eps_j_mono_raw, eps_uncond_mono = mono_noise.chunk(2)
        eps_j_mono = guided_eps(eps_j_mono_raw, eps_uncond_mono, ctx.guidance_scale)
        x0_mono = tweedie_mean(mono_latents, alpha_bar_t, eps_j_mono)
        mono_latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0_mono, eps=eps_j_mono,
        )

    elapsed = time.time() - t0

    write_decoded_image(decode_latents(ctx.models, latents).cpu(), cell_dir / "poe.png")
    write_decoded_image(decode_latents(ctx.models, mono_latents).cpu(), cell_dir / "mono.png")

    write_json(cell_dir / "meta.json", {
        "pair": [prompt_a, prompt_b],
        "pair_slug": slug,
        "seed": seed,
        "split": split,
        "joint_prompt": joint_prompt_text,
        "guidance_scale": float(ctx.guidance_scale),
        "num_inference_steps": int(ctx.num_inference_steps),
        "height": int(cell.height),
        "width": int(cell.width),
        "branch_order": branch_order,
        "timesteps": timesteps_recorded,
        "euler_init_noise_sigma": float(euler_sigma),
        "elapsed_seconds": round(elapsed, 2),
    })

    print(f"[built] {cell_dir}  ({elapsed:.1f}s)")
    return cell_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prompt-a", required=True)
    ap.add_argument("--prompt-b", required=True)
    ap.add_argument("--joint-prompt", required=True,
                    help="Literal joint text used to compute e_J (e.g. "
                         "'a camel and a desert landscape').")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="heldout", choices=("heldout", "train"))
    ap.add_argument("--output-root", type=Path, default=DEFAULT_CACHE_ROOT,
                    help="Cache root (default: $POE_REPAIR_TRAINING_CACHE).")
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    build_cell(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        joint_prompt_text=args.joint_prompt,
        seed=args.seed,
        split=args.split,
        out_root=args.output_root,
        overwrite=args.overwrite,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
