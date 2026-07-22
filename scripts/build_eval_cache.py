"""Build a *minimal* eval-only training-cache cell.

For held-out-pair LoRA evaluation we only need:

- ``meta.json``                       (so ``CellPath.from_root`` finds the cell)
- ``embeddings.pt``                   (prompt embeddings + init_latents + sigma)
- ``residuals/step_000.pt``           (only ``x_t`` is read — eps tensors zeroed
                                       for ``load_step_raw`` shape compat)
- ``poe.png``                         (left panel of the seed-summary figure; optional)
- ``mono.png``                        (left panel of the seed-summary figure; optional)

We skip the per-step ``eps_*`` dumps that ``build_training_cache.py`` writes,
since the LoRA-on-different-pair sampler never reads them. With
``--skip-refs`` we also skip PoE/Mono sampling (~1 s/cell, no GPU
sampling — only prompt encoding + init-latent resolution).

Usage::

    python scripts/build_eval_cache.py \
        --prompt-a "a wolf" --prompt-b "a husky" \
        --joint-prompt "a wolf and a husky" --seed 9

Same env conventions as ``build_training_cache.py``: writes under
``$POE_REPAIR_TRAINING_CACHE/<split>/<pair_slug>/seed_<N>/``.
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
from poe_repair.run import MethodCtx, make_ctx
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
def build_eval_cell(
    *,
    prompt_a: str,
    prompt_b: str,
    joint_prompt_text: str,
    seed: int,
    split: str,
    out_root: Path,
    overwrite: bool,
    skip_refs: bool,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    ctx: MethodCtx | None = None,
) -> Path:
    slug = slugify(prompt_a, prompt_b)
    cell_dir = out_root / split / slug / f"seed_{seed}"
    residuals_dir = cell_dir / "residuals"
    sentinel = cell_dir / "meta.json"
    if sentinel.exists() and not overwrite:
        # Determine completeness vs the requested mode.
        has_refs = (cell_dir / "poe.png").exists() and (cell_dir / "mono.png").exists()
        if skip_refs or has_refs:
            print(f"[cached] {cell_dir}")
            return cell_dir

    if ctx is None:
        ctx = make_ctx(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    ensure_dir(residuals_dir)
    cell = cell_for(prompt_a, prompt_b, seed)

    # --- embeddings + init latent -----------------------------------------
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = encode_prompt_sdxl(
        joint_prompt_text, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)

    seq_a, pool_a = emb["seq_a"], emb["pool_a"]
    seq_b, pool_b = emb["seq_b"], emb["pool_b"]
    seq_e, pool_e = emb["seq_e"], emb["pool_e"]

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

    # --- residuals/step_000.pt (only x_t is read by sample_heldout) -------
    scheduler = ctx.scheduler
    scheduler.set_timesteps(ctx.num_inference_steps)
    timestep0 = int(scheduler.timesteps[0].item())
    x_t0 = (init_latents / euler_sigma).to(save_dtype).cpu()       # (1,4,H,W)
    eps_zero = torch.zeros_like(x_t0)
    torch.save(
        {
            "x_t": x_t0,
            "timestep": timestep0,
            "step_index": 0,
            "eps_a_raw": eps_zero,
            "eps_b_raw": eps_zero,
            "eps_j_raw": eps_zero,
            "eps_uncond": eps_zero,
        },
        residuals_dir / "step_000.pt",
    )

    timesteps_recorded: list[int] = [timestep0]

    # --- PoE + Mono reference renders (optional) --------------------------
    t0 = time.time()
    if not skip_refs:
        unet = ctx.models["unet"]
        branch_pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
        branch_pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
        branch_cond = {
            "text_embeds": branch_pool,
            "time_ids": add_time_ids(
                height=cell.height, width=cell.width, batch_size=4,
                device=ctx.device, dtype=ctx.dtype,
            ),
        }
        mono_pe = torch.cat([seq_j, seq_e], dim=0)
        mono_pool = torch.cat([pool_j, pool_e], dim=0)
        mono_cond = {
            "text_embeds": mono_pool,
            "time_ids": add_time_ids(
                height=cell.height, width=cell.width, batch_size=2,
                device=ctx.device, dtype=ctx.dtype,
            ),
        }
        latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
        mono_latents = latents.clone()
        timesteps_recorded = []
        for step_index, timestep in enumerate(scheduler.timesteps):
            latent_input = scheduler.scale_model_input(
                latents.repeat(4, 1, 1, 1), timestep,
            )
            noise = unet(
                latent_input, timestep, encoder_hidden_states=branch_pe,
                added_cond_kwargs=branch_cond, timestep_cond=None,
            ).sample
            eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)
            eps_a_g = guided_eps(eps_a_raw, eps_uncond, ctx.guidance_scale)
            eps_b_g = guided_eps(eps_b_raw, eps_uncond, ctx.guidance_scale)
            eps_p = poe_eps(eps_a_g, eps_b_g, eps_uncond)

            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=ctx.device, dtype=ctx.dtype,
            )
            x0_poe = tweedie_mean(latents, alpha_bar_t, eps_p)
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
            timesteps_recorded.append(int(timestep.item()))

        write_decoded_image(decode_latents(ctx.models, latents).cpu(),
                            cell_dir / "poe.png")
        write_decoded_image(decode_latents(ctx.models, mono_latents).cpu(),
                            cell_dir / "mono.png")
    elapsed = time.time() - t0

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
        "branch_order": ["a", "b", "j", "uncond"],
        "timesteps": timesteps_recorded,
        "euler_init_noise_sigma": float(euler_sigma),
        "elapsed_seconds": round(elapsed, 2),
        "eval_only": True,
        "has_refs": not skip_refs,
    })

    mode = "skeleton" if skip_refs else "skeleton+refs"
    print(f"[built/{mode}] {cell_dir}  ({elapsed:.1f}s)")
    return cell_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prompt-a", required=True)
    ap.add_argument("--prompt-b", required=True)
    ap.add_argument("--joint-prompt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="heldout", choices=("heldout", "train"))
    ap.add_argument("--output-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--skip-refs", action="store_true",
                    help="don't render poe.png / mono.png (~1s/cell). "
                         "Useful when only init latents are needed; "
                         "the seed-summary figure will have empty "
                         "PoE/Mono panels for this cell.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    build_eval_cell(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        joint_prompt_text=args.joint_prompt,
        seed=args.seed,
        split=args.split,
        out_root=args.output_root,
        overwrite=args.overwrite,
        skip_refs=args.skip_refs,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
