#!/usr/bin/env python
"""Fine-tune an existing adapter against the plurality objective, ReFL style.

Every adapter in this project is trained to match a cached joint-prompt prediction, so its ceiling
is that target: where the joint prompt draws two dogs, the adapter learns to draw two dogs. This
trainer optimises the picture instead. It samples a corrected run with the adapter, decodes the
clean picture a late step is heading for, scores it with
`poe_repair/rewards/plurality.py` (each region its own concept, the two regions unlike each other,
each concept in one place) and backpropagates that score into the adapter's weights. The base model
stays frozen, as everywhere else.

ReFL rather than DRaFT-K to begin with: the trajectory is sampled without gradients up to a
randomly chosen late step, and only that one step carries the graph. That keeps the memory of a
single decode rather than of a whole run, and it is the cheapest version that can work. DRaFT-K,
which keeps the last K steps, is the next thing to try if one step proves too weak a signal.

    python -m poe_repair.experiments.reward_finetune.train \\
        --checkpoint <lora_step_015000.pt> --cells <cells.json> --steps 2000

What it writes, under `--out-root/<run-id>/`: `checkpoints/reward_step_*.pt` in the same shape the
sampler already loads, `samples/` with a render per evaluation pass, and its W&B run.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
from poe_repair.experiments import _adapter_shape
from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import add_time_ids, write_decoded_image
from poe_repair.rewards.plurality import PluralityReward, soft_masks
from poe_repair.run import make_ctx
from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad

log = logging.getLogger("reward_finetune")

NOISE_PAIR = "a_cat__x__a_dog"          # whose cached starting latent every seed borrows


def guided_eps(eps_c, eps_u, w):
    return eps_u + w * (eps_c - eps_u)


def poe_eps(eps_a, eps_b, eps_u):
    return eps_a + eps_b - eps_u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="the adapter to start from; its shape is read from the file")
    ap.add_argument("--pairs", nargs="+", default=["a cat|a dog", "an elephant|a penguin"],
                    help='each as "a cat|a dog"; the cells this fine-tune optimises over')
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="small: this starts from a trained adapter and should not undo it")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--reward-step-lo", type=int, default=30,
                    help="the graph-carrying step is drawn from [lo, hi); late steps decode to a "
                         "picture the reward can actually read")
    ap.add_argument("--reward-step-hi", type=int, default=45)
    ap.add_argument("--w-identity", type=float, default=1.0)
    ap.add_argument("--w-distinct", type=float, default=1.0)
    ap.add_argument("--w-compact", type=float, default=1.0)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute UNet activations in the backward pass; needed on a 24 GB card")
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/datasets/mmolefe/poe_repair_min/outputs/reward_finetune"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-project", default="poe-repair-animals-compose")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    run_id = a.run_id or f"reward_{int(time.time())}"
    run_dir = a.out_root / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)

    ctx = make_ctx(num_inference_steps=a.num_inference_steps, guidance_scale=a.guidance_scale)
    unet = ctx.models["unet"]
    info = _adapter_shape.attach(unet, a.checkpoint, adapter_name="lora")
    log.info("adapter: %s modules at rank %s from step %s", info["n_matched"], info["rank"],
             info["checkpoint_step"])
    (run_dir / "adapter.json").write_text(json.dumps({**info, "args": vars(a)}, indent=1,
                                                     default=str))

    if a.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        log.info("gradient checkpointing on")
    lora_params = [p for n, p in unet.named_parameters() if "lora_" in n]
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)
    opt = torch.optim.AdamW(lora_params, lr=a.lr)
    reward = PluralityReward(ctx.device, w_identity=a.w_identity, w_distinct=a.w_distinct,
                             w_compact=a.w_compact)

    import wandb
    wandb.init(project=a.wandb_project, name=run_id, mode=a.wandb_mode,
               config={**vars(a), "adapter": info}, tags=["reward", "refl"])

    pairs = [tuple(x.strip() for x in p.split("|")) for p in a.pairs]
    rng = random.Random(0)
    scheduler = ctx.scheduler
    t0 = time.perf_counter()

    for step in range(1, a.steps + 1):
        prompt_a, prompt_b = rng.choice(pairs)
        seed = rng.choice(a.seeds)
        cell = cell_for(prompt_a, prompt_b, seed)
        emb = encode_pair(cell, ctx)
        init_latents, euler_sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)

        pe3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0)
        pool3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0)
        cond3 = {"text_embeds": pool3,
                 "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=3,
                                          device=ctx.device, dtype=ctx.dtype)}

        def branch_parts(x16, timestep):
            latent_input = scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
            noise = unet(latent_input, timestep, encoder_hidden_states=pe3,
                         added_cond_kwargs=cond3, timestep_cond=None).sample
            ea_raw, eb_raw, eu = noise.chunk(3)
            return (guided_eps(ea_raw, eu, a.guidance_scale),
                    guided_eps(eb_raw, eu, a.guidance_scale), eu)

        scheduler.set_timesteps(a.num_inference_steps)
        latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
        k = rng.randrange(a.reward_step_lo, a.reward_step_hi)

        # Sample the run without gradients up to the chosen step: only that step carries the graph.
        with torch.no_grad():
            for i, timestep in enumerate(scheduler.timesteps):
                if i >= k:
                    break
                ea, eb, eu = branch_parts(latents, timestep)
                eps = poe_eps(ea, eb, eu)
                latents = scheduler.step(eps, timestep, latents).prev_sample

        timestep = scheduler.timesteps[k]
        ea, eb, eu = branch_parts(latents, timestep)
        eps = poe_eps(ea, eb, eu)
        ab = scheduler.alphas_cumprod[int(timestep.item())].to(ctx.device, torch.float32)
        x0 = (latents.float() - (1 - ab).sqrt() * eps.float()) / ab.sqrt()
        image = decode_latents_with_grad(ctx.models["vae"], x0.to(ctx.dtype))
        with torch.no_grad():
            x0_a = (latents.float() - (1 - ab).sqrt() * ea.float()) / ab.sqrt()
            x0_b = (latents.float() - (1 - ab).sqrt() * eb.float()) / ab.sqrt()
            mask_a, mask_b = soft_masks(x0_a, x0_b)
        total, terms = reward.score(image, prompt_a=prompt_a, prompt_b=prompt_b,
                                    mask_a=mask_a, mask_b=mask_b)

        loss = -total                      # the optimizer minimises; the objective is maximised
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(lora_params, a.grad_clip)
        opt.step()

        payload = {"reward/total": terms.total, "reward/identity": terms.identity,
                   "reward/distinct": terms.distinct, "reward/compact": terms.compact,
                   "train/grad_norm": float(gnorm), "train/step_sampled": k,
                   "train/pair": f"{prompt_a} | {prompt_b}", "train/seed": seed,
                   "train/elapsed_s": time.perf_counter() - t0}
        wandb.log(payload, step=step)
        if step % 20 == 0:
            log.info("step %d/%d reward=%.4f (identity %.3f distinct %.3f compact %.3f) k=%d",
                     step, a.steps, terms.total, terms.identity, terms.distinct, terms.compact, k)

        if step % a.sample_every == 0:
            with torch.no_grad():
                out = run_dir / "samples" / f"step_{step:06d}_{cell.pair_slug}_seed{seed:02d}.png"
                write_decoded_image(image.detach().cpu(), out)
            wandb.log({"samples/latest": wandb.Image(str(out))}, step=step)

        if step % a.ckpt_every == 0 or step == a.steps:
            state = {n: p.detach().cpu().clone() for n, p in unet.named_parameters()
                     if "lora_" in n}
            torch.save({"lora_state": state, "step": step,
                        "from_checkpoint": str(a.checkpoint)},
                       run_dir / "checkpoints" / f"reward_step_{step:06d}.pt")
            log.info("checkpoint saved at step %d", step)

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
