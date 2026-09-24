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
from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
from poe_repair.methods._sampling import add_time_ids, write_decoded_image
from poe_repair.training_cache import CellPath
from poe_repair.rewards.plurality import PluralityReward, soft_masks
from poe_repair.run import make_ctx
from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad

log = logging.getLogger("reward_finetune")

NOISE_PAIR = "a_cat__x__a_dog"          # whose cached starting latent every seed borrows

# The cells every sample pass renders, the same set the pooled training runs track: two pairs the
# adapter trained on and two it did not, so a sheet shows both at once and is comparable with
# theirs. (quadrant, prompt_a, prompt_b, seed)
TRACKING_SET = [
    ("in_in", "a lion", "a meerkat", 1),
    ("in_in", "a typewriter", "a cactus", 1),
    ("out_out", "a cat", "a dog", 9),
    ("out_out", "a cat", "a dog", 10),
    ("out_out", "an elephant", "a penguin", 9),
    ("out_out", "an elephant", "a penguin", 10),
]


def guided_eps(eps_c, eps_u, w):
    return eps_u + w * (eps_c - eps_u)


def poe_eps(eps_a, eps_b, eps_u):
    return eps_a + eps_b - eps_u


def _strip(lora_png: Path, cell, seed: int, prompt_a: str, prompt_b: str, step: int) -> Path | None:
    """The three panels every run is read against: joint prompt, plain product, this adapter.

    The first two come from the cell's own training cache, which stored them when the cell was
    built, so nothing is re-rendered for them.
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        cache = CellPath.from_root(cell.pair_slug, int(seed))
    except Exception as exc:                      # a cell the cache does not hold
        log.warning("no cached references for %s seed %s (%s)", cell.pair_slug, seed, exc)
        return None
    mono, poe = cache.root / "mono.png", cache.root / "poe.png"
    if not (mono.exists() and poe.exists()):
        return None
    T, pad = 512, 28
    canvas = Image.new("RGB", (3 * T, T + pad), "white")
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        f = ImageFont.load_default()
    for i, (path, label) in enumerate(((mono, "joint prompt (target)"), (poe, "plain product"),
                                       (lora_png, f"adapter at reward step {step}"))):
        canvas.paste(Image.open(path).convert("RGB").resize((T, T), Image.LANCZOS), (i * T, pad))
        d.text((i * T + 6, 6), f"{label}", fill="black", font=f)
    out = lora_png.with_name(lora_png.stem + "__strip.png")
    canvas.save(out)
    return out


def _render_cell(ctx, unet, scheduler, a, prompt_a: str, prompt_b: str, seed: int,
                 run_dir: Path, step: int) -> Path:
    """One tracking cell: render it at 50 steps with the adapter, then build its three-panel strip."""
    cell = cell_for(prompt_a, prompt_b, seed)
    emb = encode_pair(cell, ctx)
    init_latents, euler_sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
    with torch.no_grad():
        res = run_lora_langevin_windowed_poe(
            init_latents=init_latents, models=ctx.models, scheduler=scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"], guidance_scale=a.guidance_scale,
            num_inference_steps=a.num_inference_steps, height=cell.height, width=cell.width,
            euler_init_noise_sigma=euler_sigma, device=ctx.device, dtype=ctx.dtype,
            lambda_value=1.0, k=0, c=0.0, corrector_window=None, noise_seed=seed,
            lora_adapter_name="lora", lambda_window=(0, a.num_inference_steps),
            corrector_score="frozen")
    out = run_dir / "samples" / f"step_{step:06d}_{cell.pair_slug}_seed{seed:02d}.png"
    write_decoded_image(res.image, out)
    return _strip(out, cell, seed, prompt_a, prompt_b, step) or out


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
    ap.add_argument("--mask-at", type=int, default=8,
                    help="the step the concept masks are read at. By the late steps the two "
                         "branches agree almost everywhere, so masks taken there cover the whole "
                         "picture and the distinctness term reads a flat zero.")
    ap.add_argument("--w-identity", type=float, default=1.0)
    ap.add_argument("--w-distinct", type=float, default=1.0)
    ap.add_argument("--w-compact", type=float, default=1.0)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute UNet activations in the backward pass; needed on a 24 GB card")
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/datasets/mmolefe/poe_repair_min/outputs/reward_finetune"))
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="a reward checkpoint to continue from; the step counter continues too")
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
        # diffusers checks `self.training and self.gradient_checkpointing`, so in eval mode the
        # flag does nothing at all. The UNet has no batch norm and the adapter has no dropout, so
        # train mode changes nothing about the arithmetic here.
        unet.train()
        log.info("gradient checkpointing on (unet in train mode so it takes effect)")
    start_step = 0
    if a.resume_from is not None:
        prev = torch.load(str(a.resume_from), map_location="cpu", weights_only=False)
        loaded = 0
        with torch.no_grad():
            by_name = dict(unet.named_parameters())
            for n, v in prev["lora_state"].items():
                if n in by_name:
                    by_name[n].copy_(v.to(by_name[n].device, by_name[n].dtype))
                    loaded += 1
        start_step = int(prev.get("step", 0))
        log.info("resumed %d tensors from %s at step %d", loaded, a.resume_from, start_step)

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

    for step in range(start_step + 1, a.steps + 1):
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

        cond1 = {"text_embeds": None,
                 "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=1,
                                          device=ctx.device, dtype=ctx.dtype)}

        def branch_parts(x16, timestep, *, batched: bool):
            """The three guided pieces of one step.

            Batched (one UNet call on three copies) while no gradient is kept, and one call per
            branch on the step that carries the graph: three activations at 1024 do not fit a
            24 GB card at once, and the arithmetic is identical either way.
            """
            if batched:
                latent_input = scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
                noise = unet(latent_input, timestep, encoder_hidden_states=pe3,
                             added_cond_kwargs=cond3, timestep_cond=None).sample
                ea_raw, eb_raw, eu = noise.chunk(3)
            else:
                outs = []
                for seq, pool in ((emb["seq_a"], emb["pool_a"]), (emb["seq_b"], emb["pool_b"]),
                                  (emb["seq_e"], emb["pool_e"])):
                    latent_input = scheduler.scale_model_input(x16, timestep)
                    outs.append(unet(latent_input, timestep, encoder_hidden_states=seq,
                                     added_cond_kwargs={"text_embeds": pool,
                                                        "time_ids": cond1["time_ids"]},
                                     timestep_cond=None).sample)
                ea_raw, eb_raw, eu = outs
            return (guided_eps(ea_raw, eu, a.guidance_scale),
                    guided_eps(eb_raw, eu, a.guidance_scale), eu)

        scheduler.set_timesteps(a.num_inference_steps)
        latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
        k = rng.randrange(a.reward_step_lo, a.reward_step_hi)

        # Sample the run without gradients up to the chosen step: only that step carries the graph.
        mask_a = mask_b = None
        with torch.no_grad():
            for i, timestep in enumerate(scheduler.timesteps):
                if i >= k:
                    break
                ea, eb, eu = branch_parts(latents, timestep, batched=True)
                if i == a.mask_at:
                    ab_m = scheduler.alphas_cumprod[int(timestep.item())].to(ctx.device, torch.float32)
                    def _x0(e):
                        return (latents.float() - (1 - ab_m).sqrt() * e.float()) / ab_m.sqrt()
                    mask_a, mask_b = soft_masks(_x0(poe_eps(ea, eb, eu)), _x0(ea), _x0(eb))
                eps = poe_eps(ea, eb, eu)
                latents = scheduler.step(eps, timestep, latents).prev_sample

        # The sampled part is done; give its memory back before the step that keeps a graph.
        torch.cuda.empty_cache()
        timestep = scheduler.timesteps[k]
        ea, eb, eu = branch_parts(latents, timestep, batched=False)
        eps = poe_eps(ea, eb, eu)
        ab = scheduler.alphas_cumprod[int(timestep.item())].to(ctx.device, torch.float32)
        x0 = (latents.float() - (1 - ab).sqrt() * eps.float()) / ab.sqrt()
        image = decode_latents_with_grad(ctx.models["vae"], x0.to(ctx.dtype))
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
            # The tracking set, as the three panels every other run is read against: the joint
            # prompt this cell was cached with, the plain product it repairs, and this adapter.
            unet.eval()
            for quadrant, pa, pb, sd in TRACKING_SET:
                try:
                    strip = _render_cell(ctx, unet, scheduler, a, pa, pb, sd, run_dir, step)
                except Exception as exc:
                    log.warning("sample %s %s x %s seed %s failed: %s", quadrant, pa, pb, sd, exc)
                    continue
                slug = cell_for(pa, pb, sd).pair_slug
                wandb.log({f"samples/{quadrant}/{slug}/seed_{sd:02d}": wandb.Image(str(strip))},
                          step=step)
            if a.gradient_checkpointing:
                unet.train()

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
