"""Attend-and-Excite-equivalent intervention on plain PoE (mechanism study, plan 02).

DRAFT FOR REVIEW — not yet run. This is the independent comparison point plan 03's
headline read depends on: a test-time attention-optimization baseline, so the LoRA's
attention shift has something to be measured against.

What it does, per seed:

  Runs plain PoE (no LoRA adapter — never attached). At the commitment-window steps
  (5..25, from group-a-failure.md `probe.commit_window = (5, 25)`), it does 1-2
  gradient steps on the LATENT (not the UNet weights) to sharpen each concept token's
  cross-attention peak — the Attend-and-Excite move, transplanted onto the PoE
  3-branch marginals:

      L = Σ_tokens  relu( 1 − max_spatial_attn(token) )          (--aae-reduce sum)

  where the tokens are cat@branchA and dog@branchB, the SAME token spec plan 01
  captured (`CAT_DOG_TOKEN_INDICES`). Gradient flows latent → to_q/to_k → softmax
  attention → loss, via `_CrossAttnRecorder(keep_grad=True)`.

  After the (possibly updated) latent, it captures attention in plan 01's schema
  (`step_XXX_token_<key>.pt`), so plan 03 reads AAE and LoRA the same way, and
  decodes the final image for the visual composition label.

Output (mirrors plan 01):
  /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/aae_equiv/<slug>/seed_<N>/
      attn_maps/step_XXX_token_{cat,dog}_branch_poe.pt
      sample_seed_<N>.png
  + aae_equiv/<slug>/capture_manifest.json

Usage::

    # smoke test one seed first (mandatory GPU preflight)
    python -m poe_repair.experiments.mechanism_study.aae_intervene --seeds 9

    # full run
    python -m poe_repair.experiments.mechanism_study.aae_intervene --seeds 1-12
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import (
    CAT_DOG_TOKEN_INDICES,
    _parse_seeds,
)
from poe_repair.methods._sampling import (
    _CrossAttnRecorder,
    add_time_ids,
    write_decoded_image,
)
from poe_repair.runtime import (
    decode_latents,
    ddim_prev_from_x0_eps,
    ensure_dir,
    guided_eps,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    poe_eps,
    tweedie_mean,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

log = logging.getLogger(__name__)

# Same shared /datasets tree plan 01 pinned its captures to.
DEFAULT_ATTN_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism")

# Commitment window (inclusive) from group-a-failure.md `probe.commit_window = (5, 25)`,
# with anchors (7, 15, 22) inside it. The intervention only fires on these steps —
# outside the window the trajectory is either not yet committed or no longer correctable.
COMMIT_WINDOW = (5, 25)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aae_intervene")
    ap.add_argument("--seeds", default="1-12", help="e.g. '1-12' or '1,4,9'")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--regime-name", default="aae_equiv")
    # --- intervention knobs (the parts most worth reviewing) ------------------
    ap.add_argument("--commit-lo", type=int, default=COMMIT_WINDOW[0],
                    help="first step (inclusive) the intervention fires on")
    ap.add_argument("--commit-hi", type=int, default=COMMIT_WINDOW[1],
                    help="last step (inclusive) the intervention fires on")
    ap.add_argument("--grad-steps", type=int, default=1,
                    help="gradient steps on the latent per window step (AAE uses 1-2)")
    ap.add_argument("--aae-step-size", type=float, default=20.0,
                    help="latent update scale: x_t <- x_t - step_size * grad")
    ap.add_argument("--aae-reduce", choices=("sum", "max"), default="sum",
                    help="combine per-token losses (plan spec = sum)")
    ap.add_argument("--aae-renorm-tokens", type=int, default=4,
                    help="softmax-renorm the steering map over the first N real "
                         "words so 'peak' is a share in [0,1] (AAE canon). "
                         "For solo 'a cat'/'a dog' prompts use 4.")
    # --- capture knobs: keep these matching plan 01's LoRA capture --------------
    ap.add_argument("--attn-resolution", type=int, default=32)
    ap.add_argument("--capture-renorm-tokens", type=int, default=None,
                    help="renorm for the SAVED maps; default None matches "
                         "lora_lambda1 so plan 03 compares like-for-like")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    return ap


def _agg(rec: _CrossAttnRecorder, spec: dict, res: int, *, renorm, keep_grad):
    """One token's aggregated spatial map, plan-01 aggregation settings."""
    return rec.aggregate_token_map(
        int(spec["token_index"]),
        target_hw=(res, res),
        branch_index=int(spec["branch_index"]),
        agg_resolution=res,
        text_token_count=renorm,
        drop_bos=(renorm is not None),
        keep_grad=keep_grad,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("MECH_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)
    seeds = _parse_seeds(p.seeds)
    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    res = int(p.attn_resolution)
    window = range(int(p.commit_lo), int(p.commit_hi) + 1)
    log.info("device=%s dtype=%s regime=%s window=%s grad_steps=%d step=%g seeds=%s",
             device, dtype, p.regime_name, (p.commit_lo, p.commit_hi),
             p.grad_steps, p.aae_step_size, seeds)

    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)
    unet = models["unet"]
    unet.eval()  # no LoRA adapter is ever attached: this is plain PoE

    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt

    class _CfgShim:
        cell = _PromptShim()

    emb = encode_all_prompts(_CfgShim(), models, device, dtype)
    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT
    out_root = (Path(p.out_root) if p.out_root
                else DEFAULT_ATTN_ROOT / p.regime_name / p.pair_slug)

    # 3-branch (A, B, ∅) conditioning, identical to run_lora_residual_inject.
    pe_3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0)
    pool_3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0)
    cond_3 = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(height=int(p.height), width=int(p.width),
                                 batch_size=3, device=device, dtype=dtype),
    }

    def _forward_branches(latents, timestep, *, recorder):
        """One 3-branch forward under `recorder`; returns (eps_a, eps_b, eps_∅)."""
        latent_in = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        with recorder as rec:
            noise = unet(latent_in, timestep, encoder_hidden_states=pe_3,
                         added_cond_kwargs=cond_3, timestep_cond=None).sample
        return rec, noise.chunk(3)

    def _aae_update(latents, timestep):
        """1-2 latent gradient steps to sharpen each concept token's peak."""
        latents = latents.detach().requires_grad_(True)
        last_loss = None
        for _ in range(int(p.grad_steps)):
            latent_in = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
            with _CrossAttnRecorder(unet, keep_grad=True) as rec:
                unet(latent_in, timestep, encoder_hidden_states=pe_3,
                     added_cond_kwargs=cond_3, timestep_cond=None)
                terms = []
                for spec in CAT_DOG_TOKEN_INDICES.values():
                    amap = _agg(rec, spec, res, renorm=p.aae_renorm_tokens, keep_grad=True)
                    if amap is not None:
                        terms.append(torch.relu(1.0 - amap.max()))
                if not terms:
                    break
                stacked = torch.stack(terms)
                loss = stacked.sum() if p.aae_reduce == "sum" else stacked.max()
            grad = torch.autograd.grad(loss, latents)[0]
            latents = (latents - float(p.aae_step_size) * grad).detach().requires_grad_(True)
            last_loss = float(loss.detach())
        return latents.detach(), last_loss

    manifest: list[dict] = []
    for s in seeds:
        seed_dir = ensure_dir(out_root / f"seed_{int(s)}")
        attn_dir = ensure_dir(seed_dir / "attn_maps")
        cell = CellPath.from_root(p.pair_slug, int(s), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(p.euler_sigma),
        )
        scheduler.set_timesteps(int(p.num_inference_steps))
        latents = (init / float(p.euler_sigma)).to(device=device, dtype=dtype)
        loss_trace: list[float | None] = []

        for step_index, timestep in enumerate(scheduler.timesteps):
            # --- intervention (only inside the commitment window) -------------
            if step_index in window:
                with torch.enable_grad():
                    latents, lstep = _aae_update(latents, timestep)
                loss_trace.append(lstep)
            else:
                loss_trace.append(None)

            # --- capture (plan-01 schema) + eps for the DDIM step -------------
            with torch.no_grad():
                rec, (eps_a_raw, eps_b_raw, eps_uncond) = _forward_branches(
                    latents, timestep,
                    recorder=_CrossAttnRecorder(unet, keep_grad=False),
                )
                for key, spec in CAT_DOG_TOKEN_INDICES.items():
                    amap = _agg(rec, spec, res, renorm=p.capture_renorm_tokens,
                                keep_grad=False)
                    if amap is None:
                        continue
                    torch.save(
                        {"map": amap.float().cpu(), "spec": dict(spec),
                         "step_index": int(step_index), "timestep": int(timestep.item())},
                        attn_dir / f"step_{step_index:03d}_token_{key}.pt",
                    )
                eps_a = guided_eps(eps_a_raw, eps_uncond, float(p.guidance_scale))
                eps_b = guided_eps(eps_b_raw, eps_uncond, float(p.guidance_scale))
                eps_poe = poe_eps(eps_a, eps_b, eps_uncond)

                # --- DDIM step -----------------------------------------------
                alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                    device=device, dtype=dtype)
                x0 = tweedie_mean(latents, alpha_bar_t, eps_poe)
                latents = ddim_prev_from_x0_eps(
                    scheduler=scheduler, timestep=timestep,
                    step_index=step_index, x0=x0, eps=eps_poe,
                )

        with torch.no_grad():
            image = decode_latents(models, latents).cpu()
        write_decoded_image(image, seed_dir / f"sample_seed_{int(s)}.png")
        n_files = len(list(attn_dir.glob("step_*_token_*.pt")))
        fired = [i for i, v in enumerate(loss_trace) if v is not None]
        manifest.append({
            "seed": int(s), "attn_dir": str(attn_dir), "n_attn_files": n_files,
            "window_steps": [fired[0], fired[-1]] if fired else [],
            "loss_first": next((v for v in loss_trace if v is not None), None),
            "loss_last": next((v for v in reversed(loss_trace) if v is not None), None),
        })
        log.info("seed=%d wrote %d attn files; loss %.4g → %.4g over %d window steps",
                 int(s), n_files,
                 manifest[-1]["loss_first"] or float("nan"),
                 manifest[-1]["loss_last"] or float("nan"), len(fired))

    write_json(out_root / "capture_manifest.json", {
        "regime": p.regime_name, "pair_slug": p.pair_slug, "method": "aae_equiv",
        "commit_window": [int(p.commit_lo), int(p.commit_hi)],
        "grad_steps": int(p.grad_steps), "aae_step_size": float(p.aae_step_size),
        "aae_reduce": p.aae_reduce, "aae_renorm_tokens": p.aae_renorm_tokens,
        "capture_renorm_tokens": p.capture_renorm_tokens,
        "token_indices": CAT_DOG_TOKEN_INDICES,
        "attn_resolution": res, "num_inference_steps": int(p.num_inference_steps),
        "guidance_scale": float(p.guidance_scale), "seeds": seeds, "samples": manifest,
    })
    log.info("done — %d seeds, manifest at %s", len(seeds),
             out_root / "capture_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
