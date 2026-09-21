"""Apply one fix to the adapter at inference, and save what it draws.

The adapter is trained to predict a correction, and nothing in that target says which picture
should come out. So the picture is an accident of the fit: at fixed starting noise it jumps between
a fused animal, a cat and a dog, and two cats as training goes on, while the loss falls smoothly.
Each fix here tries to stop that jumping, and none of them may use the joint prompt.

  --fix no-fix        the adapter on its own, the control
      cat-boost       raise how much attention the cat word gets
      dog-boost       the same for the dog word
      both-boost      raise whichever is weaker; Attend-and-Excite as published
      push-apart      make the two words' attention cover different pixels
  --correction 1.5    apply more of the adapter's own correction than usual
  --co3-steps 1       run CO3's corrector, with the adapter standing in for the joint prompt

--strength is how hard a boost or push pushes. 20 is the published setting and changes nothing
here; cat-boost needs about 50 and breaks past 200.

Why push-apart exists: Attend-and-Excite raises a subject that lost a competition inside one shared
prompt. Here the cat and the dog have separate prompts in separate branches, so neither is starved
and peak attention already starts near 0.67. The fused animal is both branches painting the same
pixels. That overlap reads 0.97 when a render fuses and 0.77 on a pair that composes 8 of 8.

Why CO3 fits: CO3 mixes Tweedie means as +2 on the joint and -(w-1)/K on each single concept, so it
needs a joint score this setting never has. The adapter's corrected prediction is a learned estimate
of exactly that, so it drops into the slot and no joint prompt is encoded.

The sampler is the per-arm composition of ``run_lora_residual_inject``, repeated here so the latent
can be changed mid-step: two forwards over (cat, dog, unconditional), adapter off then on, and
``eps = eps_off + correction * (eps_on - eps_off)``.

  python -m poe_repair.experiments.mechanism_study.guide_lora_attention \
      --checkpoint <lora_step_030000.pt> --fix cat-boost --strength 50 --seeds 10
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import (
    CAT_DOG_TOKEN_INDICES,
    _maybe_attach_lora,
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
from poe_repair import paths

log = logging.getLogger(__name__)

DEFAULT_ATTN_ROOT = paths.resolve(paths.ATTENTION_MECHANISM)
COMMIT_WINDOW = (5, 25)
# The names here are the names used in the write-up. Old names still work so earlier commands and
# render directories keep resolving.
FIXES = ("no-fix", "cat-boost", "dog-boost", "both-boost", "push-apart")
_OLD = {"none": "no-fix", "excite_cat": "cat-boost", "excite_dog": "dog-boost",
        "excite_both": "both-boost", "separate": "push-apart"}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="lora_step_*.pt from train_pooled")
    ap.add_argument("--fix", "--loss", dest="fix", choices=FIXES + tuple(_OLD), default="no-fix",
                    help="which fix to apply; 'no-fix' is the adapter on its own")
    ap.add_argument("--seeds", default="9-16")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    # Never consumed: the sampler below reads only branches A, B and the unconditional. It is
    # accepted so the prompt-encoding helper's signature is satisfied and nothing else.
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--correction", "--lambda-value", dest="correction", type=float, default=1.0,
                    help="how much of the adapter's own correction to apply. 1 is normal, 0 is the "
                         "product alone, 1.5 composes more often and drifts toward illustration")
    ap.add_argument("--commit-lo", type=int, default=COMMIT_WINDOW[0])
    ap.add_argument("--commit-hi", type=int, default=COMMIT_WINDOW[1])
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--strength", "--step-size", dest="strength", type=float, default=20.0,
                    help="how hard the fix pushes. 20 is the published Attend-and-Excite setting; "
                         "cat-boost needs about 50 to change the picture and breaks past 200")
    ap.add_argument("--renorm-tokens", type=int, default=4,
                    help="Attend-and-Excite softmax renorm over [1:N]; 4 suits 'a cat'")
    ap.add_argument("--attn-resolution", type=int, default=16,
                    help="aggregate each layer to this square; 16 is the published setting")
    # CO3 (Dutta et al., arXiv 2509.25940), with the adapter standing in for the joint prompt.
    ap.add_argument("--co3-steps", type=int, default=0,
                    help="CO3 corrector iterations per step; 0 disables CO3 entirely")
    ap.add_argument("--co3-until", type=int, default=10,
                    help="apply the CO3 corrector on step indices below this (CO3 corrects the "
                         "highest-noise steps, where the modes have not yet collided)")
    ap.add_argument("--co3-renoise", choices=("uncond", "multi"), default="uncond",
                    help="which prediction re-noises the composed Tweedie. The reference uses the "
                         "unconditional one and carries the joint as a commented alternative; at "
                         "high noise that term dominates the latent, so it is the main suspect for "
                         "renders collapsing to flat silhouettes.")
    ap.add_argument("--co3-renorm", choices=("reference", "off"), default="reference",
                    help="'reference' reproduces CO3's rescale-to-joint-max then match-latent-norm. "
                         "Those are calibrated for a single joint prediction; the adapter's joint "
                         "estimate is a PoE combination on a different scale, so 'off' skips both "
                         "and steps on the composed Tweedie directly.")
    ap.add_argument("--co3-multi-weight", type=float, default=2.0,
                    help="weight on the joint term; the K concepts share -(w-1) so the weights sum "
                         "to 1. w=1 is the plain adapter, w=2 is CO3 as published. Between them it "
                         "trades composition against how far the extrapolation leaves the image "
                         "manifold, which is what turns renders into flat silhouettes.")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--regime-name", default=None, help="default: guide_<loss>")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    ap.add_argument("--cache-root", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("MECH_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)
    seeds = _parse_seeds(p.seeds)
    device, dtype = infer_device(p.device), infer_dtype(p.dtype, infer_device(p.device))
    res = int(p.attn_resolution)
    window = range(int(p.commit_lo), int(p.commit_hi) + 1)
    p.fix = _OLD.get(p.fix, p.fix)
    regime = p.regime_name or f"guide_{p.fix}"
    log.info("device=%s dtype=%s regime=%s loss=%s window=%s grad_steps=%d step=%g lambda=%g",
             device, dtype, regime, p.fix, (p.commit_lo, p.commit_hi),
             p.grad_steps, p.strength, p.correction)

    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)
    unet = models["unet"]
    unet.eval()
    adapter_name = _maybe_attach_lora(unet, p.checkpoint)
    log.info("adapter %r attached from %s", adapter_name, p.checkpoint)

    class _PromptShim:
        prompt_a, prompt_b, joint_prompt = p.prompt_a, p.prompt_b, p.joint_prompt

    class _CfgShim:
        cell = _PromptShim()

    emb = encode_all_prompts(_CfgShim(), models, device, dtype)
    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT
    out_root = Path(p.out_root) if p.out_root else DEFAULT_ATTN_ROOT / regime / p.pair_slug

    # Branches (A, B, uncond). The joint embedding is deliberately absent from this cat.
    pe_3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0)
    pool_3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0)
    cond_3 = {"text_embeds": pool_3,
              "time_ids": add_time_ids(height=int(p.height), width=int(p.width),
                                       batch_size=3, device=device, dtype=dtype)}

    def _adapter(on: bool):
        (unet.enable_adapters if on else unet.disable_adapters)()

    def _forward(latents, timestep, *, recorder):
        latent_in = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        with recorder as rec:
            noise = unet(latent_in, timestep, encoder_hidden_states=pe_3,
                         added_cond_kwargs=cond_3, timestep_cond=None).sample
        return rec, noise.chunk(3)

    def _token_map(rec, key, *, keep_grad):
        spec = CAT_DOG_TOKEN_INDICES[key]
        return rec.aggregate_token_map(
            int(spec["token_index"]), target_hw=(res, res),
            branch_index=int(spec["branch_index"]), agg_resolution=res,
            text_token_count=p.renorm_tokens, drop_bos=True, keep_grad=keep_grad,
        )

    def _loss_from(rec):
        """The chosen attention loss, or None when a map is unavailable."""
        cat = _token_map(rec, "cat_branch_poe", keep_grad=True)
        dog = _token_map(rec, "dog_branch_poe", keep_grad=True)
        if cat is None or dog is None:
            return None
        if p.fix == "cat-boost":
            return torch.relu(1.0 - cat.max())
        if p.fix == "dog-boost":
            return torch.relu(1.0 - dog.max())
        if p.fix == "both-boost":
            return torch.stack([torch.relu(1.0 - cat.max()),
                                torch.relu(1.0 - dog.max())]).max()
        # separate: the two maps as spatial distributions, penalise their shared mass
        c = cat.clamp_min(0) / (cat.clamp_min(0).sum() + 1e-8)
        d = dog.clamp_min(0) / (dog.clamp_min(0).sum() + 1e-8)
        return torch.minimum(c, d).sum()

    def _update(latents, timestep):
        """Gradient steps on the latent against the adapter's own attention."""
        latents = latents.detach().requires_grad_(True)
        last = None
        for _ in range(int(p.grad_steps)):
            _adapter(True)          # guide the corrected forward, not the frozen one
            with _CrossAttnRecorder(unet, keep_grad=True) as rec:
                latent_in = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
                unet(latent_in, timestep, encoder_hidden_states=pe_3,
                     added_cond_kwargs=cond_3, timestep_cond=None)
                loss = _loss_from(rec)
                if loss is None:
                    break
            grad = torch.autograd.grad(loss, latents)[0]
            latents = (latents - float(p.strength) * grad).detach().requires_grad_(True)
            last = float(loss.detach())
        return latents.detach(), last

    def _co3_correct(latents, timestep):
        """CO3's contrastive-Tweedie corrector, with the adapter in place of the joint prompt.

        CO3 samples from p(x|C)^2 / prod_k p(x|c_k)^(1/K), which it realises as a weighted sum of
        Tweedie means: +2 on the joint, -1/K on each concept. It needs the joint score, which this
        setting never has. The adapter supplies it: eps_poe_frozen + delta_hat is exactly a learned
        estimate of the joint's prediction, so it substitutes into the joint slot and the joint
        prompt is still never encoded.

        Guidance is applied once, here, rather than inside the composition as the reference does,
        because the adapter's output is already a guided PoE combination. The algebra is otherwise
        the reference's: normalise the composed Tweedie to the joint's own scale, re-noise with the
        unconditional prediction, and match the latent's norm.
        """
        at = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        sig = (1 - at).sqrt()
        # The weights must sum to 1 or the composed Tweedie is scaled wrongly. CO3 publishes
        # +2 on the joint with the K concepts sharing -1, so in general the concepts share
        # -(w - 1). w = 1 is the plain adapter with no contrast; w = 2 is CO3 as published.
        k_w = -(float(p.co3_multi_weight) - 1.0) / 2.0
        for _ in range(int(p.co3_steps)):
            _adapter(False)
            _, (a_f, b_f, e_f) = _forward(
                latents, timestep, recorder=_CrossAttnRecorder(unet, keep_grad=False))
            g_a = guided_eps(a_f, e_f, float(p.guidance_scale))
            g_b = guided_eps(b_f, e_f, float(p.guidance_scale))
            eps_frozen = poe_eps(g_a, g_b, e_f)
            _adapter(True)
            _, (a_l, b_l, e_l) = _forward(
                latents, timestep, recorder=_CrossAttnRecorder(unet, keep_grad=False))
            eps_multi = poe_eps(guided_eps(a_l, e_l, float(p.guidance_scale)),
                                guided_eps(b_l, e_l, float(p.guidance_scale)), e_l)
            eps_multi = eps_frozen + float(p.correction) * (eps_multi - eps_frozen)

            tw_multi = latents - sig * eps_multi
            composed = float(p.co3_multi_weight) * tw_multi
            for g in (g_a, g_b):
                composed = composed + k_w * (latents - sig * g)
            renoise = e_f if p.co3_renoise == "uncond" else eps_multi
            if p.co3_renorm == "reference":
                r = tw_multi.max()
                composed = composed / composed.max() * r
                x2 = composed + sig * renoise
                latents = x2 * (latents.norm() / x2.norm())
            else:
                latents = composed + sig * renoise
        return latents

    manifest: list[dict] = []
    for s in seeds:
        seed_dir = ensure_dir(out_root / f"seed_{int(s)}")
        cell = CellPath.from_root(p.pair_slug, int(s), cache_root=cache_root)
        init = load_pinned_init_latents(cell, device=device, dtype=dtype,
                                        euler_init_noise_sigma=float(p.euler_sigma))
        scheduler.set_timesteps(int(p.num_inference_steps))
        latents = (init / float(p.euler_sigma)).to(device=device, dtype=dtype)
        trace: list[float | None] = []

        for step_index, timestep in enumerate(scheduler.timesteps):
            if int(p.co3_steps) > 0 and step_index < int(p.co3_until):
                with torch.no_grad():
                    latents = _co3_correct(latents, timestep)
            if p.fix != "no-fix" and step_index in window:
                with torch.enable_grad():
                    latents, l = _update(latents, timestep)
                trace.append(l)
            else:
                trace.append(None)

            with torch.no_grad():
                _adapter(False)
                _, (a_f, b_f, e_f) = _forward(
                    latents, timestep, recorder=_CrossAttnRecorder(unet, keep_grad=False))
                eps_frozen = poe_eps(guided_eps(a_f, e_f, float(p.guidance_scale)),
                                     guided_eps(b_f, e_f, float(p.guidance_scale)), e_f)
                if float(p.correction) == 0.0:
                    eps_t = eps_frozen
                else:
                    _adapter(True)
                    _, (a_l, b_l, e_l) = _forward(
                        latents, timestep, recorder=_CrossAttnRecorder(unet, keep_grad=False))
                    eps_lora = poe_eps(guided_eps(a_l, e_l, float(p.guidance_scale)),
                                       guided_eps(b_l, e_l, float(p.guidance_scale)), e_l)
                    eps_t = eps_frozen + float(p.correction) * (eps_lora - eps_frozen)

                alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                    device=device, dtype=dtype)
                x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
                latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep,
                                                step_index=step_index, x0=x0, eps=eps_t)

        with torch.no_grad():
            image = decode_latents(models, latents).cpu()
        write_decoded_image(image, seed_dir / f"sample_seed_{int(s)}.png")
        fired = [v for v in trace if v is not None]
        manifest.append({"seed": int(s), "fix": p.fix,
                         "loss_first": fired[0] if fired else None,
                         "loss_last": fired[-1] if fired else None,
                         "window_steps": len(fired)})
        log.info("seed=%d fix %s: %s → %s over %d window steps", int(s), p.fix,
                 "n/a" if not fired else "%.4g" % fired[0],
                 "n/a" if not fired else "%.4g" % fired[-1], len(fired))

    write_json(out_root / "guide_manifest.json", {
        "regime": regime, "fix": p.fix, "checkpoint": str(p.checkpoint),
        "correction": float(p.correction), "commit_window": [p.commit_lo, p.commit_hi],
        "grad_steps": int(p.grad_steps), "strength": float(p.strength),
        "attn_resolution": res, "renorm_tokens": p.renorm_tokens,
        "co3_steps": int(p.co3_steps), "co3_until": int(p.co3_until),
        "co3_multi_weight": float(p.co3_multi_weight), "co3_renorm": p.co3_renorm,
        "co3_renoise": p.co3_renoise,
        "prompt_a": p.prompt_a, "prompt_b": p.prompt_b,
        "joint_prompt_used": False, "seeds": manifest,
    })
    log.info("done — %d seeds, out %s", len(manifest), out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
