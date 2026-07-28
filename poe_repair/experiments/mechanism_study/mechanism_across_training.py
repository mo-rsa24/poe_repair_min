"""Mechanism across TRAINING checkpoints (for the training-slider-driven tabs).

For each checkpoint of the seed-9 LoRA, capture a reduced per-generation
mechanism summary so ONE training slider can drive both the outcome and the
mechanism. Loads SDXL once, attaches each checkpoint's weights via
load_lora_state, and at a few denoise steps captures:
  weight/content maps (cat,dog), value-direction cosine, delta split,
  and the forming PoE/LoRA images.

Output: one JSON keyed by checkpoint step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import _maybe_attach_lora
from poe_repair.experiments.mechanism_study.mechanism_timeseries import (
    TOKENS, _capture, _thumb_uri, _q,
)
from poe_repair.methods._sampling import (
    add_time_ids, guided_eps, poe_eps, tweedie_mean, ddim_prev_from_x0_eps,
)
from poe_repair.runtime import (
    infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


def _parse_ints(a):
    out = []
    for p in a.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mechanism_across_training")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--steps-ckpt", required=True,
                    help="training steps, e.g. 12500,15000,...")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--capture-steps", default="5,15,25,35,45",
                    help="denoise steps to capture per checkpoint")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args(argv)

    ckpts = _parse_ints(args.steps_ckpt)
    want = set(_parse_ints(args.capture_steps))
    ckpt_dir = Path(args.ckpt_dir)
    R, N = int(args.res), int(args.num_inference_steps)
    gs = float(args.guidance_scale)

    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype)
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    unet = models["unet"]
    present = [c for c in ckpts
               if (ckpt_dir / f"lora_step_{c:06d}.pt").exists()]
    adapter = _maybe_attach_lora(unet, str(ckpt_dir / f"lora_step_{present[0]:06d}.pt"))

    class _P:
        prompt_a, prompt_b, joint_prompt = "a cat", "a dog", "a cat and a dog"

    class _C:
        cell = _P()

    emb = encode_all_prompts(_C(), models, device, dtype)
    cell = CellPath.from_root(args.pair_slug, int(args.seed),
                              cache_root=DEFAULT_CACHE_ROOT)
    init = load_pinned_init_latents(
        cell, device=device, dtype=dtype,
        euler_init_noise_sigma=float(args.euler_sigma))
    pe_3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0)
    pool_3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0)
    cond_3 = {"text_embeds": pool_3,
              "time_ids": add_time_ids(height=1024, width=1024, batch_size=3,
                                       device=device, dtype=dtype)}

    def off():
        if hasattr(unet, "disable_adapters"):
            try: unet.disable_adapters()
            except ValueError: pass

    def on():
        if hasattr(unet, "enable_adapters"):
            try: unet.enable_adapters()
            except ValueError: pass
        if hasattr(unet, "set_adapter"):
            try: unet.set_adapter(adapter)
            except Exception: pass

    unet.eval()
    torch.set_grad_enabled(False)
    per_ckpt = {}
    for ci in present:
        sd = torch.load(ckpt_dir / f"lora_step_{ci:06d}.pt",
                        map_location="cpu", weights_only=False)
        lora_trainer.load_lora_state(unet, sd["lora_state"])
        latents = (init / float(args.euler_sigma)).to(device=device, dtype=dtype)
        steps_out = []
        for si, timestep in enumerate(scheduler.timesteps):
            latent_input_3 = scheduler.scale_model_input(
                latents.repeat(3, 1, 1, 1), timestep)
            off()
            if si in want:
                moff = _capture(unet, pe_3, cond_3, latent_input_3, timestep, R)
            noise = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                         added_cond_kwargs=cond_3, timestep_cond=None).sample
            ea, eb, eu = noise.chunk(3)
            eps_poe = poe_eps(guided_eps(ea, eu, gs), guided_eps(eb, eu, gs), eu)
            ab = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
            if si in want:
                on()
                mon = _capture(unet, pe_3, cond_3, latent_input_3, timestep, R)
                noise_l = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                               added_cond_kwargs=cond_3, timestep_cond=None).sample
                la, lb, lu = noise_l.chunk(3)
                eps_lora = poe_eps(guided_eps(la, lu, gs), guided_eps(lb, lu, gs), lu)
                delta = (eps_lora - eps_poe)[0]
                dmag = torch.nn.functional.interpolate(
                    delta.float().norm(dim=0)[None, None], size=(R, R),
                    mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                rec = {"step": si, "timestep": int(timestep.item())}
                for tok in TOKENS:
                    rec[f"{tok}_weight_on"] = mon[tok]["weight"]
                    rec[f"{tok}_content_off"] = moff[tok]["content"]
                    rec[f"{tok}_content_on"] = mon[tok]["content"]
                    vo, vn = moff[tok]["value_vec"], mon[tok]["value_vec"]
                    rec[f"{tok}_value_cos"] = (
                        float(np.dot(vo, vn) / (np.linalg.norm(vo) * np.linalg.norm(vn) + 1e-9))
                        if vo is not None and vn is not None else 1.0)
                rec["delta"] = dmag
                rec["delta_left"] = float(dmag[:, :R // 2].sum())
                rec["delta_right"] = float(dmag[:, R // 2:].sum())
                x0_poe = tweedie_mean(latents, ab, eps_poe)
                x0_lora = tweedie_mean(latents, ab, eps_lora)
                rec["img_poe"] = _thumb_uri(models, x0_poe)
                rec["img_lora"] = _thumb_uri(models, x0_lora)
                steps_out.append(rec)
                del noise_l, eps_lora, delta, x0_lora
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            off()
            x0 = tweedie_mean(latents, ab, eps_poe)
            latents = ddim_prev_from_x0_eps(
                scheduler=scheduler, timestep=timestep, step_index=si, x0=x0, eps=eps_poe)
        per_ckpt[ci] = steps_out
        print(f"[across] ckpt {ci}: {len(steps_out)} steps captured", flush=True)

    # global vmax per map key
    keys_map = [f"{t}_weight_on" for t in TOKENS] + \
               [f"{t}_content_off" for t in TOKENS] + \
               [f"{t}_content_on" for t in TOKENS] + ["delta"]
    def gmax(k):
        vals = [float(np.max(s[k])) for recs in per_ckpt.values() for s in recs
                if s[k] is not None]
        return max(vals) if vals else 1.0
    vmax = {k: gmax(k) for k in keys_map}

    data = {"seed": args.seed, "res": R, "checkpoints": present, "vmax": vmax,
            "capture_steps": sorted(want), "by_ckpt": {}}
    for ci in present:
        out_steps = []
        for s in per_ckpt[ci]:
            e = {"step": s["step"], "timestep": s["timestep"],
                 "delta_left": round(s["delta_left"], 2),
                 "delta_right": round(s["delta_right"], 2),
                 "img_poe": s["img_poe"], "img_lora": s["img_lora"]}
            for k in keys_map:
                e[k] = "" if s[k] is None else _q(s[k], vmax[k])
            for t in TOKENS:
                e[f"{t}_value_cos"] = round(s[f"{t}_value_cos"], 4)
            out_steps.append(e)
        data["by_ckpt"][str(ci)] = out_steps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"[across] wrote {out} ({out.stat().st_size/1e6:.2f} MB, "
          f"{len(present)} ckpts × {len(want)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
