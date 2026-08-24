"""Per-step mechanism capture for the interactive tabs (plan 04, all steps).

Runs seed 9's plain-PoE trajectory and, at EVERY denoise step, captures the
reduced quantities the interactive artifact needs (all small, so no OOM):

  per token (cat, dog), adapter OFF and ON at the same x_t:
    weight[32,32]   — where the word looks
    content[32,32]  — what the word paints (|Σ attn·value|)
    value_cos       — cosine(value_vec off, value_vec on): 1=same content
                      direction, <1 = the LoRA rotated what the word carries
  delta[32,32]      — |ε_LoRA − ε_PoE| per latent cell (the correction)

Exports one compact JSON (uint8-quantized maps) for the sweep viewer's
mechanism tabs to render live against the denoise-step slider.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import _maybe_attach_lora
from poe_repair.methods._sampling import (
    _CrossAttnRecorder, add_time_ids, guided_eps, poe_eps,
    tweedie_mean, ddim_prev_from_x0_eps, decode_latents,
)
from poe_repair.runtime import (
    infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

TOKENS = {"cat": {"branch": 0, "tok": 2}, "dog": {"branch": 1, "tok": 2}}


def _q(arr, vmax):
    """uint8-quantize a [R,R] map to [0,vmax] → base64."""
    q = np.clip(arr / (vmax + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


def _thumb_uri(models, latents, side=192):
    """Decode a latent to a small JPEG data-URI ('image so far')."""
    import io
    from PIL import Image
    img = decode_latents(models, latents).squeeze(0)   # [3,H,W] in [0,1]-ish
    arr = img.permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
    im = Image.fromarray((arr * 255).astype(np.uint8))
    if im.width > side:
        im = im.resize((side, side), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _capture(unet, pe_3, cond_3, latent_input_3, timestep, res):
    with torch.no_grad(), _CrossAttnRecorder(
        unet, keep_grad=False, track_values=True
    ) as rec:
        unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
             added_cond_kwargs=cond_3, timestep_cond=None)
        out = {}
        for name, spec in TOKENS.items():
            w = rec.aggregate_token_map(
                spec["tok"], target_hw=(res, res),
                branch_index=spec["branch"], agg_resolution=res)
            c = rec.aggregate_painted_content(
                branch_index=spec["branch"], token_index=spec["tok"],
                agg_resolution=res)
            vv = rec.token_value_vector(spec["tok"], branch_index=spec["branch"])
            out[name] = {
                "weight": None if w is None else w.float().numpy(),
                "content": None if c is None else c.float().numpy(),
                "value_vec": None if vv is None else vv.float().numpy(),
            }
        rec.attn_maps = []; rec.value_maps = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mechanism_timeseries")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args(argv)

    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype)
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    unet = models["unet"]
    adapter = _maybe_attach_lora(unet, args.checkpoint)

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

    scheduler.set_timesteps(int(args.num_inference_steps))
    latents = (init / float(args.euler_sigma)).to(device=device, dtype=dtype)
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

    N = int(args.num_inference_steps)
    R = int(args.res)
    steps = []          # per-step reduced record (raw arrays first, quantize after)
    unet.eval()
    torch.set_grad_enabled(False)
    for si, timestep in enumerate(scheduler.timesteps):
        latent_input_3 = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        off()
        moff = _capture(unet, pe_3, cond_3, latent_input_3, timestep, R)
        noise = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                     added_cond_kwargs=cond_3, timestep_cond=None).sample
        ea, eb, eu = noise.chunk(3)
        eps_poe = poe_eps(guided_eps(ea, eu, args.guidance_scale),
                          guided_eps(eb, eu, args.guidance_scale), eu)
        on()
        mon = _capture(unet, pe_3, cond_3, latent_input_3, timestep, R)
        noise_l = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                       added_cond_kwargs=cond_3, timestep_cond=None).sample
        la, lb, lu = noise_l.chunk(3)
        eps_lora = poe_eps(guided_eps(la, lu, args.guidance_scale),
                           guided_eps(lb, lu, args.guidance_scale), lu)
        delta = (eps_lora - eps_poe)[0]                # [4,H,W]
        dmag = delta.float().norm(dim=0)               # [H,W] latent res (128)
        dmag = torch.nn.functional.interpolate(
            dmag[None, None], size=(R, R), mode="bilinear",
            align_corners=False)[0, 0].cpu().numpy()

        rec = {"step": si, "timestep": int(timestep.item())}
        for tok in TOKENS:
            rec[f"{tok}_weight_on"] = mon[tok]["weight"]
            rec[f"{tok}_content_off"] = moff[tok]["content"]
            rec[f"{tok}_content_on"] = mon[tok]["content"]
            vo, vn = moff[tok]["value_vec"], mon[tok]["value_vec"]
            if vo is not None and vn is not None:
                cos = float(np.dot(vo, vn) /
                            (np.linalg.norm(vo) * np.linalg.norm(vn) + 1e-9))
            else:
                cos = 1.0
            rec[f"{tok}_value_cos"] = cos
        rec["delta"] = dmag
        rec["delta_left"] = float(dmag[:, :R // 2].sum())
        rec["delta_right"] = float(dmag[:, R // 2:].sum())
        # "image so far" = Tweedie x̂0 decoded, for PoE (off) and LoRA (on)
        ab = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0_poe = tweedie_mean(latents, ab, eps_poe)
        x0_lora = tweedie_mean(latents, ab, eps_lora)
        rec["img_poe"] = _thumb_uri(models, x0_poe)
        rec["img_lora"] = _thumb_uri(models, x0_lora)
        steps.append(rec)
        del noise, noise_l, eps_lora, delta, x0_lora
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x0 = x0_poe
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=si, x0=x0, eps=eps_poe)
        print(f"[timeseries] step {si} done", flush=True)

    # global vmax per map type for quantization
    def gmax(key):
        return max(float(np.max(s[key])) for s in steps if s[key] is not None)
    keys_map = [f"{t}_weight_on" for t in TOKENS] + \
               [f"{t}_content_off" for t in TOKENS] + \
               [f"{t}_content_on" for t in TOKENS] + ["delta"]
    vmax = {k: gmax(k) for k in keys_map}

    data = {"seed": args.seed, "res": R, "n_steps": N,
            "tokens": list(TOKENS), "vmax": vmax,
            "timesteps": [s["timestep"] for s in steps],
            "steps": []}
    for s in steps:
        e = {"step": s["step"], "timestep": s["timestep"],
             "delta_left": round(s["delta_left"], 2),
             "delta_right": round(s["delta_right"], 2)}
        for k in keys_map:
            e[k] = "" if s[k] is None else _q(s[k], vmax[k])
        for t in TOKENS:
            e[f"{t}_value_cos"] = round(s[f"{t}_value_cos"], 4)
        e["img_poe"] = s["img_poe"]
        e["img_lora"] = s["img_lora"]
        data["steps"].append(e)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"[timeseries] wrote {out} ({out.stat().st_size/1e6:.2f} MB, {N} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
