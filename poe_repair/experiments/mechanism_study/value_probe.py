"""Painted-content (value) probe: does the LoRA change WHAT is written, not where?

Plan 04 task 2. Runs one seed's denoising trajectory (the plain-PoE path, so
x_t matches the λ=0 capture) and at chosen denoising steps captures, for the
same latent state, both the adapter-OFF and adapter-ON attention:
  - the cross-attention WEIGHT map for cat/dog (where the word looks)
  - the painted-CONTENT map for cat/dog (Σ attn·value: what the word writes)

If the fix lives in the values, the content maps differ between OFF and ON more
than the weight maps do. Writes one .pt per (step) with both regimes' maps.

Usage::

    python -m poe_repair.experiments.mechanism_study.value_probe \
        --checkpoint <lora_step_062500.pt> --seed 9 --steps 10,25,40,49
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import _maybe_attach_lora
from poe_repair.methods import _sampling as S
from poe_repair.methods._sampling import (
    _CrossAttnRecorder, add_time_ids, guided_eps, poe_eps,
    tweedie_mean, ddim_prev_from_x0_eps,
)
from poe_repair.runtime import (
    ensure_dir, infer_device, infer_dtype, load_ddim_scheduler,
    load_sdxl_models, write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

DEFAULT_OUT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/value_probe"
)
# solo-subject token index for "a cat"/"a dog"; branch 0=cat(A), 1=dog(B).
TOKENS = {"cat": {"branch": 0, "tok": 2}, "dog": {"branch": 1, "tok": 2}}


def _parse_ints(a):
    out = []
    for p in a.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def _self_entropy(rec, res, branch):
    """Per-pixel self-attention entropy at res×res: how spread each pixel's
    attention to other pixels is. Low = tight grouping (this pixel binds to a
    small region → object-like); high = diffuse. Returns [res,res] or None."""
    sa = rec.aggregate_self_attention(target_hw=(res, res), branch_index=branch)
    if sa is None:
        return None
    p = sa.float().clamp_min(1e-12)          # [HW, HW] row-stochastic
    ent = -(p * p.log()).sum(dim=-1)         # [HW]
    return ent.reshape(res, res).detach().cpu()


def _capture(unet, pe_3, cond_3, latent_input_3, timestep, res, self_attn=False):
    """Return {tok:{'weight','content'}, '_self_ent':[R,R]|None} for this forward."""
    with torch.no_grad(), _CrossAttnRecorder(
        unet, keep_grad=False, track_values=True, track_self_attn=self_attn
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
            out[name] = {
                "weight": None if w is None else w.float().cpu(),
                "content": None if c is None else c.float().cpu(),
            }
        out["_self_ent"] = _self_entropy(rec, res, 0) if self_attn else None
        # free the big GPU-side lists before the next forward
        rec.attn_maps = []; rec.value_maps = []; rec.self_attn_maps = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="value_probe")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--steps", default="10,25,40,49")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--self-attn", action="store_true",
                    help="also capture self-attention entropy (heavier; off by default)")
    args = ap.parse_args(argv)

    want = set(_parse_ints(args.steps))
    out_root = ensure_dir(
        Path(args.out_root) if args.out_root
        else DEFAULT_OUT / args.pair_slug / f"seed_{args.seed}"
    )
    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype)
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    unet = models["unet"]
    adapter = _maybe_attach_lora(unet, args.checkpoint)

    class _P:
        prompt_a, prompt_b, joint_prompt = (
            args.prompt_a, args.prompt_b, args.joint_prompt)

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

    records = []
    unet.eval()
    torch.set_grad_enabled(False)   # whole trajectory is inference-only
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input_3 = scheduler.scale_model_input(
            latents.repeat(3, 1, 1, 1), timestep)
        # frozen forward (adapter OFF) — drives the trajectory (plain PoE)
        off()
        if step_index in want:
            maps_off = _capture(unet, pe_3, cond_3, latent_input_3, timestep, args.res, args.self_attn)
        noise = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                     added_cond_kwargs=cond_3, timestep_cond=None).sample
        ea, eb, eu = noise.chunk(3)
        eps_poe = poe_eps(guided_eps(ea, eu, args.guidance_scale),
                          guided_eps(eb, eu, args.guidance_scale), eu)
        # LoRA forward (adapter ON) at the SAME x_t — for the comparison only
        if step_index in want:
            on()
            maps_on = _capture(unet, pe_3, cond_3, latent_input_3, timestep, args.res, args.self_attn)
            # LoRA-on guided PoE eps at the same x_t → Δ-field = eps_lora - eps_poe
            noise_l = unet(latent_input_3, timestep, encoder_hidden_states=pe_3,
                           added_cond_kwargs=cond_3, timestep_cond=None).sample
            la, lb, lu = noise_l.chunk(3)
            eps_lora = poe_eps(guided_eps(la, lu, args.guidance_scale),
                               guided_eps(lb, lu, args.guidance_scale), lu)
            delta = (eps_lora - eps_poe)                 # [1,4,H,W] the correction
            rec = {"step_index": step_index, "timestep": int(timestep.item())}
            for name in TOKENS:
                for reg, mm in [("off", maps_off), ("on", maps_on)]:
                    rec[f"{name}_{reg}_weight"] = mm[name]["weight"]
                    rec[f"{name}_{reg}_content"] = mm[name]["content"]
            rec["self_ent_off"] = maps_off["_self_ent"]
            rec["self_ent_on"] = maps_on["_self_ent"]
            rec["delta"] = delta.float().cpu()           # for the Δ vector field
            rec["x_t"] = latents.float().cpu()
            torch.save(rec, out_root / f"step_{step_index:03d}_valuemaps.pt")
            records.append({"step": step_index, "timestep": int(timestep.item())})
            print(f"[value_probe] step={step_index} captured off+on", flush=True)
            del noise_l, la, lb, lu, eps_lora, delta, maps_off, maps_on
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # advance trajectory with the frozen (plain-PoE) eps
        off()
        ab = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, ab, eps_poe)
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_poe)

    write_json(out_root / "value_probe_manifest.json", {
        "seed": args.seed, "pair_slug": args.pair_slug,
        "checkpoint": args.checkpoint, "steps": sorted(want),
        "res": args.res, "records": records})
    print(f"[value_probe] done — {len(records)} steps → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
