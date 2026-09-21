#!/usr/bin/env python
"""Mismatch against energy along the adapter's own corrected trajectory (plan 19, scope 01).

Starts from the cached initial latent of each held-out cat × dog seed, steps the latent with
eps_PoE_frozen + lambda * r_hat exactly as the showcase sampler does, and at every state the run
actually visits asks the frozen UNet for the true correction there:

    r_true = gs * (eps_j − eps_a − eps_b + eps_uncond)        four frozen branches
    r_hat  = eps_PoE_lora − eps_PoE_frozen                      three LoRA branches

and records, per step, the two norms, their cosine, the squared mismatch ||r_hat − r_true||^2,
the Girsanov weight gamma_k of that step, and the relative drift of the visited state from the
cached plain-PoE state. Energy is 0.5 * gamma_k * ||.||^2 in nats, the same unit as
control_energy_from_cache.py, so the adapter's spend, the target's spend and the mismatch are
on one scale. The final latent is decoded so the render can be looked at.

Writes <out>/on_policy_energy.json (per seed, per step, plus per-seed totals) and
<out>/seed_<n>_corrected.png. Extends scripts/showcase/on_policy_fit_cosine.py, which records
the cosine and the norm ratio but not the absolute norms this read needs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair._sdxl.metrics import ddim_prev_from_x0_eps, tweedie_mean  # noqa: E402
from poe_repair._sdxl.runtime import decode_latents  # noqa: E402
from poe_repair.experiments.one_pair_one_seed import trainer as T  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.config import RunConfig  # noqa: E402
from poe_repair.methods._sampling import add_time_ids, write_decoded_image  # noqa: E402
from poe_repair.runtime import infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models  # noqa: E402
from poe_repair.training_cache import resolve_cells  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="label written into the json (e.g. 'baseline 30,050')")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--lora-key", default="lora_state", choices=("lora_state", "lora_state_ema"))
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10, 11, 12, 13, 14, 15, 16])
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig()
    cfg.lora.rank = args.rank
    cfg.lora.alpha = args.alpha if args.alpha is not None else args.rank
    device = infer_device("cuda"); dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(cfg.model_id)
    scheduler.set_timesteps(args.num_inference_steps)
    unet = models["unet"]
    T.attach_lora(unet, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    T.load_lora_state(unet, ckpt[args.lora_key])
    unet.eval()
    gs = float(cfg.sampler.guidance_scale); H, W = cfg.sampler.height, cfg.sampler.width
    lo, hi = cfg.probe.commit_window
    lam = float(args.lambda_value)
    n = int(args.num_inference_steps)
    print(f"[onpolicy-energy] checkpoint step={ckpt.get('step')} rank={args.rank} key={args.lora_key} "
          f"lambda={lam} pair={args.pair} seeds={args.seeds}", flush=True)

    def adapters(on: bool) -> None:
        if on:
            unet.enable_adapters() if hasattr(unet, "enable_adapters") else unet.enable_adapter_layers()
        else:
            unet.disable_adapters() if hasattr(unet, "disable_adapters") else unet.disable_adapter_layers()

    timesteps = [int(t.item()) for t in scheduler.timesteps]
    gamma = T.control_energy_weights(timesteps, scheduler.alphas_cumprod, num_inference_steps=n).tolist()

    rows = []
    t0 = time.time()
    for cell in resolve_cells(args.pair, args.seeds, split=args.split):
        emb = torch.load(cell.root / "embeddings.pt", map_location="cpu", weights_only=False)
        seq4 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_j"], emb["seq_uncond"]], 0).to(device=device, dtype=dtype)
        pool4 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_j"], emb["pool_uncond"]], 0).to(device=device, dtype=dtype)
        cond4 = {"text_embeds": pool4, "time_ids": add_time_ids(height=H, width=W, batch_size=4, device=device, dtype=dtype)}
        seq3, pool3 = seq4[[0, 1, 3]], pool4[[0, 1, 3]]
        cond3 = {"text_embeds": pool3, "time_ids": add_time_ids(height=H, width=W, batch_size=3, device=device, dtype=dtype)}
        latents = (emb["init_latents"] / float(emb["euler_init_noise_sigma"])).to(device=device, dtype=dtype)
        cached = {int(e.step_index): e for e in T.load_cached_steps(cell, guidance_scale=gs)}

        for step_index, timestep in enumerate(scheduler.timesteps):
            with torch.no_grad():
                x_in = scheduler.scale_model_input(latents, timestep)
                tb4 = torch.full((4,), int(timestep.item()), device=device, dtype=torch.long)
                adapters(False)
                n4 = unet(x_in.repeat(4, 1, 1, 1), tb4, encoder_hidden_states=seq4, added_cond_kwargs=cond4).sample.float()
                ea, eb, ej, eu = n4[0:1], n4[1:2], n4[2:3], n4[3:4]
                poe_frozen = T._compose(ea, eb, eu, gs, "poe", 0.5)
                r_true = gs * (ej - ea - eb + eu)
                adapters(True)
                n3 = unet(x_in.repeat(3, 1, 1, 1), tb4[:3], encoder_hidden_states=seq3, added_cond_kwargs=cond3).sample.float()
                poe_lora = T._compose(n3[0:1], n3[1:2], n3[2:3], gs, "poe", 0.5)
                r_hat = poe_lora - poe_frozen

                g = float(gamma[step_index])
                hat_sq = float((r_hat.double() ** 2).sum()); true_sq = float((r_true.double() ** 2).sum())
                mis_sq = float(((r_hat - r_true).double() ** 2).sum())
                cos = float(torch.nn.functional.cosine_similarity(r_hat.flatten(), r_true.flatten(), dim=0))
                c = cached.get(step_index)
                drift = None
                if c is not None:
                    xc = c.x_t.to(device=device, dtype=torch.float32)
                    drift = float((latents.float() - xc).norm() / (xc.norm() + 1e-8))

                eps_t = poe_frozen + lam * r_hat
                alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=torch.float32)
                x0 = tweedie_mean(latents.float(), alpha_bar_t, eps_t)
                latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                                x0=x0, eps=eps_t).to(dtype)

            bucket = "early" if step_index < lo else ("commit" if step_index < hi else "late")
            rows.append({"seed": int(cell.seed), "step": step_index, "timestep": int(timestep.item()), "bucket": bucket,
                         "gamma": g, "cos": cos,
                         "r_hat_norm": hat_sq ** 0.5, "r_true_norm": true_sq ** 0.5,
                         "energy_hat_nats": 0.5 * g * hat_sq, "energy_true_nats": 0.5 * g * true_sq,
                         "mismatch_nats": 0.5 * g * mis_sq, "state_drift_rel": drift})
        with torch.no_grad():
            write_decoded_image(decode_latents(models, latents).cpu(), out / f"seed_{cell.seed}_corrected.png")
        r = [x for x in rows if x["seed"] == cell.seed]
        print(f"[onpolicy-energy] seed={cell.seed} energy hat/true/mismatch = "
              f"{sum(x['energy_hat_nats'] for x in r):.1f}/{sum(x['energy_true_nats'] for x in r):.1f}/"
              f"{sum(x['mismatch_nats'] for x in r):.1f} nats, cos early/late "
              f"{st.mean([x['cos'] for x in r if x['bucket']=='early']):.3f}/"
              f"{st.mean([x['cos'] for x in r if x['bucket']=='late']):.3f}, drift@49 {r[-1]['state_drift_rel']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    per_seed = {}
    for s in args.seeds:
        r = [x for x in rows if x["seed"] == s]
        if not r:
            continue
        per_seed[str(s)] = {
            "energy_hat_nats": round(sum(x["energy_hat_nats"] for x in r), 3),
            "energy_true_nats": round(sum(x["energy_true_nats"] for x in r), 3),
            "mismatch_nats": round(sum(x["mismatch_nats"] for x in r), 3),
            "energy_hat_late_nats": round(sum(x["energy_hat_nats"] for x in r if x["bucket"] == "late"), 3),
            "cos_mean": round(st.mean(x["cos"] for x in r), 4),
            "state_drift_rel_final": r[-1]["state_drift_rel"],
        }
    summary = {k: round(st.mean(v[k] for v in per_seed.values()), 3)
               for k in ("energy_hat_nats", "energy_true_nats", "mismatch_nats", "energy_hat_late_nats", "cos_mean")}
    (out / "on_policy_energy.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "step": ckpt.get("step"), "tag": args.tag, "rank": args.rank,
        "lora_key": args.lora_key, "lambda": lam, "pair": args.pair, "seeds": list(args.seeds),
        "gamma": gamma, "timesteps": timesteps,
        "definition": "energies are 0.5 * gamma_k * ||.||^2 in nats with gamma_k = (1 - abar_k/abar_prev)/(1 - abar_k); "
                      "r_true is the joint-prompt correction at the state the corrected run actually visits",
        "summary_mean_over_seeds": summary, "per_seed": per_seed, "rows": rows}, indent=1))
    print("[onpolicy-energy] summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
