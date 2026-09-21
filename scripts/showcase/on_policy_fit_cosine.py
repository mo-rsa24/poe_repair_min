#!/usr/bin/env python
"""On-policy fit of a pooled LoRA checkpoint: cos(delta_hat, true delta_t) along the corrected
trajectory the adapter actually produces at inference (task 1.3 of the
improving-the-pooled-lora-run idea walk).

The cache holds x_t on the *uncorrected* PoE trajectory and fit_cosine_on_cache.py measures the
adapter there. This script instead starts from the same cached x_T and prompt embeddings, steps
the latent with eps_PoE_frozen + lambda * delta_hat (the showcase sampler's rule), and at every
step asks the frozen UNet for the true target at the state actually reached:

    delta_t^true = gs * (eps_j - eps_a - eps_b + eps_uncond)      four frozen branches
    delta_hat    = eps_PoE_lora - eps_PoE_frozen                   three LoRA branches

and records cos(delta_hat, delta_t^true), the norm ratio, and the drift of the reached state from
the cached state at the same step. Decodes the final latent so the render can be eyeballed.

Writes <out>/on_policy_fit.json (per seed, per step) and <out>/on_policy_fit_table.md.
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
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10, 11, 12, 13, 14, 15, 16])
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig()
    cfg.lora.rank, cfg.lora.alpha = args.rank, args.alpha
    device = infer_device("cuda"); dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(cfg.model_id)
    scheduler.set_timesteps(args.num_inference_steps)
    unet = models["unet"]
    T.attach_lora(unet, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    T.load_lora_state(unet, ckpt["lora_state"])
    unet.eval()
    gs = float(cfg.sampler.guidance_scale); H, W = cfg.sampler.height, cfg.sampler.width
    lo, hi = cfg.probe.commit_window
    lam = float(args.lambda_value)
    print(f"[onpolicy] checkpoint step={ckpt.get('step')} rank={args.rank} lambda={lam} "
          f"pair={args.pair} seeds={args.seeds} commit_window={(lo, hi)}", flush=True)

    def adapters(on: bool) -> None:
        if on:
            unet.enable_adapters() if hasattr(unet, "enable_adapters") else unet.enable_adapter_layers()
        else:
            unet.disable_adapters() if hasattr(unet, "disable_adapters") else unet.disable_adapter_layers()

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
                d_true = gs * (ej - ea - eb + eu)
                adapters(True)
                n3 = unet(x_in.repeat(3, 1, 1, 1), tb4[:3], encoder_hidden_states=seq3, added_cond_kwargs=cond3).sample.float()
                poe_lora = T._compose(n3[0:1], n3[1:2], n3[2:3], gs, "poe", 0.5)
                d_hat = poe_lora - poe_frozen

                cos = float(torch.nn.functional.cosine_similarity(d_hat.flatten(), d_true.flatten(), dim=0))
                ratio = float(d_hat.norm() / (d_true.norm() + 1e-8))
                c = cached.get(step_index)
                drift = None; cos_cached_target = None
                if c is not None:
                    xc = c.x_t.to(device=device, dtype=torch.float32)
                    drift = float((latents.float() - xc).norm() / (xc.norm() + 1e-8))
                    cos_cached_target = float(torch.nn.functional.cosine_similarity(
                        d_true.flatten(), c.delta_t.to(device=device, dtype=torch.float32).flatten(), dim=0))

                eps_t = poe_frozen + lam * d_hat
                alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=torch.float32)
                x0 = tweedie_mean(latents.float(), alpha_bar_t, eps_t)
                latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                                x0=x0, eps=eps_t).to(dtype)

            bucket = "early" if step_index < lo else ("commit" if step_index < hi else "late")
            rows.append({"pair": args.pair, "seed": int(cell.seed), "step": step_index, "bucket": bucket,
                         "cos": cos, "norm_ratio": ratio, "state_drift_rel": drift,
                         "cos_true_vs_cached_target": cos_cached_target})
        with torch.no_grad():
            write_decoded_image(decode_latents(models, latents).cpu(), out / f"seed_{cell.seed}_corrected.png")
        r = [x for x in rows if x["seed"] == cell.seed]
        print(f"[onpolicy] seed={cell.seed} cos early/commit/late = "
              f"{st.mean([x['cos'] for x in r if x['bucket']=='early']):.3f}/"
              f"{st.mean([x['cos'] for x in r if x['bucket']=='commit']):.3f}/"
              f"{st.mean([x['cos'] for x in r if x['bucket']=='late']):.3f}  "
              f"drift@49={r[-1]['state_drift_rel']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    (out / "on_policy_fit.json").write_text(json.dumps({"checkpoint": args.checkpoint, "step": ckpt.get("step"),
                                                        "rank": args.rank, "lambda": lam, "rows": rows}, indent=1))

    def m(vals):
        vals = [v for v in vals if v is not None]
        return f"{st.mean(vals):.3f}" if vals else "-"
    lines = ["| seed | cos all | early | commit | late | ‖Δ̂‖/‖Δ‖ | drift@10 | drift@25 | drift@49 | cos(true, cached target) late |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in args.seeds:
        r = [x for x in rows if x["seed"] == s]
        if not r: continue
        d = {x["step"]: x["state_drift_rel"] for x in r}
        lines.append(f"| {s} | {m([x['cos'] for x in r])} | {m([x['cos'] for x in r if x['bucket']=='early'])} | "
                     f"{m([x['cos'] for x in r if x['bucket']=='commit'])} | {m([x['cos'] for x in r if x['bucket']=='late'])} | "
                     f"{m([x['norm_ratio'] for x in r])} | {d.get(10, 0):.3f} | {d.get(25, 0):.3f} | {d.get(49, 0):.3f} | "
                     f"{m([x['cos_true_vs_cached_target'] for x in r if x['bucket']=='late'])} |")
    lines.append(f"| **mean** | **{m([x['cos'] for x in rows])}** | {m([x['cos'] for x in rows if x['bucket']=='early'])} | "
                 f"{m([x['cos'] for x in rows if x['bucket']=='commit'])} | {m([x['cos'] for x in rows if x['bucket']=='late'])} | "
                 f"{m([x['norm_ratio'] for x in rows])} | | | | {m([x['cos_true_vs_cached_target'] for x in rows if x['bucket']=='late'])} |")
    # per-step mean curve, for the plot beside run 1's cached-state curve
    lines.append("\n| step | mean cos over seeds | mean drift |\n|---|---|---|")
    for k in range(args.num_inference_steps):
        r = [x for x in rows if x["step"] == k]
        lines.append(f"| {k} | {m([x['cos'] for x in r])} | {m([x['state_drift_rel'] for x in r])} |")
    (out / "on_policy_fit_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


if __name__ == "__main__":
    main()
