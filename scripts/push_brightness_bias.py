#!/usr/bin/env python
"""Does the adapter's push carry a brightness bias?

Plan 06 of scope 09, the follow-on to task 1.1. Seed 9 of a_cat__x__a_dog renders with its
blacks lifted from 14 to 40 on a 0-to-255 scale, which is what haze is. A push whose average
over the picture is not zero would do exactly that, so this measures the average of the push
the adapter applies, against the average of the push the picture actually needed.

Both are measured per latent channel and per denoising step. The comparison that matters is
r_hat's mean against r's mean: the needed correction has whatever mean it has, and the question
is whether the adapter adds one of its own on top.

    python scripts/push_brightness_bias.py --checkpoint pool43_all50_40k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from poe_repair._sdxl.metrics import guided_eps, poe_eps          # noqa: E402
from poe_repair.methods._sampling import add_time_ids             # noqa: E402
from poe_repair.run import make_ctx                               # noqa: E402
from poe_repair.training_cache import CellPath, load_step_raw     # noqa: E402

# Same three the sweep and the shrinkage read use.
CHECKPOINTS = {
    "v1_freeze_null_r16_s0_25_30k": (
        "/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/"
        "v1_freeze_null_r16_s0_25/checkpoints/lora_step_030000.pt", 16),
    "pool43_all50_40k": (
        "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_040000.pt", 32),
}
PAIR = "a_cat__x__a_dog"
GUIDANCE = 7.5
HEIGHT = WIDTH = 1024
OUT_DIR = REPO / "artifacts/results/designing-the-correction-loss/why-the-renders-are-not-crisp"


def _freeze_null_of(ckpt: str) -> bool:
    cfg = Path(ckpt).parent.parent / "config.json"
    return bool(json.loads(cfg.read_text()).get("freeze_null") or False)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[9])
    a = ap.parse_args()

    ckpt, rank = CHECKPOINTS[a.checkpoint]
    freeze_null = _freeze_null_of(ckpt)
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    free, total = torch.cuda.mem_get_info(0)
    if (total - free) / 1e9 > 1.0:
        raise SystemExit("device already in use, refusing to share")

    import lambda_boundary_probe as lbp
    ctx = make_ctx()
    unet = ctx.models["unet"]
    lbp.LORA_RANK = lbp.LORA_ALPHA = rank
    lbp._attach_and_load_lora(unet, Path(ckpt))
    unet.eval()
    device, dtype = ctx.device, ctx.dtype
    print(f"{a.checkpoint}: rank {rank}, freeze_null={freeze_null}", flush=True)

    rows = []
    for seed in a.seeds:
        cell = CellPath.from_root(PAIR, seed, split="heldout")
        emb = torch.load(cell.root / "embeddings.pt", map_location="cpu", weights_only=False)
        seq_3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_uncond"]], dim=0).to(device=device, dtype=dtype)
        pool_3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_uncond"]], dim=0).to(device=device, dtype=dtype)
        for f in cell.step_files():
            c = load_step_raw(f)
            x_t = c["x_t"].to(device=device, dtype=dtype).repeat(3, 1, 1, 1)
            ts = torch.tensor([c["timestep"]], device=device, dtype=torch.long).repeat(3)
            cond = {"text_embeds": pool_3,
                    "time_ids": add_time_ids(height=HEIGHT, width=WIDTH, batch_size=3,
                                             device=device, dtype=dtype)}
            out = unet(ctx.scheduler.scale_model_input(x_t, ts[0]), ts,
                       encoder_hidden_states=seq_3, added_cond_kwargs=cond,
                       timestep_cond=None).sample.float()
            ea_l, eb_l, eu_l = out[0:1], out[1:2], out[2:3]
            ea, eb = c["eps_a_raw"].to(device), c["eps_b_raw"].to(device)
            ej, eu = c["eps_j_raw"].to(device), c["eps_uncond"].to(device)
            eps_poe_frozen = poe_eps(guided_eps(ea, eu, GUIDANCE), guided_eps(eb, eu, GUIDANCE), eu)
            r = guided_eps(ej, eu, GUIDANCE) - eps_poe_frozen
            u = eu if freeze_null else eu_l
            eps_poe_lora = poe_eps(guided_eps(ea_l, u, GUIDANCE), guided_eps(eb_l, u, GUIDANCE), u)
            r_hat = eps_poe_lora - eps_poe_frozen
            # Blacks can lift without the average moving, if the push pulls dark areas up and
            # bright areas down. So compare the picture the frozen composition is heading for
            # against the one the adapted composition is heading for, and measure their spread.
            # Tweedie: x0 = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t).
            abar = ctx.scheduler.alphas_cumprod.to(device)[int(c["timestep"])].float()
            x1 = c["x_t"].to(device).float()
            x0_frozen = (x1 - (1 - abar).sqrt() * eps_poe_frozen) / abar.sqrt()
            x0_lora = (x1 - (1 - abar).sqrt() * eps_poe_lora) / abar.sqrt()
            def spread(t):
                f = t.reshape(-1)
                lo, hi = torch.quantile(f, 0.01), torch.quantile(f, 0.99)
                return float((hi - lo).item())
            rows.append({
                "seed": seed, "step_index": c["step_index"], "timestep": c["timestep"],
                # The contrast of the picture each composition is heading for, as the gap between
                # its 1st and 99th percentile. Lower means a flatter picture.
                "x0_spread_frozen": spread(x0_frozen), "x0_spread_lora": spread(x0_lora),
                "x0_std_frozen": float(x0_frozen.std().item()),
                "x0_std_lora": float(x0_lora.std().item()),
                # Mean over the spatial grid, one number per latent channel. A push that averages
                # to zero moves detail around; one that does not shifts the whole picture.
                "r_mean_per_channel": [float(v) for v in r.mean(dim=(0, 2, 3))],
                "r_hat_mean_per_channel": [float(v) for v in r_hat.mean(dim=(0, 2, 3))],
                "r_std": float(r.std().item()), "r_hat_std": float(r_hat.std().item()),
            })
        print(f"seed {seed}: {len(cell.step_files())} steps", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"push-brightness-bias.{a.checkpoint}.json"
    out.write_text(json.dumps({"checkpoint_name": a.checkpoint, "checkpoint": ckpt,
                               "rank": rank, "freeze_null": freeze_null, "pair": PAIR,
                               "what_this_measures": "the spatial mean of the needed push (r) and "
                                                     "of the adapter's push (r_hat), per latent "
                                                     "channel, per denoising step",
                               "rows": rows}, indent=2))
    print(f"wrote {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
