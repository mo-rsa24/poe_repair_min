#!/usr/bin/env python
"""Where the trained adapter lands, in the plane product-of-experts already lived in.

`composition_plane_coords.py` builds, per cached state, the affine plane through eps_u spanned
by (eps_a - eps_u) and (eps_b - eps_u). Every prediction the FROZEN composition can make lies in
it. This script runs a trained adapter at the same cached states and projects its three branches
into that same frozen frame, so the question "did the adapter move the composition toward the
joint, and did it leave the plane to do it?" becomes a position on a picture.

The frame is deliberately the FROZEN one. Measuring the adapter against a plane it helped define
would hide exactly the thing being asked.

Writes <out>/adapter_in_plane.json. Needs a GPU; about three forwards per state.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from correction_span_common import load_step, GUIDANCE, PAIR, SEEDS          # noqa: E402
from poe_repair.run import make_ctx                                          # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl                            # noqa: E402
from poe_repair.experiments.one_pair_one_seed.trainer import (               # noqa: E402
    attach_lora, load_lora_state)
from poe_repair.methods._sampling import add_time_ids                        # noqa: E402

CKPT = ("/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/"
        "r16_s0_25_plain/checkpoints/lora_step_030000.pt")
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/composition_plane")
STEPS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 49]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    ctx = make_ctx(num_inference_steps=50, guidance_scale=GUIDANCE)
    unet, dev, dt = ctx.models["unet"], ctx.device, ctx.dtype

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    class _L:  # the attach helper only reads cfg.lora
        pass
    lo = _L()
    lcfg = cfg["lora"] if isinstance(cfg, dict) else cfg.lora
    for k in ("rank", "alpha", "dropout", "target_modules", "init", "adapter_name"):
        setattr(lo, k, lcfg[k] if isinstance(lcfg, dict) else getattr(lcfg, k))
    shim = _L(); shim.lora = lo
    info = attach_lora(unet, shim)
    load_lora_state(unet, ck["lora_state"])
    unet.eval()
    print(f"[lora] {info['matched_modules']} modules, step {ck['step']}", flush=True)

    seq = {}
    for tag, prompt in (("a", "a cat"), ("b", "a dog"), ("u", "")):
        seq[tag] = encode_prompt_sdxl(prompt, models=ctx.models, device=dev, dtype=dt)

    out = {"ckpt": args.ckpt, "step_trained": int(ck["step"]), "pair": PAIR, "seeds": {}}
    for sd in seeds:
        rows, t0 = [], time.time()
        for st in STEPS:
            s = load_step(sd, st)
            # the frozen frame, exactly as the plane figure builds it
            u0 = s.eps_u.reshape(-1).double()
            da = s.eps_a.reshape(-1).double() - u0
            db = s.eps_b.reshape(-1).double() - u0
            e1 = da / da.norm().clamp_min(1e-12)
            p = db - (db @ e1) * e1
            e2 = p / p.norm().clamp_min(1e-12)

            x = s.x_t.to(device=dev, dtype=dt)
            tt = torch.tensor([s.timestep] * 3, device=dev, dtype=torch.long)
            pe = torch.cat([seq["a"][0], seq["b"][0], seq["u"][0]], 0).to(device=dev, dtype=dt)
            pool = torch.cat([seq["a"][1], seq["b"][1], seq["u"][1]], 0).to(device=dev, dtype=dt)
            cond = {"text_embeds": pool,
                    "time_ids": add_time_ids(height=1024, width=1024, batch_size=3,
                                             device=dev, dtype=dt)}
            with torch.no_grad():
                pred = unet(x.repeat(3, 1, 1, 1), tt, encoder_hidden_states=pe,
                            added_cond_kwargs=cond, timestep_cond=None).sample.float()
            # the cached frame lives on CPU; bring the predictions to it
            aT, bT, uT = [pred[i].reshape(-1).double().cpu() for i in range(3)]

            def xy(v):
                return [float(v @ e1), float(v @ e2)]

            # everything measured from the FROZEN origin u0
            A, B, U = xy(aT - u0), xy(bT - u0), xy(uT - u0)
            Tt = (aT + bT - uT) - u0
            Txy = xy(Tt)
            out_of = float((Tt - (Txy[0] * e1 + Txy[1] * e2)).norm())
            rows.append({"s": st, "a": [round(v, 3) for v in A], "b": [round(v, 3) for v in B],
                         "u": [round(v, 3) for v in U], "T": [round(v, 3) for v in Txy],
                         "o": round(out_of, 3)})
            torch.cuda.empty_cache()
        out["seeds"][str(sd)] = rows
        print(f"[seed {sd}] {len(rows)} states ({time.time()-t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adapter_in_plane.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"[done] -> {OUT/'adapter_in_plane.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
