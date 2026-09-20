#!/usr/bin/env python
"""Cat and dog, seed 1: one adapter draws the early steps, another draws the late ones.

Composition is decided in the first ten steps and the picture is finished in the last forty.
On this seed the original-pool adapter composes and draws badly, and the 43-cell adapter draws
well and does not compose. This hands the early steps to the first and the late steps to the
second, which needs no training, no joint prompt, and nothing about the pair.

    python scripts/seed1_two_adapters.py --render
    python scripts/seed1_two_adapters.py --sheet
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "showcase"))
O = Path("/datasets/mmolefe/poe_repair_min/outputs")
WORK = O / "seed1_two_adapters"
PAIR, SEED = "a_cat__x__a_dog", 1

EARLY = (O / "showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt", 32, "composes")
LATE  = (O / "showcase/improve_r32/pool43-all50/checkpoints/lora_step_040000.pt", 32, "draws well")
SWITCHES = (0, 10, 15, 20, 25, 50)   # 0 = the late adapter alone, 50 = the early adapter alone


def png_for(sw: int) -> Path:
    return WORK / f"switch_at_{sw:02d}.png"


def render() -> None:
    free, total = torch.cuda.mem_get_info(0)
    if (total - free) / 1e9 > 1.0:
        raise SystemExit(f"device holds {(total-free)/1e9:.1f} GB, refusing to share")
    from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.run import make_ctx
    import lambda_boundary_probe as lbp

    ctx = make_ctx(); unet = ctx.models["unet"]
    # Two adapters on one U-Net, under different names, so the sampler can pick per step.
    for ckpt, rank, name in ((EARLY[0], EARLY[1], "early"), (LATE[0], LATE[1], "late")):
        lbp.LORA_RANK = lbp.LORA_ALPHA = rank
        lbp.LORA_ADAPTER_NAME = name
        info = lbp._attach_and_load_lora(unet, ckpt)
        if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
            raise SystemExit(f"adapter {name}: matched {info['n_matched']}, loaded {info['n_loaded']}")
        print(f"attached '{name}': rank {rank}, step {info['checkpoint_step']}", flush=True)

    cell = cell_from_slug(PAIR, SEED)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    for sw in SWITCHES:
        out = png_for(sw)
        if out.exists():
            continue
        t0 = time.time()
        res = run_lora_langevin_windowed_poe(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype, lambda_value=1.0, k=0, c=0.0,
            corrector_window=None, noise_seed=SEED,
            lora_adapter_name="early", lambda_window=(0, 50), corrector_score="frozen",
            lora_adapter_name_late="late", adapter_switch_at=sw,
        )
        # The handover has to have happened where its name says, or the tile is mislabelled.
        names = [r.get("adapter_name") for r in res.extras["per_step"]]
        n_early = sum(1 for n in names if n == "early")
        if n_early != sw:
            raise SystemExit(f"switch {sw}: the early adapter drew {n_early} steps, not {sw}")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(res.image, out)
        del res; gc.collect(); torch.cuda.empty_cache()
        print(f"switch at {sw}: early drew {n_early} steps ({time.time()-t0:.0f}s)", flush=True)


def sheet() -> None:
    T, PAD, TOP = 300, 10, 38
    have = [sw for sw in SWITCHES if png_for(sw).exists()]
    W = PAD + len(have) * (T + PAD); H = TOP + T + PAD
    c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
    for j, sw in enumerate(have):
        x = PAD + j * (T + PAD)
        if sw == 0:   lab, sub = "draws well, alone", "the 43-cell adapter, all 50 steps"
        elif sw == 50: lab, sub = "composes, alone", "the original adapter, all 50 steps"
        else:          lab, sub = f"hand over at step {sw}", "composes first, then draws well"
        d.text((x, 6), lab, fill="black"); d.text((x, 20), sub, fill="#666666")
        c.paste(Image.open(png_for(sw)).convert("RGB").resize((T, T), Image.LANCZOS), (x, TOP))
    out = REPO / "artifacts/results/composing-unseen-pairs/seed1-two-adapters.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    c.save(out); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true"); ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.sheet: sheet()
    elif a.render: render()
    else: raise SystemExit("pass --render or --sheet")
