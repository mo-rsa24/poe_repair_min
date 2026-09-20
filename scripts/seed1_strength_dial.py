#!/usr/bin/env python
"""Cat and dog, seed 1: the adapter's correction turned up and down at render time.

Every render so far used the correction at exactly the strength it was trained at. That strength
is a number the sampler multiplies the correction by, and it needs no training, no joint prompt,
and no knowledge of the pair. Two adapters are swept: the one that draws well and does not
compose on this seed, and the one that composes and draws badly.

    python scripts/seed1_strength_dial.py --render
    python scripts/seed1_strength_dial.py --sheet
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "showcase"))
O = Path("/datasets/mmolefe/poe_repair_min/outputs")
WORK = O / "seed1_strength_dial"
PAIR, SEED = "a_cat__x__a_dog", 1
LAMBDAS = (0.6, 0.8, 1.0, 1.3, 1.6, 2.0)
ADAPTERS = [
    ("draws well, does not compose", "showcase/improve_r32/pool43-all50", 40000, 32),
    ("composes, draws badly",        "showcase/phase1_r32_100k",          30050, 32),
]


def png_for(tag: str, lam: float) -> Path:
    return WORK / tag / f"lambda_{lam}.png"


def render(which: int) -> None:
    label, run, step, rank = ADAPTERS[which]
    tag = run.split("/")[-1]
    free, total = torch.cuda.mem_get_info(0)
    if (total - free) / 1e9 > 1.0:
        raise SystemExit(f"device holds {(total-free)/1e9:.1f} GB, refusing to share")
    from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
    from poe_repair.methods._sampling import write_decoded_image
    from poe_repair.run import make_ctx
    import lambda_boundary_probe as lbp

    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = rank
    info = lbp._attach_and_load_lora(ctx.models["unet"], O / run / "checkpoints" / f"lora_step_{step:06d}.pt")
    if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
        raise SystemExit(f"{label}: matched {info['n_matched']}, loaded {info['n_loaded']}")
    print(f"{label}: rank {rank}, step {info['checkpoint_step']}", flush=True)
    cell = cell_from_slug(PAIR, SEED)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    for lam in LAMBDAS:
        out = png_for(tag, lam)
        if out.exists():
            continue
        t0 = time.time()
        res = run_lora_langevin_windowed_poe(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype, lambda_value=float(lam), k=0, c=0.0,
            corrector_window=None, noise_seed=SEED, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
            lambda_window=(0, 50), corrector_score="frozen",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(res.image, out)
        del res; gc.collect(); torch.cuda.empty_cache()
        print(f"{tag} lambda {lam} ({time.time()-t0:.0f}s)", flush=True)


def sheet() -> None:
    T, PAD, TOP, LEFT = 300, 10, 24, 240
    rows = [(lab, run.split("/")[-1]) for lab, run, _, _ in ADAPTERS]
    W = LEFT + len(LAMBDAS) * (T + PAD) + PAD
    H = TOP + len(rows) * (T + PAD) + PAD
    c = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(c)
    for j, lam in enumerate(LAMBDAS):
        d.text((LEFT + j * (T + PAD), 6), f"strength {lam}" + ("  (as trained)" if lam == 1.0 else ""), fill="black")
    for i, (lab, tag) in enumerate(rows):
        y = TOP + i * (T + PAD)
        d.text((8, y + T // 2), lab, fill="black")
        for j, lam in enumerate(LAMBDAS):
            p = png_for(tag, lam)
            if p.exists():
                c.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (LEFT + j * (T + PAD), y))
    out = REPO / "artifacts/results/composing-unseen-pairs/seed1-correction-strength.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    c.save(out); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=int); ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.sheet: sheet()
    elif a.render is not None: render(a.render)
    else: raise SystemExit("pass --render <0|1> or --sheet")
