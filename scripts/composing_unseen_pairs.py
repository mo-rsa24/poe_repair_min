#!/usr/bin/env python
"""Sixteen pairs the adapter has never trained on, six seeds each, for the paper's figure.

Ten of them are object pairs matched to the seven the pool did train on. Six are look-alike
species, where plain composition reliably draws one animal instead of two, so the before picture
carries the argument without a measurement.

Adapter only on this pass: a seed search is cheap and the no-adapter counterpart is worth
rendering only for the seeds that win. One checkpoint, one process.

    python scripts/composing_unseen_pairs.py --render        # 96 renders, resumes if interrupted
    python scripts/composing_unseen_pairs.py --sheets        # one contact sheet per pair
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

CKPT = ("/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_040000.pt")
RANK = 32
WORK = Path("/datasets/mmolefe/poe_repair_min/outputs/composing_unseen_pairs")
FILED = REPO / "artifacts/results/composing-unseen-pairs"

# (pair, seeds). Object pairs have twelve cached seeds; the look-alikes have eight, and
# wolf x husky only four, so its row is shorter rather than padded with seeds that do not exist.
OBJECTS = [
    ("a_briefcase__x__a_ceramic_bowl", (1, 2, 3, 4, 5, 6)),
    ("a_mug__x__a_wine_glass", (1, 2, 3, 4, 5, 6)),
    ("a_suitcase__x__a_desk_fan", (1, 2, 3, 4, 5, 6)),
    ("a_microwave__x__a_potted_plant", (1, 2, 3, 4, 5, 6)),
    ("a_feather_pillow__x__a_cast_iron_pan", (1, 2, 3, 4, 5, 6)),
    ("a_drum_set__x__a_snowman", (1, 2, 3, 4, 5, 6)),
    ("a_lab_microscope__x__a_hay_bale", (1, 2, 3, 4, 5, 6)),
    ("a_bathtub__x__a_streetlamp", (1, 2, 3, 4, 5, 6)),
    ("a_chessboard__x__a_lantern", (1, 2, 3, 4, 5, 6)),
    ("a_birdcage__x__a_watering_can", (1, 2, 3, 4, 5, 6)),
]
LOOKALIKES = [
    ("an_eagle__x__a_hawk", (1, 2, 3, 4, 5, 6)),
    ("a_leopard__x__a_jaguar", (1, 2, 3, 4, 5, 6)),
    ("a_frog__x__a_toad", (1, 2, 3, 4, 5, 6)),
    ("a_goose__x__a_swan", (1, 2, 3, 4, 5, 6)),
    ("a_seal__x__a_walrus", (1, 2, 3, 4, 5, 6)),
    ("a_wolf__x__a_husky", (1, 2, 3, 4)),
]
GROUPS = [("objects", OBJECTS), ("look-alike species", LOOKALIKES)]


def png_for(pair: str, seed: int) -> Path:
    return WORK / pair / f"seed_{seed}__adapter.png"


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

    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], Path(CKPT))
    if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
        raise SystemExit("the adapter matched or loaded nothing; this would render the frozen model")
    print(f"adapter: rank {RANK}, {info['n_matched']} sites, step {info['checkpoint_step']}", flush=True)

    todo = [(p, s) for _, grp in GROUPS for p, seeds in grp for s in seeds]
    for n, (pair, seed) in enumerate(todo, 1):
        out = png_for(pair, seed)
        if out.exists():
            continue
        t0 = time.time()
        cell = cell_from_slug(pair, seed)
        init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
        emb = encode_pair(cell, ctx)
        res = run_lora_langevin_windowed_poe(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype, lambda_value=1.0, k=0, c=0.0, corrector_window=None,
            noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
            lambda_window=(0, 50), corrector_score="frozen",
        )
        on = sum(1 for r in res.extras["per_step"] if r["adapter_on"])
        if on != 50:
            raise SystemExit(f"{pair} seed {seed}: adapter on {on} steps, not 50")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(res.image, out)
        del res
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{n}/{len(todo)}] {pair} seed {seed} ({time.time()-t0:.0f}s)", flush=True)


def sheets() -> None:
    FILED.mkdir(parents=True, exist_ok=True)
    T, PAD, TOP = 300, 10, 22
    for gname, grp in GROUPS:
        rows = [(p, s) for p, s in grp if any(png_for(p, x).exists() for x in s)]
        if not rows:
            continue
        cols = max(len(s) for _, s in rows)
        W = 250 + cols * (T + PAD) + PAD
        H = PAD + len(rows) * (T + PAD) + TOP
        canvas = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(canvas)
        for j in range(cols):
            d.text((250 + j * (T + PAD), 6), f"seed {j+1}", fill="black")
        for i, (pair, seeds) in enumerate(rows):
            y = TOP + PAD + i * (T + PAD)
            d.text((8, y + T // 2), pair.replace("__x__", "  x  "), fill="black")
            for j, s in enumerate(seeds):
                p = png_for(pair, s)
                if p.exists():
                    canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS),
                                 (250 + j * (T + PAD), y))
        out = FILED / f"{gname.replace(' ', '-')}-six-seeds.png"
        canvas.save(out)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sheets", action="store_true")
    a = ap.parse_args()
    if a.render:
        render()
    elif a.sheets:
        sheets()
    else:
        raise SystemExit("pass --render or --sheets")
