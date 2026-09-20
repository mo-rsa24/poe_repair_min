#!/usr/bin/env python
"""Cat and dog, seed 1, rendered once by every adapter this project has trained.

One checkpoint per process, because attaching a second adapter to the same U-Net stacks it on
the first. The shell loop below calls --render once per row, then --sheet assembles.

    for i in $(seq 0 21); do python scripts/seed1_across_adapters.py --render $i; done
    python scripts/seed1_across_adapters.py --sheet
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

O = Path("/datasets/mmolefe/poe_repair_min/outputs")
PAIR, SEED = "a_cat__x__a_dog", 1
WORK = O / "seed1_across_adapters"

# (label, run directory, checkpoint step, rank). The step is the last save unless an earlier one
# is the one this project actually reads, which is noted where it applies.
ROWS = [
    ("no adapter",                 None,                                            None,   32),
    ("original, rank 8, 450k",     "showcase/phase1_r8_450k",                       450000,  8),
    ("original, rank 16",          "showcase/phase1_r16_100k",                      30000,  16),
    ("original, rank 32",          "showcase/phase1_r32_100k",                      30050,  32),
    ("original + weight decay",    "showcase/phase1_r32_wd0.1_100k",                30000,  32),
    ("original + weight average",  "showcase/phase1_r32_wd0.1_ema0.999_40k",        30000,  32),
    ("judged on the picture",      "showcase/phase1_r32_x0loss_40k",                25000,  32),
    ("charged for its size",       "showcase/phase1_r32_energy0.05_from30050_40k",  40050,  32),
    ("43 cells, all 50 steps",     "showcase/improve_r32/pool43-all50",             40000,  32),
    ("43 cells, first 25 steps",   "showcase/improve_r32/pool43-early25",           40000,  32),
    ("43 cells, extra-dir charge", "showcase/improve_r32/pool43-all50-orth3",       40000,  32),
    ("23 cells, animals only",     "showcase/improve_r32/pool43-animals",           30000,  32),
    ("54 cells",                   "showcase/improve_r32/v54d_P",                   30000,  32),
    ("182 cells",                  "showcase/improve_r32/v54d_C1_allseeds",         30000,  32),
    ("72 cells",                   "showcase/improve_r32/v56_wd0",                  10000,  32),
    ("empty branch from cache",    "correction_loss_variants/v1_freeze_null_r16_s0_25",            30000, 16),
    ("only the empty branch",      "correction_loss_variants/v3_adapt_null_only_r16_s0_25",        30000, 16),
    ("target is a picture",        "correction_loss_variants/v6_render_teacher_r16_s0_25",         30000, 16),
    ("picture + cached empty",     "correction_loss_variants/v6a_render_teacher_null_frozen_r16_s0_25", 30000, 16),
    ("54-cell control",            "correction_loss_variants/r16_s0_25_plain",       30000, 16),
    ("54-cell + drift penalty",    "correction_loss_variants/r16_s0_25_anchor",      30000, 16),
    ("54-cell + plural prompt",    "correction_loss_variants/r16_s0_25_plurality",   30000, 16),
    ("54-cell + connective",       "correction_loss_variants/r16_s0_25_connective",  30000, 16),
]


def png_for(i: int) -> Path:
    return WORK / f"{i:02d}.png"


def render(i: int) -> None:
    label, run, step, rank = ROWS[i]
    out = png_for(i)
    if out.exists():
        print(f"{i:02d} {label}: already rendered", flush=True)
        return
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
    if run is None:
        adapter_steps = 0                      # the control: the adapter is never on
        lbp.LORA_RANK = lbp.LORA_ALPHA = rank
        lbp._attach_and_load_lora(ctx.models["unet"], Path(
            O / "showcase/improve_r32/pool43-all50/checkpoints/lora_step_040000.pt"))
    else:
        adapter_steps = 50
        ckpt = O / run / "checkpoints" / f"lora_step_{step:06d}.pt"
        if not ckpt.exists():
            raise SystemExit(f"{i:02d} {label}: no checkpoint at {ckpt}")
        lbp.LORA_RANK = lbp.LORA_ALPHA = rank
        info = lbp._attach_and_load_lora(ctx.models["unet"], ckpt)
        if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
            raise SystemExit(f"{i:02d} {label}: matched {info['n_matched']}, loaded {info['n_loaded']}")
        print(f"{i:02d} {label}: rank {rank}, step {info['checkpoint_step']}", flush=True)

    cell = cell_from_slug(PAIR, SEED)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    res = run_lora_langevin_windowed_poe(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype, lambda_value=1.0, k=0, c=0.0, corrector_window=None,
        noise_seed=SEED, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
        lambda_window=(0, adapter_steps), corrector_score="frozen",
    )
    on = sum(1 for r in res.extras["per_step"] if r["adapter_on"])
    if on != adapter_steps:
        raise SystemExit(f"{i:02d} {label}: adapter on {on} steps, not {adapter_steps}")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_decoded_image(res.image, out)
    del res
    gc.collect(); torch.cuda.empty_cache()
    print(f"{i:02d} {label}: wrote {out}", flush=True)


def sheet() -> None:
    have = [(i, r) for i, r in enumerate(ROWS) if png_for(i).exists()]
    cols, T, PAD, TOP = 6, 300, 10, 30
    rows = (len(have) + cols - 1) // cols
    W = PAD + cols * (T + PAD)
    H = PAD + rows * (T + TOP + PAD)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for n, (i, (label, run, step, rank)) in enumerate(have):
        cx, cy = n % cols, n // cols
        x = PAD + cx * (T + PAD)
        y = PAD + cy * (T + TOP + PAD)
        d.text((x, y + 4), f"{label}", fill="black")
        d.text((x, y + 16), f"rank {rank}" + (f", step {step:,}" if step else ""), fill="#666666")
        canvas.paste(Image.open(png_for(i)).convert("RGB").resize((T, T), Image.LANCZOS), (x, y + TOP))
    out = REPO / "seed1_across_every_adapter.png"
    canvas.save(out)
    print(f"wrote {out}  ({len(have)} of {len(ROWS)} rendered)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=int)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, (label, run, step, rank) in enumerate(ROWS):
            print(f"{i:02d}  {label:30} {run or '(control)'}  step={step}  rank={rank}")
    elif a.sheet:
        sheet()
    elif a.render is not None:
        render(a.render)
    else:
        raise SystemExit("pass --render <i>, --sheet or --list")
