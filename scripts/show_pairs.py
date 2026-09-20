#!/usr/bin/env python
"""Render a handful of pairs with the step-40,000 adapter, beside plain composition.

Two sheets, both written to the repo root so they are one click from the file tree. Each column
is one cell: the top tile is plain product-of-experts with no adapter, the bottom tile is the same
seed with the adapter on the whole path. The gap between the two rows is what the adapter buys.

    python scripts/show_pairs.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell   # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug        # noqa: E402
from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe    # noqa: E402
from poe_repair.methods._sampling import write_decoded_image                   # noqa: E402
from poe_repair.run import make_ctx                                            # noqa: E402

CKPT = ("/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_040000.pt")
RANK = 32

# Sheet 1: the pair the project is judged on, three seeds nobody has rendered with this checkpoint.
CATDOG = [("a_cat__x__a_dog", s) for s in (1, 4, 6)]
# Sheet 2: four pairs the adapter has never trained on, three of them objects rather than animals.
FOUR = [("a_chessboard__x__a_lantern", 1),
        ("a_briefcase__x__a_ceramic_bowl", 2),
        ("a_birdcage__x__a_watering_can", 1),
        ("a_cat__x__a_horse", 3)]

WORK = Path("/datasets/mmolefe/poe_repair_min/outputs/show_pairs")


def render_one(ctx, lbp, pair: str, seed: int, adapter_steps: int) -> Path:
    png = WORK / pair / f"seed_{seed}__adapter_on_{adapter_steps}.png"
    if png.exists():
        return png
    png.parent.mkdir(parents=True, exist_ok=True)
    cell = cell_from_slug(pair, seed)
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_lora_langevin_windowed_poe(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        lambda_value=1.0, k=0, c=0.0, corrector_window=None,
        noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
        lambda_window=(0, adapter_steps), corrector_score="frozen",
    )
    # The window has to have selected what its name claims, or the tile is mislabelled.
    on = sum(1 for r in out.extras["per_step"] if r["adapter_on"])
    if on != adapter_steps:
        raise SystemExit(f"{pair} seed {seed}: adapter on for {on} steps, not {adapter_steps}")
    write_decoded_image(out.image, png)
    del out
    gc.collect(); torch.cuda.empty_cache()
    return png


def sheet(cells, out_path: Path, ctx, lbp) -> None:
    T, PAD, TOP, LEFT = 330, 12, 26, 96
    W = LEFT + len(cells) * (T + PAD) + PAD
    H = TOP + 2 * (T + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, TOP + T // 2), "no adapter", fill="black")
    d.text((8, TOP + T + PAD + T // 2), "adapter", fill="black")
    for j, (pair, seed) in enumerate(cells):
        x = LEFT + j * (T + PAD)
        d.text((x, 8), f"{pair.replace('__x__', ' x ')}  seed {seed}", fill="black")
        for i, steps in enumerate((0, 50)):
            p = render_one(ctx, lbp, pair, seed, steps)
            print(f"[{pair} seed {seed}] adapter on {steps} steps -> {p}", flush=True)
            canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS),
                         (x, TOP + i * (T + PAD)))
    canvas.save(out_path)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    free, total = torch.cuda.mem_get_info(0)
    if (total - free) / 1e9 > 1.0:
        raise SystemExit(f"device already holds {(total-free)/1e9:.1f} GB, refusing to share")
    import lambda_boundary_probe as lbp
    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], Path(CKPT))
    if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
        raise SystemExit("the adapter matched or loaded nothing; this would render the frozen model")
    print(f"adapter: rank {RANK}, {info['n_matched']} sites, step {info['checkpoint_step']}", flush=True)
    sheet(CATDOG, REPO / "cat_and_dog_seeds_1_4_6.png", ctx, lbp)
    sheet(FOUR, REPO / "four_unseen_pairs.png", ctx, lbp)


if __name__ == "__main__":
    main()
