#!/usr/bin/env python
"""Method comparison grids: rows are seeds, columns are Mono | PoE | PoE + LoRA.

One figure per (prompt set, adapter). Two prompt sets:

  catdog   "a cat" x "a dog". Mono is the joint prompt "a cat and a dog"; PoE is the plain
           product of experts with no adapter. Both are the cached renders in the training
           cache cell (heldout/a_cat__x__a_dog/seed_N/{mono,poe}.png, 50 DDIM steps, w=7.5),
           and the LoRA column starts from that cell's cached x_T and text embeddings, so all
           three columns share the same initial noise.
  dogdog   "a dog" x "a dog". Mono is the joint prompt "a dog and a dog"; PoE is
           PoE("a dog","a dog") with no adapter; both rendered here from the seed's x_T, the
           same way dog_x_dog_probe.py renders them.

Four adapters, each a rank-r cross-attention LoRA on SDXL's attn2 q/k/v, loaded from the
checkpoint the ADAPTERS table names. The LoRA column is the pre-registered setting from
decisions-taken-here.md: lambda=1, active over denoising steps 0-10 of 50.

Seeds 1 and 2 are in every adapter's training seed pool and seeds 9 and 12 are held out. For
the three pooled adapters "a cat" x "a dog" is a held-out pair, so every cat x dog cell is
unseen regardless of seed; for the cross-seed adapter (trained on cat x dog seeds 1-8) seeds 1
and 2 are training cells and 9 and 12 are held-out cells.

Stages (each is one process; the adapters are loaded one per process because peft cannot
swap a rank-8 adapter for a rank-16 one in place):

    python scripts/showcase/method_grid.py --baselines            # dogdog Mono + PoE, no adapter
    python scripts/showcase/method_grid.py --adapter r8_240k      # LoRA column, both prompt sets
    python scripts/showcase/method_grid.py --compose              # all 8 grids from what is on disk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SEEDS = (1, 2, 9, 12)
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/method_grid")
CACHE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
CATDOG_CELL_ROOT = CACHE_ROOT / "heldout" / "a_cat__x__a_dog"

SHOWCASE = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase")
CROSS_SEED = Path(
    "/datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog"
    "/taskB__k04_ep2000_resumed__wandb-pueuo7bl"
)

# name -> (checkpoint, W&B run, column label). Checkpoint steps are the exact files on disk;
# the rank-16 and rank-32 runs were resumed once, which is why their step counts carry the
# +18 / +50 offset.
ADAPTERS = {
    "r8_240k": dict(
        ckpt=SHOWCASE / "phase1_r8_450k/checkpoints/lora_step_240000.pt",
        wandb="prime_lab/poe-repair-animals-compose/n1w4fw5b",
        label="LoRA rank 8, step 240k",
    ),
    "r16_50k": dict(
        ckpt=SHOWCASE / "phase1_r16_100k/checkpoints/lora_step_050018.pt",
        wandb="prime_lab/poe-repair-animals-compose/gcej8ib9",
        label="LoRA rank 16, step 50k",
    ),
    "r32_60k": dict(
        ckpt=SHOWCASE / "phase1_r32_100k/checkpoints/lora_step_060050.pt",
        wandb="prime_lab/poe-repair-animals-compose/6xc2l8ix",
        label="LoRA rank 32, step 60k",
    ),
    "crossseed": dict(
        ckpt=CROSS_SEED / "checkpoints/lora_step_100000.pt",
        wandb="prime_lab/poe-repair-cross-seed/pueuo7bl",
        label="LoRA cross-seed (cat x dog, seeds 1-8), step 100k",
    ),
}

PROMPT_SETS = {
    "catdog": dict(prompt_a="a cat", prompt_b="a dog", title="a cat  x  a dog"),
    "dogdog": dict(prompt_a="a dog", prompt_b="a dog", title="a dog  x  a dog"),
}

WINDOW_ON_STEPS = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
LAMBDA = 1.0


# --------------------------------------------------------------------------- rendering


def _ctx():
    from poe_repair.run import make_ctx
    return make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)


def _dogdog_inputs(ctx, seed: int):
    """x_T from the seed, prompt embeddings for "a dog", the joint prompt, and empty."""
    from poe_repair.config import RunConfig
    from poe_repair.methods._sampling import initial_latents_for_pair
    from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl

    cfg = RunConfig()
    cell = PairSeedCell(
        pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / "dog_x_dog_probe",
        pair_slug="dog_x_dog_probe",
        prompt_a="a dog", prompt_b="a dog",
        seed=seed, regime="collision", height=1024, width=1024, grid_assets={},
    )
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_a, pool_a = enc("a dog")
    seq_j, pool_j = enc("a dog and a dog")
    seq_e, pool_e = enc("")
    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
        height=1024, width=1024, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    return common, (seq_a, pool_a), (seq_j, pool_j)


def _catdog_inputs(ctx, seed: int):
    """The cached cell's x_T and text embeddings, so the LoRA column shares the initial noise
    of the cached mono.png and poe.png."""
    import torch

    emb = torch.load(CATDOG_CELL_ROOT / f"seed_{seed}" / "embeddings.pt",
                     map_location=ctx.device, weights_only=False)
    t = lambda k: emb[k].to(device=ctx.device, dtype=ctx.dtype)
    common = dict(
        init_latents=t("init_latents"), models=ctx.models, scheduler=ctx.scheduler,
        seq_e=t("seq_uncond"), pool_e=t("pool_uncond"),
        guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
        height=1024, width=1024, euler_init_noise_sigma=float(emb["euler_init_noise_sigma"]),
        device=ctx.device, dtype=ctx.dtype,
    )
    return common, (t("seq_a"), t("pool_a")), (t("seq_b"), t("pool_b"))


def render_baselines(seeds) -> None:
    """dogdog Mono and PoE columns, no adapter attached. catdog's baselines are the cached
    mono.png / poe.png and are not re-rendered."""
    from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image

    ctx = _ctx()
    mono_dir = OUT_ROOT / "dogdog" / "mono"
    poe_dir = OUT_ROOT / "dogdog" / "poe"
    mono_dir.mkdir(parents=True, exist_ok=True)
    poe_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in seeds:
        common, (seq_a, pool_a), (seq_j, pool_j) = _dogdog_inputs(ctx, seed)
        out = run_cfg(seq_cond=seq_j, pool_cond=pool_j, **common)
        write_decoded_image(out.image, mono_dir / f"seed_{seed}.png")
        out = run_cfg_poe(seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a, **common)
        write_decoded_image(out.image, poe_dir / f"seed_{seed}.png")
        manifest.append({"seed": seed, "mono_prompt": "a dog and a dog",
                         "poe_prompts": ["a dog", "a dog"]})
        print(f"[method_grid] dogdog baselines seed={seed} done", flush=True)
    (OUT_ROOT / "dogdog" / "baselines_manifest.json").write_text(json.dumps({
        "guidance_scale": GUIDANCE_SCALE, "num_inference_steps": NUM_INFERENCE_STEPS,
        "adapter": None, "cells": manifest}, indent=2))


def _attach_and_load(unet, ckpt_path: Path) -> str:
    import torch
    from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["lora_state"]
    meta = ckpt["config"]["lora"]
    lora_cfg = LoRAConfig(
        rank=int(meta["rank"]), alpha=int(meta["alpha"]), dropout=0.0,
        target_modules=tuple(meta["target_modules"]), init="gaussian",
        adapter_name=str(meta["adapter_name"]),
    )
    info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=lora_cfg))
    lora_trainer.load_lora_state(unet, state)
    print(f"[method_grid] loaded {ckpt_path} rank={meta['rank']} alpha={meta['alpha']} "
          f"step={ckpt.get('step')} n_tensors={len(state)} matched={info.get('n_matched')}",
          flush=True)
    return lora_cfg.adapter_name


def render_adapter(name: str, seeds, prompt_sets) -> None:
    from poe_repair.methods._sampling import run_lora_residual_inject_masked, write_decoded_image

    spec = ADAPTERS[name]
    ctx = _ctx()
    adapter_name = _attach_and_load(ctx.models["unet"], spec["ckpt"])
    mask = [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)

    for ps in prompt_sets:
        out_dir = OUT_ROOT / ps / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cells = []
        for seed in seeds:
            if ps == "catdog":
                common, (seq_a, pool_a), (seq_b, pool_b) = _catdog_inputs(ctx, seed)
            else:
                common, (seq_a, pool_a), _ = _dogdog_inputs(ctx, seed)
                seq_b, pool_b = seq_a, pool_a
            out = run_lora_residual_inject_masked(
                seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                cfg_mask=mask, composition_mode="with_prompt",
                lambda_value=LAMBDA, lora_adapter_name=adapter_name, **common,
            )
            png = out_dir / f"seed_{seed}.png"
            write_decoded_image(out.image, png)
            cells.append({"seed": seed, "png": str(png),
                          "max_delta_norm": max(out.extras["delta_norm_per_step"])})
            print(f"[method_grid] {ps} {name} seed={seed} max||r_hat||="
                  f"{cells[-1]['max_delta_norm']:.3f}", flush=True)
        (out_dir / "manifest.json").write_text(json.dumps({
            "adapter": name, "checkpoint": str(spec["ckpt"]), "wandb": spec["wandb"],
            "lambda": LAMBDA, "window_on_steps": WINDOW_ON_STEPS,
            "num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE,
            "prompts": [PROMPT_SETS[ps]["prompt_a"], PROMPT_SETS[ps]["prompt_b"]],
            "cells": cells}, indent=2))


# --------------------------------------------------------------------------- composing


def _panel_path(ps: str, column: str, seed: int) -> Path:
    if ps == "catdog" and column in ("mono", "poe"):
        return CATDOG_CELL_ROOT / f"seed_{seed}" / f"{column}.png"
    return OUT_ROOT / ps / column / f"seed_{seed}.png"


def compose(ps: str, adapter: str, seeds, thumb: int = 384) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    spec = ADAPTERS[adapter]
    pset = PROMPT_SETS[ps]
    columns = [
        ("mono", f"Mono\n\"{pset['prompt_a']} and {pset['prompt_b']}\""),
        ("poe", "PoE\nno adapter"),
        (adapter, f"PoE + {spec['label']}\nlambda={LAMBDA:g}, steps 0-{WINDOW_ON_STEPS}"),
    ]
    title_h, header_h, label_w, gap = 44, 56, 96, 6
    W = label_w + len(columns) * (thumb + gap)
    H = title_h + header_h + len(seeds) * (thumb + gap)
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((label_w, 10), f"{pset['title']}   rows: seeds, columns: methods, same x_T per row",
              fill="black", font=font(20))
    for c, (_, label) in enumerate(columns):
        x = label_w + c * (thumb + gap)
        draw.multiline_text((x + 6, title_h + 4), label, fill="black", font=font(15), spacing=2)

    missing = []
    for r, seed in enumerate(seeds):
        y = title_h + header_h + r * (thumb + gap)
        draw.text((8, y + thumb // 2 - 10), f"seed {seed}", fill="black", font=font(16))
        for c, (col, _) in enumerate(columns):
            x = label_w + c * (thumb + gap)
            p = _panel_path(ps, col, seed)
            if p.exists():
                panel = Image.open(p).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
            else:
                panel = Image.new("RGB", (thumb, thumb), "lightgray")
                missing.append(str(p))
            grid.paste(panel, (x, y))

    out = OUT_ROOT / "figures" / f"{ps}__{adapter}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"[method_grid] wrote {out}" + (f"  MISSING {len(missing)}: {missing}" if missing else ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true", help="render dogdog Mono and PoE columns")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), help="render this adapter's LoRA column")
    ap.add_argument("--compose", action="store_true", help="compose every grid from what is on disk")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--prompt-sets", default=",".join(PROMPT_SETS))
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    prompt_sets = args.prompt_sets.split(",")
    if not (args.baselines or args.adapter or args.compose):
        ap.error("pass --baselines, --adapter NAME, or --compose")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.baselines:
        render_baselines(seeds)
    if args.adapter:
        render_adapter(args.adapter, seeds, prompt_sets)
    if args.compose:
        for ps in prompt_sets:
            for adapter in ADAPTERS:
                compose(ps, adapter, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
