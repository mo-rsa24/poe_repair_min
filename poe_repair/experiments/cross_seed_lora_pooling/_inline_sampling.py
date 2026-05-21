"""Inline per-epoch sampling for the cross-seed pooled trainer.

Used during training to render one PNG per seed at the current LoRA
state, plus a composed contact sheet for at-a-glance review. Designed
to be cheap (20 DDIM steps by default) so it can run every N epochs
without dominating wall-clock time. The post-hoc, full-quality renderer
(``scripts/cross_seed_lora_pooling/render_per_epoch.py``) handles the
final 50-step / 1024² pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir
from poe_repair.training_cache import CellPath


@dataclass
class InlineSamplerContext:
    seeds: list[int]
    init_by_seed: dict[int, torch.Tensor]
    embeddings: dict[str, torch.Tensor]
    device: torch.device | str
    dtype: torch.dtype
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    euler_init_noise_sigma: float
    train_pool: set[int]
    held_out: set[int]


def build_context(
    *,
    seeds: list[int],
    pair_slug: str,
    cache_root: Path,
    embeddings: dict[str, torch.Tensor],
    device,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    euler_init_noise_sigma: float,
    train_pool: set[int],
    held_out: set[int],
) -> InlineSamplerContext:
    """Load pinned init latents for every requested seed (once, at startup)."""
    init_by_seed: dict[int, torch.Tensor] = {}
    for s in seeds:
        cell = CellPath.from_root(pair_slug, int(s), cache_root=cache_root)
        init_by_seed[int(s)] = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(euler_init_noise_sigma),
        )
    return InlineSamplerContext(
        seeds=list(seeds),
        init_by_seed=init_by_seed,
        embeddings=embeddings,
        device=device,
        dtype=dtype,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        euler_init_noise_sigma=float(euler_init_noise_sigma),
        train_pool=set(int(x) for x in train_pool),
        held_out=set(int(x) for x in held_out),
    )


def sample_all_seeds(
    *,
    unet,
    models,
    scheduler,
    ctx: InlineSamplerContext,
    out_dir: Path,
    lambda_value: float = 1.0,
    lora_adapter_name: str = "lora",
) -> dict[int, Path]:
    """Render one PNG per seed under ``out_dir``. Returns {seed: png_path}."""
    ensure_dir(out_dir)
    was_training = unet.training
    unet.eval()
    paths: dict[int, Path] = {}
    try:
        with torch.inference_mode():
            for s in ctx.seeds:
                png = out_dir / f"sample_seed_{int(s):02d}.png"
                out = run_lora_residual_inject(
                    init_latents=ctx.init_by_seed[int(s)],
                    models=models,
                    scheduler=scheduler,
                    seq_a=ctx.embeddings["seq_a"], pool_a=ctx.embeddings["pool_a"],
                    seq_b=ctx.embeddings["seq_b"], pool_b=ctx.embeddings["pool_b"],
                    seq_j=ctx.embeddings["seq_j"], pool_j=ctx.embeddings["pool_j"],
                    seq_e=ctx.embeddings["seq_e"], pool_e=ctx.embeddings["pool_e"],
                    guidance_scale=ctx.guidance_scale,
                    num_inference_steps=ctx.num_inference_steps,
                    height=ctx.height, width=ctx.width,
                    euler_init_noise_sigma=ctx.euler_init_noise_sigma,
                    device=ctx.device, dtype=ctx.dtype,
                    lambda_value=float(lambda_value),
                    lora_adapter_name=lora_adapter_name,
                    record_eps_path=None,
                )
                write_decoded_image(out.image, png)
                paths[int(s)] = png
    finally:
        if was_training:
            unet.train()
    return paths


def compose_contact_sheet(
    *,
    pngs_by_seed: dict[int, Path],
    seeds: list[int],
    train_pool: set[int],
    held_out: set[int],
    title: str,
    thumb: int = 320,
) -> Image.Image:
    """Single-row contact sheet, one column per seed, with group annotations.

    Layout: a title strip on top, then one row of thumbnails. Each column
    label says ``seed NN (train)`` or ``seed NN (held)`` so the audience
    can see in-distribution vs held-out behaviour at a glance.
    """
    n = len(seeds)
    pad = 12
    label_h = 26
    title_h = 32
    W = pad + n * (thumb + pad)
    H = title_h + label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw.text((pad, 6), title, fill="black", font=font_title)

    for j, s in enumerate(seeds):
        x = pad + j * (thumb + pad)
        group = ("train" if int(s) in train_pool
                 else "held" if int(s) in held_out else "other")
        label = f"seed {int(s):02d} ({group})"
        draw.text((x, title_h + 4), label, fill="black", font=font_label)
        png = pngs_by_seed.get(int(s))
        if png is None or not Path(png).exists():
            draw.rectangle(
                [x, title_h + label_h + pad,
                 x + thumb, title_h + label_h + pad + thumb],
                outline="red",
            )
            draw.text((x + 6, title_h + label_h + pad + thumb // 2 - 8),
                      "missing", fill="red", font=font_label)
            continue
        im = Image.open(png).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(im, (x, title_h + label_h + pad))
    return canvas
