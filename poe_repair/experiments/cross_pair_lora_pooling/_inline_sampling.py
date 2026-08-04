"""Inline per-epoch sampling for the cross-pair pooled trainer.

Renders one PNG per (pair, seed) cell in a configurable list at the
current LoRA state, plus a composed contact sheet. Designed cheap (20
DDIM steps by default) so it can run every N epochs without dominating
the long training wall-clock. Each cell knows which pair's embeddings
to use, so the same renderer covers in-pair and held-pair cells in one
pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge import (
    _alpha_fit,
    _cos,
)
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir
from poe_repair.training_cache import CellPath, mean_delta_per_step, resolve_cells


@dataclass
class InlineCell:
    quadrant: str            # in_in / in_out / out_in / out_out
    pair_slug: str
    seed: int
    init_latents: torch.Tensor


@dataclass
class InlineContext:
    cells: list[InlineCell]
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]]
    device: torch.device | str
    dtype: torch.dtype
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    euler_init_noise_sigma: float


def build_context(
    *,
    plan: list[tuple[str, str, int]],     # (quadrant, pair_slug, seed)
    cache_root: Path,
    embeddings_by_pair: dict[str, dict[str, torch.Tensor]],
    device,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    euler_init_noise_sigma: float,
) -> InlineContext:
    """Load each cell's pinned init latent once at startup."""
    cells: list[InlineCell] = []
    for quadrant, pair, seed in plan:
        cell = CellPath.from_root(pair, int(seed), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(euler_init_noise_sigma),
        )
        cells.append(InlineCell(
            quadrant=quadrant, pair_slug=pair, seed=int(seed),
            init_latents=init,
        ))
    return InlineContext(
        cells=cells,
        embeddings_by_pair=embeddings_by_pair,
        device=device, dtype=dtype,
        height=int(height), width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        euler_init_noise_sigma=float(euler_init_noise_sigma),
    )


def build_pool_mean_cache(
    *,
    train_pairs: list[str],
    train_seeds: list[int],
    cache_root: Path,
) -> dict[int, torch.Tensor]:
    """Δ̄_t: mean Δ_t across the whole training pool (all train pairs × train
    seeds), keyed by step_index. This is the fixed reference direction for
    direction-cosine / fraction-of-distance-reached — static w.r.t. LoRA
    weights, so it is built once at startup, not per eval step.
    """
    cells: list[CellPath] = []
    for pair in train_pairs:
        cells.extend(resolve_cells(pair, train_seeds, cache_root=cache_root))
    deltas, step_indices, _ = mean_delta_per_step(cells)
    return {int(si): t for si, t in zip(step_indices, deltas)}


def direction_metrics(
    where_applied: dict[int, dict[str, torch.Tensor]],
    pool_mean: dict[int, torch.Tensor],
) -> dict[str, float]:
    """Per-cell direction-cosine and fraction-of-distance-reached, averaged
    over whatever steps were recorded in ``where_applied`` (see
    ``record_delta_at_steps``). Reuses ``_cos``/``_alpha_fit`` from
    ``task_d_bridge`` rather than redefining them.

    direction-cosine: cos(Δ̂_t, Δ̄_t) — does this cell's correction point the
    same way as the pool-mean correction?
    fraction-of-distance-reached: alpha_fit(Δ̂_t, Δ̄_t) — the best-fit scalar
    projecting Δ̂_t onto Δ̄_t's direction; ~1.0 means the correction travelled
    the full pool-mean distance, ~0.4 is the observed plateau.
    """
    cosines, alphas = [], []
    for si, rec in where_applied.items():
        v_bar = pool_mean.get(int(si))
        if v_bar is None:
            continue
        delta_hat = rec["delta_hat"]
        cosines.append(_cos(delta_hat, v_bar))
        alphas.append(_alpha_fit(delta_hat, v_bar))
    if not cosines:
        return {"direction_cosine": float("nan"), "frac_distance_reached": float("nan")}
    return {
        "direction_cosine": float(sum(cosines) / len(cosines)),
        "frac_distance_reached": float(sum(alphas) / len(alphas)),
    }


def sample_all_cells(
    *,
    unet,
    models,
    scheduler,
    ctx: InlineContext,
    out_dir: Path,
    lambda_value: float = 1.0,
    lora_adapter_name: str = "lora",
    record_delta_at_steps: list[int] | None = None,
) -> list[tuple[InlineCell, Path, dict[int, dict[str, torch.Tensor]]]]:
    """Render one PNG per cell under ``out_dir``.

    Returns ``[(cell, png_path, where_applied_cache)]``. ``where_applied_cache``
    is ``{}`` unless ``record_delta_at_steps`` is given, in which case it maps
    ``step_index -> {"delta_hat": Tensor, ...}`` (see
    ``run_lora_residual_inject``'s ``where_applied_cache`` extra) — the
    in-memory per-step ``Δ̂`` the direction-cosine / distance-reached metrics
    are computed from, with no disk round-trip.
    """
    ensure_dir(out_dir)
    was_training = unet.training
    unet.eval()
    out_pairs: list[tuple[InlineCell, Path, dict[int, dict[str, torch.Tensor]]]] = []
    try:
        with torch.inference_mode():
            for cell in ctx.cells:
                emb = ctx.embeddings_by_pair[cell.pair_slug]
                png = (out_dir
                       / f"{cell.quadrant}__{cell.pair_slug}__seed{cell.seed:02d}.png")
                out = run_lora_residual_inject(
                    init_latents=cell.init_latents,
                    models=models, scheduler=scheduler,
                    seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                    seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                    seq_j=emb["seq_j"], pool_j=emb["pool_j"],
                    seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                    guidance_scale=ctx.guidance_scale,
                    num_inference_steps=ctx.num_inference_steps,
                    height=ctx.height, width=ctx.width,
                    euler_init_noise_sigma=ctx.euler_init_noise_sigma,
                    device=ctx.device, dtype=ctx.dtype,
                    lambda_value=float(lambda_value),
                    lora_adapter_name=lora_adapter_name,
                    record_eps_path=None,
                    record_delta_at_steps=record_delta_at_steps,
                )
                write_decoded_image(out.image, png)
                where_applied = out.extras.get("where_applied_cache", {}) or {}
                out_pairs.append((cell, png, where_applied))
    finally:
        if was_training:
            unet.train()
    return out_pairs


def compose_triptych(
    *,
    mono_path: Path,
    poe_path: Path,
    lora_path: Path,
    title: str,
    lora_label: str = "LoRA",
    thumb: int = 320,
) -> Image.Image:
    """Three panels side by side: Mono (target) | PoE (default) | LoRA (corrected).

    Mono and PoE are the fixed cache references (they do not change over training);
    only the LoRA panel evolves. Lets you read compose-vs-blend against both the
    good-composer target and the broken PoE baseline, over training steps.
    """
    panels = [("Mono (target)", mono_path), ("PoE (default)", poe_path),
              (lora_label, lora_path)]
    pad, label_h, title_h = 8, 22, 26
    W = len(panels) * thumb + (len(panels) + 1) * pad
    H = title_h + label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(255, 255, 255))
    for i, (label, p) in enumerate(panels):
        x = pad + i * (thumb + pad)
        draw.text((x, title_h), label, fill=(200, 200, 200))
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
            off = (x + (thumb - im.width) // 2, title_h + label_h + (thumb - im.height) // 2)
            canvas.paste(im, off)
        except Exception:
            draw.rectangle([x, title_h + label_h, x + thumb, title_h + label_h + thumb],
                           outline=(90, 90, 90))
            draw.text((x + 8, title_h + label_h + 8), "(missing)", fill=(150, 90, 90))
    return canvas


def compose_contact_sheet(
    *,
    rendered: list[tuple[InlineCell, Path]],
    title: str,
    thumb: int = 256,
) -> Image.Image:
    """Grid contact sheet: rows = quadrants, columns = cells in each quadrant."""
    by_q: dict[str, list[tuple[InlineCell, Path]]] = {}
    for cell, png in rendered:
        by_q.setdefault(cell.quadrant, []).append((cell, png))
    quadrant_order = ["in_in", "in_out", "out_in", "out_out"]
    quadrants_present = [q for q in quadrant_order if q in by_q]
    if not quadrants_present:
        return Image.new("RGB", (thumb, thumb), "white")

    pad = 8
    label_h = 22
    title_h = 32
    row_label_w = 160
    max_cols = max(len(by_q[q]) for q in quadrants_present)
    W = row_label_w + pad + max_cols * (thumb + pad)
    H = title_h + len(quadrants_present) * (label_h + thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw.text((pad, 6), title, fill="black", font=font_title)
    y = title_h
    for q in quadrants_present:
        draw.text((pad, y + thumb // 2), q, fill="black", font=font_title)
        for j, (cell, png) in enumerate(by_q[q]):
            x = row_label_w + pad + j * (thumb + pad)
            label = f"{cell.pair_slug}\nseed {cell.seed:02d}"
            draw.text((x, y), label, fill="black", font=font_label)
            if not Path(png).exists():
                draw.rectangle([x, y + label_h, x + thumb, y + label_h + thumb],
                               outline="red")
                continue
            im = Image.open(png).convert("RGB").resize(
                (thumb, thumb), Image.LANCZOS,
            )
            canvas.paste(im, (x, y + label_h))
        y += label_h + thumb + pad
    return canvas
