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

from poe_repair.experiments.held_out_seeds.task_d_bridge import (
    _alpha_fit,
    _cos,
)
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import ensure_dir
from poe_repair.training_cache import (
    CellPath,
    iter_cell_deltas,
    mean_delta_per_step,
    resolve_cells,
)


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
    # V1 (--freeze-null): sample the empty branch with the adapter off, matching how the
    # adapter was trained. Without it training and sampling disagree and the render shows a
    # configuration that was never trained.
    freeze_null: bool = False
    adapt_null_only: bool = False


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
    freeze_null: bool = False,
    adapt_null_only: bool = False,
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
        freeze_null=bool(freeze_null),
        adapt_null_only=bool(adapt_null_only),
        euler_init_noise_sigma=float(euler_init_noise_sigma),
    )


def build_pool_mean_cache(
    *,
    train_pairs: list[str],
    train_seeds: list[int],
    cache_root: Path,
    cells_by_pair: dict[str, list[int]] | None = None,
) -> dict[int, torch.Tensor]:
    """Δ̄_t: mean Δ_t across the whole training pool (all train pairs × train
    seeds), keyed by step_index. This is the fixed reference direction for
    direction-cosine / fraction-of-distance-reached — static w.r.t. LoRA
    weights, so it is built once at startup, not per eval step.
    """
    # ``cells_by_pair`` carries an explicit per-pair seed list, for a pool that is not a cross
    # product. Without it this walked every train pair against every train seed and asked for cells
    # that were never rendered: a pool where one pair keeps seeds 1 and 2 and another keeps 17 to 19
    # has no cell at most of those crossings.
    cells: list[CellPath] = []
    for pair in train_pairs:
        seeds = (cells_by_pair or {}).get(pair, train_seeds)
        cells.extend(resolve_cells(pair, list(seeds), cache_root=cache_root))
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


# ---------------------------------------------------------------------------
# Tracking-set additions (plan 06: extend-the-tracking-set).
#
# Bucket boundaries and representative steps mirror RunConfig.probe
# (commit_window=(5,25), where_applied_steps=(7,15,22)) — the same split
# already used by train/loss_bucket/*, so "commit" means the same thing
# everywhere in this codebase rather than two different windows. These
# module-level defaults exist for standalone use; the trainer passes
# cfg.probe.commit_window / cfg.probe.where_applied_steps explicitly so a
# future change to RunConfig doesn't silently drift out of sync with this
# file. Callers with a different --sample-num-inference-steps still get
# whichever steps actually got recorded (buckets/reps just have fewer
# points, never a crash).
# ---------------------------------------------------------------------------
TRACKING_BUCKETS_DEFAULT = {"early": (0, 5), "commit": (5, 25), "late": (25, 49)}
TRACKING_REP_STEPS_DEFAULT = (7, 15, 22)
TRACKING_SPECTRAL_STEP_DEFAULT = 22  # last of the representative steps,
                                      # inside the commit window.


def learned_vs_actual_cosine(
    where_applied: dict[int, dict[str, torch.Tensor]],
    actual_by_step: dict[int, torch.Tensor],
    *,
    buckets: dict[str, tuple[int, int]] = TRACKING_BUCKETS_DEFAULT,
    rep_steps: tuple[int, ...] = TRACKING_REP_STEPS_DEFAULT,
) -> dict[str, float]:
    """cos(learned Δ̂_t, cached actual Δ_t) per recorded step, bucketed
    early/commit/late plus three named representative steps.

    "actual" is the cached ground-truth Δ_t for this exact (pair, seed) cell,
    read from the training cache in closed form (``iter_cell_deltas``, no
    forward pass) — the same target the LoRA is trained to approximate, so
    this says which part of the trajectory the correction has actually
    learned versus is still guessing at.
    """
    per_step_cos: dict[int, float] = {}
    for si, rec in where_applied.items():
        actual = actual_by_step.get(int(si))
        if actual is None:
            continue
        per_step_cos[int(si)] = _cos(rec["delta_hat"], actual)
    out: dict[str, float] = {}
    for name, (lo, hi) in buckets.items():
        vals = [c for s, c in per_step_cos.items() if lo <= s <= hi]
        if vals:
            out[f"bucket_{name}"] = float(sum(vals) / len(vals))
    for s in rep_steps:
        if s in per_step_cos:
            out[f"step_{s:02d}"] = per_step_cos[s]
    return out


def embedding_drift(
    *,
    lora_path: Path,
    mono_path: Path,
    poe_path: Path,
    embedders,
) -> dict[str, float]:
    """distance-to-mono minus distance-to-poe, DINOv2 and CLIP, one render.

    Reuses the compose-scorer's already-loaded embedders (the same ones
    ``eval/compose_rate`` calls every epoch) — no new backbone load, no UNet
    forward pass. Positive means this render sits closer to the broken PoE
    baseline than to the mono target; negative means it has moved toward the
    target. The instance-count scorer, not this, still owns the compose-vs-
    blend label.
    """
    if not (mono_path.exists() and poe_path.exists()):
        return {}
    out: dict[str, float] = {}
    for space in ("dino", "clip"):
        embed = getattr(embedders, space)
        e_lora, e_mono, e_poe = embed([lora_path, mono_path, poe_path])
        d_mono = 1.0 - float((e_lora.reshape(-1) * e_mono.reshape(-1)).sum())
        d_poe = 1.0 - float((e_lora.reshape(-1) * e_poe.reshape(-1)).sum())
        out[space] = d_mono - d_poe
    return out


def spectral_share(
    deltas: list[torch.Tensor],
    *,
    top_k: int = 3,
) -> dict[str, float]:
    """Top-k singular-value share of the stacked LEARNED deltas at one fixed
    step, across whichever cells were rendered this epoch. Labelled
    diagnostic: the paper's low-rank claim is about the cached targets
    (r_t), never about this LoRA-output read. Fewer than 2 cells returns {}
    (a single vector has no spectrum to speak of).
    """
    if len(deltas) < 2:
        return {}
    mat = torch.stack([d.flatten().float() for d in deltas], dim=0)  # (N, D)
    try:
        s = torch.linalg.svdvals(mat)
    except Exception:
        return {}
    total = float((s ** 2).sum())
    if total <= 0:
        return {}
    k = min(top_k, int(s.numel()))
    return {
        "top1_share": float((s[0] ** 2) / total),
        f"top{k}_share": float((s[:k] ** 2).sum() / total),
    }


def divergence_step_profile(
    where_applied: dict[int, dict[str, torch.Tensor]],
    *,
    rep_steps: tuple[int, ...] = TRACKING_REP_STEPS_DEFAULT,
) -> dict[str, float]:
    """‖Δ̂_t‖ at the three named representative steps, on the trajectory this
    render actually followed.

    Closed-loop, labelled so: unlike ``learned_vs_actual_cosine``, this is
    not compared against the cached teacher-forced trajectory, so it can
    diverge from that trajectory's x_t at the same step index once the LoRA
    correction accumulates over the rollout.
    """
    out: dict[str, float] = {}
    for s in rep_steps:
        rec = where_applied.get(int(s))
        if rec is not None:
            out[f"step_{s:02d}"] = float(rec["delta_hat"].float().norm())
    return out


def actual_delta_by_step(cell_pair_slug: str, cell_seed: int, *,
                          split: str, cache_root: Path) -> dict[int, torch.Tensor]:
    """The cached ground-truth Δ_t for one cell, keyed by step index.
    Returns {} if this cell has no cached residuals (e.g. a cell added to
    the sample plan that was never run through the caching pass) rather than
    raising, so a missing cache degrades the cosine read, not the epoch.
    """
    try:
        cell = CellPath.from_root(cell_pair_slug, int(cell_seed),
                                   split=split, cache_root=cache_root)
    except FileNotFoundError:
        return {}
    return {int(e["step_index"]): e["delta"] for e in iter_cell_deltas(cell)}


def tracking_reads(
    *,
    cell: InlineCell,
    split: str,
    where_applied: dict[int, dict[str, torch.Tensor]],
    cache_root: Path,
    lora_path: Path,
    mono_path: Path,
    poe_path: Path,
    embedders,
    buckets: dict[str, tuple[int, int]] = TRACKING_BUCKETS_DEFAULT,
    rep_steps: tuple[int, ...] = TRACKING_REP_STEPS_DEFAULT,
) -> dict[str, float]:
    """The four plan-06 additions for one rendered cell, flattened to one
    {name: float} dict. The caller (train_pooled.py) prefixes every key with
    ``eval/tracking/`` and this cell's identity, and separately builds the
    cross-cell spectral-share read at whichever step it chooses to fix (see
    ``TRACKING_SPECTRAL_STEP_DEFAULT``). Pass ``buckets``/``rep_steps`` from
    ``cfg.probe`` so this always agrees with train/loss_bucket/*.
    """
    out: dict[str, float] = {}
    actual_by_step = actual_delta_by_step(
        cell.pair_slug, cell.seed, split=split, cache_root=cache_root,
    )
    if actual_by_step:
        out.update({
            f"learned_actual_cosine/{k}": v
            for k, v in learned_vs_actual_cosine(
                where_applied, actual_by_step,
                buckets=buckets, rep_steps=rep_steps,
            ).items()
        })
    out.update({
        f"embedding_drift/{k}": v
        for k, v in embedding_drift(
            lora_path=lora_path, mono_path=mono_path, poe_path=poe_path,
            embedders=embedders,
        ).items()
    })
    out.update({
        f"divergence_profile/{k}": v
        for k, v in divergence_step_profile(where_applied, rep_steps=rep_steps).items()
    })
    return out


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
    on_cell_done=None,
) -> list[tuple[InlineCell, Path, dict[int, dict[str, torch.Tensor]]]]:
    """Render one PNG per cell under ``out_dir``.

    Returns ``[(cell, png_path, where_applied_cache)]``. ``where_applied_cache``
    is ``{}`` unless ``record_delta_at_steps`` is given, in which case it maps
    ``step_index -> {"delta_hat": Tensor, ...}`` (see
    ``run_lora_residual_inject``'s ``where_applied_cache`` extra) — the
    in-memory per-step ``Δ̂`` the direction-cosine / distance-reached metrics
    are computed from, with no disk round-trip.

    ``on_cell_done(cell, png_path, where_applied)``, if given, is called
    right after each cell renders — so a caller can build that cell's
    triptych, compute its tracking reads, and log to W&B immediately,
    instead of waiting for every cell in ``ctx.cells`` to finish first. A
    long cell list (152 cells x 50 DDIM steps) otherwise means total
    silence in W&B until the very last cell is done.
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
                    freeze_null=bool(getattr(ctx, "freeze_null", False)),
                    adapt_null_only=bool(getattr(ctx, "adapt_null_only", False)),
                    record_eps_path=None,
                    record_delta_at_steps=record_delta_at_steps,
                )
                write_decoded_image(out.image, png)
                # Second render of the same cell, same seed, same everything, except the
                # correction is switched off after T/2 (lambda_schedule="first_half"). At
                # lambda 0 the step is plain guided PoE, so the composition stays on and only
                # the adapter stops. This is the second row of the comparison grid.
                png_half = (out_dir
                            / f"{cell.quadrant}__{cell.pair_slug}__seed{cell.seed:02d}__half.png")
                out_half = run_lora_residual_inject(
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
                    lambda_schedule="first_half",
                    lora_adapter_name=lora_adapter_name,
                    freeze_null=bool(getattr(ctx, "freeze_null", False)),
                    adapt_null_only=bool(getattr(ctx, "adapt_null_only", False)),
                    record_eps_path=None,
                    record_delta_at_steps=None,
                )
                write_decoded_image(out_half.image, png_half)
                where_applied = out.extras.get("where_applied_cache", {}) or {}
                out_pairs.append((cell, png, where_applied))
                if on_cell_done is not None:
                    on_cell_done(cell, png, where_applied)
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


def compose_two_row_grid(
    *,
    mono_path: Path,
    poe_path: Path,
    lora_full_path: Path,
    lora_half_path: Path,
    title: str,
    lora_label: str = "LoRA",
    thumb: int = 320,
) -> Image.Image:
    """Two rows of three panels: Mono (target) | PoE (default) | LoRA.

    Row 1 is the adapter on for the whole path. Row 2 is the adapter on over 0..T/2 and off
    over T/2..T, which is plain guided PoE for the tail. Mono and PoE are the fixed cache
    references and are the same in both rows; only the third column differs, so the grid reads
    as one question: does keeping the correction past the halfway point help or hurt.
    """
    rows = [
        ("LoRA on the whole path", lora_full_path),
        ("LoRA on 0 to T/2, off T/2 to T", lora_half_path),
    ]
    cols = ["Mono (target)", "PoE (default)", lora_label]
    pad, label_h, title_h, row_h = 8, 22, 26, 20
    W = 3 * thumb + 4 * pad
    H = title_h + len(rows) * (row_h + label_h + thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(255, 255, 255))
    y = title_h
    for row_label, lora_p in rows:
        draw.text((pad, y), row_label, fill=(235, 200, 120))
        y += row_h
        for i, (col_label, p) in enumerate(zip(cols, [mono_path, poe_path, lora_p])):
            x = pad + i * (thumb + pad)
            draw.text((x, y), col_label, fill=(200, 200, 200))
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb, thumb))
                off = (x + (thumb - im.width) // 2, y + label_h + (thumb - im.height) // 2)
                canvas.paste(im, off)
            except Exception:
                draw.rectangle([x, y + label_h, x + thumb, y + label_h + thumb],
                               outline=(90, 90, 90))
                draw.text((x + 8, y + label_h + 8), "(missing)", fill=(150, 90, 90))
        y += label_h + thumb + pad
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
