"""Render a per-seed summary figure for a finished pooled-trainer run.

Layout (one figure per ``--seed``)::

    +----------------+ +-----------------------------------------+
    |  PoE (green)   | |  ep_a  ep_b  ep_c  ep_d  ep_e            |
    |                | |  ep_f  ep_g  ep_h  ep_i  ep_j            |
    +----------------+ |  ep_k  ep_l  ep_m  ep_n  ep_o            |
    |  Mono (red)    | |  ep_p  ep_q  ep_r  ep_s  ep_t            |
    +----------------+ +-----------------------------------------+

The 4x5 right panel is filled by epoch samples chosen at equal intervals
across the recorded epochs available on disk (up to the latest one). Per-epoch
samples are read from ``<run-dir>/samples/per_epoch/epoch_NNNN_step_MMMMMM/
sample_seed_NN.png`` — i.e. whatever ``render_per_epoch.py`` already produced.
No model is loaded; this is a pure composition script.

PoE / Mono references and the (train-pool) vs (held-out) row label are
sourced the same way ``render_per_epoch.py`` sources them — training-cache
``CellPath`` + ``pooled_meta.json``.

Usage::

    python -m scripts.cross_seed_lora_pooling.render_seed_summary --seed 1
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from poe_repair import paths

from poe_repair.experiments.held_out_seeds.seed_pool import load_seed_pool
from poe_repair.runtime import ensure_dir
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


log = logging.getLogger("render_seed_summary")


DEFAULT_RUN_DIR = (
    paths.resolve(paths.HELD_OUT_SEEDS_INDEX)
    / "task_b_learning_curve/k04__ep2000_resumed"
)

EPOCH_DIR_RE = re.compile(r"epoch_(\d+)_step_(\d+)")


def list_epoch_dirs(per_epoch_root: Path, seed: int) -> list[tuple[int, int, Path]]:
    """Return [(epoch, step, dir)] for every per_epoch subdir that has the
    requested seed's sample PNG, sorted by epoch ascending."""
    out: list[tuple[int, int, Path]] = []
    if not per_epoch_root.is_dir():
        return out
    for sub in sorted(per_epoch_root.iterdir()):
        if not sub.is_dir():
            continue
        m = EPOCH_DIR_RE.fullmatch(sub.name)
        if not m:
            continue
        png = sub / f"sample_seed_{int(seed):02d}.png"
        if not png.exists():
            continue
        out.append((int(m.group(1)), int(m.group(2)), sub))
    out.sort(key=lambda x: x[0])
    return out


def merge_epoch_dirs(
    primary: list[tuple[int, int, Path]],
    parent: list[tuple[int, int, Path]],
) -> list[tuple[int, int, Path]]:
    """Merge two epoch lists by epoch, keeping the primary entry on collision
    (the resumed run is the canonical source for epochs that overlap the
    parent — both produce the same image at the resume point, since the
    resumed run starts from the parent's final checkpoint)."""
    by_epoch: dict[int, tuple[int, int, Path]] = {e: (e, s, d) for e, s, d in parent}
    by_epoch.update({e: (e, s, d) for e, s, d in primary})
    return sorted(by_epoch.values(), key=lambda x: x[0])


def pick_indices(n_available: int, n_pick: int) -> list[int | None]:
    """Unique indices in ``[0, n_available-1]`` of length ``n_pick``.

    If ``n_available >= n_pick``: use ``np.linspace`` rounded to ints, then
    resolve any collisions by sliding to a neighbour (forward, falling back
    to backward at the right edge) so every picked index is distinct.

    If ``n_available < n_pick``: return all ``n_available`` indices padded
    with ``None`` to length ``n_pick``. The caller leaves those grid cells
    blank — duplicates are explicitly disallowed."""
    if n_available <= 0:
        return [None] * n_pick
    if n_available < n_pick:
        return list(range(n_available)) + [None] * (n_pick - n_available)
    raw = np.linspace(0, n_available - 1, n_pick)
    chosen: list[int] = []
    used: set[int] = set()
    for x in raw:
        i = int(round(x))
        # Forward-slide on collision, fall back to backward at the boundary.
        while i in used and i < n_available - 1:
            i += 1
        while i in used and i > 0:
            i -= 1
        chosen.append(i)
        used.add(i)
    return chosen  # type: ignore[return-value]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _paste_thumb(canvas: Image.Image, png: Path, box: tuple[int, int, int, int]) -> None:
    """Paste ``png`` resized to (w, h) at (x, y). Box = (x, y, w, h)."""
    x, y, w, h = box
    try:
        im = Image.open(png).convert("RGB").resize((w, h), Image.LANCZOS)
        canvas.paste(im, (x, y))
    except Exception as e:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
        draw.text((x + 4, y + 4), f"MISS {png.name}: {e}", fill="red",
                  font=_font(11))


def _draw_border(canvas: Image.Image, box: tuple[int, int, int, int],
                 color: str, width: int) -> None:
    x, y, w, h = box
    ImageDraw.Draw(canvas).rectangle(
        (x - width, y - width, x + w + width, y + h + width),
        outline=color, width=width,
    )


def compose(seed: int,
            row_group: str,
            poe_png: Path,
            mono_png: Path,
            epoch_picks: list[tuple[int, Path] | None],
            *,
            left_thumb: int = 512,
            right_thumb: int = 256,
            border_w: int = 14,
            pad: int = 14,
            caption_h: int = 22,
            title_h: int = 40,
            gutter: int = 36) -> Image.Image:
    """Build the composite figure.

    Right grid: 4 rows × 5 cols (rows = vertical = 4, cols = horizontal = 5).
    Left panel: PoE on top, Mono below — each at ``left_thumb`` square.
    ``left_thumb`` is chosen so the left panel's image stack height matches
    the right panel's grid height: 2 * left_thumb ≈ 4 * right_thumb.
    """
    n_rows, n_cols = 4, 5
    assert len(epoch_picks) == n_rows * n_cols, (
        f"expected {n_rows * n_cols} epoch picks, got {len(epoch_picks)}"
    )

    # Left panel internal geometry.
    left_block_w = left_thumb + 2 * border_w
    left_block_h = 2 * (left_thumb + 2 * border_w) + caption_h + pad
    # Right panel internal geometry.
    right_block_w = n_cols * right_thumb + (n_cols + 1) * pad
    right_block_h = n_rows * (right_thumb + caption_h) + (n_rows + 1) * pad

    body_h = max(left_block_h, right_block_h)
    W = pad + left_block_w + gutter + right_block_w + pad
    H = title_h + pad + body_h + pad

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # Title.
    title_text = f"seed {int(seed)} ({row_group})"
    title_font = _font(20)
    tw = draw.textlength(title_text, font=title_font)
    draw.text((max(pad, (W - tw) // 2), 10), title_text, fill="black",
              font=title_font)

    # Left panel: PoE (green) top, Mono (red) bottom.
    left_x = pad + border_w
    body_top = title_h + pad
    # Vertically center the left block within body if it's shorter than right.
    left_y0 = body_top + max(0, (body_h - left_block_h) // 2) + border_w

    # PoE
    poe_box = (left_x, left_y0, left_thumb, left_thumb)
    _paste_thumb(canvas, poe_png, poe_box)
    _draw_border(canvas, poe_box, "#2ca02c", border_w)
    cap_font = _font(16)
    draw.text((left_x, left_y0 + left_thumb + border_w + 4),
              "PoE", fill="#2ca02c", font=cap_font)

    # Mono
    mono_y0 = left_y0 + left_thumb + 2 * border_w + caption_h + pad
    mono_box = (left_x, mono_y0, left_thumb, left_thumb)
    _paste_thumb(canvas, mono_png, mono_box)
    _draw_border(canvas, mono_box, "#d62728", border_w)
    draw.text((left_x, mono_y0 + left_thumb + border_w + 4),
              "Mono", fill="#d62728", font=cap_font)

    # Right panel: 4x5 grid of epoch picks.
    right_x0 = pad + left_block_w + gutter
    right_y0 = body_top + max(0, (body_h - right_block_h) // 2)
    cell_font = _font(13)
    for i, pick in enumerate(epoch_picks):
        r = i // n_cols
        c = i % n_cols
        x = right_x0 + pad + c * (right_thumb + pad)
        y = right_y0 + pad + r * (right_thumb + caption_h + pad)
        if pick is None:
            draw.rectangle((x, y, x + right_thumb, y + right_thumb),
                           outline="#dddddd", width=1)
            continue
        epoch, png = pick
        _paste_thumb(canvas, png, (x, y, right_thumb, right_thumb))
        cap = f"ep {epoch}"
        cw = draw.textlength(cap, font=cell_font)
        draw.text((x + max(0, (right_thumb - cw) // 2),
                   y + right_thumb + 4),
                  cap, fill="black", font=cell_font)

    return canvas


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="render_seed_summary")
    ap.add_argument("--seed", required=True, type=int,
                    help="seed to render (single seed per invocation)")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                    help=f"finished pooled run directory (default: {DEFAULT_RUN_DIR})")
    ap.add_argument("--parent-run-dir", type=Path, default=None,
                    help=("optional parent run directory to harvest extra "
                          "early-epoch samples from (e.g. k04__ep200, which "
                          "k04__ep2000_resumed was resumed from). If omitted, "
                          "auto-detected as the sibling whose name matches "
                          "the run-dir's prefix without the '_resumed' suffix "
                          "and at a lower epoch count. Pass an empty string "
                          "to disable."))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=("output directory; default: "
                          "<run-dir>/samples/per_seed_summary"))
    ap.add_argument("--per-epoch-subdir", default="samples/per_epoch",
                    help="relative to --run-dir")
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="training cache root (default: env / package default)")
    ap.add_argument("--pair-slug", default=None,
                    help="override pair slug; default: read from "
                         "<run-dir>/config.json then pooled_meta then seed_pool")
    ap.add_argument("--seed-pool-path", default=None)
    ap.add_argument("--left-thumb", type=int, default=512)
    ap.add_argument("--right-thumb", type=int, default=256)
    ap.add_argument("--border-w", type=int, default=14)
    return ap


_RESUMED_NAME_RE = re.compile(r"^(?P<prefix>.+?)__ep(?P<n>\d+)_resumed$")
_PARENT_NAME_RE = re.compile(r"^(?P<prefix>.+?)__ep(?P<n>\d+)$")


def _auto_detect_parent(run_dir: Path) -> Path | None:
    """For a run dir named ``<prefix>__ep<N>_resumed`` find a sibling named
    ``<prefix>__ep<M>`` with ``M < N``. Returns the largest such sibling,
    or ``None`` if no match exists."""
    m = _RESUMED_NAME_RE.match(run_dir.name)
    if not m:
        return None
    prefix = m.group("prefix")
    n_self = int(m.group("n"))
    candidates: list[tuple[int, Path]] = []
    for sib in run_dir.parent.iterdir():
        if not sib.is_dir() or sib == run_dir:
            continue
        sm = _PARENT_NAME_RE.match(sib.name)
        if not sm or sm.group("prefix") != prefix:
            continue
        n_sib = int(sm.group("n"))
        if n_sib < n_self:
            candidates.append((n_sib, sib))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _resolve_pair_slug(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            slug = data.get("cell", {}).get("pair_slug")
            if slug:
                return str(slug)
        except Exception:
            pass
    # Fall back to seed pool's default.
    return load_seed_pool(None).pair_slug


def _resolve_group(run_dir: Path, seed: int, seed_pool_path: str | None) -> str:
    pooled_meta_path = run_dir / "pooled_meta.json"
    if pooled_meta_path.exists():
        m = json.loads(pooled_meta_path.read_text())
        trained = {int(s) for s in m.get("seeds_used", [])}
        held = {int(s) for s in m.get("held_out", [])}
    else:
        pool = load_seed_pool(seed_pool_path)
        trained = {int(s) for s in pool.train_pool}
        held = {int(s) for s in pool.held_out}
    if seed in trained:
        return "train-pool"
    if seed in held:
        return "held-out"
    return "other"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)

    run_dir: Path = p.run_dir.resolve()
    seed = int(p.seed)

    per_epoch_root = run_dir / p.per_epoch_subdir
    primary = list_epoch_dirs(per_epoch_root, seed)
    if not primary:
        log.error("no per-epoch samples for seed %d under %s", seed, per_epoch_root)
        return 2

    # Resolve parent run for early-epoch samples (e.g. k04__ep200 for a
    # k04__ep2000_resumed primary). Empty-string flag disables.
    parent_arg = p.parent_run_dir
    if parent_arg is not None and str(parent_arg) == "":
        parent_dir: Path | None = None
    elif parent_arg is not None:
        parent_dir = Path(parent_arg).resolve()
    else:
        parent_dir = _auto_detect_parent(run_dir)
        if parent_dir is not None:
            log.info("auto-detected parent run: %s", parent_dir)

    parent_epochs: list[tuple[int, int, Path]] = []
    if parent_dir is not None:
        parent_epochs = list_epoch_dirs(parent_dir / p.per_epoch_subdir, seed)
        log.info("parent epochs for seed %d: %s",
                 seed, [e for e, _, _ in parent_epochs])

    epoch_dirs = merge_epoch_dirs(primary, parent_epochs)
    log.info("merged: %d unique epochs (range %d..%d) for seed %d",
             len(epoch_dirs), epoch_dirs[0][0], epoch_dirs[-1][0], seed)

    idxs = pick_indices(len(epoch_dirs), 20)
    epoch_picks: list[tuple[int, Path] | None] = []
    for i in idxs:
        if i is None:
            epoch_picks.append(None)
        else:
            epoch_picks.append((
                epoch_dirs[i][0],
                epoch_dirs[i][2] / f"sample_seed_{seed:02d}.png",
            ))
    log.info("picked epochs: %s",
             [pk[0] if pk is not None else None for pk in epoch_picks])

    pair_slug = _resolve_pair_slug(run_dir, p.pair_slug)
    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT
    cell = CellPath.from_root(pair_slug, seed, cache_root=cache_root)
    poe_png = cell.root / "poe.png"
    mono_png = cell.root / "mono.png"
    if not poe_png.exists():
        log.error("missing PoE ref: %s", poe_png)
        return 3
    if not mono_png.exists():
        log.error("missing Mono ref: %s", mono_png)
        return 3

    group = _resolve_group(run_dir, seed, p.seed_pool_path)
    log.info("seed %d → group=%s pair=%s", seed, group, pair_slug)

    out_dir = ensure_dir(p.out_dir if p.out_dir else run_dir / "samples" / "per_seed_summary")
    out_path = out_dir / f"summary_seed_{seed:02d}.png"

    fig = compose(
        seed=seed,
        row_group=group,
        poe_png=poe_png,
        mono_png=mono_png,
        epoch_picks=epoch_picks,
        left_thumb=int(p.left_thumb),
        right_thumb=int(p.right_thumb),
        border_w=int(p.border_w),
    )
    fig.save(out_path)
    log.info("wrote → %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
