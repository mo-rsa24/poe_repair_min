"""Build contact-sheet PNGs from finished Task B / Task C runs.

Task B contact sheet (rows = held-out seeds, cols = k):
    layout:
        seed_09  | k=1a | k=1b | k=4 | k=8 |
        seed_10  | ...
        ...

Task C comparison sheet (rows = held-out seeds, cols = condition):
    layout:
        seed_09  | PoE | pooled-k8 | per-seed | mono |
        seed_10  | ...

Usage::

    python -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet \
        --task B [--out-root ...]
    python -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet \
        --task C [--task-b-k8-run NAME] [--step0-dir ...]

The script assumes the launcher's directory layout. Missing cells are
filled with a blank placeholder + label rather than failing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool


log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "cross_seed_lora_pooling"


def _placeholder(text: str, *, size: tuple[int, int] = (512, 512)) -> Image.Image:
    img = Image.new("RGB", size, color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((10, 10), f"[missing]\n{text}", fill=(200, 200, 200), font=font)
    return img


def _label_row(images: list[Image.Image], labels: list[str], *, thumb: int = 384,
               gap: int = 6, header: int = 28) -> Image.Image:
    """One row of thumbnails with column labels above each."""
    cells: list[Image.Image] = []
    for img, lab in zip(images, labels):
        w, h = thumb, thumb
        thumb_img = img.resize((w, h), Image.LANCZOS)
        cell = Image.new("RGB", (w, h + header), color=(255, 255, 255))
        draw = ImageDraw.Draw(cell)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((6, 4), lab, fill=(0, 0, 0), font=font)
        cell.paste(thumb_img, (0, header))
        cells.append(cell)
    total_w = sum(c.width for c in cells) + gap * (len(cells) - 1)
    total_h = max(c.height for c in cells)
    out = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    x = 0
    for c in cells:
        out.paste(c, (x, 0))
        x += c.width + gap
    return out


def _row_label(text: str, *, height: int, width: int = 88) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, height // 2 - 10), text, fill=(0, 0, 0), font=font)
    return img


def _stack_rows(rows: list[Image.Image], row_labels: list[str], *, gap: int = 6) -> Image.Image:
    labels = [_row_label(lab, height=r.height) for r, lab in zip(rows, row_labels)]
    full_rows: list[Image.Image] = []
    for lab, body in zip(labels, rows):
        w = lab.width + body.width
        h = max(lab.height, body.height)
        merged = Image.new("RGB", (w, h), color=(255, 255, 255))
        merged.paste(lab, (0, 0))
        merged.paste(body, (lab.width, 0))
        full_rows.append(merged)
    w = max(r.width for r in full_rows)
    h = sum(r.height for r in full_rows) + gap * (len(full_rows) - 1)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    y = 0
    for r in full_rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


def _load_or_placeholder(path: Path, label: str) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    log.warning("missing: %s", path)
    return _placeholder(label)


def build_task_b(out_root: Path, *, epochs: int = 200) -> Path:
    pool = load_seed_pool()
    out_dir = out_root / "task_b_learning_curve"
    cols = [
        ("k=1 (pick=1)", out_dir / f"k01_pick1__ep{epochs}" / "samples" / "heldout"),
        ("k=1 (pick=5)", out_dir / f"k01_pick5__ep{epochs}" / "samples" / "heldout"),
        ("k=4",          out_dir / f"k04__ep{epochs}"       / "samples" / "heldout"),
        ("k=8",          out_dir / f"k08__ep{epochs}"       / "samples" / "heldout"),
    ]
    rows: list[Image.Image] = []
    row_labels: list[str] = []
    for s in pool.held_out:
        imgs = []
        for label, samples_dir in cols:
            p = samples_dir / f"sample_seed_{int(s):02d}.png"
            imgs.append(_load_or_placeholder(p, f"{label}\nseed {s}"))
        rows.append(_label_row(imgs, [c[0] for c in cols]))
        row_labels.append(f"seed_{int(s):02d}")
    sheet = _stack_rows(rows, row_labels)
    target = out_dir / "contact_sheet.png"
    sheet.save(target)
    log.info("Task B contact sheet → %s", target)
    return target


def build_task_c(out_root: Path, *, epochs_pooled: int = 200,
                 epochs_per_seed: int = 1600, step0_dir: Path | None = None) -> Path:
    pool = load_seed_pool()
    pooled_dir = (out_root / "task_b_learning_curve"
                  / f"k08__ep{epochs_pooled}" / "samples" / "heldout")
    out_dir = out_root / "task_c_per_seed_ceiling"
    step0 = step0_dir or (out_root / "step0_prescreen")
    rows: list[Image.Image] = []
    row_labels: list[str] = []
    for s in pool.held_out:
        poe_path  = step0 / f"seed_{int(s):02d}" / "poe.png"
        mono_path = step0 / f"seed_{int(s):02d}" / "mono.png"
        pooled_path = pooled_dir / f"sample_seed_{int(s):02d}.png"
        per_seed_path = (out_dir / f"per_seed_s{int(s)}__ep{epochs_per_seed}"
                         / "samples" / "ceiling"
                         / f"sample_seed_{int(s):02d}.png")
        col_labels = ["PoE", "pooled-k8", "per-seed", "mono"]
        imgs = [
            _load_or_placeholder(poe_path,      "PoE"),
            _load_or_placeholder(pooled_path,   "pooled-k8"),
            _load_or_placeholder(per_seed_path, "per-seed"),
            _load_or_placeholder(mono_path,     "mono"),
        ]
        rows.append(_label_row(imgs, col_labels))
        row_labels.append(f"seed_{int(s):02d}")
    sheet = _stack_rows(rows, row_labels)
    target = out_dir / "comparison_sheet.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    log.info("Task C comparison sheet → %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("B", "C"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--epochs", type=int, default=1600,
                    help="Task B: pooled epochs")
    ap.add_argument("--epochs-per-seed", type=int, default=1600,
                    help="Task C: per-seed (epoch-parity) epochs")
    ap.add_argument("--step0-dir", default=None,
                    help="Task C: where to pull PoE and mono columns from")
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    if args.task == "B":
        build_task_b(out_root, epochs=args.epochs)
    else:
        step0 = Path(args.step0_dir) if args.step0_dir else None
        build_task_c(out_root, epochs_pooled=args.epochs,
                     epochs_per_seed=args.epochs_per_seed, step0_dir=step0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
