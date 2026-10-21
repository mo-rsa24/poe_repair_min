#!/usr/bin/env python
"""Every sampling strategy we have, on the same cells, through each v58 adapter.

One row per adapter and seed, one column per strategy, all from the same starting noise, so what
differs down a column is the adapter and what differs across a row is only how the sampler was run.
The first two columns are the references: the joint-prompt render the correction aims at, and the
plain product it repairs.

The best-of-8 column is not one render. It is the eight nearby starting noises drawn as a small
contact patch, so the choice a selection rule would make is visible rather than hidden behind it.

    python scripts/sampling_strategy_grid.py <tiles_dir> <out.png>
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAIR = "a_cat__x__a_dog"
SEEDS = [9, 10]
ADAPTERS = [
    ("plain", "v58-01-plain", "cross-attention, step 42,500"),
    ("contrast", "v58-02-contrast", "cross-attention + contrast term, step 22,500"),
    ("self", "v58-03-self", "cross + self-attention, step 50,000"),
]
# (column key, header, note). "bo8" is drawn as a patch of the eight candidates.
COLUMNS = [
    ("mono", "joint prompt", "the target"),
    ("poe", "plain product", "what is repaired"),
    ("base", "the adapter alone", "correction on all 50 steps"),
    ("tail", "clean tail", "adapter for 20 steps, frozen model finishes"),
    ("lang", "Langevin corrector", "20 steps each over steps 20 to 50"),
    ("co3", "CO3 corrector", "6 steps, 5 iterations each"),
    ("superdiff", "SuperDiff", "its own balancing sampler, our adapter attached"),
    ("bo8", "best of 8", "eight nearby starting noises"),
]
T, GAP, LEFT, HEAD, FOOT = 300, 8, 250, 96, 22


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def tile_path(tiles: Path, col: str, adapter: str, seed: int) -> Path | None:
    d = tiles / PAIR / f"seed{seed:02d}"
    if col in ("mono", "poe"):
        return d / f"{col}.png"
    if col == "superdiff":
        hits = sorted((tiles / "superdiff" / adapter).rglob(f"seed_{seed}/**/*.png"))
        return hits[0] if hits else None
    return d / f"ours_{adapter}{col}_w{20 if col == 'tail' else 50}.png"


def patch_of_candidates(tiles: Path, adapter: str, seed: int, size: int) -> Image.Image | None:
    """The eight best-of candidates as a 3 by 3 patch, the ninth square left blank."""
    files = sorted((tiles / PAIR / f"seed{seed:02d}").glob(f"ours_{adapter}bo8_w50_c*.png"))
    if not files:
        return None
    cell = size // 3
    patch = Image.new("RGB", (size, size), "white")
    for i, f in enumerate(files[:9]):
        patch.paste(Image.open(f).convert("RGB").resize((cell, cell), Image.LANCZOS),
                    ((i % 3) * cell, (i // 3) * cell))
    return patch


if __name__ == "__main__":
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [(a, k, note, s) for a, k, note in ADAPTERS for s in SEEDS]
    W = LEFT + len(COLUMNS) * (T + GAP)
    H = HEAD + len(rows) * (T + GAP + FOOT) + 30
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_head, f_row, f_small = font(22), font(12), font(13), font(11)
    d.text((14, 12), "Every sampling strategy, on cat and dog, through each v58 adapter",
           fill="black", font=f_title)
    d.text((14, 40),
           "Same starting noise across a row. The contrast adapter is at step 22,500 against 42,500 "
           "and 50,000 for the others, because its card is slower, so its row is younger rather "
           "than worse.", fill="#444444", font=f_head)

    for j, (key, head, note) in enumerate(COLUMNS):
        x = LEFT + j * (T + GAP)
        d.text((x + 3, HEAD - 38), head, fill="black", font=f_head)
        for k, line in enumerate(textwrap.wrap(note, 42)):
            d.text((x + 3, HEAD - 22 + k * 13), line, fill="#555555", font=f_small)

    record = {"pair": PAIR, "rows": [], "columns": [c[0] for c in COLUMNS], "tiles": {}}
    for i, (adapter, run, note, seed) in enumerate(rows):
        y = HEAD + i * (T + GAP + FOOT)
        for k, line in enumerate(textwrap.wrap(f"{run}, seed {seed}", 28)):
            d.text((8, y + T // 2 - 20 + k * 16), line, fill="black", font=f_row)
        d.text((8, y + T // 2 + 16), note, fill="#666666", font=f_small)
        record["rows"].append({"adapter": adapter, "run": run, "seed": seed})
        for j, (key, _, _) in enumerate(COLUMNS):
            x = LEFT + j * (T + GAP)
            img = (patch_of_candidates(tiles, adapter, seed, T) if key == "bo8"
                   else None)
            if img is None:
                p = tile_path(tiles, key, adapter, seed)
                if p and p.exists():
                    img = Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS)
                    record["tiles"].setdefault(f"seed_{seed:02d}/{adapter}", {})[key] = str(p)
            if img is None:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + 10, y + T // 2), "not rendered", fill="#999999", font=f_small)
            else:
                canvas.paste(img, (x, y))
    d.text((14, H - 22),
           "Feynman-Kac steering is still rendering and gets its own column when it lands.",
           fill="#444444", font=f_head)
    canvas.save(out)
    (out.with_suffix(".json")).write_text(json.dumps(record, indent=1))
    print("wrote", out)
