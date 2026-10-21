#!/usr/bin/env python
"""The across-adapters tiles regrouped: seeds as rows, adapters as columns grouped by what the run changed.

Reads the tiles `scripts/across_adapters.py --render` wrote (`<tiles>/seedNN/<index>.png`, the index
being the row of that script's ROWS table) and draws one sheet. Reading along a row shows whether
a seed keeps its look across adapters; the two marked columns are the adapters that do not.

    python scripts/across_adapters_by_family.py <tiles_dir> <out.png> [--v2]
    python scripts/across_adapters_by_family.py <tiles_dir> <out_dir> --paper

--v2 is the paper version: no title or subtitle (the caption carries them) and only the columns
the paper speaks to. Its step-window group names only the steps; the first-10-steps, 40k column
is v57-09-x0-seeds, which also differs in loss and seeds (see the card).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

T, GAP, GROUP_GAP, LEFT, HEAD = 200, 6, 26, 150, 190
SEEDS = [(9, "seed 9"), (11, "seed 11"), (12, "seed 12")]
MARK = {17, 18}  # the two adapters trained against filtered renders

# (group heading, [(ROWS index in across_adapters.py, tile label)])
GROUPS = [
    ("No adapter", [(0, "plain product")]),
    ("Original recipe: rank and regularisation", [
        (1, "rank 8, 450k"), (2, "rank 16, 30k"), (3, "rank 32, 30k"),
        (4, "+ weight decay"), (5, "+ weight averaging")]),
    ("Original recipe, different loss", [
        (6, "loss on the predicted picture, 25k"), (7, "penalty on correction size, 40k"),
        (23, "43 cells, first 10 steps, picture loss, 7k")]),
    ("Which cells it trains on (rank 32)", [
        (8, "43 cells, all 50 steps"), (9, "43 cells, first 25 steps"),
        (10, "43 cells + orthogonality penalty"), (11, "23 cells, animals only"),
        (12, "54 cells"), (13, "182 cells"), (14, "72 cells, 10k")]),
    ("What the target is (rank 16)", [
        (15, "empty branch from cache"), (16, "only the empty branch trained"),
        (17, "target is a filtered picture"), (18, "filtered picture + cached empty branch")]),
    ("Extra loss terms, 54 cells (rank 16)", [
        (19, "no extra term"), (20, "+ drift penalty"), (21, "+ plural prompt"),
        (22, "+ connective term")]),
]

GROUPS_V2 = [
    ("No adapter", [(0, "PoE")]),
    ("Rank", [
        (1, "rank 8"), (2, "rank 16"), (3, "rank 32")]),
    ("Different loss", [(6, "loss on the predicted picture, 25k")]),
    ("Which steps it trains on", [
        (25, "first 10 steps"), (9, "first 25 steps"), (8, "all 50 steps")]),
    ("What the target is", [
        (15, "empty branch from cache"), (16, "only the empty branch trained")]),
]


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def wrap_px(text: str, f, width: int) -> list[str]:
    """Wrap on words so no line is wider than `width` pixels in font `f`."""
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if cur and f.getlength(trial) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur] if cur else lines


# The paper figure: file name to its bands, stacked top to bottom (one band is one row of groups).
PAPER_FIGURES = {
    "adapters-cat-dog-rank-and-steps.png": [[GROUPS_V2[0], GROUPS_V2[1], GROUPS_V2[3]]],
}


def paper(tiles: Path, out: Path, bands: list) -> None:
    """Each band a row of groups, lettered to read in print.

    Lettering is sized for an image 1,680 px wide printed at full text width (34 px is then about
    8 pt). A wider image scales its lettering up by the same ratio, so it prints at that size too.
    """
    t, gap, ggap = 300, 8, 34
    cols = max(sum(len(g[1]) for g in b) for b in bands)
    groups_n = max(len(b) for b in bands)
    sc = max(1.0, (64 + cols * (t + gap) + (groups_n - 1) * ggap) / 1680)
    left = round(64 * sc)
    # Headings and labels are capped at the largest size that keeps "No adapter" and "first 25 steps"
    # on one line inside a single 300 px column.
    f_head, f_label, f_seed = font(min(round(42 * sc), 52)), font(min(round(34 * sc), 42)), font(round(38 * sc))
    line_h, head_h = round(40 * sc), round(50 * sc)
    widths = [left + sum(len(g[1]) for g in b) * (t + gap) + (len(b) - 1) * ggap for b in bands]
    # Each band's header is as tall as its longest label, so no label is cut.
    def head_lines(heading, items):
        return wrap_px(heading, f_head, len(items) * (t + gap) - gap)
    heads = [max(len(head_lines(h, it)) for h, it in b) * head_h + 16
             + max(len(wrap_px(lab, f_label, t - 6)) for _, items in b for _, lab in items) * line_h + 14
             for b in bands]
    band_hs = [h + len(SEEDS) * (t + gap) for h in heads]
    img = Image.new("RGB", (max(widths) + 8, sum(band_hs) + 36 * (len(bands) - 1) + 10), "white")
    d = ImageDraw.Draw(img)
    y0 = 0
    for band, hh, bh in zip(bands, heads, band_hs):
        head = y0 + hh
        label_y = y0 + 16 + max(len(head_lines(h, it)) for h, it in band) * head_h
        x = left
        for heading, items in band:
            gw = len(items) * (t + gap) - gap
            d.rectangle((x, y0 + 4, x + gw, y0 + 9), fill="#888888")
            for k, line in enumerate(head_lines(heading, items)):
                d.text((x, y0 + 14 + k * head_h), line, fill="black", font=f_head)
            for i, (idx, label) in enumerate(items):
                tx = x + i * (t + gap)
                for k, line in enumerate(wrap_px(label, f_label, t - 6)):
                    d.text((tx, label_y + k * line_h), line, fill="#222222", font=f_label)
                for r, (seed, _) in enumerate(SEEDS):
                    ty = head + r * (t + gap)
                    p = tiles / f"seed{seed:02d}" / f"{idx:02d}.png"
                    img.paste(Image.open(p).convert("RGB").resize((t, t), Image.LANCZOS), (tx, ty))
            x += gw + gap + ggap
        for r, (_, name) in enumerate(SEEDS):
            lab = Image.new("RGBA", (t, round(56 * sc)), (255, 255, 255, 0))
            ImageDraw.Draw(lab).text((t // 2, round(28 * sc)), name, fill="black", font=f_seed, anchor="mm")
            lab = lab.rotate(90, expand=True)
            img.paste(lab, (2, head + r * (t + gap)), lab)
        y0 += bh + 36
    img.save(out)
    print("wrote", out, img.size)


def main():
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])
    if "--paper" in sys.argv[3:]:            # out is a folder: one file per paper figure
        out.mkdir(parents=True, exist_ok=True)
        for name, groups in PAPER_FIGURES.items():
            paper(tiles, out / name, groups)
        return
    v2 = "--v2" in sys.argv[3:]
    groups = GROUPS_V2 if v2 else GROUPS
    top = 14 if v2 else 84          # where the group rules sit; v2 has no title above them
    head = top + HEAD - 84
    n = sum(len(g[1]) for g in groups)
    W = LEFT + n * (T + GAP) + (len(groups) - 1) * GROUP_GAP + 20
    H = head + len(SEEDS) * (T + GAP) + 20
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    if not v2:
        d.text((14, 12), "The seed sets the look; only the two adapters trained on filtered pictures "
               "override it", fill="black", font=font(26))
        d.text((14, 50), "Cat and dog, three seeds no adapter trained on. Each row is one seed, every tile "
               "in a row starts from the same noise, so a change along a row is the adapter. Seed 12 draws "
               "a photograph and seed 11 a pencil sketch even with no adapter, and the adapters mostly "
               "follow. A cell is one pair at one seed; 30k means the checkpoint at training step 30,000.",
               fill="#444444", font=font(14))
    x = LEFT
    for heading, items in groups:
        gw = len(items) * (T + GAP) - GAP
        d.rectangle((x, top, x + gw, top + 4), fill="#888888")
        for k, line in enumerate(textwrap.wrap(heading, max(12, gw // 9))):
            d.text((x, top + 10 + k * 18), line, fill="black", font=font(15))
        for i, (idx, label) in enumerate(items):
            tx = x + i * (T + GAP)
            for k, line in enumerate(textwrap.wrap(label, 26)):
                d.text((tx, head - 40 + k * 15), line,
                       fill="#b3261e" if idx in MARK else "#333333", font=font(12))
            for r, (seed, _) in enumerate(SEEDS):
                ty = head + r * (T + GAP)
                p = tiles / f"seed{seed:02d}" / f"{idx:02d}.png"
                if p.exists():
                    img.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (tx, ty))
                else:
                    d.rectangle((tx, ty, tx + T, ty + T), outline="#cccccc")
                    d.text((tx + 10, ty + T // 2), "not rendered", fill="#999999", font=font(12))
            if idx in MARK:
                d.rectangle((tx - 3, head - 3, tx + T + 2, head + len(SEEDS) * (T + GAP) - GAP + 2),
                            outline="#b3261e", width=3)
        x += gw + GAP + GROUP_GAP
    for r, (_, name) in enumerate(SEEDS):
        d.text((14, head + r * (T + GAP) + T // 2 - 10), name, fill="black", font=font(18))
    img.save(out)
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
