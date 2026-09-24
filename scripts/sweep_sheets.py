#!/usr/bin/env python
"""Pass 2 of the figure sweep: one sheet per run, checkpoints down the page, probe cells across.

Pass 1 renders every checkpoint of a run on six fixed cells. This turns those tiles into one image
per run so the checkpoints worth keeping can be chosen by eye, and writes `shortlist_template.json`
listing every run and step it found, ready to be cut down by hand into the shortlist pass 3 reads.

The first two rows of every sheet are the references: the joint prompt the correction aims at, and
the plain product it repairs. Without them a row of adapter renders cannot be judged.

    python scripts/sweep_sheets.py <tiles_dir> <out_dir>
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELLS = [("a_cat__x__a_dog", 9), ("a_cat__x__a_dog", 42),
         ("a_chess_board__x__an_hourglass", 9), ("a_chess_board__x__an_hourglass", 42),
         ("a_dog__x__a_dog", 9), ("a_dog__x__a_dog", 42)]
TILE = re.compile(r"^ours_(?P<run>.+)_s(?P<step>\d+)_w50\.png$")
T, GAP, LEFT, HEAD = 230, 6, 210, 66


def font(size: int):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def found(tiles: Path) -> dict[str, set[int]]:
    """run -> the steps that left at least one probe tile behind."""
    steps: dict[str, set[int]] = defaultdict(set)
    for p in tiles.glob("*/seed*/ours_*_w50.png"):
        m = TILE.match(p.name)
        if m:
            steps[m.group("run")].add(int(m.group("step")))
    return steps


def sheet(tiles: Path, run: str, steps: list[int], out: Path) -> dict:
    rows = [("joint prompt", "mono"), ("plain product", "poe")] + [(f"step {s:,}", s) for s in steps]
    W = LEFT + len(CELLS) * (T + GAP)
    H = HEAD + len(rows) * (T + GAP)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_title, f_head, f_row = font(19), font(11), font(12)
    d.text((12, 10), f"{run}: which checkpoint to take a figure cell from", fill="black", font=f_title)
    d.text((12, 34), "rows are checkpoints of this run, columns are the six probe cells; the top two "
                     "rows are the references every row is judged against", fill="#444444", font=f_head)
    for j, (pair, seed) in enumerate(CELLS):
        x = LEFT + j * (T + GAP)
        d.text((x + 2, HEAD - 16), f"{pair.replace('__x__', ' + ').replace('_', ' ')}, seed {seed}",
               fill="#333333", font=f_head)

    record = {"run": run, "steps": steps, "cells": [f"{p}/seed{s:02d}" for p, s in CELLS], "tiles": {}}
    for i, (label, key) in enumerate(rows):
        y = HEAD + i * (T + GAP)
        d.text((8, y + T // 2 - 8), label, fill="black", font=f_row)
        for j, (pair, seed) in enumerate(CELLS):
            x = LEFT + j * (T + GAP)
            name = key if isinstance(key, str) else f"ours_{run}_s{key:06d}_w50"
            p = tiles / pair / f"seed{seed:02d}" / f"{name}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                record["tiles"].setdefault(str(label), {})[f"{pair}/seed{seed:02d}"] = str(p)
            else:
                d.rectangle((x, y, x + T, y + T), outline="#cccccc")
                d.text((x + 8, y + T // 2), "not rendered", fill="#999999", font=f_head)
    canvas.save(out)
    return record


if __name__ == "__main__":
    tiles, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_by_run = found(tiles)
    if not steps_by_run:
        raise SystemExit(f"no probe tiles under {tiles}")

    template = {}
    for run, steps in sorted(steps_by_run.items()):
        ordered = sorted(steps)
        rec = sheet(tiles, run, ordered, out_dir / f"which-checkpoint-{run}.png")
        (out_dir / f"which-checkpoint-{run}.json").write_text(json.dumps(rec, indent=1))
        template[run] = ordered
        print(f"{run:28s} {len(ordered)} checkpoints -> which-checkpoint-{run}.png")

    # Cut this down by hand: keep only the run and step pairs worth the paper's full cell set.
    (out_dir / "shortlist_template.json").write_text(json.dumps(template, indent=1))
    print("wrote shortlist_template.json with every run and step found")
