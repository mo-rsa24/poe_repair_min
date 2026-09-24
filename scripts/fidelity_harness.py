#!/usr/bin/env python
"""Every fidelity fix we have, on the same cells, one job each, landing in one sheet per cell.

The adapter restores plurality and costs picture quality, and there are several candidate fixes
that do not need retraining. This runs them all on the same pairs and seeds from the same starting
noise, so the sheet answers "which one wins" by eye rather than by argument.

The treatments, and what each one is testing:

``plain``            the adapter as it ships, the thing every other column is judged against.
``guidance10/12``    a higher classifier-free scale. The correction's reachable part damps the two
                     experts from 7.5 to roughly 1 to 3, so raising the scale tests whether the
                     haze is that damping showing up as lost contrast.
``reweight``         those damped weights applied explicitly in the sampler instead, which is what
                     the joint prompt's own prediction does.
``refine25/40``      renoise the finished picture and repaint it with the adapter off. Composition
                     is already settled, so this asks whether texture comes back when the weights
                     drawing it were never damped.
``tail20``           the adapter for the first 20 steps only, the frozen model for the rest.
``bo8``              eight nearby starting noises, one tile each, the selection made visible.

    python scripts/fidelity_harness.py submit --checkpoint <pt>      # on the cluster
    python scripts/fidelity_harness.py sheets <tiles_dir> <out_dir>  # anywhere the tiles are
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# (tag, extra flags). The tag names the tile, so a sheet can find it.
TREATMENTS = [
    ("plain", []),
    ("guid10", ["--guidance", "10"]),
    ("guid12", ["--guidance", "12"]),
    ("reweight", ["--expert-weights", "3.0", "3.0"]),
    ("refine25", ["--refine-strength", "0.25", "--refine-guidance", "7.5"]),
    ("refine40", ["--refine-strength", "0.40", "--refine-guidance", "10"]),
    ("tail20", ["--window", "20"]),
    ("bo8", ["--candidates", "8", "--jitter", "0.1"]),
]
REFERENCES = [("mono", ["--column", "mono"]), ("poe", ["--column", "poe"])]
PAIRS = ["a cat|a dog", "an elephant|a penguin", "a tiger|a dog", "a chess board|an hourglass"]
SEEDS = ["9", "10"]

REPO = "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min"
PY = "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/fidelity_harness"


def submit(checkpoint: str, out_root: str) -> None:
    Path(f"{out_root}/logs").mkdir(parents=True, exist_ok=True)
    base = (f'cd {REPO} && POE_REPO=$PWD FIGURE_CANDIDATES_OUT={out_root} {PY} '
            f'{REPO}/scripts/figure_candidates.py --pairs ' +
            " ".join(f'"{p}"' for p in PAIRS) + " --seeds " + " ".join(SEEDS))
    jobs = []
    for tag, flags in REFERENCES:
        cmd = f'{base} {" ".join(flags)}' + (f' --checkpoint {checkpoint}' if tag == "poe" else "")
        jobs.append((f"fh_{tag}", cmd))
    for tag, flags in TREATMENTS:
        window = "" if "--window" in flags else "--window 50"
        cmd = (f'{base} --column ours {window} --checkpoint {checkpoint} --tag {tag} '
               f'{" ".join(flags)}')
        jobs.append((f"fh_{tag}", cmd))
    for name, cmd in jobs:
        sb = ["sbatch", "--parsable", "-p", "batch", "-t", "04:00:00",
              "--exclude=mscluster124,mscluster129", "-J", name,
              "-o", f"{out_root}/logs/%x-%j.out", "-e", f"{out_root}/logs/%x-%j.err",
              "--wrap", cmd]
        jid = subprocess.run(sb, capture_output=True, text=True).stdout.strip()
        print(f"{name:14s} {jid}")


def sheets(tiles: Path, out_dir: Path) -> None:
    """One sheet per cell: the two references, then every treatment that has landed."""
    from PIL import Image, ImageDraw, ImageFont
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [("mono", "joint prompt"), ("poe", "plain product")] + [
        (f"ours_{t}_w{'20' if t == 'tail20' else '50'}", t) for t, _ in TREATMENTS]
    T, G, TOP = 320, 6, 34
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    made = []
    for pair_dir in sorted(p for p in tiles.iterdir() if p.is_dir()):
        for seed_dir in sorted(pair_dir.glob("seed*")):
            canvas = Image.new("RGB", (len(cols) * (T + G), TOP + T + 4), "white")
            d = ImageDraw.Draw(canvas)
            d.text((4, 8), f"{pair_dir.name.replace('__x__', ' + ')}  {seed_dir.name}",
                   fill="black", font=font)
            found = 0
            for j, (stem, label) in enumerate(cols):
                x = j * (T + G)
                p = seed_dir / f"{stem}.png"
                if not p.exists() and stem.startswith("ours_bo8"):
                    p = seed_dir / f"{stem}_c00.png"
                d.text((x + 4, 20), label, fill="#444444", font=font)
                if p.exists():
                    canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS),
                                 (x, TOP))
                    found += 1
                else:
                    d.rectangle((x, TOP, x + T, TOP + T), outline="#cccccc")
            name = f"{pair_dir.name}__{seed_dir.name}.png"
            canvas.save(out_dir / name)
            made.append({"sheet": name, "columns": [c for c, _ in cols], "tiles_found": found})
    (out_dir / "sheets.json").write_text(json.dumps({"cells": made, "treatments": [t for t, _ in TREATMENTS]},
                                                    indent=1))
    print(f"wrote {len(made)} sheets to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("submit"); s1.add_argument("--checkpoint", required=True)
    s1.add_argument("--out-root", default=OUT)
    s2 = sub.add_parser("sheets"); s2.add_argument("tiles", type=Path); s2.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.checkpoint, a.out_root)
    else:
        sheets(a.tiles, a.out)
