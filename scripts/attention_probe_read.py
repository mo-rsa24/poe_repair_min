#!/usr/bin/env python
"""Where does each branch's concept token look, and is it inside its own animal?

Reads a capture written by `poe_repair.experiments.mechanism_study.capture_attention`
(one `step_XXX_token_<key>.pt` per step per branch token, a 32x32 map each) together with the
final image the same run wrote, finds each concept's box on that image with GroundingDINO, and
reports the share of each token's attention mass that falls inside its own concept's box and
inside the other concept's box, per step.

    python scripts/attention_probe_read.py --capture <dir with seed_N/> --seed 9 \
        --a "a cat" --b "a dog" --out <dir>

The shares are on the 32x32 grid, so a box is rounded to grid cells. Written for the walk on
whether an attention loss has anything to push on; it decides nothing on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics  # noqa: E402

TOKENS = {"a": "cat_branch_poe", "b": "dog_branch_poe"}   # the capture's fixed keys: branch 0 and 1


def top_box(image: Path, query: str, device):
    dets = vmetrics.detect_boxes(image, [query], device=device)
    dets = [d for d in dets if d["label"] == query.lower()]
    if not dets:
        return None
    return max(dets, key=lambda d: d["confidence"])


def box_mask(box, size, grid):
    x0, y0, x1, y1 = box
    m = torch.zeros(grid, grid)
    s = grid / size
    m[int(y0 * s):max(int(y1 * s + 0.999), int(y0 * s) + 1), int(x0 * s):max(int(x1 * s + 0.999), int(x0 * s) + 1)] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", type=Path, required=True)
    p = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dir = p.capture / f"seed_{p.seed}"
    image = seed_dir / f"sample_seed_{p.seed}.png"
    im = Image.open(image).convert("RGB")
    boxes = {"a": top_box(image, p.a, device), "b": top_box(image, p.b, device)}
    print(f"boxes on {image.name}: {p.a} -> {boxes['a']}, {p.b} -> {boxes['b']}", flush=True)
    if boxes["a"] is None or boxes["b"] is None:
        print("a concept was not found on the image; nothing to measure against", flush=True)
    maps = sorted((seed_dir / "attn_maps").glob("step_*_token_*.pt"))
    if not maps:
        sys.exit(f"no attention maps under {seed_dir}")
    grid = torch.load(maps[0])["map"].shape[-1]
    masks = {k: (box_mask(v["box"], im.width, grid) if v else None) for k, v in boxes.items()}
    rows = []
    for f in maps:
        d = torch.load(f)
        m = d["map"].float()
        m = m / m.sum().clamp_min(1e-8)
        key = f.name.split("_token_")[1][:-3]
        which = "a" if key == TOKENS["a"] else "b"
        row = {"step": d["step_index"], "token": which}
        for k in ("a", "b"):
            row[f"in_{k}"] = float((m * masks[k]).sum()) if masks[k] is not None else None
        rows.append(row)
    rows.sort(key=lambda r: (r["token"], r["step"]))

    def avg(tok, lo, hi, k):
        v = [r[f"in_{k}"] for r in rows if r["token"] == tok and lo <= r["step"] < hi and r[f"in_{k}"] is not None]
        return sum(v) / len(v) if v else float("nan")

    print(f"\n{'token':10s} {'steps':8s} {'in own box':>11s} {'in other box':>13s}")
    for tok, name in (("a", p.a), ("b", p.b)):
        other = "b" if tok == "a" else "a"
        for lo, hi in ((0, 10), (10, 25), (25, 50)):
            print(f"{name:10s} {lo:2d}-{hi:<5d} {avg(tok, lo, hi, tok):11.3f} {avg(tok, lo, hi, other):13.3f}")
    frac = {k: (float(masks[k].mean()) if masks[k] is not None else None) for k in masks}
    print(f"box share of the frame (a uniform map would score this): a {frac['a']}, b {frac['b']}")

    p.out.mkdir(parents=True, exist_ok=True)
    d = ImageDraw.Draw(im)
    for k, col in (("a", "yellow"), ("b", "cyan")):
        if boxes[k]:
            d.rectangle(boxes[k]["box"], outline=col, width=6)
            d.text((boxes[k]["box"][0] + 8, boxes[k]["box"][1] + 8), {"a": p.a, "b": p.b}[k], fill=col)
    im.save(p.out / f"boxes_seed_{p.seed}.png")
    # the two tokens' maps at steps 0, 5, 10, 25, 49, side by side, for the eye
    steps = [0, 5, 10, 25, 49]
    T = 128
    sheet = Image.new("L", (T * len(steps), T * 2), 0)
    for j, st in enumerate(steps):
        for i, tok in enumerate(("a", "b")):
            f = seed_dir / "attn_maps" / f"step_{st:03d}_token_{TOKENS[tok]}.pt"
            if f.exists():
                m = torch.load(f)["map"].float()
                m = (m / m.max().clamp_min(1e-8) * 255).byte()
                sheet.paste(Image.fromarray(m.numpy()).resize((T, T)), (j * T, i * T))
    sheet.save(p.out / f"maps_seed_{p.seed}.png")
    json.dump({"a": p.a, "b": p.b, "boxes": boxes, "box_frame_share": frac, "rows": rows},
              open(p.out / f"attention_shares_seed_{p.seed}.json", "w"), indent=1)
    print(f"wrote {p.out}")


if __name__ == "__main__":
    main()
