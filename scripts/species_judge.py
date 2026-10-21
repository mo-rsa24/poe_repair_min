#!/usr/bin/env python
"""Can a score see species? Count the bodies, then name each one.

The validated instance count (GroundingDINO asked for "animal", NMS, the 0.25 size floor) says
how many bodies a render holds. It cannot tell a cat beside a dog from two dogs. This adds one
step: each body's crop is scored by CLIP against the pair's two concept names, and the body is
labelled with the higher one, or called a hybrid when the two scores sit within HYBRID_MARGIN.

    composed    two or more bodies, at least one labelled a and one labelled b, none hybrid
    two_of_one  two or more bodies, all the same label
    hybrid      two or more bodies and at least one hybrid, or one body scoring as a hybrid
    one_body    fewer than two bodies

Run against labels.json (by-eye verdicts) and it prints the agreement, by pair, on the confident
rows, and writes every tile annotated with its boxes and labels so a disagreement can be looked at.

    python scripts/species_judge.py --labels <labels.json> --tiles <ablation_grid> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageStat

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances  # noqa: E402

# A body whose two concept scores sit closer than this is a hybrid. Chosen before the run, on
# no data: CLIP's zero-shot probabilities over two classes are near 0.5 each when the crop
# carries both animals' features. It is reported beside every verdict so it can be judged.
HYBRID_MARGIN = 0.10
# ViT-L/14 rather than the reward's base-patch32: the first pass scored a white cat beside a
# white dog as a dog at 0.84 with the small model. Cached on the cluster already.
CLIP_NAME = "openai/clip-vit-large-patch14"


class Namer:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    @torch.no_grad()
    def probs(self, crop: Image.Image, names: list[str]) -> list[float]:
        # Three ways, not two: the third names the hybrid outright, so a striped dog has
        # somewhere to go other than "tiger".
        a, b = names
        texts = [f"a photo of {a}", f"a photo of {b}",
                 f"a photo of an animal that is half {a.split()[-1]} and half {b.split()[-1]}"]
        inp = self.proc(text=texts, images=crop, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inp).logits_per_image[0]
        return logits.softmax(-1).tolist()


def judge(tile: Path, a: str, b: str, namer: Namer, device) -> dict:
    n, boxes = count_instances(tile, device=device)
    im = Image.open(tile).convert("RGB")
    bodies = []
    mean = tuple(int(v) for v in ImageStat.Stat(im).mean)
    for i, d in enumerate(boxes):
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        crop = im.crop((x0, y0, x1, y1))
        # Paint the other bodies out of this crop with the image mean, so the dog's flank does
        # not sit inside the cat's box. A body wholly inside another is left alone.
        for j, o in enumerate(boxes):
            if j == i:
                continue
            ox0, oy0, ox1, oy1 = (int(v) for v in o["box"])
            if ox0 <= x0 and oy0 <= y0 and ox1 >= x1 and oy1 >= y1:
                continue
            ix0, iy0, ix1, iy1 = max(x0, ox0), max(y0, oy0), min(x1, ox1), min(y1, oy1)
            if ix1 > ix0 and iy1 > iy0:
                ImageDraw.Draw(crop).rectangle((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0), fill=mean)
        pa, pb, ph = namer.probs(crop, [a, b])
        if ph >= max(pa, pb) or abs(pa - pb) < HYBRID_MARGIN:
            label = "hybrid"
        else:
            label = a if pa > pb else b
        bodies.append({"box": [x0, y0, x1, y1], "p_a": round(pa, 3), "p_b": round(pb, 3),
                       "p_hybrid": round(ph, 3), "label": label})
    labels = [x["label"] for x in bodies]
    if n < 2:
        verdict = "hybrid" if labels == ["hybrid"] else "one_body"
    elif "hybrid" in labels:
        verdict = "hybrid"
    elif a in labels and b in labels:
        verdict = "composed"
    else:
        verdict = "two_of_one"
    return {"n_bodies": n, "bodies": bodies, "verdict": verdict}


def annotate(tile: Path, res: dict, out: Path):
    im = Image.open(tile).convert("RGB")
    d = ImageDraw.Draw(im)
    for body in res["bodies"]:
        x0, y0, x1, y1 = body["box"]
        d.rectangle((x0, y0, x1, y1), outline="yellow", width=6)
        d.text((x0 + 8, y0 + 8), f'{body["label"]}  {body["p_a"]:.2f}/{body["p_b"]:.2f}/h{body["p_hybrid"]:.2f}', fill="yellow")
    d.text((10, im.height - 30), res["verdict"], fill="yellow")
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--tiles", type=Path, required=True, help="the ablation grid root")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    namer = Namer(device)
    cells = json.load(open(a.labels))["cells"]
    rows = []
    for c in cells:
        tile = a.tiles / c["slug"] / f"seed{c['seed']:02d}" / f"ours_{c['run']}_s{c['step']:06d}_w50.png"
        if not tile.exists():
            print(f"missing {tile}", flush=True)
            rows.append({**c, "verdict_score": "missing"})
            continue
        res = judge(tile, c["a"], c["b"], namer, device)
        annotate(tile, res, a.out / "annotated" / f"{c['slug']}_s{c['seed']:02d}_{c['run']}_{c['step']}.png")
        rows.append({**c, "verdict_score": res["verdict"], "n_bodies": res["n_bodies"], "bodies": res["bodies"]})
        flag = "" if res["verdict"] == c["verdict"] else "  <-- disagrees"
        print(f"{c['slug']:28s} seed {c['seed']:2d} {c['run']:22s} {c['step']:6d}  eye {c['verdict']:10s} "
              f"score {res['verdict']:10s} bodies {res['n_bodies']}{'' if c['confident'] else ' (eye unsure)'}{flag}",
              flush=True)

    conf = [r for r in rows if r["confident"] and r["verdict_score"] != "missing"]
    agree = sum(r["verdict_score"] == r["verdict"] for r in conf)
    by_pair = defaultdict(Counter)
    for r in conf:
        by_pair[r["slug"]][r["verdict_score"] == r["verdict"]] += 1
    print(f"\nagreement on confident rows: {agree} of {len(conf)}")
    for slug, cnt in by_pair.items():
        print(f"  {slug:28s} {cnt[True]} agree, {cnt[False]} disagree")
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump({"hybrid_margin": HYBRID_MARGIN, "clip": CLIP_NAME, "agree": agree, "confident_rows": len(conf),
               "rows": rows}, open(a.out / "verdicts.json", "w"), indent=1)
    print(f"wrote {a.out / 'verdicts.json'}")


if __name__ == "__main__":
    main()
