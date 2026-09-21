"""Per-concept read of a run's sheet tiles: is each composing tile one cat and one dog?

The validated compose scorer counts instances of a generic "animal" query, so a count of 2
means two animals and says nothing about which two. This asks the detector for each concept
by name and reports, per tile, how many boxes of each concept survive the same confidence,
size and overlap rules the compose scorer uses.

    <co3 python> scripts/fk_steering/identity_check.py <sheet sidecar json> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CONF = 0.30
NMS_IOU = 0.5
MIN_BOX_FRACTION = 0.12
CONCEPTS = {"a_cat__x__a_dog": ["a cat", "a dog"],
            "a_butterfly__x__a_flower_meadow": ["a butterfly", "a flower meadow"]}


def per_concept(path: Path, queries: list[str], device) -> dict:
    from PIL import Image
    from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics
    with Image.open(path) as im:
        min_side = MIN_BOX_FRACTION * max(im.size)
    out = {}
    for q in queries:
        ds = [d for d in vmetrics.detect_boxes(path, [q], box_threshold=0.20, text_threshold=0.20, device=device)
              if d["confidence"] >= CONF and max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= min_side]
        ds.sort(key=lambda d: -d["confidence"])
        keep = []
        for d in ds:
            if all(vmetrics.box_iou(d["box"], k["box"]) < NMS_IOU for k in keep):
                keep.append(d)
        out[q] = {"n": len(keep), "max_conf": round(max((d["confidence"] for d in keep), default=0.0), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--columns", nargs="+", default=None, help="only these column keys (default: all)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    side = json.loads(args.sidecar.read_text())
    pair = next(p for p in CONCEPTS if p in args.sidecar.name)
    rows = []
    for t in side["tiles"]:
        if args.columns and t["column"] not in args.columns:
            continue
        r = per_concept(Path(t["source"]), CONCEPTS[pair], device)
        both = all(v["n"] >= 1 for v in r.values())
        rows.append({"seed": t["row_seed"], "column": t["column"], "animal_count": t["count"],
                     "per_concept": r, "both_present": bool(both)})
        print(f"seed {t['row_seed']:>2} {t['column']:<24} animals={t['count']} "
              + " ".join(f"{q}:{v['n']}({v['max_conf']:.2f})" for q, v in r.items())
              + ("  BOTH" if both else "  NOT BOTH"), flush=True)
    dst = args.sidecar.with_name(args.sidecar.stem + "_identity.json")
    by_col: dict[str, list[bool]] = {}
    for r in rows:
        by_col.setdefault(r["column"], []).append(r["both_present"])
    summary = {c: {"both_present_fraction": sum(v) / len(v), "n": len(v)} for c, v in by_col.items()}
    dst.write_text(json.dumps({"sidecar": str(args.sidecar), "pair": pair, "concepts": CONCEPTS[pair],
                               "rule": f"conf>={CONF}, longer box side >= {MIN_BOX_FRACTION} of the image, NMS IoU {NMS_IOU}, per-concept query",
                               "by_column": summary, "rows": rows}, indent=1))
    print("\nboth concepts present, by column:")
    for c, v in summary.items():
        print(f"  {c}: {v['both_present_fraction']:.3f} of {v['n']}")
    print("wrote", dst)


if __name__ == "__main__":
    main()
