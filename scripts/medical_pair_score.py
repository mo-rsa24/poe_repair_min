"""Detector for the pair test in plans/10-does-roentgen-draw-two-findings.

calibrate: run the CheXpert-only TorchXRayVision model on real NIH images, report AUROC per
finding (positives carry the finding, negatives lack it) and the score that gives
CALL_SPECIFICITY on the negatives. Writes thresholds.json.
score: apply those thresholds to folders of generated images and report, per prompt folder,
each finding's drawn-rate and the both-rate for each pair.

Run on mscluster107 with the co3 python.
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image

from medical_pair_bar import CALL_SPECIFICITY

FINDINGS = ["Cardiomegaly", "Pneumothorax", "Effusion"]
PAIRS = {
    "cardiomegaly_pneumothorax": ("Cardiomegaly", "Pneumothorax"),
    "cardiomegaly_effusion": ("Cardiomegaly", "Effusion"),
}
TRANSFORM = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])


def load_model(weights):
    assert torch.cuda.is_available(), "no CUDA: a CPU run would crawl without any error"
    model = xrv.models.DenseNet(weights=weights).cuda().eval()
    return model, {f: model.pathologies.index(f) for f in FINDINGS}


@torch.no_grad()
def probabilities(model, idx, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            a = np.array(Image.open(p).convert("L")).astype(np.float32)
            a = xrv.datasets.normalize(a, 255)[None]
            imgs.append(torch.from_numpy(TRANSFORM(a)))
        out.append(model(torch.stack(imgs).cuda()).cpu().numpy())
    out = np.concatenate(out)
    return {f: out[:, j] for f, j in idx.items()}


def auroc(pos, neg):
    """Probability a random positive scores above a random negative (rank form)."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate(args):
    model, idx = load_model(args.weights)
    labels = {r["Image Index"]: set(r["Finding Labels"].split("|"))
              for r in csv.DictReader(open(args.labels))}
    paths = sorted(p for p in pathlib.Path(args.images).glob("*.png") if p.name in labels)
    print(len(paths), "real NIH images")
    probs = probabilities(model, idx, paths)
    result = {}
    for f in FINDINGS:
        has = np.array([f in labels[p.name] for p in paths])
        pos, neg = probs[f][has], probs[f][~has]
        result[f] = {
            "n_pos": int(has.sum()), "n_neg": int((~has).sum()),
            "auroc": auroc(pos, neg),
            "threshold": float(np.quantile(neg, CALL_SPECIFICITY)),
        }
        print(f, result[f])
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))


def score(args):
    model, idx = load_model(args.weights)
    thr = json.loads(pathlib.Path(args.thresholds).read_text())
    summary = {}
    for d in sorted(p for p in pathlib.Path(args.generated).iterdir() if p.is_dir()):
        paths = sorted(d.glob("seed_*.png"))
        probs = probabilities(model, idx, paths)
        drawn = {f: probs[f] > thr[f]["threshold"] for f in FINDINGS}
        row = {"n": len(paths), **{f"{f}_drawn": float(drawn[f].mean()) for f in FINDINGS},
               **{f"{f}_mean_score": float(probs[f].mean()) for f in FINDINGS}}
        if d.name in PAIRS:
            a, b = PAIRS[d.name]
            row["both_rate"] = float((drawn[a] & drawn[b]).mean())
        summary[d.name] = row
        print(d.name, row)
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--images", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--weights", default="densenet121-res224-chex")
    s = sub.add_parser("score")
    s.add_argument("--generated", required=True)
    s.add_argument("--thresholds", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--weights", default="densenet121-res224-chex")
    a = ap.parse_args()
    calibrate(a) if a.cmd == "calibrate" else score(a)
