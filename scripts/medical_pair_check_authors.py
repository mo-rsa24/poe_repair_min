"""Check our detector on the RoentGen-v2 authors' released synthetic images.

Reads one parquet shard of stanfordmimi/RoentGen-v2-synthetic-dataset, scores every image with the
TorchXRayVision DenseNet-121 (the model the paper used for prompt alignment), and reports per
finding the AUROC against the label stored with the image, the drawn-rate at our calibrated
threshold among labelled-positive images, and the most common impressions of the positives.
Two downsamplings are compared: the paper's area interpolation and our crop-and-resize.

Run on mscluster107 with the co3 python.
"""
import argparse
import collections
import io
import json

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
from PIL import Image

from medical_pair_score import auroc

FINDINGS = {"Cardiomegaly": "Cardiomegaly", "Pneumothorax": "Pneumothorax", "Effusion": "Pleural Effusion"}
OURS = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])


def prep(img, how):
    a = xrv.datasets.normalize(np.array(img.convert("L")).astype(np.float32), 255)[None]
    if how == "area":
        return F.interpolate(torch.from_numpy(a)[None], size=(224, 224), mode="area")[0]
    return torch.from_numpy(OURS(a))


@torch.no_grad()
def scores(model, idx, imgs, how, batch=64):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([prep(im, how) for im in imgs[i:i + batch]]).cuda()
        out.append(model(x).cpu().numpy())
    out = np.concatenate(out)
    return {f: out[:, j] for f, j in idx.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="densenet121-res224-all")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA: a CPU run would crawl without any error"
    model = xrv.models.DenseNet(weights=args.weights).cuda().eval()
    idx = {f: model.pathologies.index(f) for f in FINDINGS}
    thr = json.load(open(args.thresholds))
    table = pq.read_table(args.shard).to_pandas()
    print(len(table), "rows")
    imgs = [Image.open(io.BytesIO(d["bytes"])) for d in table["image"]]
    result = {}
    for how in ("area", "ours"):
        sc = scores(model, idx, imgs, how)
        for f, col in FINDINGS.items():
            y = table[col].to_numpy() == 1
            result[f"{how}/{f}"] = {
                "n_pos": int(y.sum()), "auroc": auroc(sc[f][y], sc[f][~y]),
                "drawn_rate_among_pos": float((sc[f][y] > thr[f]["threshold"]).mean()),
                "drawn_rate_among_neg": float((sc[f][~y] > thr[f]["threshold"]).mean()),
            }
            print(how, f, {k: round(v, 3) if isinstance(v, float) else v for k, v in result[f"{how}/{f}"].items()})
    for f, col in FINDINGS.items():
        c = collections.Counter(table.loc[table[col] == 1, "impression"].str.strip())
        print("top impressions with", col, c.most_common(8))
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
