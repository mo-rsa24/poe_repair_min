#!/usr/bin/env python3
"""Render a browsable gallery from the liver CECT volumes (Luo et al. 2025).

One PNG per patient: the four phases (P, C1, C2, C3) side by side at the slice through the
lesion, lesion outline in red, soft-tissue window. Controls (no lesion mask) get the middle of
the liver. Output is a folder of ordinary PNGs plus index.html, small enough to copy to a laptop
and scroll there, so nothing has to stay open on the shared cluster.

  python scripts/liver_cect_gallery.py                  8 patients per class, evenly spread
  python scripts/liver_cect_gallery.py --n 20           more per class
  python scripts/liver_cect_gallery.py --all            every patient (361)
  python scripts/liver_cect_gallery.py --ids P0200 P0300

Needs nibabel, numpy, matplotlib. Phases can have different slice counts, so each phase picks
its own slice. Orientation follows the file's array order; pass --flip-lr if left and right look
mirrored against what you expect.
"""
import argparse, csv, html, os
from collections import defaultdict
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = ["P", "C1", "C2", "C3"]
CLASSES = ["HCC", "ICC", "CHCC", "CN"]
WL, WW = 60, 400  # soft-tissue window in HU: level and width


def read_rows(root):
    pts = defaultdict(lambda: {"phases": {}})
    with open(os.path.join(root, "raw", "patient_data.csv")) as f:
        for r in csv.DictReader(f):
            p = pts[r["patient_id"]]
            p.update(cls=r["cancer_type"], age=r["age"], sex=r["gender"])
            p["phases"][r["phase"]] = {k: (None if r[k] == "NA" else os.path.join(root, r[k]))
                                       for k in ("ct_path", "mask_path", "liver_mask_path")}
    return pts


def pick_slice(mask_path, liver_path):
    """Slice with the most lesion; for controls, the middle of the liver."""
    if mask_path:
        m = np.asarray(nib.load(mask_path).dataobj) > 0
        area = m.sum(axis=(0, 1))
        if area.max() > 0:
            return int(area.argmax())
    lv = np.asarray(nib.load(liver_path).dataobj) > 0
    zs = np.where(lv.sum(axis=(0, 1)) > 0)[0]
    return int(zs[len(zs) // 2])


def render_phase(ax, ph, rec, flip):
    z = pick_slice(rec["mask_path"], rec["liver_mask_path"])
    ct_img = nib.load(rec["ct_path"])
    ct = np.asarray(ct_img.dataobj[:, :, z], dtype=float)
    lv = np.asarray(nib.load(rec["liver_mask_path"]).dataobj[:, :, z]) > 0
    ls = (np.asarray(nib.load(rec["mask_path"]).dataobj[:, :, z]) > 0) if rec["mask_path"] else None
    lo, hi = (WL - WW / 2, WL + WW / 2) if ct.min() < 0 else np.percentile(ct, (1, 99))
    img = np.clip((ct - lo) / (hi - lo), 0, 1)
    xs, ys = np.where(lv) if lv.any() else (np.array([0, ct.shape[0] - 1]), np.array([0, ct.shape[1] - 1]))
    x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, ct.shape[0])
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, ct.shape[1])
    view = lambda a: (np.fliplr(np.rot90(a[x0:x1, y0:y1])) if flip else np.rot90(a[x0:x1, y0:y1]))
    ax.imshow(view(img), cmap="gray", vmin=0, vmax=1)
    if ls is not None and ls.any():
        ax.contour(view(ls.astype(float)), levels=[0.5], colors="red", linewidths=0.9)
    ax.set_title(f"{ph}  slice {z}/{ct_img.shape[2]}", fontsize=9)
    ax.axis("off")


def render_patient(pid, p, out_dir, flip):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=80)
    for ax, ph in zip(axes, PHASES):
        if ph in p["phases"]:
            render_phase(ax, ph, p["phases"][ph], flip)
        else:
            ax.axis("off")
    fig.suptitle(f"{pid}   {p['cls']}   age {p['age']}   {p['sex']}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, p["cls"], f"{pid}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def pick_ids(pts, n, ids, everything):
    if ids:
        return ids
    chosen = []
    for c in CLASSES:
        members = sorted(k for k, v in pts.items() if v["cls"] == c)
        idx = range(len(members)) if everything else np.linspace(0, len(members) - 1, min(n, len(members))).astype(int)
        chosen += [members[i] for i in idx]
    return chosen


def write_index(out_dir, done):
    by = defaultdict(list)
    for pid, cls in done:
        by[cls].append(pid)
    parts = ["<!doctype html><meta charset=utf-8><title>Liver CECT gallery</title>",
             "<style>body{font:14px sans-serif;margin:16px}img{width:100%;max-width:1280px;display:block}"
             "h2{margin-top:28px}figure{margin:0 0 14px}</style><h1>Liver CECT gallery</h1>"]
    for c in CLASSES:
        if by[c]:
            parts.append(f"<h2>{c} ({len(by[c])})</h2>")
            parts += [f'<figure><a href="{c}/{pid}.png"><img loading=lazy src="{c}/{pid}.png"></a></figure>'
                      for pid in sorted(by[c])]
    open(os.path.join(out_dir, "index.html"), "w").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/datasets/mmolefe/poe_repair_min/data/liver-cect")
    ap.add_argument("--out", default=None, help="default: <root>/gallery")
    ap.add_argument("--n", type=int, default=8, help="patients per class")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--flip-lr", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "gallery")
    pts = read_rows(a.root)
    todo = pick_ids(pts, a.n, a.ids, a.all)
    done = []
    for i, pid in enumerate(todo, 1):
        try:
            render_patient(pid, pts[pid], out, a.flip_lr)
            done.append((pid, pts[pid]["cls"]))
            print(f"[{i}/{len(todo)}] {pid} {pts[pid]['cls']}", flush=True)
        except Exception as e:  # keep going; report at the end
            print(f"[{i}/{len(todo)}] {pid} FAILED: {e!r}", flush=True)
    write_index(out, done)
    print(f"rendered {len(done)} of {len(todo)} -> {out}/index.html")


if __name__ == "__main__":
    main()
