"""Pack the training-checkpoint sweep into one JSON for the sweep viewer.

Reads <root>/step_<N>/seed_<S>/{attn_maps,sample.png} and emits::

    {
      "pair_slug","seed","tokens":["cat","dog"],"res":32,"n_steps":50,
      "checkpoints":[12500,...,100000],
      "vmax":{"cat":..,"dog":..},                # global across all ckpts
      "timesteps":[...],
      "data": { "<ckpt>": {
          "sample":"data:image/jpeg;..","mass":{"cat":[..],"dog":[..]},
          "maps":{"cat":["<b64>",..],"dog":[..]},
          "sep":[<50 peak-sep px>]               # cat-vs-dog per denoise step
      }, ... }
    }
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

DEFAULT_ROOT = paths.resolve(paths.ATTENTION_MECHANISM) / "lora_train_sweep"
_STEP_RE = re.compile(r"step_(\d+)_token_(.+)\.pt$")
TOKEN_MAP = {"cat_branch_poe": "cat", "dog_branch_poe": "dog"}


def _load_ckpt_seed(adir: Path, keys):
    per = {k: {} for k in keys}
    ts = {}
    for f in sorted(adir.glob("step_*_token_*.pt")):
        m = _STEP_RE.search(f.name)
        if not m:
            continue
        step, key = int(m.group(1)), m.group(2)
        if key not in keys:
            continue
        sd = torch.load(f, weights_only=False)
        per[key][step] = sd["map"].float().numpy()
        ts.setdefault(step, int(sd["timestep"]))
    return per, ts


def _sample_uri(png: Path, max_side: int):
    if not png.exists():
        return None
    from PIL import Image
    img = Image.open(png).convert("RGB")
    w, h = img.size
    sc = max_side / max(w, h)
    if sc < 1:
        img = img.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _peak(m):
    i = np.unravel_index(m.reshape(32, 32).argmax(), (32, 32))
    return i


def main(argv=None):
    ap = argparse.ArgumentParser(prog="export_sweep_data")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--root", default=None)
    ap.add_argument("--sample-max-side", type=int, default=384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else DEFAULT_ROOT / args.pair_slug
    keys = list(TOKEN_MAP.keys())
    ckpts = sorted(
        int(re.match(r"step_(\d+)$", d.name).group(1))
        for d in root.glob("step_*")
        if re.match(r"step_(\d+)$", d.name)
        and (d / f"seed_{args.seed}" / "attn_maps").is_dir()
    )
    if not ckpts:
        raise SystemExit(f"no step_*/seed_{args.seed} dirs under {root}")

    loaded = {}
    n_steps = 0
    vmax = {"cat": 0.0, "dog": 0.0}
    ts_ref = {}
    for c in ckpts:
        adir = root / f"step_{c}" / f"seed_{args.seed}" / "attn_maps"
        per, ts = _load_ckpt_seed(adir, keys)
        loaded[c] = per
        ts_ref.update(ts)
        for k in keys:
            for st, arr in per[k].items():
                n_steps = max(n_steps, st + 1)
                vmax[TOKEN_MAP[k]] = max(vmax[TOKEN_MAP[k]], float(arr.max()))

    res = 32
    data = {}
    for c in ckpts:
        per = loaded[c]
        entry = {"sample": _sample_uri(
            root / f"step_{c}" / f"seed_{args.seed}" / "sample.png",
            args.sample_max_side), "mass": {}, "maps": {}, "sep": []}
        real = {}
        for k in keys:
            tok = TOKEN_MAP[k]
            vm = vmax[tok] or 1.0
            maps_b64, mass, rlist = [], [], []
            for st in range(n_steps):
                arr = per[k].get(st)
                if arr is None:
                    maps_b64.append("")
                    mass.append(0.0)
                    rlist.append(None)
                    continue
                res = int(round(arr.shape[-1]))
                mass.append(round(float(arr.sum()), 4))
                q = np.clip(arr / vm * 255.0, 0, 255).astype(np.uint8)
                maps_b64.append(base64.b64encode(q.tobytes()).decode())
                rlist.append(arr)
            entry["maps"][tok] = maps_b64
            entry["mass"][tok] = mass
            real[tok] = rlist
        for st in range(n_steps):
            a, b = real["cat"][st], real["dog"][st]
            if a is None or b is None:
                entry["sep"].append(None)
            else:
                pa, pb = _peak(a), _peak(b)
                entry["sep"].append(round(
                    float(((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)**0.5), 2))
        data[str(c)] = entry

    out = {
        "pair_slug": args.pair_slug, "seed": args.seed,
        "tokens": ["cat", "dog"], "res": res, "n_steps": n_steps,
        "checkpoints": ckpts, "vmax": vmax,
        "timesteps": [ts_ref.get(i, 0) for i in range(n_steps)],
        "data": data,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[export_sweep] {op} ({op.stat().st_size/1e6:.2f} MB) "
          f"ckpts={len(ckpts)} steps={n_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
