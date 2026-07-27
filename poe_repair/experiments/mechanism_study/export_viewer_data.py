"""Pack captured attention + decoded samples into one JSON for the artifact.

Layout of the emitted JSON::

    {
      "regime": "plain_poe",
      "pair_slug": "...",
      "tokens": ["cat", "dog"],
      "n_steps": 50,
      "res": 32,
      "seeds": [1, 2, ...],
      "vmax": {"cat": <float>, "dog": <float>},   # global per-token scale
      "timesteps": [981, ...],                     # sigma schedule, per step
      "data": {
        "<seed>": {
          "sample": "data:image/jpeg;base64,...",  # overlay base (may be null)
          "mass": {"cat": [<50 floats>], "dog": [...]},  # Σ map per step
          "maps": {"cat": ["<b64 uint8 1024>", ...], "dog": [...]}
        }, ...
      }
    }

Maps are quantized to uint8 in [0, vmax_token] and base64-encoded row-major
(32×32). The page rescales with vmax to recover the real value.
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

DEFAULT_ATTN_ROOT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
)
_STEP_RE = re.compile(r"step_(\d+)_token_(.+)\.pt$")

# save_key → short display token
TOKEN_MAP = {"cat_branch_poe": "cat", "dog_branch_poe": "dog"}


def _seed_dirs(root: Path) -> list[int]:
    seeds = []
    for d in sorted(root.glob("seed_*")):
        m = re.match(r"seed_(\d+)$", d.name)
        if m and (d / "attn_maps").is_dir():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def _load_seed(root: Path, seed: int, keys: list[str]):
    adir = root / f"seed_{seed}" / "attn_maps"
    per_key_steps: dict[str, dict[int, np.ndarray]] = {k: {} for k in keys}
    timesteps: dict[int, int] = {}
    for f in sorted(adir.glob("step_*_token_*.pt")):
        m = _STEP_RE.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        key = m.group(2)
        if key not in keys:
            continue
        sd = torch.load(f, weights_only=False)
        per_key_steps[key][step] = sd["map"].float().numpy()
        timesteps.setdefault(step, int(sd["timestep"]))
    return per_key_steps, timesteps


def _sample_data_uri(root: Path, seed: int, max_side: int) -> str | None:
    png = root / f"seed_{seed}" / "sample.png"
    if not png.exists():
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    img = Image.open(png).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_viewer_data")
    ap.add_argument("--regime", default="plain_poe")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--sample-max-side", type=int, default=384,
                    help="downscale sample images to this longest edge")
    ap.add_argument("--out", required=True, help="output .json path")
    args = ap.parse_args(argv)

    root = DEFAULT_ATTN_ROOT / args.regime / args.pair_slug
    keys = list(TOKEN_MAP.keys())
    seeds = _seed_dirs(root)
    if not seeds:
        raise SystemExit(f"no seed dirs under {root}")

    # Pass 1: load everything, find n_steps + per-token global vmax.
    loaded: dict[int, dict] = {}
    n_steps = 0
    vmax = {TOKEN_MAP[k]: 0.0 for k in keys}
    timesteps_ref: dict[int, int] = {}
    for s in seeds:
        per_key_steps, ts = _load_seed(root, s, keys)
        loaded[s] = per_key_steps
        timesteps_ref.update(ts)
        for k in keys:
            for step, arr in per_key_steps[k].items():
                n_steps = max(n_steps, step + 1)
                vmax[TOKEN_MAP[k]] = max(vmax[TOKEN_MAP[k]], float(arr.max()))

    # Pass 2: quantize + pack.
    data: dict[str, dict] = {}
    for s in seeds:
        per_key_steps = loaded[s]
        seed_entry: dict = {
            "sample": _sample_data_uri(root, s, args.sample_max_side),
            "mass": {}, "maps": {},
        }
        for k in keys:
            tok = TOKEN_MAP[k]
            vm = vmax[tok] if vmax[tok] > 0 else 1.0
            maps_b64: list[str] = []
            mass: list[float] = []
            for step in range(n_steps):
                arr = per_key_steps[k].get(step)
                if arr is None:
                    maps_b64.append("")
                    mass.append(0.0)
                    continue
                mass.append(round(float(arr.sum()), 4))
                q = np.clip(arr / vm * 255.0, 0, 255).astype(np.uint8)
                maps_b64.append(
                    base64.b64encode(q.tobytes()).decode("ascii")
                )
            seed_entry["maps"][tok] = maps_b64
            seed_entry["mass"][tok] = mass
        data[str(s)] = seed_entry

    res = 32
    # infer res from first map
    for s in seeds:
        for k in keys:
            for arr in loaded[s][k].values():
                res = int(round(arr.shape[-1]))
                break
            break
        break

    out = {
        "regime": args.regime,
        "pair_slug": args.pair_slug,
        "tokens": [TOKEN_MAP[k] for k in keys],
        "n_steps": n_steps,
        "res": res,
        "seeds": seeds,
        "vmax": vmax,
        "timesteps": [timesteps_ref.get(i, 0) for i in range(n_steps)],
        "data": data,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1e6
    n_samples = sum(1 for s in seeds if data[str(s)]["sample"])
    print(f"[export] {out_path}  ({size_mb:.2f} MB)  "
          f"seeds={len(seeds)} steps={n_steps} samples_embedded={n_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
