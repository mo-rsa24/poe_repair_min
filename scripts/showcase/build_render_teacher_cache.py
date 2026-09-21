"""V6 task 1.1 and 1.2: resolve each pool cell to the render a person picked, encode it once
through the frozen VAE, and cache the latent.

The picked render is ``mono.png``: the joint-prompt image, the one the by-eye pass judged to show
both concepts as separate things. ``poe.png`` beside it is the product-of-experts render, which is
the chimera this project exists to fix, and is never the teacher.

    python scripts/showcase/build_render_teacher_cache.py \
        --cells artifacts/_shared/cross_pair_pool_configs/cells_v57.json

Writes one ``.pt`` per cell under artifacts/caches/render_teacher_cache/<pair>/seed_<n>.pt and
prints the count. A cell that resolves to no render is named and counted, never skipped silently:
a smaller corpus that nobody noticed is the failure this print exists to prevent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair._sdxl.runtime import load_sdxl_models
from poe_repair.experiments.twisted_smc.data import _encode_png
from poe_repair.training_cache import CellPath, DEFAULT_CACHE_ROOT

OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/render_teacher_cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--image-name", default="mono.png",
                    help="which render in the cell is the teacher. mono.png is the joint-prompt "
                         "image the by-eye pass picked; poe.png is the chimera.")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    args = ap.parse_args()

    cells = json.load(open(args.cells))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device: the VAE encode would run on the CPU with no error")
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=torch.float16)
    vae = models["vae"]

    written, missing, total_bytes = 0, [], 0
    for pair, seeds in sorted(cells.items()):
        for seed in seeds:
            cell = CellPath.from_root(pair, int(seed), split="train",
                                      cache_root=DEFAULT_CACHE_ROOT)
            src = cell.root / args.image_name
            if not src.exists():
                missing.append(f"{pair} seed {seed}: no {args.image_name}")
                continue
            z = _encode_png(vae, src, device)
            dst = out_root / pair / f"seed_{int(seed)}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"latent": z, "pair": pair, "seed": int(seed),
                        "source": str(src), "image_name": args.image_name}, dst)
            total_bytes += dst.stat().st_size
            written += 1

    want = sum(len(v) for v in cells.values())
    print(f"cells named by {Path(args.cells).name}: {want}")
    print(f"latents written: {written}  ({total_bytes / 1e6:.1f} MB under {out_root})")
    if missing:
        print(f"UNRESOLVED: {len(missing)}")
        for m in missing:
            print("   ", m)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
