#!/usr/bin/env python
"""What each expert believes the picture is, at every step, from the cache.

The training cache stores all four raw UNet outputs per step: prompt A alone,
prompt B alone, the joint prompt, and the unconditional pass. From those and the
latent it carried, the Tweedie estimate x̂_0 can be formed under each one without
running the UNet again. Decoding those is the conditioned-versus-unconditional
view: five pictures per step showing

    uncond   what the model draws with no prompt at all
    A        what "a cat" alone pulls towards
    B        what "a dog" alone pulls towards
    PoE      the sum of the two experts, the broken composition
    joint    the single joint prompt, the target

Read across a row and you watch the PoE column fuse into one animal while the
joint column keeps two. That is the failure this project is about, shown rather
than asserted, and it costs no sampling: the UNet outputs are already on disk.

All five views at one step come from the same x_t, so any difference between
them is the prompt and nothing else.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_expert_frames.py
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_expert_frames.py --stride 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean
from poe_repair.experiments.interaction_term import cross_grid as cg
from poe_repair.experiments.interaction_term.cache import _alphas_cumprod, load_cell

OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/experts")
VIEWS = ("uncond", "a", "b", "poe", "joint")
FRAME_PX = 512


def to_uint8(img: torch.Tensor):
    """decode_latents returns [0,1]; treating it as [-1,1] washes out contrast.

    The same mistake is called out in write_decoded_image, which is where this
    convention is pinned. Kept as one helper so the two decode scripts cannot
    drift apart on it.
    """
    arr = img.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.permute(1, 2, 0).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="comma-separated (default: the 3 cross pairs)")
    ap.add_argument("--seeds", help="comma-separated (default: the 3 cross seeds)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--px", type=int, default=FRAME_PX)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pairs = args.pairs.split(",") if args.pairs else list(cg.PAIRS)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(cg.SEEDS))

    from PIL import Image

    from poe_repair._sdxl.runtime import decode_latents
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    ab_all = _alphas_cumprod()
    print(f"{len(pairs)} pair(s) x {len(seeds)} seed(s), every {args.stride} steps")
    print(f"device: {torch.cuda.get_device_name(0)}")

    n_written = n_skipped = 0
    t0 = time.time()
    index: list[dict] = []

    for pair in pairs:
        for seed in seeds:
            try:
                cell = load_cell(pair, seed)
            except FileNotFoundError as e:
                print(f"  skip {pair} seed {seed}: {e}", file=sys.stderr)
                continue
            w = cell.guidance_scale
            steps = list(range(0, cell.n_steps, args.stride))
            if steps[-1] != cell.n_steps - 1:
                steps.append(cell.n_steps - 1)

            out_dir = args.out / pair / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict] = []

            for s in steps:
                x_t = cell.x_t[s]
                ab = ab_all[int(cell.timesteps[s])].to(ctx.device, ctx.dtype)
                u = cell.eps_uncond[s]
                ea = guided_eps(cell.eps_a_raw[s], u, w)
                eb = guided_eps(cell.eps_b_raw[s], u, w)
                ej = guided_eps(cell.eps_j_raw[s], u, w)
                eps_by_view = {
                    "uncond": u, "a": ea, "b": eb,
                    "poe": poe_eps(ea, eb, u), "joint": ej,
                }
                row = {"step": int(s), "timestep": int(cell.timesteps[s])}
                for view in VIEWS:
                    path = out_dir / f"step_{s:03d}_{view}.png"
                    row[view] = str(path)
                    if path.exists() and not args.overwrite:
                        n_skipped += 1
                        continue
                    x0 = tweedie_mean(
                        x_t.to(ctx.device, ctx.dtype), ab,
                        eps_by_view[view].to(ctx.device, ctx.dtype),
                    )
                    img = decode_latents(ctx.models, x0).cpu()
                    Image.fromarray(to_uint8(img)).resize(
                        (args.px, args.px), Image.LANCZOS).save(path)
                    n_written += 1
                rows.append(row)

            (out_dir / "experts.json").write_text(json.dumps({
                "pair": pair, "seed": seed,
                "prompt_a": cell.meta.get("pair", ["", ""])[0],
                "prompt_b": cell.meta.get("pair", ["", ""])[1],
                "guidance_scale": w,
                "n_steps": cell.n_steps,
                "views": list(VIEWS),
                "px": args.px,
                "rows": rows,
            }, indent=2))
            index.append({"pair": pair, "seed": seed,
                          "path": str(out_dir / "experts.json"),
                          "n_steps_decoded": len(rows)})
            print(f"  {pair} seed {seed}: {len(rows)} steps x {len(VIEWS)} views",
                  flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(
        {"views": list(VIEWS), "cells": index}, indent=2))
    print(f"\n{n_written} frames written, {n_skipped} skipped, "
          f"{(time.time() - t0) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
