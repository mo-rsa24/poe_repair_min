#!/usr/bin/env python
"""Decode the per-step latents every sampled cell already saved.

The sampler writes ``latent_trajectory.pt`` beside each image: the latent at
every one of the 51 points along the run. Turning those into pictures needs the
VAE and nothing else, so the frame-by-frame view of what a window does costs no
sampling at all. This decodes them.

Cost is 0.36s a frame and it does not batch: the VAE peaks at 36 GiB decoding
eight 1024x1024 frames at once and runs no faster per frame, so it goes one at a
time. That is why the default takes every fifth step rather than all 51. Pass
``--all-steps`` for the handful of cells worth seeing at full rate.

Frames are written at 512px, which is what the inspector shows; the full-size
image beside them is the one that was scored.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_trajectory_frames.py --root cross
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_trajectory_frames.py \
        --root cross --cells 'a_cat__x__a_dog/seed_9/call__r*' --all-steps
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term")
FRAME_PX = 512


def steps_to_decode(n_frames: int, stride: int, all_steps: bool) -> list[int]:
    """Which points along the run to decode.

    The last frame is always included: it is the finished picture, and a strip
    that stops at step 45 invites the reader to think the run ended there.
    """
    if all_steps:
        return list(range(n_frames))
    keep = list(range(0, n_frames, stride))
    if keep[-1] != n_frames - 1:
        keep.append(n_frames - 1)
    return keep


def find_cells(root: Path, pattern: str | None) -> list[Path]:
    cells = sorted(p.parent for p in root.rglob("latent_trajectory.pt"))
    if pattern:
        cells = [c for c in cells
                 if fnmatch.fnmatch(str(c.relative_to(root / "pairs")), pattern)]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="cross",
                    help="'cross', 'window', or an absolute path")
    ap.add_argument("--cells", help="glob over <pair>/seed_N/<cell_id>")
    ap.add_argument("--source", choices=("x0", "xt"), default="x0",
                    help="x0: what the model thinks the picture is at that "
                         "step (default, legible from the first frame). "
                         "xt: the noisy latent actually carried.")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--all-steps", action="store_true")
    ap.add_argument("--px", type=int, default=FRAME_PX)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if "/" in args.root else OUT_ROOT / args.root
    cells = find_cells(root, args.cells)
    if args.limit:
        cells = cells[:args.limit]
    if not cells:
        print(f"no cells with a saved trajectory under {root}", file=sys.stderr)
        return 2

    from PIL import Image

    from poe_repair._sdxl.runtime import decode_latents
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    print(f"{len(cells)} cell(s) under {root}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    n_frames_written = n_skipped = 0
    t0 = time.time()
    for ci, cell in enumerate(cells):
        blob = torch.load(cell / "latent_trajectory.pt", map_location="cpu",
                          weights_only=True)
        # x̂_0 is what the model believes the finished picture is at that step,
        # so a strip of it reads from the first frame. x_t is the noisy latent
        # actually being carried, which looks like noise until near the end and
        # cannot show where two runs part company.
        if args.source == "x0" and "x0_estimates" in blob:
            traj = blob["x0_estimates"]
        else:
            if args.source == "x0":
                print(f"  {cell.name}: no x0_estimates saved, using the noisy "
                      f"latent instead", file=sys.stderr)
            traj = blob["trajectories"]      # [T+1,1,4,h,w]
        keep = steps_to_decode(traj.shape[0], args.stride, args.all_steps)
        frames_dir = cell / "frames"
        frames_dir.mkdir(exist_ok=True)

        written: list[dict] = []
        for s in keep:
            path = frames_dir / f"step_{s:03d}.png"
            if path.exists() and not args.overwrite:
                n_skipped += 1
                written.append({"step": s, "path": str(path)})
                continue
            lat = traj[s].to(ctx.device, ctx.dtype)
            img = decode_latents(ctx.models, lat).cpu()
            arr = (img.clamp(-1, 1) + 1).mul(127.5).round().to(torch.uint8)
            arr = arr[0].permute(1, 2, 0).numpy()
            Image.fromarray(arr).resize((args.px, args.px), Image.LANCZOS).save(path)
            written.append({"step": s, "path": str(path)})
            n_frames_written += 1

        (frames_dir / "frames.json").write_text(json.dumps({
            "source": args.source,
            "n_steps": int(blob.get("num_steps", traj.shape[0] - 1)),
            "n_frames_total": int(traj.shape[0]),
            "stride": None if args.all_steps else args.stride,
            "px": args.px,
            "frames": written,
        }, indent=2))

        if (ci + 1) % 10 == 0 or ci == len(cells) - 1:
            el = time.time() - t0
            rate = el / max(n_frames_written, 1)
            print(f"  {ci + 1}/{len(cells)} cells, {n_frames_written} frames "
                  f"decoded, {n_skipped} already there, {rate:.2f}s per frame",
                  flush=True)

    print(f"\n{n_frames_written} frames written, {n_skipped} skipped, "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
