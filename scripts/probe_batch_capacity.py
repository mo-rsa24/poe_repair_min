#!/usr/bin/env python
"""How many cells fit on the card at once, measured rather than guessed.

The window sampler runs four UNet branches (A, B, J, uncond) for one cell. A
batched runner stacks N cells into one call, so the UNet sees 4N. This script
times one UNet forward at a range of N and reports peak memory, so the grid size
can be chosen against what the card does rather than against what it should do.

Loads the real UNet at the real resolution. Nothing is saved and no image comes
out; this only measures.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/probe_batch_capacity.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.run import make_ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="1,2,4,6,8,10,12",
                    help="comma-separated cell counts to try")
    ap.add_argument("--reps", type=int, default=3, help="timed forwards per size")
    args = ap.parse_args()

    ctx = make_ctx()
    unet = ctx.models["unet"]
    device, dtype = ctx.device, ctx.dtype
    h = w = 128            # latent side for 1024x1024
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {torch.cuda.get_device_name(0)}  {total:.1f} GiB total")
    print(f"steps per image: {ctx.num_inference_steps}\n")

    from poe_repair.methods._sampling import add_time_ids

    seq_dim = 2048
    print(f"{'cells':>6} {'branches':>9} {'peak GiB':>9} {'s/forward':>10} "
          f"{'s/image(50 steps)':>18}")
    for n in [int(x) for x in args.cells.split(",")]:
        b = 4 * n
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            lat = torch.randn(b, 4, h, w, device=device, dtype=dtype)
            pe = torch.randn(b, 77, seq_dim, device=device, dtype=dtype)
            cond = {
                "text_embeds": torch.randn(b, 1280, device=device, dtype=dtype),
                "time_ids": add_time_ids(height=1024, width=1024, batch_size=b,
                                         device=device, dtype=dtype),
            }
            t = torch.tensor(500, device=device)
            with torch.no_grad():
                unet(lat, t, encoder_hidden_states=pe,
                     added_cond_kwargs=cond, timestep_cond=None).sample
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.reps):
                    unet(lat, t, encoder_hidden_states=pe,
                         added_cond_kwargs=cond, timestep_cond=None).sample
                torch.cuda.synchronize()
                dt = (time.time() - t0) / args.reps
            peak = torch.cuda.max_memory_allocated() / 2**30
            per_image = dt * ctx.num_inference_steps / n
            print(f"{n:>6} {b:>9} {peak:>9.1f} {dt:>10.3f} {per_image:>18.1f}")
            del lat, pe, cond
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>6} {b:>9} {'OOM':>9}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
