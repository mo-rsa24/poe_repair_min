#!/usr/bin/env python
"""Sample a whole grid in one process, at a fixed batch size.

The single-cell command line spawns a process per image, so SDXL is loaded from
disk 288 times to make 288 pictures: about 31s of wall clock for 17s of compute.
This loads the model once and keeps it, which is where the time goes back.

Batching itself is not the win. On an A6000 the UNet is compute-bound from about
four cells upward, so eight cells per call runs at the same seconds-per-image as
one. The batch size is fixed rather than tuned, and short final groups are padded
to it, because the same UNet returns slightly different numbers at different
batch shapes; letting the last group run narrower would put a systematic
difference into whichever cells happened to land there.

Two grids:
    --grid cross    the conditioning x correction cross plus the dense timing
                    strip, 3 pairs (poe_repair/.../cross_grid.py)
    --grid window   the 8-pair sliding-window sweep that feeds F4
                    (poe_repair/.../window_grid.py)

Resumable: a cell whose image exists is skipped, so Ctrl-C and re-run continues.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_cross_sweep.py --grid window
    CUDA_VISIBLE_DEVICES=1 python scripts/run_cross_sweep.py --grid cross
    CUDA_VISIBLE_DEVICES=1 python scripts/run_cross_sweep.py --grid cross --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.experiments.interaction_term import cross_grid as cg
from poe_repair.experiments.interaction_term import window_grid as wg
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.methods._window_batch import run_window_batch, schedule_masks
from poe_repair.run import make_ctx

OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term")
DISK_LIMIT_PCT = 90


def window_jobs() -> list[dict]:
    """The 8-pair F4 sweep, in the directory layout the scorer already reads."""
    out = []
    for pair in wg.PAIRS:
        for seed in wg.SEEDS:
            for a, b in wg.windows():
                out.append({
                    "pair": pair, "seed": int(seed), "block": "window",
                    "cell_id": f"teacher_residual_const_lam100_w{a}-{b}",
                    "cond_tag": "all", "cond_window": None, "cond_outside": False,
                    "corr": f"{a}-{b}", "corr_window": (a, b), "lambda_max": 1.0,
                })
    return out


def cell_dir(root: Path, job: dict) -> Path:
    return root / "pairs" / job["pair"] / f"seed_{job['seed']}" / job["cell_id"]


def image_path(root: Path, job: dict) -> Path:
    d = cell_dir(root, job)
    # The window grid keeps its original naming so the existing scorer, strip
    # and manifest keep working without a translation layer.
    if job["block"] == "window":
        return d / f"{job['cell_id']}.png"
    return d / "image.png"


def summary_path(root: Path, job: dict) -> Path:
    d = cell_dir(root, job)
    if job["block"] == "window":
        return d / f"summary_{job['cell_id']}.json"
    return d / "summary.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("cross", "window"), required=True)
    ap.add_argument("--batch", type=int, default=cg.BATCH)
    ap.add_argument("--steps", type=int, default=wg.NUM_STEPS)
    ap.add_argument("--pairs", help="comma-separated override")
    ap.add_argument("--seeds", help="comma-separated override")
    ap.add_argument("--dense-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, help="stop after this many cells")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and what would run, touch nothing")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="skip latent_trajectory.pt (6.7MB per cell); the "
                         "per-step frames come from those files, so only pass "
                         "this if the frames are not wanted")
    args = ap.parse_args()

    if args.grid == "cross":
        root = OUT_ROOT / "cross"
        pairs = args.pairs.split(",") if args.pairs else cg.PAIRS
        seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                 else cg.SEEDS)
        jobs = cg.jobs(pairs=pairs, seeds=seeds, dense_stride=args.dense_stride)
    else:
        root = OUT_ROOT / "window"
        jobs = window_jobs()
        if args.pairs:
            keep = set(args.pairs.split(","))
            jobs = [j for j in jobs if j["pair"] in keep]
        if args.seeds:
            keep_s = {int(s) for s in args.seeds.split(",")}
            jobs = [j for j in jobs if j["seed"] in keep_s]

    todo = [j for j in jobs
            if args.overwrite or not image_path(root, j).exists()]
    n_existing = len(jobs) - len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"grid: {args.grid}   output: {root}")
    print(f"{len(jobs)} cells in the grid, {n_existing} already on disk, "
          f"{len(todo)} to run"
          + (f" (limited from {len(jobs) - n_existing})" if args.limit else ""))
    print(f"batch {args.batch} (padded), {args.steps} steps")
    est = len(todo) * 17 / 3600
    print(f"estimate: {est:.1f} hours at 17s per cell")

    if args.dry_run:
        from collections import Counter
        print("\nby block:", dict(Counter(j["block"] for j in todo)))
        print("by pair: ", dict(Counter(j["pair"] for j in todo)))
        print("\nfirst 5 cells to run:")
        for j in todo[:5]:
            print(f"  {j['pair']} seed {j['seed']} {j['cell_id']}"
                  f"  cond={j['cond_window']} corr={j['corr_window']}"
                  f" lam={j['lambda_max']}")
        return 0

    if not todo:
        print("nothing to do")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    pct = 100 * usage.used / usage.total
    print(f"disk: {root} at {pct:.0f}% used")
    if pct >= DISK_LIMIT_PCT:
        print(f"ERROR: over {DISK_LIMIT_PCT}% full, aborting.", file=sys.stderr)
        return 4

    ctx = make_ctx(num_inference_steps=args.steps)
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"gpu {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # Text embeddings depend only on the pair; the starting noise depends on the
    # pair and the seed. Both are computed once and reused across every cell
    # that shares them, which also guarantees a row of the grid really does
    # start from identical noise.
    emb_cache: dict[str, dict] = {}
    init_cache: dict[tuple[str, int], tuple] = {}

    def cell_inputs(pair: str, seed: int):
        if pair not in emb_cache:
            c = cell_from_slug(pair, seed)
            e = encode_pair(c, ctx)
            seq_j, pool_j = get_joint_embeds(c, ctx)
            emb_cache[pair] = dict(cell=c, seq_j=seq_j, pool_j=pool_j, **e)
        if (pair, seed) not in init_cache:
            c = cell_from_slug(pair, seed)
            init_cache[(pair, seed)] = init_latents_for_cell(c, ctx)
        return emb_cache[pair], init_cache[(pair, seed)]

    n_done = n_fail = 0
    t_start = time.time()
    for gi in range(0, len(todo), args.batch):
        group = todo[gi:gi + args.batch]
        real = len(group)
        # Pad to a fixed batch shape; the padding cells are sampled and thrown
        # away rather than shrinking the batch.
        padded = group + [group[-1]] * (args.batch - real)

        cats = {k: [] for k in
                ("seq_a", "pool_a", "seq_b", "pool_b", "seq_j", "pool_j",
                 "seq_e", "pool_e")}
        inits, cond_rows, lam_rows = [], [], []
        sigma = None
        for j in padded:
            emb, (init, sig) = cell_inputs(j["pair"], j["seed"])
            sigma = sig
            for k in cats:
                cats[k].append(emb[k])
            inits.append(init)
            on, lm = schedule_masks(
                num_steps=args.steps,
                cond_window=j["cond_window"],
                corr_window=j["corr_window"],
                lambda_max=j["lambda_max"],
                cond_outside=j["cond_outside"],
            )
            cond_rows.append(on)
            lam_rows.append(lm)

        try:
            out = run_window_batch(
                init_latents=torch.cat(inits, dim=0),
                models=ctx.models, scheduler=ctx.scheduler,
                **{k: torch.cat(v, dim=0) for k, v in cats.items()},
                cond_on=torch.tensor(cond_rows, dtype=torch.bool),
                lam=torch.tensor(lam_rows, dtype=torch.float32),
                guidance_scale=ctx.guidance_scale,
                num_inference_steps=args.steps,
                height=emb_cache[padded[0]["pair"]]["cell"].height,
                width=emb_cache[padded[0]["pair"]]["cell"].width,
                euler_init_noise_sigma=sigma,
                device=ctx.device, dtype=ctx.dtype,
                save_trajectory=not args.no_trajectory,
            )
        except Exception as e:                      # noqa: BLE001
            print(f"  FAILED group at {gi}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            n_fail += real
            continue

        for i, j in enumerate(group):
            d = cell_dir(root, j)
            d.mkdir(parents=True, exist_ok=True)
            write_decoded_image(out.images[i], image_path(root, j))
            if not args.no_trajectory:
                # .clone() is load-bearing: a slice of the batch tensor keeps
                # the whole batch's storage, so saving it writes every cell's
                # trajectory into every cell's file (52MB instead of 6.7MB).
                torch.save(
                    {"trajectories": out.trajectories[:, i:i + 1].clone(),
                     "x0_estimates": out.x0_estimates[:, i:i + 1].clone(),
                     "timesteps": out.timesteps,
                     "num_steps": int(args.steps)},
                    d / "latent_trajectory.pt",
                )
            summary_path(root, j).write_text(json.dumps({
                "method": j["cell_id"],
                "pair_slug": j["pair"], "seed": j["seed"],
                "pair": [emb_cache[j["pair"]]["cell"].prompt_a,
                         emb_cache[j["pair"]]["cell"].prompt_b],
                "block": j["block"],
                "cond_tag": j["cond_tag"],
                "cond_window": (None if j["cond_window"] is None
                                else list(j["cond_window"])),
                "cond_outside": j["cond_outside"],
                "correction_window": (None if j["corr_window"] is None
                                      else list(j["corr_window"])),
                "corr": j["corr"],
                "lambda_max": j["lambda_max"],
                "guidance_scale": ctx.guidance_scale,
                "num_inference_steps": int(args.steps),
                "batch_size": int(args.batch),
                "image_path": str(image_path(root, j)),
                **out.extras[i],
            }, indent=2))
            n_done += 1

        elapsed = time.time() - t_start
        rate = elapsed / max(n_done, 1)
        left = (len(todo) - n_done) * rate / 60
        print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(todo)} cells  "
              f"{rate:.1f}s each  ~{left:.0f} min left  "
              f"({group[0]['pair']} seed {group[0]['seed']} {group[0]['cell_id']})",
              flush=True)

    print(f"\nfinished: {n_done} written, {n_fail} failed, "
          f"{(time.time() - t_start) / 60:.0f} min")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
