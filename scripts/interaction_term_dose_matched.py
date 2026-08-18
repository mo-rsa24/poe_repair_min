#!/usr/bin/env python
"""Separate WHEN the correction lands from HOW MUCH of it lands.

The nine-window timing sweep confounds two things. The windows differ in when
the correction is applied, and they also differ in how much correction gets
delivered, because the correction's own size grows through the run: on
cat x dog it averages 0.48 over steps 0 to 14 and 1.33 over steps 20 to 40.
So "early wins" and "small wins" cannot be told apart from that sweep alone.

Two experiments untie them. Both hold the pair, the seed, the prompt and the
window width fixed, and vary one thing.

  --mode matched   Experiment 3. Rescale the dose inside each window so every
                   window delivers the same total correction. Only the timing
                   differs. If early still wins, timing is doing the work; if
                   the early advantage shrinks, part of the original result was
                   about dose.

  --mode swap      Experiment 4. Four cells per seed: the early window at its
                   own dose and at the late window's dose, and the late window
                   likewise. If small-early composes and large-late does not,
                   timing wins outright.

These cells must be comparable with the nine-window sweep, so they run through
the same sampler it used: ``run_window_batch``, which already takes lambda as an
[cells, steps] tensor. Building that tensor is the whole of the mechanism.

Routing them through ``teacher_residual.run`` instead would look equivalent and
is not. `scripts/check_fixed_schedule.py` compared the two paths on one cell with
byte-identical per-step lambdas and got a mean difference of 2.6 grey levels of
255: the sweep runs a different conditioning path, recorded in its summaries as
``block=window, cond_tag=all``. A dose-matched cell built the other way would
differ from the sweep for reasons having nothing to do with dose.

    python scripts/interaction_term_dose_matched.py --mode matched --dry-run
    python scripts/interaction_term_dose_matched.py --mode matched --seeds 9,10,11,12
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._sampling import write_decoded_image  # noqa: E402
from poe_repair.methods._window_batch import run_window_batch  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from snr_collapse import curve_for  # noqa: E402  the committed size measure

PAIR = "a_cat__x__a_dog"
WIDTH = 10
STRIDE = 5
STEPS = 50
BATCH = 8            # the window sweep's batch; cells only match it at this shape
EXP_NAME = "interaction_term/dose_matched"
OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose_matched")


def window_lambdas(win: tuple[int, int], scale: float) -> np.ndarray:
    """One cell's lambda per step: `scale` inside the window, 0 outside.

    This is `schedule_masks(corr_window=win, lambda_max=scale)` with the prompt
    on at every step, written out so the scale is visibly the only thing that
    varies between cells.
    """
    lam = np.zeros(STEPS, dtype=float)
    lam[win[0]:win[1]] = float(scale)
    return lam


def windows() -> list[tuple[int, int]]:
    return [(s, s + WIDTH) for s in range(0, STEPS - WIDTH + 1, STRIDE)]


def size_per_step(seed: int) -> np.ndarray:
    """The correction's size at each step, same measure F3 and F4b draw."""
    y = np.asarray(curve_for(PAIR, seed)[3], dtype=float)
    if len(y) != STEPS:
        raise SystemExit(f"cached curve for seed {seed} has {len(y)} steps, "
                         f"expected {STEPS}")
    return y


def matched_lambdas(size: np.ndarray, win: tuple[int, int],
                    target: float) -> tuple[np.ndarray, float]:
    """Lambda inside `win` scaled so the window delivers `target` total size.

    Delivered total is sum over the window of lambda_t * size_t. Constant lambda
    inside the window keeps the shape of the dose and changes only its scale, so
    the comparison against the unmatched sweep stays a single-axis one.
    """
    lam = np.zeros(STEPS, dtype=float)
    a, b = win
    delivered_at_one = float(size[a:b].sum())
    if delivered_at_one <= 0:
        raise SystemExit(f"window {win} delivers no correction at lambda 1")
    scale = target / delivered_at_one
    lam[a:b] = scale
    return lam, scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("matched", "swap"), required=True)
    ap.add_argument("--seeds", default="9,10,11,12")
    ap.add_argument("--pair", default=PAIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="print every cell and its lambdas, sample nothing")
    ap.add_argument("--max-scale", type=float, default=4.0,
                    help="refuse a cell whose matched dose exceeds this, since a "
                         "very large lambda leaves the regime the sweep measured")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    wins = windows()

    # The reference dose every matched window must deliver: what the earliest
    # window delivers at full strength. Chosen because it is the window that
    # composes, so the matched sweep asks "can a later window do this well with
    # the same amount", not "can it do it with more".
    plan = []
    for seed in seeds:
        size = size_per_step(seed)
        ref_win = wins[0]
        target = float(size[ref_win[0]:ref_win[1]].sum())
        if args.mode == "matched":
            chosen = wins
        else:
            chosen = [wins[0], wins[-1]]
        for win in chosen:
            if args.mode == "matched":
                lam, scale = matched_lambdas(size, win, target)
                tag = f"matched_w{win[0]}-{win[1]}"
                plan.append((seed, win, lam, scale, tag, target))
            else:
                own = float(size[win[0]:win[1]].sum())
                for donor in (wins[0], wins[-1]):
                    donor_total = float(size[donor[0]:donor[1]].sum())
                    lam, scale = matched_lambdas(size, win, donor_total)
                    tag = (f"swap_w{win[0]}-{win[1]}"
                           f"_dose_of_w{donor[0]}-{donor[1]}")
                    plan.append((seed, win, lam, scale, tag, donor_total))

    print(f"mode {args.mode}: {len(plan)} cells over {len(seeds)} seeds\n")
    print(f"{'seed':>5} {'window':>9} {'lambda':>8} {'delivers':>9}  tag")
    over = []
    for seed, win, lam, scale, tag, target in plan:
        print(f"{seed:>5} {str(list(win)):>9} {scale:>8.3f} {target:>9.2f}  {tag}")
        if scale > args.max_scale:
            over.append((seed, win, scale, tag))
    if over:
        print()
        for seed, win, scale, tag in over:
            print(f"refusing {tag} seed {seed}: matched dose is {scale:.2f}x full "
                  f"strength, above --max-scale {args.max_scale}")
        print("A window whose correction is small needs a large lambda to deliver "
              "the same total, and a large lambda leaves the regime the original "
              "sweep measured. Raise --max-scale deliberately or accept the gap.")
        return 1

    if args.dry_run:
        print("\ndry run: nothing sampled")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(num_inference_steps=STEPS)
    written = []

    # Batched per seed: one seed's cells share an init latent and its prompt
    # embeddings, and run_window_batch samples them together.
    for seed in seeds:
        seed_rows = [p for p in plan if p[0] == seed]
        cell = cell_from_slug(args.pair, seed)
        init, euler_sigma = init_latents_for_cell(cell, ctx)
        emb = encode_pair(cell, ctx)
        seq_j, pool_j = get_joint_embeds(cell, ctx)
        # Chunked into groups of exactly BATCH, the last padded by repeating its
        # final cell, exactly as run_cross_sweep pads its groups. The same UNet
        # returns different numbers at different batch shapes, so a cell run at any
        # other N does not reproduce the window sweep's image, while the same cell
        # at N=8 is byte-identical to it, verified on all 8 duplicate cells in
        # --mode swap. A seed with 9 cells is therefore two batches, not one of 9.
        groups = [seed_rows[i:i + BATCH] for i in range(0, len(seed_rows), BATCH)]

        for rows in groups:
            n = len(rows)
            pad = BATCH - n
            lam_rows = [r[2] for r in rows] + [rows[-1][2]] * pad
            lam = torch.tensor(np.stack(lam_rows), dtype=torch.float32)
            cond_on = torch.ones((BATCH, STEPS), dtype=torch.bool)  # prompt on throughout
            n_batch = BATCH

            def rep(t):
                return t.repeat(n_batch, *([1] * (t.dim() - 1)))

            out = run_window_batch(
                init_latents=init.repeat(n_batch, 1, 1, 1),
                models=ctx.models, scheduler=ctx.scheduler,
                seq_a=rep(emb["seq_a"]), pool_a=rep(emb["pool_a"]),
                seq_b=rep(emb["seq_b"]), pool_b=rep(emb["pool_b"]),
                seq_j=rep(seq_j), pool_j=rep(pool_j),
                seq_e=rep(emb["seq_e"]), pool_e=rep(emb["pool_e"]),
                cond_on=cond_on, lam=lam,
                guidance_scale=ctx.guidance_scale,
                num_inference_steps=STEPS,
                height=cell.height, width=cell.width,
                euler_init_noise_sigma=euler_sigma,
                device=ctx.device, dtype=ctx.dtype,
                save_trajectory=False,
            )

            for i, (sd, win, lam_i, scale, tag, target) in enumerate(rows):
                d = OUT_DIR / "pairs" / args.pair / f"seed_{sd}" / tag
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{tag}.png"
                write_decoded_image(out.images[i], path)
                (d / f"summary_{tag}.json").write_text(json.dumps({
                    "pair": args.pair, "seed": sd, "tag": tag,
                    "window": list(win), "lambda_inside": scale,
                    "delivered_total": target,
                    "lambda_per_step": [float(v) for v in lam_i],
                    "sampler": "run_window_batch, the same path as the window sweep",
                }, indent=2))
                written.append({"seed": sd, "window": list(win), "tag": tag,
                                "lambda_inside": scale, "delivered_total": target,
                                "image": str(path)})
                print(f"wrote  seed {sd:<3d} {tag:<34s} {path}")
        gc.collect()
        torch.cuda.empty_cache()

    manifest = OUT_DIR / f"{args.mode}_manifest.json"
    manifest.write_text(json.dumps({
        "mode": args.mode, "pair": args.pair, "seeds": seeds,
        "width": WIDTH, "stride": STRIDE, "steps": STEPS,
        "reference": "the earliest window at full strength; every cell delivers "
                     "that same total correction",
        "measure": "delivered total = sum over the window of lambda_t * size_t, "
                   "size being the same per-step measure F3 and F4b draw",
        "cells": written,
    }, indent=2))
    print(f"\nmanifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
