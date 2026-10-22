#!/usr/bin/env python
"""Plain product-of-experts over any number of prompts, one tile per seed, beside the pair sheet's tiles.

The repo's two-concept loop (poe_repair/methods/_sampling.py, run_cfg_poe) with the sum taken over N
conditions: eps = eps_empty + sum_i w (eps_i - eps_empty). Each row starts from the same noise as
scripts/concept_pair_sheet.py's tiles at that seed (init_latents_for_cell on `a_cat__x__a_dog`, which
falls back to noise drawn from the seed number when that cache is absent), so the tile sits in the
same row as theirs.

    python scripts/concept_multi_poe.py --conds dog cat single --seeds 1 2 3 4 5 6 7 8

Tiles land as <OUT>/<cond1>__x__<cond2>__x__.../seedNN/poe.png with a JSON sidecar; an existing tile
is skipped.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from concept_pair_sheet import NOISE_PAIR, OUT, concept_slug, write_sidecar  # noqa: E402


def render(conds: list[str], seeds: list[int]) -> None:
    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    slug = "__x__".join(concept_slug(c) for c in conds)
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    embs = [enc(c) for c in conds] + [enc("")]
    seq = torch.cat([e[0] for e in embs], dim=0)
    pool = torch.cat([e[1] for e in embs], dim=0)
    n = len(embs)
    unet, sched, w = ctx.models["unet"], ctx.scheduler, ctx.guidance_scale
    for seed in seeds:
        out = OUT / slug / f"seed{seed:02d}" / "poe.png"
        if out.exists():
            print(f"{out}: already rendered", flush=True)
            continue
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        h, wd = cell.height, cell.width
        cond = {"text_embeds": pool, "time_ids": add_time_ids(height=h, width=wd, batch_size=n,
                                                              device=ctx.device, dtype=ctx.dtype)}
        sched.set_timesteps(ctx.num_inference_steps)
        lat = (init / sigma).to(device=ctx.device, dtype=ctx.dtype)
        with torch.no_grad():
            for i, t in enumerate(sched.timesteps):
                eps = unet(sched.scale_model_input(lat.repeat(n, 1, 1, 1), t), t,
                           encoder_hidden_states=seq, added_cond_kwargs=cond, timestep_cond=None).sample
                e_u = eps[-1:]
                e = e_u + w * sum(eps[k:k + 1] - e_u for k in range(n - 1))
                abar = sched.alphas_cumprod[int(t.item())].to(device=ctx.device, dtype=ctx.dtype)
                x0 = tweedie_mean(lat, abar, e)
                lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=e)
            image = decode_latents(ctx.models, lat).cpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(image, out)
        write_sidecar(out, {"conditions": conds, "seed": seed, "column": "poe", "noise_from": NOISE_PAIR,
                            "steps": ctx.num_inference_steps, "guidance": w, "size": [wd, h]})
        gc.collect()
        torch.cuda.empty_cache()
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    render(a.conds, a.seeds)
