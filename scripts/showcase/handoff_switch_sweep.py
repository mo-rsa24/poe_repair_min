"""Task 8.2: where does softness live? Sweep the step at which the adapter hands off to the
frozen model, rendering the switch points the clean-tail grid did not (20 and 30 exist).

Same machinery, constants and checkpoint as `scripts/corrector_window_sweep.py --clean-tail`
(rank 32 @ 30050, lambda 1.2, k=0 so no corrector, cat x dog seeds 9 to 16), so the new renders
sit beside the old ones in `clean_tail/<pair>/seed_N/` under the same filename pattern and every
downstream read treats them alike. Scores each render with the compose count and the DINOv2
distance to the seed's joint-prompt render, the clean-tail finding's primary read.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import corrector_window_sweep as cws  # noqa: E402

CUTOFFS = (25, 35, 40, 45)
PAIR = cws.FAILING_PAIR
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/crispness")


def main() -> None:
    import lambda_boundary_probe as lbp
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
    cws._disk_guard(cws.OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cws.make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = cws.ADAPTER_RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], cws.ADAPTER_CHECKPOINT)
    print(f"switch sweep: rank {cws.ADAPTER_RANK} n_matched={info['n_matched']} "
          f"checkpoint_step={info['checkpoint_step']} host={cws._host()}", flush=True)
    c = cws.picked_c()
    rows = []
    for seed in cws.SHEET_SEEDS:
        cell = cws.cell_from_slug(PAIR, seed)
        init_latents, euler_sigma = cws.init_latents_for_cell(cell, ctx)
        emb = cws.encode_pair(cell, ctx)
        for cutoff in CUTOFFS:
            d = cws.OUT / "clean_tail" / PAIR / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            png = d / f"lambda_{cws.TAIL_LAMBDA}_on0-{cutoff}_k000_{cws.CLEAN_TAIL_SCORE}_w{cws.TAIL_WINDOW[0]}-{cws.TAIL_WINDOW[1]}.png"
            t0 = time.time()
            if not png.exists():
                out = run_lora_langevin_windowed_poe(
                    init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                    seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                    seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                    guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                    height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                    device=ctx.device, dtype=ctx.dtype,
                    lambda_value=cws.TAIL_LAMBDA, k=0, c=c, corrector_window=cws.TAIL_WINDOW,
                    noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
                    lambda_window=(0, cutoff), corrector_score=cws.CLEAN_TAIL_SCORE,
                )
                cws.write_decoded_image(out.image, png)
                del out; gc.collect(); torch.cuda.empty_cache()
            rows.append({"pair": PAIR, "seed": seed, "cutoff": cutoff, "png": str(png)})
            print(f"[{time.strftime('%H:%M:%S')}] seed {seed} cutoff {cutoff} ({time.time() - t0:.0f}s)", flush=True)
    # Score every switch point, old and new, so the sweep is one table.
    all_rows = []
    for seed in cws.SHEET_SEEDS:
        for cutoff in (20, 25, 30, 35, 40, 45):
            png = cws.OUT / "clean_tail" / PAIR / f"seed_{seed}" / f"lambda_{cws.TAIL_LAMBDA}_on0-{cutoff}_k000_{cws.CLEAN_TAIL_SCORE}_w{cws.TAIL_WINDOW[0]}-{cws.TAIL_WINDOW[1]}.png"
            if not png.exists():
                continue
            r = {"seed": seed, "cutoff": cutoff, "png": str(png)}
            r.update(cws.score_png(png, PAIR))
            r["dino_dist_to_mono"] = cws.dino_dist_to_mono(png, PAIR, seed)
            all_rows.append(r)
        png = cws.OUT / "tail" / PAIR / f"seed_{seed}" / f"lambda_{cws.TAIL_LAMBDA}_k000_w{cws.TAIL_WINDOW[0]}-{cws.TAIL_WINDOW[1]}.png"
        r = {"seed": seed, "cutoff": 50, "png": str(png)}
        r.update(cws.score_png(png, PAIR)); r["dino_dist_to_mono"] = cws.dino_dist_to_mono(png, PAIR, seed)
        all_rows.append(r)
    (OUT / "handoff_switch_sweep.json").write_text(json.dumps(all_rows, indent=1))
    print("wrote", OUT / "handoff_switch_sweep.json", flush=True)


if __name__ == "__main__":
    main()
