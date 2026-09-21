#!/usr/bin/env python
"""In-distribution fit of a pooled LoRA checkpoint, measured on the cache itself.

For every cell in the pool (train pairs x train seeds, held-out pairs x held-out seeds) and a
subsample of cached steps, run the LoRA UNet on the exact cached (x_t, t) the trainer saw, form
delta_hat = eps_PoE_lora - eps_PoE_frozen, and take cos(delta_hat, cached delta_t). No sampling,
so the state is the cached state and the number is the fit, not a render-time proxy.

Writes <out>/fit_cosine.json (per cell, per step) and <out>/fit_cosine_table.md (per pair:
mean cosine overall and per bucket early/commit/late, using cfg.probe.commit_window).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.experiments.cross_pair_lora_pooling.pair_pool import load_pair_pool  # noqa: E402
from poe_repair.experiments.cross_pair_lora_pooling.pair_prompts import load_pair_prompts  # noqa: E402
from poe_repair.experiments.cross_pair_lora_pooling.seed_pool import load_seed_pool  # noqa: E402
from poe_repair.experiments.one_pair_one_seed import trainer as T  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.config import RunConfig  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts  # noqa: E402
from poe_repair.methods._sampling import add_time_ids  # noqa: E402
from poe_repair.runtime import infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models  # noqa: E402
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, resolve_cells  # noqa: E402

SCOPE = Path(__file__).resolve().parent.parent.parent / "artifacts/results/does-the-fix-reach-unseen-pairs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--step-stride", type=int, default=5, help="use cached steps 2, 7, 12, ... (stride 5 -> 10 per cell)")
    ap.add_argument("--step-offset", type=int, default=2)
    ap.add_argument("--pair-pool", default=str(SCOPE / "pair_pool.yaml"))
    ap.add_argument("--seed-pool-path", default=str(SCOPE / "seed_pool.yaml"))
    ap.add_argument("--pair-prompts", default=str(SCOPE / "pair_prompts.yaml"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pool = load_pair_pool(args.pair_pool)
    seeds = load_seed_pool(args.seed_pool_path)
    prompts = load_pair_prompts(args.pair_prompts, pair_pool=pool)

    cfg = RunConfig()
    cfg.lora.rank, cfg.lora.alpha = args.rank, args.alpha
    device = infer_device("cuda"); dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(cfg.model_id)
    unet = models["unet"]
    T.attach_lora(unet, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    T.load_lora_state(unet, ckpt["lora_state"])
    unet.eval()
    gs = float(cfg.sampler.guidance_scale); H, W = cfg.sampler.height, cfg.sampler.width
    lo, hi = cfg.probe.commit_window
    print(f"[fit] checkpoint step={ckpt.get('step')} rank={args.rank} commit_window={(lo, hi)}", flush=True)

    plan = [(p, "train", list(seeds.train_pool)) for p in pool.train] + \
           [(p, "heldout", list(seeds.held_out)) for p in pool.heldout]
    rows = []
    t0 = time.time()
    for pair, split, seed_list in plan:
        pp = prompts[pair]
        cfg.cell.prompt_a, cfg.cell.prompt_b, cfg.cell.joint_prompt = pp.prompt_a, pp.prompt_b, pp.joint_prompt
        emb = encode_all_prompts(cfg, models, device, dtype)
        seq3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], 0).to(device=device, dtype=dtype)
        pool3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], 0).to(device=device, dtype=dtype)
        cond = {"text_embeds": pool3, "time_ids": add_time_ids(height=H, width=W, batch_size=3, device=device, dtype=dtype)}
        for cell in resolve_cells(pair, seed_list, split=split):
            try:
                steps = T.load_cached_steps(cell, guidance_scale=gs)
            except FileNotFoundError as exc:
                print(f"[fit] MISSING {pair} seed={cell.seed}: {exc}", flush=True); continue
            for e in steps:
                if (int(e.step_index) - args.step_offset) % args.step_stride != 0:
                    continue
                with torch.no_grad():
                    x = e.x_t.to(device=device, dtype=dtype).repeat(3, 1, 1, 1)
                    tb = torch.full((3,), int(e.timestep), device=device, dtype=torch.long)
                    noise = unet(x, tb, encoder_hidden_states=seq3, added_cond_kwargs=cond).sample.float()
                    ea, eb, eu = noise[0:1], noise[1:2], noise[2:3]
                    poe_lora = T._compose(ea, eb, eu, gs, "poe", 0.5)
                    ea0 = e.eps_a_raw.to(device=device, dtype=torch.float32)
                    eb0 = e.eps_b_raw.to(device=device, dtype=torch.float32)
                    eu0 = e.eps_uncond.to(device=device, dtype=torch.float32)
                    poe_frozen = T._compose(ea0, eb0, eu0, gs, "poe", 0.5)
                    d_hat = (poe_lora - poe_frozen).flatten()
                    d = e.delta_t.to(device=device, dtype=torch.float32).flatten()
                    cos = float(torch.nn.functional.cosine_similarity(d_hat, d, dim=0))
                    ratio = float(d_hat.norm() / (d.norm() + 1e-8))
                si = int(e.step_index)
                bucket = "early" if si < lo else ("commit" if si < hi else "late")
                rows.append({"pair": pair, "split": split, "seed": int(cell.seed), "step": si,
                             "bucket": bucket, "cos": cos, "norm_ratio": ratio})
            print(f"[fit] {split} {pair} seed={cell.seed} done ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)

    (out / "fit_cosine.json").write_text(json.dumps({"checkpoint": args.checkpoint, "step": ckpt.get("step"),
                                                     "rank": args.rank, "rows": rows}, indent=1))
    # per-pair table
    import statistics as st
    def m(vals): return f"{st.mean(vals):.3f}" if vals else "-"
    lines = ["| split | pair | n | cos all | early | commit | late | ‖Δ̂‖/‖Δ‖ |", "|---|---|---|---|---|---|---|---|"]
    for split in ("train", "heldout"):
        for pair in (pool.train if split == "train" else pool.heldout):
            r = [x for x in rows if x["pair"] == pair and x["split"] == split]
            if not r: continue
            lines.append(f"| {split} | {pair} | {len(r)} | {m([x['cos'] for x in r])} | "
                         f"{m([x['cos'] for x in r if x['bucket']=='early'])} | {m([x['cos'] for x in r if x['bucket']=='commit'])} | "
                         f"{m([x['cos'] for x in r if x['bucket']=='late'])} | {m([x['norm_ratio'] for x in r])} |")
        r = [x for x in rows if x["split"] == split]
        lines.append(f"| **{split} mean** | | {len(r)} | **{m([x['cos'] for x in r])}** | "
                     f"{m([x['cos'] for x in r if x['bucket']=='early'])} | {m([x['cos'] for x in r if x['bucket']=='commit'])} | "
                     f"{m([x['cos'] for x in r if x['bucket']=='late'])} | {m([x['norm_ratio'] for x in r])} |")
    (out / "fit_cosine_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
