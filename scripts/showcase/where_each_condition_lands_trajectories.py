#!/usr/bin/env python
"""Per-step frames for the animated "where does each condition land" page.

For cat×dog, held-out seeds 9 to 16, six conditions:
    solo_a    "a cat"                   plain CFG
    solo_b    "a dog"                   plain CFG
    joint     "a cat and a dog"         plain CFG
    poe       A + B − null              the windowed LoRA sampler at λ 0 (identical to plain PoE)
    lora_1.0  PoE + 1.0 × correction    rank-32 LoRA, step 30050, all 50 steps
    lora_1.2  PoE + 1.2 × correction    same adapter

Every run starts from the pinned initial latents in the training cache (heldout
split, step_000 x_t), 50 DDIM steps, guidance 7.5, 1024×1024, so runs differ only
by prompt and by whether the correction is injected. At each of the FRAME_STEPS
the model's running estimate of the finished image (the Tweedie mean
x̂0 = (x_t − √(1−ᾱ_t) ε_t) / √ᾱ_t, using the guided ε_t the sampler actually stepped
with) is decoded through the VAE and saved at 256 px. The final decoded image is
saved as step_050.png.

Writes:
    <out-root>/<condition>/seed_<n>/step_<kk>.png   (kk in FRAME_STEPS and 50)
    <out-root>/frames_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/showcase"))

from poe_repair._sdxl.metrics import tweedie_mean  # noqa: E402
from poe_repair._sdxl.runtime import decode_latents  # noqa: E402
from poe_repair.methods._sampling import run_cfg  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.training_cache import CellPath  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents  # noqa: E402
import lambda_boundary_probe as lbp  # noqa: E402
from lambda_window_grid import run_lora_residual_inject_windowed_poe  # noqa: E402

TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames")
CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
LORA_RANK = 32

PAIR_SLUG = "a_cat__x__a_dog"
PROMPT_A, PROMPT_B, PROMPT_J = "a cat", "a dog", "a cat and a dog"
HELD_OUT_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
EULER_INIT_NOISE_SIGMA = 1.0
FRAME_STEPS = (0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 49)
THUMB = 256
CONDITIONS = ("solo_a", "solo_b", "joint", "poe", "lora_1.0", "lora_1.2")


def _save(img01: torch.Tensor, path: Path) -> None:
    arr = (img01[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).resize((THUMB, THUMB), Image.LANCZOS).save(path)


def dump_frames(out: "SamplerOutputs", ctx, cond_dir: Path) -> list[dict]:
    """Decode x̂0 at FRAME_STEPS from the tracker, plus the final image."""
    cond_dir.mkdir(parents=True, exist_ok=True)
    tr = out.tracker
    frames = []
    for k in FRAME_STEPS:
        z = tr.trajectories[k].to(device=ctx.device, dtype=ctx.dtype)
        eps = tr.velocities[k].to(device=ctx.device, dtype=ctx.dtype)
        t = int(tr.timesteps[k].item())
        ab = ctx.scheduler.alphas_cumprod[t].to(device=ctx.device, dtype=ctx.dtype)
        x0 = tweedie_mean(z, ab, eps)
        img = decode_latents(ctx.models, x0)
        p = cond_dir / f"step_{k:03d}.png"
        _save(img, p)
        frames.append({"step": k, "timestep": t, "png": str(p)})
        del z, eps, x0, img
    p = cond_dir / "step_050.png"
    _save(out.image, p)
    frames.append({"step": 50, "timestep": 0, "png": str(p)})
    torch.cuda.empty_cache()
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--seeds", default=",".join(str(s) for s in HELD_OUT_SEEDS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c for c in args.conditions.split(",") if c.strip()]

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc(PROMPT_A)
    seq_b, pool_b = enc(PROMPT_B)
    seq_j, pool_j = enc(PROMPT_J)

    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                  height=1024, width=1024, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
                  device=ctx.device, dtype=ctx.dtype)
    inits = {}
    for seed in seeds:
        cell = CellPath.from_root(PAIR_SLUG, seed, split="heldout", cache_root=TRAINING_CACHE)
        inits[seed] = load_pinned_init_latents(cell, device=ctx.device, dtype=ctx.dtype,
                                               euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA)
    manifest = []

    def _run_one(cond: str, seed: int) -> None:
        cond_dir = out_root / cond / f"seed_{seed}"
        if (cond_dir / "step_050.png").exists():
            print(f"[skip] {cond_dir} done", flush=True)
            return
        t0 = time.time()
        if cond in ("solo_a", "solo_b", "joint"):
            seq, pool = {"solo_a": (seq_a, pool_a), "solo_b": (seq_b, pool_b), "joint": (seq_j, pool_j)}[cond]
            out = run_cfg(init_latents=inits[seed], seq_cond=seq, pool_cond=pool, **common)
        else:
            lam = {"poe": 0.0, "lora_1.0": 1.0, "lora_1.2": 1.2}[cond]
            out = run_lora_residual_inject_windowed_poe(
                init_latents=inits[seed], seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                cfg_mask=[True] * NUM_INFERENCE_STEPS, lambda_value=lam,
                lora_adapter_name=lbp.LORA_ADAPTER_NAME, **common)
        frames = dump_frames(out, ctx, cond_dir)
        manifest.append({"seed": seed, "condition": cond, "frames": frames,
                         "elapsed_s": round(time.time() - t0, 1)})
        print(f"[{cond}] seed={seed} {len(frames)} frames ({time.time() - t0:.1f}s)", flush=True)
        del out
        torch.cuda.empty_cache()

    # Pass 1: the three single-prompt references, with NO adapter attached to the UNet.
    # (The windowed sampler leaves the adapter enabled when it returns, so attaching first
    # would render the references with the correction active.)
    ref_conds = [c for c in conds if c in ("solo_a", "solo_b", "joint")]
    probe_conds = [c for c in conds if c not in ("solo_a", "solo_b", "joint")]
    for seed in seeds:
        for cond in ref_conds:
            _run_one(cond, seed)

    # Pass 2: attach the adapter once, then PoE (λ 0) and the corrected runs.
    if probe_conds:
        lbp.LORA_RANK = lbp.LORA_ALPHA = LORA_RANK
        info = lbp._attach_and_load_lora(ctx.models["unet"], CHECKPOINT)
        print(f"[frames] LoRA rank={LORA_RANK} attached: n_matched={info['n_matched']} "
              f"n_loaded={info['n_loaded']} checkpoint_step={info['checkpoint_step']}", flush=True)
        for seed in seeds:
            for cond in probe_conds:
                _run_one(cond, seed)

    (out_root / "frames_manifest.json").write_text(json.dumps({
        "pair_slug": PAIR_SLUG, "seeds": seeds, "conditions": conds, "frame_steps": list(FRAME_STEPS) + [50],
        "num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE, "thumb_px": THUMB,
        "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK,
        "x0_rule": "Tweedie mean from the tracker's z_t and the guided eps_t the sampler stepped with",
        "runs": manifest,
    }, indent=2))
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
