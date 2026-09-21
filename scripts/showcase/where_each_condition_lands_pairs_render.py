#!/usr/bin/env python
"""Task 4 of plan 06 (scope 05): the five arms on the eight held-out animal pairs,
with per-step frames and the full latent trajectory saved.

Per pair, per held-out seed 9 to 16, from the pinned initial latents in the training
cache (heldout split, step_000 x_t), 50 DDIM steps, guidance 7.5, 1024 square:

    solo_a    "<a>"                      plain CFG
    solo_b    "<b>"                      plain CFG
    joint     "<a> and <b>"              plain CFG
    poe       A + B - null               the windowed LoRA sampler at lambda 0
    lora_1.2  PoE + 1.2 x correction     rank-32 LoRA, step 30050, all 50 steps

The three references render before the adapter is attached (the windowed sampler
leaves the adapter enabled on the UNet when it returns). Reuses dump_frames from
where_each_condition_lands_trajectories.py, so the frame steps and the Tweedie rule
are identical to the cat x dog run.

Writes:
    <out-root>/<pair_slug>/<condition>/seed_<n>/step_<kk>.png     (13 saved steps + step_050)
    <out-root>/<pair_slug>/<condition>/seed_<n>/image_1024.png    the finished image, full size
    <out-root>/<pair_slug>/<condition>/seed_<n>/latent_trajectory.pt
    <out-root>/<pair_slug>/frames_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/showcase"))

from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.training_cache import CellPath  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents  # noqa: E402
import lambda_boundary_probe as lbp  # noqa: E402
from lambda_window_grid import run_lora_residual_inject_windowed_poe  # noqa: E402
import where_each_condition_lands_trajectories as traj  # noqa: E402

OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/pairs")
PAIR_PROMPTS = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/pair_prompts.json")
HELD_OUT_PAIRS = (
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk", "a_seal__x__a_walrus",
    "a_goose__x__a_swan", "a_cow__x__a_buffalo", "a_cat__x__a_dog", "an_elephant__x__a_penguin",
)
CONDITIONS = ("solo_a", "solo_b", "joint", "poe", "lora_1.2")
LAMBDA = {"poe": 0.0, "lora_1.0": 1.0, "lora_1.2": 1.2}


def _save_trajectory(out, path: Path) -> None:
    tr = out.tracker
    torch.save({
        "trajectories": tr.trajectories.to(torch.float16),
        "velocities": tr.velocities.to(torch.float16),
        "sigmas": tr.sigmas, "timesteps": tr.timesteps, "num_steps": int(tr.num_steps),
    }, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--pairs", default=",".join(HELD_OUT_PAIRS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in traj.HELD_OUT_SEEDS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--no-trajectory", action="store_true", help="skip latent_trajectory.pt (about 100 MB per run)")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    pairs = [p for p in args.pairs.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c for c in args.conditions.split(",") if c.strip()]
    prompts = json.loads(PAIR_PROMPTS.read_text())
    for p in pairs:
        if p not in prompts:
            raise SystemExit(f"no prompts for {p} in {PAIR_PROMPTS}")

    ctx = make_ctx(num_inference_steps=traj.NUM_INFERENCE_STEPS, guidance_scale=traj.GUIDANCE_SCALE)
    enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    common = dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=seq_e, pool_e=pool_e,
                  guidance_scale=traj.GUIDANCE_SCALE, num_inference_steps=traj.NUM_INFERENCE_STEPS,
                  height=1024, width=1024, euler_init_noise_sigma=traj.EULER_INIT_NOISE_SIGMA,
                  device=ctx.device, dtype=ctx.dtype)

    ref_conds = [c for c in conds if c in ("solo_a", "solo_b", "joint")]
    probe_conds = [c for c in conds if c not in ("solo_a", "solo_b", "joint")]
    adapter_attached = False

    def run_pass(pair: str, cond_list: list[str], encoded: dict, inits: dict, manifest: list) -> None:
        for seed in seeds:
            for cond in cond_list:
                cond_dir = out_root / pair / cond / f"seed_{seed}"
                if (cond_dir / "image_1024.png").exists():
                    print(f"[skip] {cond_dir} done", flush=True)
                    continue
                t0 = time.time()
                if cond in ("solo_a", "solo_b", "joint"):
                    seq, pool = encoded[cond]
                    out = run_cfg(init_latents=inits[seed], seq_cond=seq, pool_cond=pool, **common)
                else:
                    out = run_lora_residual_inject_windowed_poe(
                        init_latents=inits[seed],
                        seq_a=encoded["solo_a"][0], pool_a=encoded["solo_a"][1],
                        seq_b=encoded["solo_b"][0], pool_b=encoded["solo_b"][1],
                        cfg_mask=[True] * traj.NUM_INFERENCE_STEPS, lambda_value=LAMBDA[cond],
                        lora_adapter_name=lbp.LORA_ADAPTER_NAME, **common)
                frames = traj.dump_frames(out, ctx, cond_dir)
                write_decoded_image(out.image, cond_dir / "image_1024.png")
                if not args.no_trajectory:
                    _save_trajectory(out, cond_dir / "latent_trajectory.pt")
                manifest.append({"pair": pair, "seed": seed, "condition": cond, "frames": frames,
                                 "image": str(cond_dir / "image_1024.png"),
                                 "elapsed_s": round(time.time() - t0, 1)})
                print(f"[{pair}] [{cond}] seed={seed} ({time.time() - t0:.1f}s)", flush=True)
                del out
                torch.cuda.empty_cache()

    # Pass 1 over every pair: references, adapter not yet attached.
    per_pair: dict[str, dict] = {}
    for pair in pairs:
        pr = prompts[pair]
        encoded = {"solo_a": enc(pr["prompt_a"]), "solo_b": enc(pr["prompt_b"]), "joint": enc(pr["joint_prompt"])}
        inits = {}
        for seed in seeds:
            cell = CellPath.from_root(pair, seed, split="heldout", cache_root=traj.TRAINING_CACHE)
            inits[seed] = load_pinned_init_latents(cell, device=ctx.device, dtype=ctx.dtype,
                                                   euler_init_noise_sigma=traj.EULER_INIT_NOISE_SIGMA)
        per_pair[pair] = {"encoded": encoded, "inits": inits, "manifest": [], "prompts": pr}
        run_pass(pair, ref_conds, encoded, inits, per_pair[pair]["manifest"])

    # Pass 2: attach the adapter once, then PoE and the corrected arm on every pair.
    if probe_conds:
        lbp.LORA_RANK = lbp.LORA_ALPHA = traj.LORA_RANK
        info = lbp._attach_and_load_lora(ctx.models["unet"], traj.CHECKPOINT)
        adapter_attached = True
        print(f"[pairs] LoRA rank={traj.LORA_RANK} attached: n_matched={info['n_matched']} "
              f"n_loaded={info['n_loaded']} checkpoint_step={info['checkpoint_step']}", flush=True)
        for pair in pairs:
            d = per_pair[pair]
            run_pass(pair, probe_conds, d["encoded"], d["inits"], d["manifest"])

    for pair in pairs:
        d = per_pair[pair]
        (out_root / pair / "frames_manifest.json").write_text(json.dumps({
            "pair_slug": pair, "prompts": d["prompts"], "seeds": seeds, "conditions": conds,
            "frame_steps": list(traj.FRAME_STEPS) + [50], "num_inference_steps": traj.NUM_INFERENCE_STEPS,
            "guidance_scale": traj.GUIDANCE_SCALE, "thumb_px": traj.THUMB,
            "checkpoint": str(traj.CHECKPOINT), "lora_rank": traj.LORA_RANK, "lambda": LAMBDA,
            "adapter_attached_before_references": False, "adapter_attached_for_probe_arms": adapter_attached,
            "init_latents": "training cache heldout split step_000 x_t",
            "x0_rule": "Tweedie mean from the tracker's z_t and the guided eps_t the sampler stepped with",
            "runs": d["manifest"],
        }, indent=2))
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
