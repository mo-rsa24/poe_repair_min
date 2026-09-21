#!/usr/bin/env python
"""The dog x dog same-prompt check (plan 02, 01-showcase-the-trained-lora).

C1 = C2 = "a dog": both PoE experts agree, so the true interaction term is near zero. If the
trained LoRA (phase1_r8_100k) learned a state-dependent rule rather than a "always add a second
animal" plurality prior, the corrected render still shows one dog and per-step ||r_hat|| stays
small on the cross-pair scale (train/delta_target_norm ~= 29.7). Two dogs in any render falsifies
the "learned a rule" claim as stated.

Everything downstream of the identity preflight is only meaningful if the preflight passes: this
script won't run the dog x dog cells unless --check-identity has already been confirmed (or
--skip-identity-check is passed explicitly, for re-runs after a passing identity check).

Identity preflight ("--check-identity"): LoRA off, PoE("a dog","a dog") must match
CFG("a dog", guidance_scale=2*7.5) within fp16 drift. This reference was settled empirically by
dog_x_dog_identity_check.py, not assumed: eps_PoE(A,A) = eps_tilde_A + eps_tilde_A - eps_empty
applies the CFG guidance term twice, so the matching single-branch reference needs double
guidance, not the project's usual joint-prompt Mono (which is off by ~4.5% relative L2, worse
than the literal same-guidance reading's ~3.3%). Confirmed at seed 9, first denoising step, to
within 0.05% relative L2 / 0.05% of ||eps_PoE(A,A)||.

Probe cells ("--run-probe"): PoE("a dog","a dog") with the LoRA on, lambda=1, active only over
denoising steps 0-10 of 50 (the pre-registered window), for each held-out seed in
phase1_r8_100k's seed pool ([9..16], from that run's seed_pool.json). Uses
run_lora_residual_inject_masked, which already returns per-step ||Delta_hat|| in its
extras["delta_norm_per_step"] -- that IS ||r_hat||, no separate accounting needed.

Scoring ("--score"): the validated instance-count scorer (scorer_validated.json: distinct
instance count >= 2 via GroundingDINO-tiny means compose / two dogs).

Usage:
    python scripts/showcase/dog_x_dog_probe.py --check-identity
    python scripts/showcase/dog_x_dog_probe.py --run-probe --seeds 9,10,11,12,13,14,15,16
    python scripts/showcase/dog_x_dog_probe.py --score --seeds 9,10,11,12,13,14,15,16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.config import RunConfig
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.methods._sampling import (
    add_time_ids,
    initial_latents_for_pair,
    run_lora_residual_inject_masked,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import PairSeedCell, encode_prompt_sdxl

CHECKPOINT_RUN_DIR = Path(
    "artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k"
)
CHECKPOINT_PATH = CHECKPOINT_RUN_DIR / "checkpoints/lora_step_100000.pt"
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe")

# From CHECKPOINT_RUN_DIR/config.json's "lora" block.
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"

# From CHECKPOINT_RUN_DIR/seed_pool.json's "held_out" list.
HELD_OUT_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)

# Pre-registered window: LoRA active over steps 0-10 of 50, per decisions-taken-here.md.
WINDOW_ON_STEPS = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
PROMPT = "a dog"

# Settled empirically by dog_x_dog_identity_check.py (2026-08-31, seed 9): the only candidate
# reference that lands within fp16 drift of eps_PoE("a dog","a dog") is single-branch CFG at
# DOUBLE the per-branch guidance scale, not the project's usual joint-prompt Mono.
IDENTITY_MONO_GUIDANCE_SCALE = 2.0 * GUIDANCE_SCALE
IDENTITY_MAX_ABS_TOL = 0.05  # eps-space; ~13x the empirical 0.0039 single-step reading, latitude for full-trajectory drift accumulation
IDENTITY_REL_L2_TOL = 0.01  # empirical single-step reading was 0.0005


def _cell(seed: int) -> PairSeedCell:
    cfg = RunConfig()
    return PairSeedCell(
        pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / "dog_x_dog_probe",
        pair_slug="dog_x_dog_probe",
        prompt_a=PROMPT, prompt_b=PROMPT,
        seed=seed, regime="collision",
        height=1024, width=1024,
        grid_assets={},
    )


def _attach_and_load_lora(unet: torch.nn.Module) -> dict:
    from types import SimpleNamespace
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig

    lora_cfg = LoRAConfig(
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0,
        target_modules=LORA_TARGET_MODULES, init="gaussian",
        adapter_name=LORA_ADAPTER_NAME,
    )
    fake_cfg = SimpleNamespace(lora=lora_cfg)
    attach_info = lora_trainer.attach_lora(unet, fake_cfg)

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state")
    if state is None:
        raise KeyError(f"{CHECKPOINT_PATH} has no 'lora_state' key (found: {list(ckpt.keys())})")
    lora_trainer.load_lora_state(unet, state)
    attach_info["n_loaded"] = len(state)
    return attach_info


@torch.no_grad()
def _within_batch_identity(seed: int, ctx) -> dict:
    """Option D: inside ONE PoE(A,A) 3-branch batched forward, does the A-slot equal the
    B-slot at every step? No external Mono reference, no cross-run batch-shape confound —
    A and B are the literal same prompt in the same batched call, so nothing should make
    their raw UNet output differ."""
    cell = _cell(seed)
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    from poe_repair.runtime import ddim_prev_from_x0_eps, guided_eps, poe_eps, tweedie_mean

    ctx.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)

    pe = torch.cat([seq_a, seq_a, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_a, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=3,
                                  device=ctx.device, dtype=ctx.dtype),
    }
    unet = ctx.models["unet"]
    per_step_max_abs = []
    for step_index, timestep in enumerate(ctx.scheduler.timesteps):
        latent_input = ctx.scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        per_step_max_abs.append(float((eps_a_raw - eps_b_raw).float().abs().max().item()))

        eps_a = guided_eps(eps_a_raw, eps_uncond, GUIDANCE_SCALE)
        eps_b = guided_eps(eps_b_raw, eps_uncond, GUIDANCE_SCALE)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        alpha_bar_t = ctx.scheduler.alphas_cumprod[int(timestep.item())].to(device=ctx.device, dtype=ctx.dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        latents = ddim_prev_from_x0_eps(
            scheduler=ctx.scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_p,
        )

    return {
        "max_abs_over_all_steps": max(per_step_max_abs),
        "per_step_max_abs": per_step_max_abs,
    }


def check_identity(seed: int) -> dict:
    """LoRA absent. Runs every candidate together rather than gating one at a time:

    D: within one PoE(A,A) batched forward, does the A-slot equal the B-slot at every step?
       (no external reference, no batch-shape confound — see _within_batch_identity)
    Three Mono candidates, each compared against PoE(A,A) across the FULL 50-step trajectory
    (not just step 0), with the per-step diff curve recorded so the drift's shape is visible:
      - same_guidance: CFG("a dog", w=7.5) — the literal pre-registered reading
      - double_guidance: CFG("a dog", w=15.0) — what the eps algebra predicts matches at step 0
      - joint_prompt: CFG("a dog and a dog", w=7.5) — this project's own canonical "mono"
        method (poe_repair/run.py run_method), and the reading raised independently in review
    """
    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)

    within_batch = _within_batch_identity(seed, ctx)

    cell = _cell(seed)
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    joint = PROMPT + " and " + PROMPT
    seq_j, pool_j = encode_prompt_sdxl(joint, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    from poe_repair.methods._sampling import run_cfg, run_cfg_poe

    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    out_poe = run_cfg_poe(seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a, **common)
    traj_poe = out_poe.tracker.trajectories

    mono_runs = {
        "same_guidance": dict(seq_cond=seq_a, pool_cond=pool_a, guidance_scale=GUIDANCE_SCALE),
        "double_guidance": dict(seq_cond=seq_a, pool_cond=pool_a, guidance_scale=IDENTITY_MONO_GUIDANCE_SCALE),
        "joint_prompt": dict(seq_cond=seq_j, pool_cond=pool_j, guidance_scale=GUIDANCE_SCALE),
    }

    candidates = {}
    for name, overrides in mono_runs.items():
        seq_cond = overrides.pop("seq_cond")
        pool_cond = overrides.pop("pool_cond")
        run_kwargs = dict(common)
        run_kwargs.update(overrides)
        out_mono = run_cfg(seq_cond=seq_cond, pool_cond=pool_cond, **run_kwargs)
        traj_mono = out_mono.tracker.trajectories
        per_step_max_abs = [
            float((traj_poe[i].float() - traj_mono[i].float()).abs().max().item())
            for i in range(traj_poe.shape[0])
        ]
        d = (out_poe.latents.float() - out_mono.latents.float())
        max_abs = float(d.abs().max().item())
        ref_norm = float(out_poe.latents.float().norm().item())
        rel_l2 = float(d.norm().item()) / ref_norm
        candidates[name] = {
            "guidance_scale": run_kwargs["guidance_scale"],
            "max_abs_latent_diff": max_abs,
            "rel_l2_latent_diff": rel_l2,
            "passed": max_abs <= IDENTITY_MAX_ABS_TOL and rel_l2 <= IDENTITY_REL_L2_TOL,
            "per_step_max_abs_latent_diff": per_step_max_abs,
        }

    result = {
        "seed": seed, "prompt": PROMPT, "joint_prompt": joint,
        "within_batch_identity": within_batch,
        "tol_max_abs": IDENTITY_MAX_ABS_TOL, "tol_rel_l2": IDENTITY_REL_L2_TOL,
        "candidates": candidates,
        "passed": within_batch["max_abs_over_all_steps"] == 0.0,
    }
    print(json.dumps({
        **{k: v for k, v in result.items() if k != "candidates"},
        "candidates": {
            name: {k: v for k, v in c.items() if k != "per_step_max_abs_latent_diff"}
            for name, c in candidates.items()
        },
    }, indent=2))
    print("within_batch per_step_max_abs:",
          [round(x, 6) for x in within_batch["per_step_max_abs"]])
    for name, c in candidates.items():
        print(f"{name} per_step_max_abs_latent_diff:",
              [round(x, 4) for x in c["per_step_max_abs_latent_diff"]])
    return result


def run_mono(seeds: list[int]) -> list[dict]:
    """Mono column: the literal joint prompt "a dog and a dog", CFG w=7.5, no PoE, no LoRA."""
    from poe_repair.methods._sampling import run_cfg

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    joint = PROMPT + " and " + PROMPT
    seq_j, pool_j = encode_prompt_sdxl(joint, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    out_dir = OUT_ROOT / "mono"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        cell = _cell(seed)
        init_latents, euler_sigma = initial_latents_for_pair(
            cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        out = run_cfg(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e,
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype,
        )
        img_path = out_dir / f"seed_{seed}.png"
        write_decoded_image(out.image, img_path)
        results.append({"seed": seed, "image_path": str(img_path), "prompt": joint})
        print(f"[dog_x_dog_probe] mono seed={seed} -> {img_path.name}")
    (out_dir / "mono_run.json").write_text(json.dumps(results, indent=2))
    return results


def run_poe(seeds: list[int]) -> list[dict]:
    """PoE column: PoE("a dog","a dog") with the LoRA off (adapter never attached)."""
    from poe_repair.methods._sampling import run_cfg_poe

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    out_dir = OUT_ROOT / "poe"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        cell = _cell(seed)
        init_latents, euler_sigma = initial_latents_for_pair(
            cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        out = run_cfg_poe(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a,
            seq_e=seq_e, pool_e=pool_e,
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype,
        )
        img_path = out_dir / f"seed_{seed}.png"
        write_decoded_image(out.image, img_path)
        results.append({"seed": seed, "image_path": str(img_path)})
        print(f"[dog_x_dog_probe] poe seed={seed} -> {img_path.name}")
    (out_dir / "poe_run.json").write_text(json.dumps(results, indent=2))
    return results


def score_column(mode: str, seeds: list[int]) -> list[dict]:
    """Score an already-rendered mono/ or poe/ column with the same validated scorer used
    for the LoRA column, so all three columns are judged the same way."""
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances,
    )

    img_dir = OUT_ROOT / mode
    scores = []
    for seed in seeds:
        img_path = img_dir / f"seed_{seed}.png"
        s = score_output_instances(img_path, PROMPT, PROMPT)
        rec = instance_score_to_dict(s)
        rec["seed"] = seed
        scores.append(rec)
        print(f"[dog_x_dog_probe] {mode} seed={seed} n_instances={rec['n_instances']} label={rec['label']}")
    out_path = img_dir / f"{mode}_scores.json"
    out_path.write_text(json.dumps(scores, indent=2))
    n_two = sum(1 for r in scores if r["n_instances"] >= 2)
    print(f"[dog_x_dog_probe] {mode}: {n_two}/{len(scores)} renders show >=2 instances")
    return scores


def run_probe(seeds: list[int]) -> list[dict]:
    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    attach_info = _attach_and_load_lora(ctx.models["unet"])
    print(f"[dog_x_dog_probe] LoRA attached+loaded: n_matched={attach_info['n_matched']} "
          f"n_loaded={attach_info['n_loaded']}")

    mask = [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)
    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    results = []
    for seed in seeds:
        cell = _cell(seed)
        init_latents, euler_sigma = initial_latents_for_pair(
            cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        out = run_lora_residual_inject_masked(
            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a,
            seq_e=seq_e, pool_e=pool_e,
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_mask=mask, composition_mode="with_prompt",
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype,
            lambda_value=1.0, lora_adapter_name=LORA_ADAPTER_NAME,
        )
        img_path = OUT_ROOT / f"seed_{seed}.png"
        write_decoded_image(out.image, img_path)
        rec = {
            "seed": seed,
            "image_path": str(img_path),
            "delta_norm_per_step": out.extras["delta_norm_per_step"],
            "max_delta_norm": max(out.extras["delta_norm_per_step"]),
        }
        results.append(rec)
        print(f"[dog_x_dog_probe] seed={seed} -> {img_path.name} "
              f"max||r_hat||={rec['max_delta_norm']:.3f}")

    write_path = OUT_ROOT / "probe_run.json"
    write_path.write_text(json.dumps(results, indent=2))
    return results


def score(seeds: list[int]) -> list[dict]:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances,
    )

    run_path = OUT_ROOT / "probe_run.json"
    per_seed = {r["seed"]: r for r in json.loads(run_path.read_text())}

    scores = []
    for seed in seeds:
        img_path = OUT_ROOT / f"seed_{seed}.png"
        s = score_output_instances(img_path, PROMPT, PROMPT)
        rec = instance_score_to_dict(s)
        rec["seed"] = seed
        rec["delta_norm_per_step"] = per_seed[seed]["delta_norm_per_step"]
        rec["max_delta_norm"] = per_seed[seed]["max_delta_norm"]
        scores.append(rec)
        print(f"[dog_x_dog_probe] seed={seed} n_instances={rec['n_instances']} "
              f"label={rec['label']} max||r_hat||={rec['max_delta_norm']:.3f}")

    out_path = OUT_ROOT / "probe_scores.json"
    out_path.write_text(json.dumps(scores, indent=2))
    n_two_dogs = sum(1 for r in scores if r["n_instances"] >= 2)
    print(f"[dog_x_dog_probe] {n_two_dogs}/{len(scores)} renders show >=2 instances")
    return scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-identity", action="store_true")
    ap.add_argument("--run-probe", action="store_true", help="render the LoRA column")
    ap.add_argument("--run-mono", action="store_true", help="render the Mono column")
    ap.add_argument("--run-poe", action="store_true", help="render the PoE (LoRA-off) column")
    ap.add_argument("--score", action="store_true", help="score the LoRA column (legacy path)")
    ap.add_argument("--score-mono", action="store_true")
    ap.add_argument("--score-poe", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in HELD_OUT_SEEDS))
    ap.add_argument("--identity-seed", type=int, default=HELD_OUT_SEEDS[0])
    args = ap.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not (args.check_identity or args.run_probe or args.run_mono or args.run_poe
            or args.score or args.score_mono or args.score_poe):
        ap.error("pass at least one of --check-identity / --run-probe / --run-mono / --run-poe / "
                  "--score / --score-mono / --score-poe")

    if args.check_identity:
        result = check_identity(args.identity_seed)
        (OUT_ROOT / "identity_check.json").write_text(json.dumps(result, indent=2))
        if not result["passed"]:
            print("[dog_x_dog_probe] IDENTITY CHECK FAILED — stop; do not trust --run-probe.",
                  file=sys.stderr)
            return 1

    if args.run_probe:
        if not (OUT_ROOT / "identity_check.json").exists():
            print("[dog_x_dog_probe] no identity_check.json on disk — run --check-identity first.",
                  file=sys.stderr)
            return 2
        prior = json.loads((OUT_ROOT / "identity_check.json").read_text())
        if not prior.get("passed"):
            print("[dog_x_dog_probe] last identity check failed — stop.", file=sys.stderr)
            return 2
        run_probe(seeds)

    if args.run_mono:
        run_mono(seeds)

    if args.run_poe:
        run_poe(seeds)

    if args.score:
        score(seeds)

    if args.score_mono:
        score_column("mono", seeds)

    if args.score_poe:
        score_column("poe", seeds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
