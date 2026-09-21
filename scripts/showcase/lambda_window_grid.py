#!/usr/bin/env python
"""Experiment C: the lambda-window injection series (plan 07, 01-showcase-the-trained-lora).

At the fixed 100k checkpoint, fixed injection window (steps 0-9), sweep the LoRA correction's
lambda across {0, 0.25, 0.5, 0.75, 1.0} on five pair/seed cells: the four F9 pairs (the four
held-out pairs rendered at seed 9 in scripts/joint_poe_adapter_grid.py) plus the repaired pair
(cat x dog at seed 1, decisions-taken-here.md's "the repaired run"):

    a_cat__x__a_dog       seed 9
    an_eagle__x__a_hawk   seed 9
    a_frog__x__a_toad     seed 9
    a_goose__x__a_swan    seed 9
    a_cat__x__a_dog       seed 1   (the repaired pair)

Unlike plan 03's lora_dose_sweep.py (a same-prompt "a dog" x "a dog" collision test with
wrong_seed/shuffled controls over held-out seeds), this grid has no controls: every pair uses
its own two-animal prompt, "real" condition only. It reuses plan 03's LoRA attach/load logic and
disk-guard/GPU-check boilerplate, but NOT its sampler
(poe_repair.methods._sampling.run_lora_residual_inject_masked): that sampler's off-window steps
run a single unconditional forward with no guidance at all (by design, for a different
compute-budget experiment elsewhere in this project), which cannot reproduce this project's
cached poe.png (guided PoE at all 50 steps, per scripts/build_training_cache.py) at lambda=0 —
exactly the identity check this plan's review file pre-registers. This script instead defines
run_lora_residual_inject_windowed_poe: off-window steps run the same full guided-PoE forward as
on-window steps, adapter disabled, so lambda=0 matches poe.png end to end and on-window steps
add lambda * delta_hat on top exactly as plan 03's sampler does.

Stages:
  1. --run-sweep: render all 5 cells x 5 lambdas (25 renders) to OUT_ROOT/renders/.
  2. --measure: per render, the validated instance-count compose read, the DINOv2/CLIP
     embedding-drift read (distance-to-mono minus distance-to-poe, using this project's already-
     cached mono.png/poe.png anchors), and the Laplacian-variance sharpness proxy. Writes
     lambda_softness.json. Also runs the lambda=0 identity check against cached poe.png.
  3. --strip: the descriptive strip for the repaired pair (cat x dog, seed 1), one frame per
     lambda, for plan 05's wall.

Usage:
    python scripts/showcase/lambda_window_grid.py --run-sweep
    python scripts/showcase/lambda_window_grid.py --measure
    python scripts/showcase/lambda_window_grid.py --strip
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    instance_score_to_dict, score_output_instances,
)
from poe_repair.experiments.compose_scorer_validation.scorer import _Embedders, _cosine_distance
from poe_repair.methods._sampling import (
    SamplerOutputs,
    add_time_ids,
    initial_latents_for_pair,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl
from poe_repair._sdxl.metrics import ddim_prev_from_x0_eps, guided_eps, poe_eps, tweedie_mean
from poe_repair._sdxl.runtime import LatentTrajectoryCollector, decode_latents

CHECKPOINT_RUN_DIR = Path(
    "artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k"
)
CHECKPOINT_PATH = CHECKPOINT_RUN_DIR / "checkpoints/lora_step_100000.pt"
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/experiment_c")
TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")

LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"

WINDOW_ON_STEPS = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# The four F9 pairs (held-out, seed 9, per scripts/joint_poe_adapter_grid.py) plus the repaired
# pair (cat x dog, seed 1, per decisions-taken-here.md "The repaired run is a paper fact").
STRIP_PAIR = "cat_dog_s1"  # the repaired pair is the named strip subject for plan 05's wall


@dataclass
class Cell:
    label: str
    slug: str          # matches the training-cache directory name, e.g. a_cat__x__a_dog
    prompt_a: str
    prompt_b: str
    query_a: str        # bare animal word for the instance-count scorer
    query_b: str
    seed: int


CELLS = [
    Cell("cat_dog_s9", "a_cat__x__a_dog", "a cat", "a dog", "cat", "dog", 9),
    Cell("eagle_hawk_s9", "an_eagle__x__a_hawk", "an eagle", "a hawk", "eagle", "hawk", 9),
    Cell("frog_toad_s9", "a_frog__x__a_toad", "a frog", "a toad", "frog", "toad", 9),
    Cell("goose_swan_s9", "a_goose__x__a_swan", "a goose", "a swan", "goose", "swan", 9),
    Cell("cat_dog_s1", "a_cat__x__a_dog", "a cat", "a dog", "cat", "dog", 1),
]


def _mask() -> list[bool]:
    return [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)


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
def run_lora_residual_inject_windowed_poe(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    cfg_mask,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    lambda_value: float = 1.0,
    lora_adapter_name: str = "lora",
) -> SamplerOutputs:
    """Like poe_repair.methods._sampling.run_lora_residual_inject_masked, except off-window
    steps run the SAME full 3-branch guided PoE (A, B, uncond) as on-window steps, just with the
    adapter disabled and no lambda term added — instead of collapsing to a single unconditional
    forward pass.

    Written for plan 07 (01-showcase-the-trained-lora, experiment C): the plan's identity check
    requires lambda=0 to reproduce the cached poe.png (guided PoE at all 50 steps, per
    scripts/build_training_cache.py), which run_lora_residual_inject_masked's "with_prompt"
    off-step mode cannot do (that mode's off-steps are unguided, by design, for a different
    compute-budget experiment elsewhere in this project). Here, on-window steps behave exactly
    as run_lora_residual_inject_masked's on-step branch; off-window steps are the same
    frozen-adapter guided-PoE forward already computed for the on-step case, just run every step.
    """
    mask_list = [bool(x) for x in (
        cfg_mask.tolist() if isinstance(cfg_mask, torch.Tensor) else list(cfg_mask)
    )]
    if len(mask_list) != int(num_inference_steps):
        raise ValueError(
            f"cfg_mask has length {len(mask_list)}, expected {num_inference_steps}"
        )

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )

    pe_3 = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond_3 = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    def _adapter_disable():
        if hasattr(unet, "disable_adapters"):
            unet.disable_adapters()
        elif hasattr(unet, "disable_adapter_layers"):
            unet.disable_adapter_layers()

    def _adapter_enable():
        if hasattr(unet, "enable_adapters"):
            unet.enable_adapters()
        elif hasattr(unet, "enable_adapter_layers"):
            unet.enable_adapter_layers()
        if hasattr(unet, "set_adapter"):
            try:
                unet.set_adapter(lora_adapter_name)
            except Exception:
                pass

    def _three_branch_forward(timestep) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_input_3 = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input_3, timestep, encoder_hidden_states=pe_3,
            added_cond_kwargs=cond_3, timestep_cond=None,
        ).sample
        return noise.chunk(3)

    delta_norm_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        _adapter_disable()
        eps_a_raw_f, eps_b_raw_f, eps_uncond_f = _three_branch_forward(timestep)
        eps_a_f = guided_eps(eps_a_raw_f, eps_uncond_f, guidance_scale)
        eps_b_f = guided_eps(eps_b_raw_f, eps_uncond_f, guidance_scale)
        eps_poe_frozen = poe_eps(eps_a_f, eps_b_f, eps_uncond_f)

        if mask_list[step_index]:
            _adapter_enable()
            eps_a_raw_l, eps_b_raw_l, eps_uncond_l = _three_branch_forward(timestep)
            eps_a_l = guided_eps(eps_a_raw_l, eps_uncond_l, guidance_scale)
            eps_b_l = guided_eps(eps_b_raw_l, eps_uncond_l, guidance_scale)
            eps_poe_lora = poe_eps(eps_a_l, eps_b_l, eps_uncond_l)
            delta_hat = eps_poe_lora - eps_poe_frozen
            eps_t = eps_poe_frozen + float(lambda_value) * delta_hat
            delta_norm_per_step.append(float(delta_hat.float().norm().item()))
        else:
            eps_t = eps_poe_frozen
            delta_norm_per_step.append(0.0)

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=dtype,
        )
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        tracker.store_step(
            step_index, latents, eps_t,
            float(step_index) / float(num_inference_steps),
            int(timestep.item()),
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_t,
        )

    _adapter_enable()
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "cfg_mask": mask_list,
            "delta_norm_per_step": delta_norm_per_step,
        },
    )


def run_sweep(smoke: bool = False) -> None:
    """smoke=True: 1 cell x 1 lambda, to prove the pipeline end to end before the real job."""
    cells = CELLS[:1] if smoke else CELLS
    lambdas = LAMBDA_GRID[:1] if smoke else LAMBDA_GRID

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    attach_info = _attach_and_load_lora(ctx.models["unet"])
    print(f"[lambda_window_grid] LoRA attached+loaded: n_matched={attach_info['n_matched']} "
          f"n_loaded={attach_info['n_loaded']}")

    mask = _mask()
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for cell in cells:
        seq_a, pool_a = encode_prompt_sdxl(
            cell.prompt_a, models=ctx.models, device=ctx.device, dtype=ctx.dtype
        )
        seq_b, pool_b = encode_prompt_sdxl(
            cell.prompt_b, models=ctx.models, device=ctx.device, dtype=ctx.dtype
        )
        pair_cell = cell_for(cell.prompt_a, cell.prompt_b, cell.seed)
        init_latents, euler_sigma = initial_latents_for_pair(
            cell=pair_cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        img_dir = OUT_ROOT / "renders" / cell.label
        img_dir.mkdir(parents=True, exist_ok=True)
        for lam in lambdas:
            img_path = img_dir / f"lambda_{lam}.png"
            out = run_lora_residual_inject_windowed_poe(
                init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                seq_e=seq_e, pool_e=pool_e,
                guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                cfg_mask=mask,
                height=pair_cell.height, width=pair_cell.width,
                euler_init_noise_sigma=euler_sigma,
                device=ctx.device, dtype=ctx.dtype,
                lambda_value=lam, lora_adapter_name=LORA_ADAPTER_NAME,
            )
            write_decoded_image(out.image, img_path)
            max_delta_norm = max(out.extras["delta_norm_per_step"])
            results.append({
                "cell": cell.label, "slug": cell.slug, "seed": cell.seed, "lambda": lam,
                "image_path": str(img_path), "max_delta_norm": max_delta_norm,
            })
            print(f"[lambda_window_grid] {cell.label} lambda={lam} -> {img_path.name} "
                  f"max||delta||={max_delta_norm:.3f}")

    run_path = OUT_ROOT / "sweep_run.json"
    run_path.write_text(json.dumps(results, indent=2))
    print(f"[lambda_window_grid] wrote {len(results)} renders to {run_path.name} "
          f"({len(cells)} cells x {len(lambdas)} lambdas)"
          + (" [SMOKE: overwritten by the real --run-sweep before use]" if smoke else ""))


def _laplacian_var(img_path: Path) -> float:
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(img)
    return float(np.var(gx[1:] - gx[:-1]) + np.var(gy[:, 1:] - gy[:, :-1]))


def measure() -> dict:
    """Measures whatever cells/lambdas are actually present in sweep_run.json — a smoke run's
    partial grid included, so this doesn't assume the full 5x5 has been rendered yet."""
    rows = json.loads((OUT_ROOT / "sweep_run.json").read_text())
    embedders = _Embedders(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    cells_present = [c for c in CELLS if any(r["cell"] == c.label for r in rows)]
    lambdas_present = sorted({r["lambda"] for r in rows})

    by_cell: dict[str, dict] = {}
    identity_check = {}
    for cell in cells_present:
        cell_rows = [r for r in rows if r["cell"] == cell.label]
        anchor_dir = TRAINING_CACHE / "heldout" / cell.slug / f"seed_{cell.seed}"
        mono_path, poe_path = anchor_dir / "mono.png", anchor_dir / "poe.png"

        by_lambda = {}
        for row in sorted(cell_rows, key=lambda r: r["lambda"]):
            out_path = Path(row["image_path"])
            inst = instance_score_to_dict(
                score_output_instances(out_path, cell.query_a, cell.query_b)
            )
            drift = {}
            for space in ("dino", "clip"):
                embed = getattr(embedders, space)
                e_out, e_mono, e_poe = embed([out_path, mono_path, poe_path])
                d_mono = _cosine_distance(e_out, e_mono)
                d_poe = _cosine_distance(e_out, e_poe)
                drift[space] = {"d_mono": d_mono, "d_poe": d_poe, "drift": d_mono - d_poe}
            by_lambda[row["lambda"]] = {
                "compose": inst["label"] == "compose",
                "n_instances": inst["n_instances"],
                "sharpness_laplacian_var": _laplacian_var(out_path),
                "embedding_drift": drift,
            }
            if row["lambda"] == 0.0:
                identity_check[cell.label] = {
                    "dino_distance_to_cached_poe": drift["dino"]["d_poe"],
                    "clip_distance_to_cached_poe": drift["clip"]["d_poe"],
                }

        by_cell[cell.label] = {
            "slug": cell.slug, "seed": cell.seed,
            "prompt_a": cell.prompt_a, "prompt_b": cell.prompt_b,
            "by_lambda": by_lambda,
        }

    mean_sharpness = {}
    compose_rate = {}
    for lam in lambdas_present:
        cells_with_lam = [c for c in cells_present if lam in by_cell[c.label]["by_lambda"]]
        vals = [by_cell[c.label]["by_lambda"][lam]["sharpness_laplacian_var"] for c in cells_with_lam]
        mean_sharpness[lam] = sum(vals) / len(vals) if vals else None
        composed = [by_cell[c.label]["by_lambda"][lam]["compose"] for c in cells_with_lam]
        compose_rate[lam] = sum(composed) / len(composed) if composed else None

    # Monotone-decrease check: sharpness (Laplacian variance) falling means MORE blur as lambda
    # rises (a sharp image has high variance, a blurred one has low variance). Only meaningful
    # for a cell with the full lambda grid rendered (a partial/smoke run is skipped).
    n_monotone = 0
    n_complete_cells = 0
    for c in cells_present:
        if set(by_cell[c.label]["by_lambda"]) != set(LAMBDA_GRID):
            continue
        n_complete_cells += 1
        seq = [by_cell[c.label]["by_lambda"][lam]["sharpness_laplacian_var"] for lam in LAMBDA_GRID]
        if all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)):
            n_monotone += 1

    out = {
        "lambdas": lambdas_present,
        "cells": by_cell,
        "identity_check_lambda0_vs_cached_poe": identity_check,
        "aggregate": {
            "mean_sharpness_laplacian_var_by_lambda": mean_sharpness,
            "compose_rate_by_lambda": compose_rate,
            "n_cells_strictly_monotone_decreasing_sharpness": n_monotone,
            "n_cells_with_full_lambda_grid": n_complete_cells,
            "n_cells_total": len(CELLS),
        },
    }
    (OUT_ROOT / "lambda_softness.json").write_text(json.dumps(out, indent=2))
    print(f"[lambda_window_grid] mean sharpness (Laplacian var) by lambda: "
          f"{json.dumps(mean_sharpness, indent=2)}")
    print(f"[lambda_window_grid] compose rate by lambda: {json.dumps(compose_rate, indent=2)}")
    print(f"[lambda_window_grid] {n_monotone}/{len(CELLS)} cells strictly monotone-decreasing "
          f"in sharpness across the lambda grid")
    print(f"[lambda_window_grid] lambda=0 identity check vs cached poe.png (DINOv2/CLIP "
          f"distance, should be small): {json.dumps(identity_check, indent=2)}")
    return out


def build_strip() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((OUT_ROOT / "sweep_run.json").read_text())
    strip_rows = sorted(
        (r for r in rows if r["cell"] == STRIP_PAIR), key=lambda r: r["lambda"]
    )
    if len(strip_rows) != len(LAMBDA_GRID):
        raise RuntimeError(
            f"expected {len(LAMBDA_GRID)} renders for {STRIP_PAIR}, found {len(strip_rows)} "
            f"— run --run-sweep first"
        )

    fig, axes = plt.subplots(1, len(strip_rows), figsize=(3.2 * len(strip_rows), 3.6))
    for ax, row in zip(axes, strip_rows):
        ax.imshow(Image.open(row["image_path"]).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"λ={row['lambda']}", fontsize=11)
    fig.suptitle("cat × dog, seed 1 (the repaired pair), fixed 100k checkpoint, window 0-9",
                 fontsize=10)
    fig.tight_layout()

    out_path = OUT_ROOT / "lambda_softness_strip.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[lambda_window_grid] wrote descriptive strip -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-sweep", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                     help="restrict --run-sweep to 1 cell x 1 lambda")
    args = ap.parse_args(argv)

    if not any([args.run_sweep, args.measure, args.strip]):
        ap.error("pass at least one of --run-sweep / --measure / --strip")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.run_sweep:
        run_sweep(smoke=args.smoke)
    if args.measure:
        measure()
    if args.strip:
        build_strip()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
