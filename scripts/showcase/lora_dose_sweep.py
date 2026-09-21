#!/usr/bin/env python
"""The LoRA's run across correction amounts (plan 03, 01-showcase-the-trained-lora).

Extends plan 02's dog x dog runner (dog_x_dog_probe.py) with a lambda sweep on the LoRA's own
output r_hat, plus two size-matched controls that should do nothing:

  - real:       lambda scales r_hat computed live, on this seed's own trajectory. Native support
                via run_lora_residual_inject_masked's lambda_value parameter.
  - wrong_seed: lambda scales a Delta_hat trajectory captured from a DIFFERENT held-out seed,
                injected into this seed's own frozen PoE trajectory.
  - shuffled:   lambda scales this seed's OWN captured Delta_hat trajectory, with its step
                assignment permuted by a fixed (not random-per-run) permutation.

Both controls reuse run_constant_residual_inject, the same precomputed-residual injector this
project already uses for the cached correction's oracle/mean_others control rows.

Stages:
  1. --capture-deltas: one real run per held-out seed at lambda=1, window 0-10 (mask steps 0-9),
     capture_delta_tensors=True. Saves each seed's Delta_hat(t) trajectory to
     OUT_ROOT/deltas/seed_<n>.pt. Delta_hat does not depend on lambda (lambda only scales the
     injection), so one capture per seed covers the whole lambda grid for the controls.
  2. --print-manifest: enumerate seeds x lambdas x conditions, print the per-condition row count
     (a flag with an empty target group is a silent no-op).
  3. --run-sweep: render every manifest row.
  4. --score: score every render with the validated instance-count scorer; write
     dose_curves_lora.json mirroring the cached-correction file's schema.
  5. --softness: per-lambda CLIP/DINOv2 sharpness proxy, labelled descriptive (task 1.5).

Usage:
    python scripts/showcase/lora_dose_sweep.py --capture-deltas
    python scripts/showcase/lora_dose_sweep.py --print-manifest
    python scripts/showcase/lora_dose_sweep.py --run-sweep
    python scripts/showcase/lora_dose_sweep.py --score
    python scripts/showcase/lora_dose_sweep.py --softness
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
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/lora_dose")
DELTA_DIR = OUT_ROOT / "deltas"

LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"

# Same held-out pool plan 02 used, so 02's lambda=1 result is this series' zero-interaction
# reference point at the real-r_hat condition.
HELD_OUT_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)

WINDOW_ON_STEPS = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
PROMPT = "a dog"

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITIONS = ("real", "wrong_seed", "shuffled")

# Fixed, not resampled per run: wrong_seed pairs each seed with its cyclic successor in the pool;
# shuffled reverses the 10-step window's assignment. Both are size-matched to the real condition
# (same seed count, same lambda grid, same number of injected steps) and fixed once so a rerun
# reproduces the same manifest.
def _wrong_seed_partner(seed: int) -> int:
    idx = HELD_OUT_SEEDS.index(seed)
    return HELD_OUT_SEEDS[(idx + 1) % len(HELD_OUT_SEEDS)]


SHUFFLE_PERMUTATION = tuple(reversed(range(WINDOW_ON_STEPS)))  # step i draws from step 9-i


def _mask() -> list[bool]:
    return [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)


def _cell(seed: int) -> PairSeedCell:
    cfg = RunConfig()
    return PairSeedCell(
        pair_dir=cfg.paths.pilot_dir / f"seed_{seed}" / "lora_dose_sweep",
        pair_slug="lora_dose_sweep",
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


def capture_deltas(seeds: list[int]) -> None:
    """One real (lambda=1) pass per seed; save its window-0..9 Delta_hat trajectory to disk."""
    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    attach_info = _attach_and_load_lora(ctx.models["unet"])
    print(f"[lora_dose_sweep] LoRA attached+loaded: n_matched={attach_info['n_matched']} "
          f"n_loaded={attach_info['n_loaded']}")

    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    mask = _mask()

    DELTA_DIR.mkdir(parents=True, exist_ok=True)
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
            capture_delta_tensors=True,
        )
        delta_by_step = out.extras["delta_by_step"]
        assert set(delta_by_step.keys()) == set(range(WINDOW_ON_STEPS)), (
            f"seed={seed}: expected {WINDOW_ON_STEPS} captured steps, got {sorted(delta_by_step)}"
        )
        save_path = DELTA_DIR / f"seed_{seed}.pt"
        torch.save({k: v.cpu() for k, v in delta_by_step.items()}, save_path)
        print(f"[lora_dose_sweep] captured seed={seed} -> {save_path.name} "
              f"({len(delta_by_step)} steps)")


def _load_delta(seed: int) -> dict[int, torch.Tensor]:
    path = DELTA_DIR / f"seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run --capture-deltas first")
    return torch.load(path, map_location="cpu")


def build_manifest(seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        for condition in CONDITIONS:
            for lam in LAMBDA_GRID:
                rows.append({"seed": seed, "condition": condition, "lambda": lam})
    return rows


def print_manifest(seeds: list[int]) -> list[dict]:
    rows = build_manifest(seeds)
    counts = {c: sum(1 for r in rows if r["condition"] == c) for c in CONDITIONS}
    print(f"[lora_dose_sweep] manifest: {len(rows)} rows "
          f"({len(seeds)} seeds x {len(LAMBDA_GRID)} lambdas x {len(CONDITIONS)} conditions)")
    for c, n in counts.items():
        print(f"[lora_dose_sweep]   condition={c}: {n} rows")
        if n == 0:
            print(f"[lora_dose_sweep]   WARNING: condition={c} has zero rows — silent no-op", file=sys.stderr)
    manifest_path = OUT_ROOT / "manifest.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"rows": rows, "counts": counts}, indent=2))
    return rows


def run_sweep(seeds: list[int]) -> None:
    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    attach_info = _attach_and_load_lora(ctx.models["unet"])
    print(f"[lora_dose_sweep] LoRA attached+loaded: n_matched={attach_info['n_matched']} "
          f"n_loaded={attach_info['n_loaded']}")

    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    mask = _mask()

    deltas_by_seed = {seed: _load_delta(seed) for seed in seeds}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        cell = _cell(seed)
        init_latents, euler_sigma = initial_latents_for_pair(
            cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        for condition in CONDITIONS:
            for lam in LAMBDA_GRID:
                img_dir = OUT_ROOT / "renders" / condition
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"seed_{seed}_lambda_{lam}.png"

                # All three conditions share the exact same masked sampler (same window, same
                # composition_mode, same off-window unconditional-only forward) so the ONLY axis
                # that differs is where Delta_hat comes from. Using a different sampler for the
                # controls (e.g. one that runs full PoE off-window) would compare two things that
                # differ in more than the injection source, which is exactly the mixed-comparison
                # failure the review file's fairness check exists to catch.
                external_delta_by_step = None
                if condition == "wrong_seed":
                    source_seed = _wrong_seed_partner(seed)
                    external_delta_by_step = deltas_by_seed[source_seed]
                elif condition == "shuffled":
                    own = deltas_by_seed[seed]
                    external_delta_by_step = {
                        i: own[SHUFFLE_PERMUTATION[i]] for i in range(WINDOW_ON_STEPS)
                    }

                out = run_lora_residual_inject_masked(
                    init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                    seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a,
                    seq_e=seq_e, pool_e=pool_e,
                    guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                    cfg_mask=mask, composition_mode="with_prompt",
                    height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                    device=ctx.device, dtype=ctx.dtype,
                    lambda_value=lam, lora_adapter_name=LORA_ADAPTER_NAME,
                    external_delta_by_step=external_delta_by_step,
                )
                max_delta_norm = max(out.extras["delta_norm_per_step"])

                write_decoded_image(out.image, img_path)
                results.append({
                    "seed": seed, "condition": condition, "lambda": lam,
                    "image_path": str(img_path), "max_delta_norm": max_delta_norm,
                })
                print(f"[lora_dose_sweep] seed={seed} condition={condition} lambda={lam} "
                      f"-> {img_path.name} max||delta||={max_delta_norm:.3f}")

    (OUT_ROOT / "sweep_run.json").write_text(json.dumps(results, indent=2))
    print(f"[lora_dose_sweep] wrote {len(results)} renders to sweep_run.json")


def score() -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances,
    )

    run_path = OUT_ROOT / "sweep_run.json"
    rows = json.loads(run_path.read_text())

    scored = []
    for row in rows:
        img_path = Path(row["image_path"])
        s = score_output_instances(img_path, PROMPT, PROMPT)
        rec = instance_score_to_dict(s)
        rec.update(row)
        scored.append(rec)

    (OUT_ROOT / "sweep_scores.json").write_text(json.dumps(scored, indent=2))

    curves = {c: [] for c in CONDITIONS}
    n_cells = {c: 0 for c in CONDITIONS}
    for condition in CONDITIONS:
        for lam in LAMBDA_GRID:
            cell_rows = [r for r in scored if r["condition"] == condition and r["lambda"] == lam]
            n_cells[condition] += len(cell_rows)
            compose_rate = (
                sum(1 for r in cell_rows if r["n_instances"] >= 2) / len(cell_rows)
                if cell_rows else 0.0
            )
            curves[condition].append(compose_rate)

    # AUC: trapezoidal, normalized to the 0-1 lambda range so it lands on the same 0-1 scale as
    # the cached correction's dose_curves.json.
    def _auc(ys: list[float]) -> float:
        xs = list(LAMBDA_GRID)
        area = sum(
            (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0 for i in range(len(xs) - 1)
        )
        span = xs[-1] - xs[0]
        return area / span if span else 0.0

    auc = {c: _auc(curves[c]) for c in CONDITIONS}

    out = {
        "scorer": "instance_count",
        "lambdas": list(LAMBDA_GRID),
        "curves": curves,
        "auc": auc,
        "n_cells": n_cells,
        "scores": scored,
    }
    (OUT_ROOT / "dose_curves_lora.json").write_text(json.dumps(out, indent=2))
    print(f"[lora_dose_sweep] AUC (area under compose-rate-vs-lambda, 0-1 scale): "
          f"{json.dumps(auc, indent=2)}")
    print(f"[lora_dose_sweep] n_cells per condition: {n_cells}")
    return out


def softness() -> dict:
    """Per-lambda sharpness proxy at the fixed checkpoint (task 1.5, experiment C's question:
    does softness track lambda, or does the same lambda always look equally soft — the latter
    would mean the blur is undertraining, not the injection)."""
    import numpy as np
    from PIL import Image

    rows = json.loads((OUT_ROOT / "sweep_run.json").read_text())
    real_rows = [r for r in rows if r["condition"] == "real"]

    by_lambda: dict[float, list[float]] = {lam: [] for lam in LAMBDA_GRID}
    for row in real_rows:
        img = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
        # Laplacian variance: a standard descriptive sharpness proxy, not a scorer contract.
        gy, gx = np.gradient(img)
        laplacian_var = float(np.var(gx[1:] - gx[:-1]) + np.var(gy[:, 1:] - gy[:, :-1]))
        by_lambda[row["lambda"]].append(laplacian_var)

    means = {lam: (sum(vs) / len(vs) if vs else None) for lam, vs in by_lambda.items()}
    out = {
        "metric": "laplacian_variance",
        "label": "descriptive, not a scorer contract",
        "mean_by_lambda": means,
    }
    (OUT_ROOT / "softness_vs_lambda.json").write_text(json.dumps(out, indent=2))
    print(f"[lora_dose_sweep] softness (Laplacian variance) by lambda: {json.dumps(means, indent=2)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-deltas", action="store_true")
    ap.add_argument("--print-manifest", action="store_true")
    ap.add_argument("--run-sweep", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--softness", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in HELD_OUT_SEEDS))
    args = ap.parse_args(argv)

    if not any([args.capture_deltas, args.print_manifest, args.run_sweep, args.score, args.softness]):
        ap.error("pass at least one of --capture-deltas / --print-manifest / --run-sweep / --score / --softness")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.capture_deltas:
        capture_deltas(seeds)

    if args.print_manifest:
        print_manifest(seeds)

    if args.run_sweep:
        run_sweep(seeds)

    if args.score:
        score()

    if args.softness:
        softness()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
