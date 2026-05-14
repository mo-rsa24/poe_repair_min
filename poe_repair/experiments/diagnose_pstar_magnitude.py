"""Magnitude diagnostic for the trained Method 2a soft prompt p*.

A4 from the fix list. Loads the trained p* and replays a small sample of
cached (cell, step) entries to answer one question:

> Does the UNet's raw output on p* match the magnitude of the training
> target  ``w · (ε_J + ε_∅ − ε_A − ε_B)``  at deployment time?

If yes, the seed-42 black-frame failure is a pure deployment-knob issue
and a λ_max sweep will fix it. If the ratio ``||p*_raw|| / ||target||`` is
systematically much larger than 1, the inference formula is injecting a
correction that's already amplified by that factor on top of λ — fix the
deployment scaling, not the method.

Reports per-step bucket and overall:
  - mean ||target||,  mean ||p*_raw||
  - mean ratio  ||p*_raw|| / ||target||
  - mean cosine(p*_raw, target)
  - mean RMSE  || p*_raw − target ||

Mirrors training-time forward exactly (no scheduler.scale_model_input,
since training fed cached x_t directly).

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m poe_repair.experiments.diagnose_pstar_magnitude
    CUDA_VISIBLE_DEVICES=1 python -m poe_repair.experiments.diagnose_pstar_magnitude \\
        --split heldout --max-samples 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.embeddings.cache_dataset import (
    TrainingCacheDataset,
    pmi_target,
)
from poe_repair.embeddings.infer import load_synthesizer
from poe_repair.methods._sampling import add_time_ids
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "training_cache"
DEFAULT_PSTAR = (
    REPO_ROOT / "checkpoints" / "residual_prompt"
    / "residual_mlp_pstar_v2" / "best.pt"
)


def _bucket(step_index: int, num_steps: int) -> str:
    frac = step_index / max(1, num_steps - 1)
    if frac < 0.2:
        return "early(0-20%)"
    if frac < 0.5:
        return "mid(20-50%)"
    return "late(50-100%)"


@torch.no_grad()
def _measure_one(
    *, synth, unet, sample,
    device, dtype,
    height: int, width: int,
    guidance_scale: float,
) -> dict:
    """One forward pass; returns per-sample numbers."""
    x_t = sample["x_t"].unsqueeze(0).to(device=device, dtype=dtype)
    timesteps = torch.tensor([int(sample["timestep"])], device=device, dtype=torch.long)
    seq_a = sample["seq_a"].unsqueeze(0).to(device=device, dtype=dtype)
    seq_b = sample["seq_b"].unsqueeze(0).to(device=device, dtype=dtype)
    seq_e = sample["seq_uncond"].unsqueeze(0).to(device=device, dtype=dtype)
    pool_a = sample["pool_a"].unsqueeze(0).to(device=device, dtype=dtype)
    pool_b = sample["pool_b"].unsqueeze(0).to(device=device, dtype=dtype)
    pool_e = sample["pool_uncond"].unsqueeze(0).to(device=device, dtype=dtype)

    out = synth(
        seq_a=seq_a, seq_b=seq_b, seq_e=seq_e,
        pool_a=pool_a, pool_b=pool_b, pool_e=pool_e,
    )
    time_ids = add_time_ids(
        height=height, width=width, batch_size=1, device=device, dtype=dtype,
    )
    eps_pstar = unet(
        x_t, timesteps,
        encoder_hidden_states=out.seq.to(dtype),
        added_cond_kwargs={
            "text_embeds": out.pooled.to(dtype), "time_ids": time_ids,
        },
        timestep_cond=None,
    ).sample

    target = pmi_target(
        {k: sample[k].unsqueeze(0).to(device=device, dtype=dtype)
         for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")},
        guidance_scale=guidance_scale,
    )

    pn = float(eps_pstar.float().norm().item())
    tn = float(target.float().norm().item())
    flat_p = eps_pstar.float().flatten()
    flat_t = target.float().flatten()
    cos = float(
        torch.nn.functional.cosine_similarity(
            flat_p.unsqueeze(0), flat_t.unsqueeze(0)
        ).item()
    )
    rmse = float((eps_pstar - target).float().pow(2).mean().sqrt().item())
    return {
        "pred_norm": pn,
        "target_norm": tn,
        "ratio_pred_over_target": pn / max(tn, 1e-8),
        "cosine": cos,
        "rmse": rmse,
        "step_index": int(sample["step_index"]),
        "timestep": int(sample["timestep"]),
        "pair_slug": sample["pair_slug"],
        "seed": int(sample["seed"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--pstar-ckpt", type=Path, default=DEFAULT_PSTAR)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    ctx = make_ctx()
    device = ctx.device
    dtype = ctx.dtype

    if not args.pstar_ckpt.exists():
        raise FileNotFoundError(f"p* checkpoint not found: {args.pstar_ckpt}")
    print(f"[diag-pstar] loading p* from {args.pstar_ckpt}")
    synth = load_synthesizer(args.pstar_ckpt, device=device, dtype=dtype)

    print(f"[diag-pstar] cache_root={args.cache_root} split={args.split}")
    ds = TrainingCacheDataset(args.cache_root, split=args.split, out_dtype=dtype)
    n = min(args.max_samples, len(ds))
    indices = torch.linspace(0, len(ds) - 1, steps=n).long().tolist()
    print(f"[diag-pstar] sampling {n} of {len(ds)} (cell, step) entries")

    per_sample: list[dict] = []
    for idx in indices:
        sample = ds[idx]
        m = _measure_one(
            synth=synth, unet=ctx.models["unet"], sample=sample,
            device=device, dtype=dtype,
            height=args.height, width=args.width,
            guidance_scale=args.guidance_scale,
        )
        per_sample.append(m)

    # ---- aggregate ----
    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        keys = ["pred_norm", "target_norm", "ratio_pred_over_target", "cosine", "rmse"]
        return {
            f"mean_{k}": float(sum(r[k] for r in rows) / len(rows))
            for k in keys
        } | {"n": len(rows)}

    overall = _agg(per_sample)
    num_steps = ds.num_inference_steps
    by_bucket: dict[str, dict] = {}
    for b in ("early(0-20%)", "mid(20-50%)", "late(50-100%)"):
        rows = [r for r in per_sample if _bucket(r["step_index"], num_steps) == b]
        by_bucket[b] = _agg(rows)

    print()
    print("=== Method 2a — p* magnitude diagnostic ===")
    print(f"checkpoint:        {args.pstar_ckpt}")
    print(f"split:             {args.split}")
    print(f"sampled:           {len(per_sample)} / {len(ds)}  "
          f"({num_steps} steps per cell)")
    print(f"guidance_scale:    {args.guidance_scale}")
    print()
    print(f"{'bucket':<14} {'n':>3} {'||p*||':>10} {'||tgt||':>10} "
          f"{'ratio':>8} {'cos':>7} {'rmse':>10}")
    for b, agg in by_bucket.items():
        if not agg:
            print(f"{b:<14} (no samples)")
            continue
        print(
            f"{b:<14} {agg['n']:>3} "
            f"{agg['mean_pred_norm']:>10.3f} "
            f"{agg['mean_target_norm']:>10.3f} "
            f"{agg['mean_ratio_pred_over_target']:>8.3f} "
            f"{agg['mean_cosine']:>7.3f} "
            f"{agg['mean_rmse']:>10.5f}"
        )
    print(
        f"{'overall':<14} {overall['n']:>3} "
        f"{overall['mean_pred_norm']:>10.3f} "
        f"{overall['mean_target_norm']:>10.3f} "
        f"{overall['mean_ratio_pred_over_target']:>8.3f} "
        f"{overall['mean_cosine']:>7.3f} "
        f"{overall['mean_rmse']:>10.5f}"
    )
    print()
    print("Interpretation:")
    print("  ratio ≈ 1.0   → magnitudes match → fix is a λ_max sweep")
    print("  ratio ≫ 1.0   → p*_raw is too big at inference → magnitude bug")
    print("  ratio ≪ 1.0   → p*_raw is too small → student under-trained")

    out_dir = ensure_dir(REPO_ROOT / "outputs" / "diagnose_pstar_magnitude")
    write_json(
        out_dir / "summary.json",
        {
            "checkpoint": str(args.pstar_ckpt),
            "split": args.split,
            "num_inference_steps": num_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height, "width": args.width,
            "overall": overall,
            "by_bucket": by_bucket,
            "samples": per_sample,
        },
    )
    print(f"[diag-pstar] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
