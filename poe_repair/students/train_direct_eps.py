"""Method 2b — direct eps-space distillation trainer.

Trains :class:`DirectEpsStudent` (a small CNN, no SDXL UNet at training
time) to predict the PMI residual ``Δ_t = w · (ε_J + ε_∅ − ε_A − ε_B)``
from ``(x_t, t, pool_a, pool_b, pool_uncond)`` along the actual PoE
trajectory of every cached cell.

Per training step does **one** student forward — no UNet calls. Targets
come from cached raw eps via ``pmi_target``.

Loss is weighted by ``step_weight(step_index)``: full weight on the
basin-selection window (steps 0–24 by default), discounted weight after.

Validation runs against the held-out cache split (cohort H —
cat × dog). Saves ``best.pt`` on best held-out RMSE and ``last.pt`` every
step. Checkpoints live under
``outputs/group_a_failure/checkpoints/direct_eps/<output_name>/``.

Usage::

    # Quick sanity run (10 steps):
    CUDA_VISIBLE_DEVICES=1 python -m poe_repair.students.train_direct_eps --smoke

    # Full overnight run:
    CUDA_VISIBLE_DEVICES=1 python -m poe_repair.students.train_direct_eps \\
        --num-steps 20000 --batch-size 4 --lr 5e-5 \\
        --output-name direct_eps_v1
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.embeddings.cache_dataset import (
    TrainingCacheDataset,
    iter_minibatches_from_cache,
    pmi_target,
    step_weight,
)
from poe_repair.students.direct_eps import (
    build_direct_eps_student,
    build_hourglass_student,
)
from poe_repair.runtime import ensure_dir, infer_device, infer_dtype, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "training_cache"


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def _step_loss(
    *,
    student,
    batch: dict,
    device: torch.device,
    train_dtype: torch.dtype,
    guidance_scale: float,
    early_window_end: int,
    late_weight: float,
    cosine_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    x_t = batch["x_t"].to(device=device, dtype=train_dtype)
    t = torch.tensor(batch["timestep"], device=device, dtype=torch.long)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    pool_a = batch["pool_a"].to(device=device, dtype=train_dtype)
    pool_b = batch["pool_b"].to(device=device, dtype=train_dtype)
    pool_uncond = batch["pool_uncond"].to(device=device, dtype=train_dtype)

    target = pmi_target(
        {k: batch[k].to(device=device, dtype=train_dtype)
         for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")},
        guidance_scale=guidance_scale,
    )

    pred = student(
        x_t=x_t, t=t,
        pool_a=pool_a, pool_b=pool_b, pool_uncond=pool_uncond,
    )

    step_idx = torch.tensor(batch["step_index"], dtype=torch.long)
    n_steps = (
        int(batch["num_inference_steps"][0])
        if isinstance(batch["num_inference_steps"], list)
        else int(batch["num_inference_steps"])
    )
    weights = step_weight(
        step_idx, num_inference_steps=n_steps,
        early_window_end=early_window_end, late_weight=late_weight,
    ).to(device=device, dtype=train_dtype)

    per_sample_sq = (pred - target).pow(2).flatten(1).mean(dim=1)
    mse_loss = (weights * per_sample_sq).sum() / weights.sum().clamp(min=1e-8)

    # Direction-alignment term: penalise (1 − cos(pred, tgt)) per sample.
    # MSE alone has a degenerate "predict near zero" minimum; the cosine
    # term forces the network to commit to a direction. Computed flat
    # over the per-sample tensor; weighted by the same step weights.
    if cosine_weight > 0.0:
        flat_pred = pred.flatten(1)
        flat_tgt = target.flatten(1)
        cos = F.cosine_similarity(flat_pred, flat_tgt, dim=1, eps=1e-8)
        per_sample_cos_loss = 1.0 - cos
        cos_loss = (weights * per_sample_cos_loss).sum() / weights.sum().clamp(min=1e-8)
        loss = mse_loss + float(cosine_weight) * cos_loss
        cos_mean = float(cos.detach().mean().item())
    else:
        loss = mse_loss
        cos_loss = torch.zeros((), device=device, dtype=train_dtype)
        cos_mean = float("nan")

    target_norm = float(target.detach().norm().item())
    pred_norm = float(pred.detach().norm().item())
    rel_err = float(
        (pred.detach() - target.detach()).norm().item()
        / max(target_norm, 1e-8)
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "mse_loss": float(mse_loss.detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "cos_mean": cos_mean,
        "target_norm": target_norm,
        "pred_norm": pred_norm,
        "rel_err": rel_err,
        "t": int(t[0].item()),
        "step_idx_sample": int(step_idx[0].item()),
        "weight_sample": float(weights[0].item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _validate(
    *,
    student,
    val_dataset,
    device: torch.device,
    train_dtype: torch.dtype,
    guidance_scale: float,
    max_samples: int = 256,
) -> dict[str, float]:
    if len(val_dataset) == 0:
        return {"val_rmse": float("nan"), "val_rel_err": float("nan")}
    student.eval()
    rmses: list[float] = []
    rels: list[float] = []
    n = min(max_samples, len(val_dataset))
    indices = torch.linspace(0, len(val_dataset) - 1, steps=n).long().tolist()
    for idx in indices:
        sample = val_dataset[idx]
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
            for k, v in sample.items()
        }
        x_t = batch["x_t"].to(device=device, dtype=train_dtype)
        t = torch.tensor([batch["timestep"][0]], device=device, dtype=torch.long)
        pool_a = batch["pool_a"].to(device=device, dtype=train_dtype)
        pool_b = batch["pool_b"].to(device=device, dtype=train_dtype)
        pool_uncond = batch["pool_uncond"].to(device=device, dtype=train_dtype)
        target = pmi_target(
            {k: batch[k].to(device=device, dtype=train_dtype)
             for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")},
            guidance_scale=guidance_scale,
        )
        pred = student(
            x_t=x_t, t=t,
            pool_a=pool_a, pool_b=pool_b, pool_uncond=pool_uncond,
        )
        rmses.append(float((pred - target).pow(2).mean().sqrt().item()))
        tnorm = float(target.norm().item())
        rels.append(float((pred - target).norm().item() / max(tnorm, 1e-8)))
    student.train()
    return {
        "val_rmse": float(sum(rmses) / len(rmses)),
        "val_rel_err": float(sum(rels) / len(rels)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Method 2b student (direct eps-space distillation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="heldout")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Restrict TRAIN to these pair slugs.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Restrict TRAIN+VAL to these seeds (for single-cell overfit diag).")
    parser.add_argument("--val-pairs", nargs="*", default=None,
                        help="Restrict VAL to these pair slugs (default: all in val split).")
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="direct_eps_v1")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    # Architecture knobs
    parser.add_argument(
        "--arch", default="flat", choices=("flat", "hourglass"),
        help="Student architecture. 'flat' = original flat ResBlock stack "
             "(small receptive field). 'hourglass' = UNet-style encoder/"
             "decoder with skips (global receptive field, ~5–6× more params).",
    )
    parser.add_argument("--body-channels", type=int, default=128,
                        help="(flat) hidden channel width.")
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=8,
                        help="(flat) number of ResBlocks.")
    parser.add_argument("--base-channels", type=int, default=128,
                        help="(hourglass) channel width at the top level.")
    parser.add_argument("--channel-mults", nargs="*", type=int, default=[1, 2, 4],
                        help="(hourglass) per-level channel multipliers.")
    parser.add_argument("--num-blocks-per-level", type=int, default=2,
                        help="(hourglass) ResBlocks per resolution level.")
    # Loss weighting
    parser.add_argument("--early-window-end", type=int, default=25)
    parser.add_argument("--late-weight", type=float, default=0.25)
    parser.add_argument(
        "--cosine-weight", type=float, default=0.0,
        help="Coefficient on (1 - cos(pred, target)) added to the MSE "
             "loss. 0 disables (MSE-only). Try 1.0–10.0 to escape the "
             "predict-near-zero local minimum where val_rel_err plateaus "
             "at ~1.0.",
    )
    # Misc
    parser.add_argument("--init-from", default=None,
                        help="Existing student checkpoint to fine-tune from.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 10 steps to sanity-check the pipeline.")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    args = parser.parse_args()

    device = infer_device(args.device)
    dtype_str = args.dtype or "fp32"
    train_dtype = infer_dtype(dtype_str, device)
    if train_dtype != torch.float32:
        # Mixed-precision adds engineering noise; default to fp32 for the
        # student. Cache reads happen in fp32 already.
        print(
            f"[2b] note: --dtype {dtype_str} requested; training in float32 "
            "for stability (override the dataset out_dtype if you want fp16)."
        )
        train_dtype = torch.float32

    out_dir = ensure_dir(
        REPO_ROOT / "outputs" / "group_a_failure" / "students" / args.output_name
    )
    ckpt_dir = ensure_dir(
        REPO_ROOT / "outputs" / "group_a_failure" / "checkpoints"
        / "direct_eps" / args.output_name
    )

    # ---- Datasets ----
    print(f"[2b] cache_root = {args.cache_root}")
    train_ds = TrainingCacheDataset(
        args.cache_root, split=args.train_split,
        pair_filter=args.pairs, seed_filter=args.seeds,
        out_dtype=train_dtype,
    )
    val_ds = None
    val_split_dir = args.cache_root / args.val_split
    if val_split_dir.exists() and any(val_split_dir.iterdir()):
        try:
            val_ds = TrainingCacheDataset(
                args.cache_root, split=args.val_split,
                pair_filter=args.val_pairs, seed_filter=args.seeds,
                out_dtype=train_dtype,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[2b] no validation set: {exc}")
    train_pairs = sorted({c.pair_slug for c in train_ds.cells})
    print(
        f"[2b] train: {train_ds.num_cells} cells, "
        f"{len(train_ds)} (cell, step) samples across pairs {train_pairs}"
    )
    if val_ds is not None:
        val_pairs = sorted({c.pair_slug for c in val_ds.cells})
        print(
            f"[2b] val:   {val_ds.num_cells} cells, "
            f"{len(val_ds)} (cell, step) samples across pairs {val_pairs}"
        )
    else:
        print("[2b] val:   none")
    print(
        f"[2b] guidance_scale={args.guidance_scale}  "
        f"early_window_end={args.early_window_end}  late_weight={args.late_weight}"
    )

    # ---- Student ----
    if args.arch == "hourglass":
        student = build_hourglass_student(
            base_channels=args.base_channels,
            channel_mults=tuple(args.channel_mults),
            num_blocks_per_level=args.num_blocks_per_level,
            cond_dim=args.cond_dim,
        ).to(device=device, dtype=train_dtype)
        arch_summary = (
            f"arch=hourglass base_channels={args.base_channels} "
            f"channel_mults={tuple(args.channel_mults)} "
            f"num_blocks_per_level={args.num_blocks_per_level} "
            f"cond_dim={args.cond_dim}"
        )
        arch_payload = {
            "base_channels": int(args.base_channels),
            "channel_mults": list(args.channel_mults),
            "num_blocks_per_level": int(args.num_blocks_per_level),
            "cond_dim": int(args.cond_dim),
        }
    else:
        student = build_direct_eps_student(
            body_channels=args.body_channels,
            cond_dim=args.cond_dim,
            num_blocks=args.num_blocks,
        ).to(device=device, dtype=train_dtype)
        arch_summary = (
            f"arch=flat body_channels={args.body_channels} "
            f"cond_dim={args.cond_dim} num_blocks={args.num_blocks}"
        )
        arch_payload = {
            "body_channels": int(args.body_channels),
            "cond_dim": int(args.cond_dim),
            "num_blocks": int(args.num_blocks),
        }
    print(f"[2b] student: {arch_summary} params={student.num_params() / 1e6:.2f}M")

    if args.init_from:
        init_path = Path(args.init_from)
        if init_path.exists():
            sd = torch.load(init_path, map_location=device)
            state = sd.get("student", sd)
            missing, unexpected = student.load_state_dict(state, strict=False)
            print(f"[2b] initialised from {init_path}")
            if missing:
                print(f"[2b] missing keys: {missing}")
            if unexpected:
                print(f"[2b] unexpected keys: {unexpected}")
        else:
            print(f"[2b] WARNING --init-from {init_path} not found; starting fresh")

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    num_steps = 10 if args.smoke else args.num_steps
    log_interval = (
        args.log_interval if args.log_interval is not None
        else max(1, min(50, num_steps // 20))
    )
    val_interval = (
        args.val_interval if args.val_interval is not None
        else max(1, min(500, num_steps // 10))
    )

    train_iter = iter_minibatches_from_cache(
        train_ds, batch_size=args.batch_size, seed=args.seed, shuffle=True,
    )

    history: list[dict] = []
    best_val_rmse = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    snapshots_dir = ckpt_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    student.train()
    t0 = time.time()
    for step in range(num_steps):
        batch = next(train_iter)
        loss, metrics = _step_loss(
            student=student, batch=batch,
            device=device, train_dtype=train_dtype,
            guidance_scale=args.guidance_scale,
            early_window_end=args.early_window_end,
            late_weight=args.late_weight,
            cosine_weight=args.cosine_weight,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optim.step()

        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - t0
            cos_str = (
                f" cos={metrics['cos_mean']:.3f}"
                if args.cosine_weight > 0.0 else ""
            )
            print(
                f"[2b] step {step+1}/{num_steps} "
                f"loss={metrics['loss']:.4f} rel_err={metrics['rel_err']:.4f}"
                f"{cos_str} "
                f"||tgt||={metrics['target_norm']:.2f} "
                f"||pred||={metrics['pred_norm']:.2f} "
                f"t={metrics['t']:>3} step_idx={metrics['step_idx_sample']:>2} "
                f"w={metrics['weight_sample']:.2f} "
                f"elapsed={elapsed/60:.1f} min"
            )
            history.append({"step": step + 1, **metrics})

        if val_ds is not None and (
            (step + 1) % val_interval == 0 or step == num_steps - 1
        ):
            vmetrics = _validate(
                student=student, val_dataset=val_ds,
                device=device, train_dtype=train_dtype,
                guidance_scale=args.guidance_scale,
            )
            print(
                f"[2b] step {step+1}  "
                f"val_rmse={vmetrics['val_rmse']:.4f} "
                f"val_rel_err={vmetrics['val_rel_err']:.4f}"
            )
            if history:
                history[-1].update(vmetrics)
            payload = {
                "student": student.state_dict(),
                "step": step + 1,
                "val_metrics": vmetrics,
                "guidance_scale": args.guidance_scale,
                "target_kind": "guided_residual_r_t",
                "trained_on_cache": True,
                "early_window_end": args.early_window_end,
                "late_weight": args.late_weight,
                "arch_kind": args.arch,
                "arch": arch_payload,
            }
            # Step-tagged snapshot at every val checkpoint — the watcher
            # script reads these to render images during training.
            snap_path = snapshots_dir / f"step_{step+1:07d}.pt"
            torch.save(payload, snap_path)
            if vmetrics["val_rmse"] < best_val_rmse:
                best_val_rmse = vmetrics["val_rmse"]
                torch.save(payload, best_path)
                print(f"[2b] saved new best -> {best_path}")
        torch.save(
            {
                "student": student.state_dict(),
                "step": step + 1,
                "guidance_scale": args.guidance_scale,
                "target_kind": "guided_residual_r_t",
                "trained_on_cache": True,
                "early_window_end": args.early_window_end,
                "late_weight": args.late_weight,
                "arch_kind": args.arch,
                "arch": arch_payload,
            },
            last_path,
        )

    write_json(out_dir / "history.json", {
        "num_steps": num_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "guidance_scale": args.guidance_scale,
        "early_window_end": args.early_window_end,
        "late_weight": args.late_weight,
        "train_pairs": train_pairs,
        "arch_kind": args.arch,
        "arch": {**arch_payload, "params": student.num_params()},
        "best_val_rmse": best_val_rmse,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "history": history,
    })
    print(f"[2b] done. best -> {best_path}  last -> {last_path}")


if __name__ == "__main__":
    main()
