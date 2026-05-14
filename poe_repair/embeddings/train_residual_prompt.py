"""Method 2a — soft-prompt distillation against the guided residual r_t.

Trains a ResidualMLP synthesiser ``p* = synth(e_A, e_B, e_∅)`` so that
the UNet's raw response on ``p*`` matches the guided PMI residual::

    target_t = w · (ε_J_raw + ε_∅ − ε_A_raw − ε_B_raw)
    pred_t   = unet(x_t, t, p*)             # raw forward, no CFG
    loss_t   = step_weight(t) · ‖ pred_t − target_t ‖²

Inputs come from the trajectory cache built by
``scripts/build_training_cache.py`` (one ``step_NNN.pt`` per cell × step,
all four raw eps cached).  The trainer never runs the 4-branch UNet pass
itself — that's already done.  Per-step cost is **one** UNet forward
(the prediction branch).

At inference, ``p*`` is fed as an extra UNet branch and its raw output is
added to guided PoE under a λ_t schedule (see
``poe_repair.composers.residual_prompt`` and
``poe_repair.methods._sampling.run_residual_prompt_inject``).

Held-out validation: the cache's ``heldout/`` split (cat × dog cells)
is reserved automatically; nothing in the train set leaks in.

Usage::

    python -m poe_repair.embeddings.train_residual_prompt \\
        --num-steps 5000 --batch-size 1 \\
        --output-name residual_mlp_pstar \\
        --init-from checkpoints/synthesizer/residual_mlp/best.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.config import RunConfig
from poe_repair.embeddings.cache_dataset import (
    TrainingCacheDataset,
    iter_minibatches_from_cache,
    pmi_target,
    step_weight,
)
from poe_repair.embeddings.synthesizer import build_synthesizer
from poe_repair.methods._sampling import add_time_ids
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_sdxl_models,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "training_cache"


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def _step_loss_residual(
    *,
    synth,
    batch: dict,
    unet,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    train_dtype: torch.dtype,
    guidance_scale: float,
    early_window_end: int,
    late_weight: float,
) -> tuple[torch.Tensor, dict]:
    """One residual-prompt step using cached (x_t, raw eps)."""
    bsz = batch["x_t"].shape[0]
    x_t = batch["x_t"].to(device=device, dtype=dtype)
    timesteps = torch.tensor(batch["timestep"], device=device, dtype=torch.long)
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)

    # Synth (forward in train_dtype; cast outputs to dtype for UNet).
    seq_a = batch["seq_a"].to(device=device, dtype=train_dtype)
    seq_b = batch["seq_b"].to(device=device, dtype=train_dtype)
    seq_e = batch["seq_uncond"].to(device=device, dtype=train_dtype)
    pool_a = batch["pool_a"].to(device=device, dtype=train_dtype)
    pool_b = batch["pool_b"].to(device=device, dtype=train_dtype)
    pool_e = batch["pool_uncond"].to(device=device, dtype=train_dtype)

    out = synth(
        seq_a=seq_a, seq_b=seq_b, seq_e=seq_e,
        pool_a=pool_a, pool_b=pool_b, pool_e=pool_e,
    )
    seq_pred = out.seq.to(dtype)
    pool_pred = out.pooled.to(dtype)

    # Time ids for the *batch* — same for every sample in the minibatch
    # because resolution is fixed.
    time_ids = add_time_ids(
        height=height, width=width, batch_size=bsz, device=device, dtype=dtype,
    )

    # Target — computed from cached raw eps. No UNet call.
    target = pmi_target(
        {k: batch[k].to(device=device, dtype=train_dtype)
         for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")},
        guidance_scale=guidance_scale,
    )

    # Prediction: 1-branch UNet on (x_t, t, p*). With grad through synth.
    eps_pstar = unet(
        x_t, timesteps,
        encoder_hidden_states=seq_pred,
        added_cond_kwargs={"text_embeds": pool_pred, "time_ids": time_ids},
        timestep_cond=None,
    ).sample.to(train_dtype)

    # Per-sample loss, weighted by step_index (early steps weight 1, late late_weight).
    step_idx = torch.tensor(batch["step_index"], dtype=torch.long)
    weights = step_weight(
        step_idx,
        num_inference_steps=int(batch["num_inference_steps"][0])
            if isinstance(batch["num_inference_steps"], list)
            else int(batch["num_inference_steps"]),
        early_window_end=early_window_end,
        late_weight=late_weight,
    ).to(device=device, dtype=train_dtype)

    per_sample_sq = (eps_pstar - target).pow(2).flatten(1).mean(dim=1)
    loss = (weights * per_sample_sq).sum() / weights.sum().clamp(min=1e-8)

    target_norm = float(target.detach().norm().item())
    pred_norm = float(eps_pstar.detach().norm().item())
    rel_err = float(
        (eps_pstar.detach() - target.detach()).norm().item()
        / max(target_norm, 1e-8)
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "target_norm": target_norm,
        "pred_norm": pred_norm,
        "rel_err": rel_err,
        "t": int(timesteps[0].item()),
        "step_idx_sample": int(step_idx[0].item()),
        "weight_sample": float(weights[0].item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Validation on held-out cache
# ---------------------------------------------------------------------------


@torch.no_grad()
def _validate_on_cache(
    *,
    synth, val_dataset, unet,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype, train_dtype: torch.dtype,
    guidance_scale: float,
    early_window_end: int,
    late_weight: float,
    max_samples: int = 64,
) -> dict[str, float]:
    """RMSE between p* prediction and target on held-out cache cells."""
    if len(val_dataset) == 0:
        return {"val_residual_rmse": float("nan"),
                "val_residual_rel_err": float("nan")}
    synth.eval()
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
        x_t = batch["x_t"].to(device=device, dtype=dtype)
        timesteps = torch.tensor([batch["timestep"][0]], device=device, dtype=torch.long)
        seq_a = batch["seq_a"].to(device=device, dtype=train_dtype)
        seq_b = batch["seq_b"].to(device=device, dtype=train_dtype)
        seq_e = batch["seq_uncond"].to(device=device, dtype=train_dtype)
        pool_a = batch["pool_a"].to(device=device, dtype=train_dtype)
        pool_b = batch["pool_b"].to(device=device, dtype=train_dtype)
        pool_e = batch["pool_uncond"].to(device=device, dtype=train_dtype)
        out = synth(
            seq_a=seq_a, seq_b=seq_b, seq_e=seq_e,
            pool_a=pool_a, pool_b=pool_b, pool_e=pool_e,
        )
        time_ids = add_time_ids(
            height=height, width=width, batch_size=1, device=device, dtype=dtype,
        )
        target = pmi_target(
            {k: batch[k].to(device=device, dtype=train_dtype)
             for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")},
            guidance_scale=guidance_scale,
        )
        eps_pstar = unet(
            x_t, timesteps,
            encoder_hidden_states=out.seq.to(dtype),
            added_cond_kwargs={
                "text_embeds": out.pooled.to(dtype), "time_ids": time_ids,
            },
            timestep_cond=None,
        ).sample.to(train_dtype)
        rmses.append(float((eps_pstar - target).pow(2).mean().sqrt().item()))
        tnorm = float(target.norm().item())
        rels.append(float((eps_pstar - target).norm().item() / max(tnorm, 1e-8)))
    synth.train()
    return {
        "val_residual_rmse": float(sum(rmses) / len(rmses)),
        "val_residual_rel_err": float(sum(rels) / len(rels)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train soft-prompt p* against guided residual r_t (Method 2a)",
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
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-from", default=None,
                        help="Existing synthesiser checkpoint to fine-tune from.")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--early-window-end", type=int, default=25,
                        help="Steps with index < this get full weight; rest get late_weight.")
    parser.add_argument("--late-weight", type=float, default=0.25)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 10 steps for sanity-checking the pipeline.")
    args = parser.parse_args()

    cfg = RunConfig()
    arch = args.arch or cfg.synth.arch
    device = infer_device(args.device)
    dtype_str = args.dtype or cfg.dtype
    dtype = infer_dtype(dtype_str, device)
    train_dtype = torch.float32
    guidance_scale = (
        float(args.guidance_scale) if args.guidance_scale is not None
        else float(cfg.guidance)
    )

    output_name = args.output_name or f"{arch}_pstar"
    out_dir = ensure_dir(cfg.paths.output_root / "residual_prompt" / output_name)
    ckpt_dir = ensure_dir(
        REPO_ROOT / "checkpoints" / "residual_prompt" / output_name
    )

    # ---- Load cache datasets ----
    print(f"[pstar] cache_root = {args.cache_root}")
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
            print(f"[pstar] no validation set: {exc}")
    train_pairs = sorted({c.pair_slug for c in train_ds.cells})
    print(
        f"[pstar] train: {train_ds.num_cells} cells, "
        f"{len(train_ds)} (cell, step) samples across pairs {train_pairs}"
    )
    if val_ds is not None:
        val_pairs = sorted({c.pair_slug for c in val_ds.cells})
        print(
            f"[pstar] val:   {val_ds.num_cells} cells, "
            f"{len(val_ds)} (cell, step) samples across pairs {val_pairs}"
        )
    else:
        print("[pstar] val:   none")
    print(f"[pstar] guidance_scale={guidance_scale}  "
          f"early_window_end={args.early_window_end}  late_weight={args.late_weight}")

    # ---- Load SDXL UNet only (text encoders not needed; we have cached embeddings) ----
    print("[pstar] loading SDXL")
    model_id = args.model_id or cfg.model_id
    models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
    unet = models["unet"]
    for p in unet.parameters():
        p.requires_grad_(False)
    unet.eval()
    if not args.no_grad_checkpoint and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # ---- Synth ----
    synth = build_synthesizer(
        arch,
        seq_dim=cfg.synth.seq_dim, pooled_dim=cfg.synth.pooled_dim,
        hidden_dim=cfg.synth.hidden_dim, num_layers=cfg.synth.num_layers,
        dropout=cfg.synth.dropout,
    ).to(device=device, dtype=train_dtype)

    init_path = (
        Path(args.init_from) if args.init_from
        else cfg.paths.synthesizer_checkpoint
    )
    if init_path and init_path.exists():
        sd = torch.load(init_path, map_location=device)
        state = sd.get("synth", sd)
        synth.load_state_dict(state, strict=False)
        print(f"[pstar] initialised from {init_path}")
    else:
        print("[pstar] starting from scratch (no init checkpoint found)")

    optim = torch.optim.AdamW(synth.parameters(), lr=args.lr, weight_decay=0.0)
    num_steps = 10 if args.smoke else args.num_steps
    log_interval = max(1, min(50, num_steps // 20))
    val_interval = max(1, min(500, num_steps // 10))

    train_iter = iter_minibatches_from_cache(
        train_ds, batch_size=args.batch_size, seed=args.seed, shuffle=True,
    )

    history: list[dict] = []
    best_val_rmse = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    synth.train()
    t0 = time.time()
    for step in range(num_steps):
        batch = next(train_iter)
        loss, metrics = _step_loss_residual(
            synth=synth, batch=batch, unet=unet,
            height=args.resolution, width=args.resolution,
            device=device, dtype=dtype, train_dtype=train_dtype,
            guidance_scale=guidance_scale,
            early_window_end=args.early_window_end,
            late_weight=args.late_weight,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(synth.parameters(), 1.0)
        optim.step()

        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"[pstar] step {step+1}/{num_steps} "
                f"loss={metrics['loss']:.4f} rel_err={metrics['rel_err']:.4f} "
                f"||tgt||={metrics['target_norm']:.2f} "
                f"||pred||={metrics['pred_norm']:.2f} "
                f"t={metrics['t']:>3} step_idx={metrics['step_idx_sample']:>2} "
                f"w={metrics['weight_sample']:.2f} "
                f"elapsed={elapsed/60:.1f} min"
            )
            history.append({"step": step + 1, **metrics})

        if val_ds is not None and ((step + 1) % val_interval == 0 or step == num_steps - 1):
            vmetrics = _validate_on_cache(
                synth=synth, val_dataset=val_ds, unet=unet,
                height=args.resolution, width=args.resolution,
                device=device, dtype=dtype, train_dtype=train_dtype,
                guidance_scale=guidance_scale,
                early_window_end=args.early_window_end,
                late_weight=args.late_weight,
            )
            print(
                f"[pstar] step {step+1}  "
                f"val_rmse={vmetrics['val_residual_rmse']:.4f} "
                f"val_rel_err={vmetrics['val_residual_rel_err']:.4f}"
            )
            if history:
                history[-1].update(vmetrics)
            if vmetrics["val_residual_rmse"] < best_val_rmse:
                best_val_rmse = vmetrics["val_residual_rmse"]
                torch.save(
                    {"synth": synth.state_dict(), "arch": arch,
                     "step": step + 1, "val_metrics": vmetrics,
                     "guidance_scale": guidance_scale,
                     "target_kind": "guided_residual_r_t",
                     "trained_on_cache": True,
                     "early_window_end": args.early_window_end,
                     "late_weight": args.late_weight},
                    best_path,
                )
                print(f"[pstar] saved new best -> {best_path}")
        torch.save(
            {"synth": synth.state_dict(), "arch": arch, "step": step + 1,
             "guidance_scale": guidance_scale,
             "target_kind": "guided_residual_r_t",
             "trained_on_cache": True,
             "early_window_end": args.early_window_end,
             "late_weight": args.late_weight},
            last_path,
        )

    write_json(out_dir / "history.json", {
        "arch": arch, "num_steps": num_steps,
        "lr": args.lr, "resolution": args.resolution,
        "batch_size": args.batch_size,
        "guidance_scale": guidance_scale,
        "early_window_end": args.early_window_end,
        "late_weight": args.late_weight,
        "train_pairs": train_pairs,
        "best_val_residual_rmse": best_val_rmse,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "history": history,
    })
    print(f"[pstar] done. best -> {best_path}  last -> {last_path}")


if __name__ == "__main__":
    main()
