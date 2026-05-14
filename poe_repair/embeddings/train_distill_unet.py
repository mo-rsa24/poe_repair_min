"""E7 — UNet-distillation training for the synthesizer (the cheaper variant).

Replaces the pure-embedding loss
    L = cosine(ê_J, e_J) + α · MSE(ê_J, e_J)
with a UNet-output distillation loss sampled over training trajectories
    L = mean_{x_t, t} ‖ ε_θ(x_t, t, ê_J) − ε_θ(x_t, t, e_J) ‖²
plus a small embedding-MSE retention term so the synthesiser stays
anchored near e_J in embedding space.

This targets the exact gap E2 measures (UNet-output divergence at given
embedding cosine). It is gradient through the UNet (frozen weights) but
NOT through the VAE or any VLM — much cheaper than ``train_vlm_reward``.

Usage (overnight on one A100):
    python -m poe_repair.embeddings.train_distill_unet \\
        --num-steps 5000 --batch-size 1 --resolution 1024 \\
        --output-name residual_mlp_distilled \\
        --init-from checkpoints/synthesizer/residual_mlp/best.pt

The output checkpoint is a drop-in replacement for the residual_mlp arch:
    checkpoints/synthesizer_distilled/residual_mlp_distilled/best.pt

Pointing ``cfg.paths.synthesizer_checkpoint`` at the new file (or
``POE_REPAIR_SYNTH_CKPT=...``) makes every Mono(ê_J) / sched-M2(ê_J) call
in the experiments use the distilled synthesiser.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.config import RunConfig
from poe_repair.embeddings.dataset import _build_pair_pool
from poe_repair.embeddings.synthesizer import build_synthesizer
from poe_repair.embeddings.train import (
    _build_holdout_set,
    _cache_key,
    _iter_minibatches_from_tensors,
    _pin,
    _precompute_pool,
    _release_text_encoders,
)
from poe_repair.runtime import (
    encode_prompt_sdxl,
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)


def _add_time_ids(
    *, height: int, width: int, batch_size: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=dtype, device=device,
    )
    return base.repeat(batch_size, 1)


def _sample_noisy_latent(
    *, scheduler, batch_size: int, height: int, width: int,
    device: torch.device, dtype: torch.dtype, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample a fresh (x_t, t) pair for distillation.

    We don't have ground-truth clean latents (this is a text-only synthesiser
    — no images involved). Instead we sample a random "clean" latent x_0 and
    a random t, then run forward diffusion. ε_target and ε_pred share the
    same x_t and t — the only thing that differs is the conditional embedding,
    which is exactly what we're trying to learn equivariance over.
    """
    H, W = height // 8, width // 8
    x0 = torch.randn(
        batch_size, 4, H, W, device=device, dtype=dtype, generator=generator,
    )
    noise = torch.randn(
        batch_size, 4, H, W, device=device, dtype=dtype, generator=generator,
    )
    # Sample a uniform integer timestep index from the trained schedule.
    timesteps = scheduler.timesteps
    idx = int(torch.randint(
        0, len(timesteps), (1,), generator=generator, device=device,
    ).item())
    t = timesteps[idx]
    alpha_bar_t = scheduler.alphas_cumprod[int(t.item())].to(device=device, dtype=dtype)
    sqrt_a = alpha_bar_t.sqrt()
    sqrt_1ma = (1.0 - alpha_bar_t).sqrt()
    x_t = sqrt_a * x0 + sqrt_1ma * noise
    return x_t, t, idx


def _step_loss_distill(
    *,
    synth,
    batch: dict[str, torch.Tensor],
    unet,
    scheduler,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    train_dtype: torch.dtype,
    generator: torch.Generator,
    retention_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single distillation step: ‖ε(x_t,t,ê_J) − ε(x_t,t,e_J)‖² + α · embed-MSE.

    UNet weights are frozen. Gradients flow through UNet activations
    (with checkpointing) → ê_J → synth weights.
    """
    # 1. Predict ê_J from cached marginals.
    out = synth(
        seq_a=batch["seq_a"], seq_b=batch["seq_b"], seq_e=batch["seq_e"],
        pool_a=batch["pool_a"], pool_b=batch["pool_b"], pool_e=batch["pool_e"],
    )
    seq_pred, pool_pred = out.seq, out.pooled
    seq_gt, pool_gt = batch["seq_j"], batch["pool_j"]

    # 2. Sample (x_t, t).
    bsz = seq_pred.shape[0]
    x_t, t, _ = _sample_noisy_latent(
        scheduler=scheduler, batch_size=bsz,
        height=height, width=width, device=device, dtype=dtype,
        generator=generator,
    )

    # 3. ε target with no grad (UNet weights frozen, no autograd graph).
    time_ids = _add_time_ids(
        height=height, width=width, batch_size=bsz, device=device, dtype=dtype,
    )
    with torch.no_grad():
        eps_target = unet(
            x_t.to(dtype), t,
            encoder_hidden_states=seq_gt.to(dtype),
            added_cond_kwargs={
                "text_embeds": pool_gt.to(dtype), "time_ids": time_ids,
            },
            timestep_cond=None,
        ).sample.float()

    # 4. ε prediction WITH grad through synth (UNet activations cached).
    eps_pred = unet(
        x_t.to(dtype), t,
        encoder_hidden_states=seq_pred.to(dtype),
        added_cond_kwargs={
            "text_embeds": pool_pred.to(dtype), "time_ids": time_ids,
        },
        timestep_cond=None,
    ).sample.float()

    unet_mse = F.mse_loss(eps_pred, eps_target)

    # 5. Retention regulariser (embedding MSE).
    embed_mse = (
        F.mse_loss(seq_pred, seq_gt.to(train_dtype))
        + F.mse_loss(pool_pred, pool_gt.to(train_dtype))
    ) * 0.5

    loss = unet_mse + float(retention_weight) * embed_mse
    metrics = {
        "loss": float(loss.detach().item()),
        "unet_mse": float(unet_mse.detach().item()),
        "embed_mse": float(embed_mse.detach().item()),
        "t": int(t.item()),
    }
    return loss, metrics


def _validate_unet_rmse(
    *, synth, holdout, unet, scheduler, height: int, width: int,
    device: torch.device, dtype: torch.dtype, train_dtype: torch.dtype,
    generator: torch.Generator, n_samples_per_pair: int = 4,
) -> dict[str, float]:
    """Held-out UNet-output RMSE between ε(x_t,t,ê_J) and ε(x_t,t,e_J).

    Faster + smaller than the full E2 audit; meant for periodic logging.
    """
    synth.eval()
    rmses: list[float] = []
    seq_cosines: list[float] = []
    with torch.no_grad():
        for entry in holdout:
            ea_seq = entry["seq_a"].to(device=device, dtype=train_dtype)
            eb_seq = entry["seq_b"].to(device=device, dtype=train_dtype)
            ee_seq = entry["seq_e"].to(device=device, dtype=train_dtype)
            ea_p = entry["pool_a"].to(device=device, dtype=train_dtype)
            eb_p = entry["pool_b"].to(device=device, dtype=train_dtype)
            ee_p = entry["pool_e"].to(device=device, dtype=train_dtype)
            ej_seq = entry["seq_j"].to(device=device, dtype=train_dtype)
            ej_p = entry["pool_j"].to(device=device, dtype=train_dtype)
            out = synth(
                seq_a=ea_seq, seq_b=eb_seq, seq_e=ee_seq,
                pool_a=ea_p, pool_b=eb_p, pool_e=ee_p,
            )
            seq_cos = float(F.cosine_similarity(
                out.seq.flatten(start_dim=1), ej_seq.flatten(start_dim=1), dim=-1,
            ).mean().item())
            seq_cosines.append(seq_cos)
            for _ in range(n_samples_per_pair):
                x_t, t, _ = _sample_noisy_latent(
                    scheduler=scheduler, batch_size=1,
                    height=height, width=width, device=device, dtype=dtype,
                    generator=generator,
                )
                time_ids = _add_time_ids(
                    height=height, width=width, batch_size=1, device=device, dtype=dtype,
                )
                eps_lit = unet(
                    x_t.to(dtype), t,
                    encoder_hidden_states=ej_seq.to(dtype),
                    added_cond_kwargs={
                        "text_embeds": ej_p.to(dtype), "time_ids": time_ids,
                    }, timestep_cond=None,
                ).sample
                eps_hat = unet(
                    x_t.to(dtype), t,
                    encoder_hidden_states=out.seq.to(dtype),
                    added_cond_kwargs={
                        "text_embeds": out.pooled.to(dtype), "time_ids": time_ids,
                    }, timestep_cond=None,
                ).sample
                rmses.append(float((eps_lit - eps_hat).pow(2).mean().sqrt().item()))
    synth.train()
    return {
        "val_unet_rmse": float(sum(rmses) / max(1, len(rmses))),
        "val_seq_cosine": float(sum(seq_cosines) / max(1, len(seq_cosines))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train synthesiser via UNet distillation (E7-cheap)")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1, help="Pairs per step (memory-bound).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--retention-weight", type=float, default=1.0,
                        help="Embed-MSE regulariser weight; keeps ê_J close to e_J.")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="UNet input resolution (lower = faster, less faithful).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--joint-template", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train-pairs", type=int, default=None)
    parser.add_argument("--n-holdout-oversample", type=int, default=None)
    parser.add_argument("--n-supercategory-oversample", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--init-from", default=None,
        help="Path to existing synthesiser checkpoint to fine-tune from.",
    )
    parser.add_argument(
        "--no-grad-checkpoint", action="store_true",
        help="Disable UNet gradient checkpointing (faster, more memory).",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny smoke run (32 pairs, 10 steps).")
    args = parser.parse_args()

    cfg = RunConfig()
    arch = args.arch or cfg.synth.arch
    device = infer_device(args.device)
    dtype_str = args.dtype or cfg.dtype
    dtype = infer_dtype(dtype_str, device)
    train_dtype = torch.float32

    output_name = args.output_name or f"{arch}_distilled"
    out_dir = ensure_dir(cfg.paths.output_root / "synthesizer_distilled" / output_name)
    ckpt_dir = ensure_dir(
        Path(__file__).resolve().parent.parent.parent / "checkpoints"
        / "synthesizer_distilled" / output_name
    )
    cache_dir = ensure_dir(
        Path(args.cache_dir) if args.cache_dir
        else cfg.paths.output_root / "synthesizer" / "cache"
    )

    n_train = args.n_train_pairs or cfg.synth.n_train_pairs
    n_holdout = args.n_holdout_oversample or cfg.synth.n_holdout_oversample
    n_super = args.n_supercategory_oversample or cfg.synth.n_supercategory_oversample
    seed = args.seed if args.seed is not None else cfg.synth.seed
    joint_template = args.joint_template or cfg.synth.joint_template
    if args.smoke:
        n_train, n_holdout, n_super = 32, 16, 16

    # 1. Pair pool + cached embeddings (reused from the regular train.py pipeline).
    pairs = _build_pair_pool(
        n_random=n_train, n_holdout_oversample=n_holdout,
        n_supercategory_oversample=n_super, seed=seed,
    )
    print(f"[distill] pair pool size = {len(pairs)}")
    model_id = args.model_id or cfg.model_id
    cache_key = _cache_key(
        pairs=pairs, joint_template=joint_template,
        model_id=model_id, dtype_str=dtype_str,
    )
    cache_path = cache_dir / f"pairs_{cache_key}.pt"

    # Always need full SDXL models here (UNet is the supervisor).
    print("[distill] loading SDXL")
    models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
    if not cache_path.exists():
        _precompute_pool(
            pairs=pairs, models=models, device=device, dtype=dtype,
            joint_template=joint_template, cache_path=cache_path,
        )
    blob = _precompute_pool(
        pairs=pairs, models=None, device=device, dtype=dtype,
        joint_template=joint_template, cache_path=cache_path,
    )
    holdout = _build_holdout_set(
        models=models, device=device, dtype=dtype, joint_template=joint_template,
    )
    _release_text_encoders(models)
    blob = _pin(blob)

    unet = models["unet"]
    for p in unet.parameters():
        p.requires_grad_(False)
    unet.eval()
    if not args.no_grad_checkpoint and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    scheduler = load_ddim_scheduler(model_id)
    scheduler.set_timesteps(cfg.num_inference_steps)

    # 2. Build synthesiser; optionally init from existing checkpoint.
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
        print(f"[distill] initialised from {init_path}")
    else:
        print("[distill] starting from scratch (no init checkpoint found)")

    optim = torch.optim.AdamW(synth.parameters(), lr=args.lr, weight_decay=0.0)
    num_steps = 10 if args.smoke else args.num_steps
    log_interval = max(1, min(50, num_steps // 20))
    val_interval = max(1, min(500, num_steps // 10))

    train_iter = _iter_minibatches_from_tensors(
        blob, batch_size=args.batch_size, device=device,
        train_dtype=train_dtype, seed=seed,
    )
    generator = torch.Generator(device=device).manual_seed(seed)

    history: list[dict] = []
    best_val_rmse = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    synth.train()
    t0 = time.time()
    for step in range(num_steps):
        batch = next(train_iter)
        loss, metrics = _step_loss_distill(
            synth=synth, batch=batch, unet=unet, scheduler=scheduler,
            height=args.resolution, width=args.resolution,
            device=device, dtype=dtype, train_dtype=train_dtype,
            generator=generator,
            retention_weight=args.retention_weight,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(synth.parameters(), 1.0)
        optim.step()
        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"[distill] step {step+1}/{num_steps} "
                f"loss={metrics['loss']:.4f} unet_mse={metrics['unet_mse']:.4f} "
                f"embed_mse={metrics['embed_mse']:.4f} t={metrics['t']:>3} "
                f"elapsed={elapsed/60:.1f} min"
            )
            history.append({"step": step + 1, **metrics})
        if (step + 1) % val_interval == 0 or step == num_steps - 1:
            vmetrics = _validate_unet_rmse(
                synth=synth, holdout=holdout, unet=unet, scheduler=scheduler,
                height=args.resolution, width=args.resolution,
                device=device, dtype=dtype, train_dtype=train_dtype,
                generator=generator,
            )
            print(
                f"[distill] step {step+1}  val_unet_rmse={vmetrics['val_unet_rmse']:.4f} "
                f"val_seq_cosine={vmetrics['val_seq_cosine']:.4f}"
            )
            history[-1].update(vmetrics)
            if vmetrics["val_unet_rmse"] < best_val_rmse:
                best_val_rmse = vmetrics["val_unet_rmse"]
                torch.save(
                    {"synth": synth.state_dict(), "arch": arch,
                     "step": step + 1, "val_metrics": vmetrics},
                    best_path,
                )
                print(f"[distill] saved new best -> {best_path}")
        torch.save(
            {"synth": synth.state_dict(), "arch": arch, "step": step + 1},
            last_path,
        )

    write_json(out_dir / "history.json", {
        "arch": arch, "num_steps": num_steps,
        "lr": args.lr, "retention_weight": args.retention_weight,
        "resolution": args.resolution, "batch_size": args.batch_size,
        "best_val_unet_rmse": best_val_rmse,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "history": history,
    })
    print(f"[distill] done. best -> {best_path}  last -> {last_path}")


if __name__ == "__main__":
    main()
