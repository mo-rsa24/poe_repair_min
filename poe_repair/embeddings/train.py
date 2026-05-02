"""Train the embedding synthesizer (text-only).

Cosine + MSE on both sequence and pooled outputs:

    L_total = w_cos * (1 - cos(seq_pred, seq_gt)) + w_mse * MSE(seq_pred, seq_gt)
            + w_cos * (1 - cos(pool_pred, pool_gt)) + w_mse * MSE(pool_pred, pool_gt)

Held-out validation uses the curated COLLISION_PAIRS list.

Fast path: the SDXL text encoders are run once over the full pair pool
upfront, the encoded tensors are cached on CPU (pinned) or disk, and the
encoders are then released. Training becomes a pure MLP loop and the
remaining bottleneck is host->device transfer per minibatch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.embeddings.dataset import _build_pair_pool
from poe_repair.embeddings.holdout_pairs import COLLISION_PAIRS
from poe_repair.embeddings.synthesizer import build_synthesizer
from poe_repair.runtime import (
    encode_prompt_sdxl,
    ensure_dir,
    infer_device,
    infer_dtype,
    load_sdxl_models,
    write_json,
)


def _step_loss(
    synth,
    batch: dict[str, torch.Tensor],
    *,
    seq_loss_weight: float,
    pooled_loss_weight: float,
    cosine_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = synth(
        seq_a=batch["seq_a"], seq_b=batch["seq_b"], seq_e=batch["seq_e"],
        pool_a=batch["pool_a"], pool_b=batch["pool_b"], pool_e=batch["pool_e"],
    )
    seq_pred, pool_pred = out.seq, out.pooled
    seq_gt, pool_gt = batch["seq_j"], batch["pool_j"]

    seq_cos = F.cosine_similarity(seq_pred.flatten(start_dim=1), seq_gt.flatten(start_dim=1), dim=-1).mean()
    pool_cos = F.cosine_similarity(pool_pred, pool_gt, dim=-1).mean()
    seq_mse = F.mse_loss(seq_pred, seq_gt)
    pool_mse = F.mse_loss(pool_pred, pool_gt)

    loss = (
        seq_loss_weight * (cosine_weight * (1.0 - seq_cos) + mse_weight * seq_mse)
        + pooled_loss_weight * (cosine_weight * (1.0 - pool_cos) + mse_weight * pool_mse)
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "seq_cos": float(seq_cos.detach().item()),
        "pool_cos": float(pool_cos.detach().item()),
        "seq_mse": float(seq_mse.detach().item()),
        "pool_mse": float(pool_mse.detach().item()),
    }
    return loss, metrics


@torch.no_grad()
def _build_holdout_set(*, models, device, dtype, joint_template: str) -> list[dict[str, torch.Tensor]]:
    holdout = []
    for a, b in COLLISION_PAIRS:
        seq_a, pool_a = encode_prompt_sdxl(a, models=models, device=device, dtype=dtype)
        seq_b, pool_b = encode_prompt_sdxl(b, models=models, device=device, dtype=dtype)
        seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
        seq_j, pool_j = encode_prompt_sdxl(
            joint_prompt(a, b, joint_template), models=models, device=device, dtype=dtype
        )
        holdout.append(
            {
                "seq_a": seq_a, "pool_a": pool_a,
                "seq_b": seq_b, "pool_b": pool_b,
                "seq_e": seq_e, "pool_e": pool_e,
                "seq_j": seq_j, "pool_j": pool_j,
            }
        )
    return holdout


def _cosine_lr(step: int, total: int, warmup: int, lr_max: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * lr_max * (1 + math.cos(math.pi * min(progress, 1.0)))


def _cache_key(*, pairs: list[tuple[str, str]], joint_template: str, model_id: str, dtype_str: str) -> str:
    h = hashlib.sha1()
    h.update(model_id.encode())
    h.update(b"\0")
    h.update(joint_template.encode())
    h.update(b"\0")
    h.update(dtype_str.encode())
    h.update(b"\0")
    for a, b in pairs:
        h.update(a.encode()); h.update(b"\0"); h.update(b.encode()); h.update(b"\0")
    return h.hexdigest()[:16]


@torch.no_grad()
def _precompute_pool(
    *,
    pairs: list[tuple[str, str]],
    models: dict,
    device: torch.device,
    dtype: torch.dtype,
    joint_template: str,
    cache_path: Path,
) -> dict[str, torch.Tensor]:
    """Encode all pairs once. Returns CPU tensors stacked along dim 0.

    Tensors:
        seq_a, seq_b, seq_j: (N, 77, 2048)
        pool_a, pool_b, pool_j: (N, 1280)
        seq_e: (77, 2048)        -- shared empty embedding
        pool_e: (1280,)
    """
    if cache_path.exists():
        print(f"[precompute] loading cache {cache_path}")
        blob = torch.load(cache_path, map_location="cpu")
        return blob

    print(f"[precompute] encoding {len(pairs)} pairs (3 prompts each) -> {cache_path}")
    seq_e_t, pool_e_t = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
    seq_e_cpu = seq_e_t.squeeze(0).cpu()
    pool_e_cpu = pool_e_t.squeeze(0).cpu()

    seq_dim = seq_e_cpu.shape[-1]
    seq_len = seq_e_cpu.shape[0]
    pool_dim = pool_e_cpu.shape[-1]
    store_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
    n = len(pairs)

    seq_a = torch.empty((n, seq_len, seq_dim), dtype=store_dtype)
    seq_b = torch.empty((n, seq_len, seq_dim), dtype=store_dtype)
    seq_j = torch.empty((n, seq_len, seq_dim), dtype=store_dtype)
    pool_a = torch.empty((n, pool_dim), dtype=store_dtype)
    pool_b = torch.empty((n, pool_dim), dtype=store_dtype)
    pool_j = torch.empty((n, pool_dim), dtype=store_dtype)

    t0 = time.time()
    for i, (a, b) in enumerate(pairs):
        sa, pa = encode_prompt_sdxl(a, models=models, device=device, dtype=dtype)
        sb, pb = encode_prompt_sdxl(b, models=models, device=device, dtype=dtype)
        sj, pj = encode_prompt_sdxl(joint_prompt(a, b, joint_template), models=models, device=device, dtype=dtype)
        seq_a[i] = sa.squeeze(0).to(store_dtype).cpu()
        seq_b[i] = sb.squeeze(0).to(store_dtype).cpu()
        seq_j[i] = sj.squeeze(0).to(store_dtype).cpu()
        pool_a[i] = pa.squeeze(0).to(store_dtype).cpu()
        pool_b[i] = pb.squeeze(0).to(store_dtype).cpu()
        pool_j[i] = pj.squeeze(0).to(store_dtype).cpu()
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - (i + 1)) / max(rate, 1e-6)
            print(f"[precompute] {i+1}/{n}  rate={rate:.1f}/s  eta={eta/60:.1f} min")

    blob = {
        "seq_a": seq_a, "seq_b": seq_b, "seq_j": seq_j,
        "pool_a": pool_a, "pool_b": pool_b, "pool_j": pool_j,
        "seq_e": seq_e_cpu.to(store_dtype),
        "pool_e": pool_e_cpu.to(store_dtype),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.rename(cache_path)
    print(f"[precompute] saved cache {cache_path} ({sum(t.numel()*t.element_size() for t in blob.values())/1e9:.2f} GB)")
    return blob


def _pin(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in tensors.items():
        try:
            out[k] = v.pin_memory()
        except RuntimeError:
            out[k] = v
    return out


def _iter_minibatches_from_tensors(
    blob: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    train_dtype: torch.dtype,
    seed: int,
):
    """Yield dicts of GPU tensors; reuses preallocated CPU index sampling."""
    n = blob["seq_a"].shape[0]
    g = torch.Generator().manual_seed(seed)
    seq_e = blob["seq_e"].to(device=device, dtype=train_dtype)
    pool_e = blob["pool_e"].to(device=device, dtype=train_dtype)
    while True:
        perm = torch.randperm(n, generator=g)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            sa = blob["seq_a"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            sb = blob["seq_b"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            sj = blob["seq_j"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            pa = blob["pool_a"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            pb = blob["pool_b"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            pj = blob["pool_j"].index_select(0, idx).to(device=device, dtype=train_dtype, non_blocking=True)
            bs = idx.shape[0]
            yield {
                "seq_a": sa, "seq_b": sb, "seq_j": sj,
                "pool_a": pa, "pool_b": pb, "pool_j": pj,
                "seq_e": seq_e.unsqueeze(0).expand(bs, -1, -1),
                "pool_e": pool_e.unsqueeze(0).expand(bs, -1),
            }


def _release_text_encoders(models: dict) -> None:
    for k in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
        if k in models:
            try:
                models[k].to("cpu")
            except Exception:
                pass
            del models[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train embedding synthesizer (M2)")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--joint-template", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train-pairs", type=int, default=None)
    parser.add_argument("--n-holdout-oversample", type=int, default=None)
    parser.add_argument("--n-supercategory-oversample", type=int, default=None)
    parser.add_argument("--cache-dir", default=None, help="Where to store the encoded-pair cache (default: <output_root>/synthesizer/cache).")
    parser.add_argument("--gpu-resident", action="store_true", help="Keep all encoded tensors on GPU (fastest; needs the cache to fit in VRAM).")
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke run (dataset of ~64, 10 steps).")
    args = parser.parse_args()

    cfg = RunConfig()
    arch = args.arch or cfg.synth.arch
    device = infer_device(args.device)
    dtype_str = args.dtype or cfg.dtype
    dtype = infer_dtype(dtype_str, device)
    train_dtype = torch.float32

    output_name = args.output_name or arch
    out_dir = ensure_dir(cfg.paths.output_root / "synthesizer" / output_name)
    ckpt_dir = ensure_dir(Path(cfg.paths.synthesizer_checkpoint).parent)
    cache_dir = ensure_dir(Path(args.cache_dir) if args.cache_dir else cfg.paths.output_root / "synthesizer" / "cache")

    n_train = args.n_train_pairs or cfg.synth.n_train_pairs
    n_holdout = args.n_holdout_oversample or cfg.synth.n_holdout_oversample
    n_super = args.n_supercategory_oversample or cfg.synth.n_supercategory_oversample
    seed = args.seed if args.seed is not None else cfg.synth.seed
    joint_template = args.joint_template or cfg.synth.joint_template
    if args.smoke:
        n_train, n_holdout, n_super = 32, 16, 16

    pairs = _build_pair_pool(
        n_random=n_train,
        n_holdout_oversample=n_holdout,
        n_supercategory_oversample=n_super,
        seed=seed,
    )
    print(f"[train] pair pool size = {len(pairs)}")

    model_id = args.model_id or cfg.model_id
    cache_key = _cache_key(pairs=pairs, joint_template=joint_template, model_id=model_id, dtype_str=dtype_str)
    cache_path = cache_dir / f"pairs_{cache_key}.pt"

    need_encoders = not cache_path.exists()
    models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype) if need_encoders else None

    if need_encoders:
        blob = _precompute_pool(
            pairs=pairs, models=models, device=device, dtype=dtype,
            joint_template=joint_template, cache_path=cache_path,
        )
        # Build holdout set while encoders are still loaded.
        holdout = _build_holdout_set(models=models, device=device, dtype=dtype, joint_template=joint_template)
        _release_text_encoders(models)
    else:
        blob = _precompute_pool(
            pairs=pairs, models=None, device=device, dtype=dtype,
            joint_template=joint_template, cache_path=cache_path,
        )
        # Cache hit -> need encoders only for the small holdout set.
        models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
        holdout = _build_holdout_set(models=models, device=device, dtype=dtype, joint_template=joint_template)
        _release_text_encoders(models)

    if args.gpu_resident:
        print("[train] moving encoded tensors to GPU")
        for k in list(blob.keys()):
            blob[k] = blob[k].to(device=device, non_blocking=True)
    else:
        blob = _pin(blob)

    synth = build_synthesizer(
        arch,
        seq_dim=cfg.synth.seq_dim,
        pooled_dim=cfg.synth.pooled_dim,
        hidden_dim=cfg.synth.hidden_dim,
        num_layers=cfg.synth.num_layers,
        dropout=cfg.synth.dropout,
    ).to(device=device, dtype=train_dtype)
    optim = torch.optim.AdamW(synth.parameters(), lr=cfg.synth.lr, weight_decay=cfg.synth.weight_decay)

    num_steps = args.num_steps if args.num_steps is not None else cfg.synth.num_steps
    if args.smoke:
        num_steps = 10
    batch_size = args.batch_size or cfg.synth.batch_size
    log_interval = max(1, min(cfg.synth.log_interval, num_steps // 20))
    val_interval = max(1, min(cfg.synth.val_interval, num_steps // 10))

    train_iter = _iter_minibatches_from_tensors(
        blob, batch_size=batch_size, device=device, train_dtype=train_dtype, seed=seed,
    )
    history: list[dict] = []
    best_val_cos = -math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    synth.train()
    t0 = time.time()
    for step in range(num_steps):
        for g in optim.param_groups:
            g["lr"] = _cosine_lr(step, num_steps, cfg.synth.warmup_steps, args.lr or cfg.synth.lr)
        batch = next(train_iter)
        loss, metrics = _step_loss(
            synth, batch,
            seq_loss_weight=cfg.synth.seq_loss_weight,
            pooled_loss_weight=cfg.synth.pooled_loss_weight,
            cosine_weight=cfg.synth.cosine_loss_weight,
            mse_weight=cfg.synth.mse_loss_weight,
        )
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(synth.parameters(), max_norm=1.0)
        optim.step()
        if step % log_interval == 0:
            dt = time.time() - t0
            rate = (step + 1) / max(dt, 1e-6)
            eta = (num_steps - step - 1) / max(rate, 1e-6)
            print(f"[train] step={step}/{num_steps} {metrics} rate={rate:.1f} step/s eta={eta/60:.1f} min")

        if step % val_interval == 0 or step == num_steps - 1:
            synth.eval()
            val_metrics = _evaluate_holdout(synth, holdout, dtype=train_dtype, device=device)
            history.append({"step": step, "train": metrics, "val": val_metrics})
            print(f"[val   ] step={step} {val_metrics}")
            cur = val_metrics["seq_cos"]
            if cur > best_val_cos:
                best_val_cos = cur
                torch.save(
                    {"state_dict": synth.state_dict(), "arch": arch, "config": asdict(cfg.synth)},
                    best_path,
                )
                print(f"[val   ] new best (seq_cos={cur:.4f}) saved to {best_path}")
            torch.save(
                {"state_dict": synth.state_dict(), "arch": arch, "config": asdict(cfg.synth)},
                last_path,
            )
            synth.train()

    write_json(out_dir / "training_log.json", history)
    print(f"[train] done. best_val seq_cos = {best_val_cos:.4f}; checkpoints in {ckpt_dir}")


@torch.no_grad()
def _evaluate_holdout(synth, holdout, *, dtype: torch.dtype, device: torch.device) -> dict[str, float]:
    seq_cos_total, pool_cos_total = 0.0, 0.0
    n = 0
    for ex in holdout:
        seq_a = ex["seq_a"].to(device=device, dtype=dtype)
        seq_b = ex["seq_b"].to(device=device, dtype=dtype)
        seq_e = ex["seq_e"].to(device=device, dtype=dtype)
        pool_a = ex["pool_a"].to(device=device, dtype=dtype)
        pool_b = ex["pool_b"].to(device=device, dtype=dtype)
        pool_e = ex["pool_e"].to(device=device, dtype=dtype)
        seq_j = ex["seq_j"].to(device=device, dtype=dtype)
        pool_j = ex["pool_j"].to(device=device, dtype=dtype)
        out = synth(seq_a=seq_a, seq_b=seq_b, seq_e=seq_e, pool_a=pool_a, pool_b=pool_b, pool_e=pool_e)
        seq_cos_total += float(F.cosine_similarity(out.seq.flatten(start_dim=1), seq_j.flatten(start_dim=1), dim=-1).mean().item())
        pool_cos_total += float(F.cosine_similarity(out.pooled, pool_j, dim=-1).mean().item())
        n += 1
    return {"seq_cos": seq_cos_total / max(n, 1), "pool_cos": pool_cos_total / max(n, 1), "n": n}


if __name__ == "__main__":
    main()
