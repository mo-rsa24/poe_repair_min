"""Render per-epoch LoRA samples for a finished pooled run.

For each checkpoint under <run>/checkpoints/, loads its LoRA weights and
samples one image per seed at lambda=1. Per-seed reference columns
(PoE and Mono) come from the training cache's pre-rendered ``poe.png``
and ``mono.png``. Outputs:

  - per-seed strip ``strip_seed_NN.png``: [PoE | epoch_0 | ... | epoch_N | Mono]
    — read horizontally to watch a single seed converge toward Mono.
  - global ``contact_sheet.png``: rows = epoch, cols = seed.
  - ``manifest.json`` enumerating every PNG.

SDXL is loaded once; checkpoint swaps just reload LoRA tensors.

Default seeds = train_pool ∪ held_out from seed_pool.yaml, so the same
strip shows both "in-distribution" (train-pool) and "held-out"
behaviour over epochs.

Usage::

    python -m scripts.cross_seed_lora_pooling.render_per_epoch \
        --run-dir /datasets/.../task_b_learning_curve/k04__ep200 \
        [--seeds 1,2,3,4,9,10,11,12]
        [--seed-groups train-pool,held-out]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool
from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


log = logging.getLogger("render_per_epoch")


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"lora_step_(\d+)\.pt$")
    out: list[tuple[int, Path]] = []
    for p in sorted(ckpt_dir.glob("lora_step_*.pt")):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="render_per_epoch")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="finished pooled-trainer run directory")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated; overrides --seed-groups")
    ap.add_argument("--seed-groups", default="train-pool,held-out",
                    help=("comma-separated subset of "
                          "{train-pool,held-out}. Default: both."))
    ap.add_argument("--out-subdir", default="samples/per_epoch",
                    help="relative to --run-dir")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32",
                             "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--seed-pool-path", default=None)
    ap.add_argument("--pair-slug-override", default=None,
                    help="when set, pull init latents and PoE/Mono ref "
                         "PNGs from the override pair's cache rather "
                         "than the checkpoint's training pair. Use "
                         "alongside --prompt-a/-b/--joint-prompt to "
                         "match the eval pair's prompts.")
    ap.add_argument("--thumb-size", type=int, default=256,
                    help="contact-sheet thumbnail edge in pixels")
    return ap


def compose_grid(rows: list[tuple[str, list[Path]]],
                 col_labels: list[str],
                 thumb: int) -> Image.Image:
    n_rows = len(rows)
    n_cols = len(col_labels)
    pad = 12
    label_w = 140
    label_h = 28
    W = label_w + n_cols * (thumb + pad) + pad
    H = label_h + n_rows * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Column labels.
    for j, cl in enumerate(col_labels):
        x = label_w + j * (thumb + pad) + pad
        draw.text((x, 6), cl, fill="black", font=font)
    # Row labels + thumbnails.
    for i, (rl, pngs) in enumerate(rows):
        y = label_h + i * (thumb + pad) + pad
        draw.text((6, y + thumb // 2 - 8), rl, fill="black", font=font)
        for j, png in enumerate(pngs):
            x = label_w + j * (thumb + pad) + pad
            try:
                im = Image.open(png).convert("RGB").resize((thumb, thumb),
                                                          Image.LANCZOS)
                canvas.paste(im, (x, y))
            except Exception as e:
                draw.text((x, y + thumb // 2 - 8), f"MISS {png.name}: {e}",
                          fill="red", font=font)
    return canvas


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)

    run_dir: Path = p.run_dir.resolve()
    ckpts = list_checkpoints(run_dir / "checkpoints")
    if not ckpts:
        log.error("no lora_step_*.pt found under %s", run_dir / "checkpoints")
        return 2
    log.info("found %d checkpoints", len(ckpts))

    pool = load_seed_pool(p.seed_pool_path)
    # If the run dir records which train-pool seeds were actually used
    # (truncated to --k), prefer that over the full pool for labeling.
    pooled_meta_path = run_dir / "pooled_meta.json"
    if pooled_meta_path.exists():
        pooled_meta = json.loads(pooled_meta_path.read_text())
        trained_seeds = [int(s) for s in pooled_meta.get(
            "seeds_used", list(pool.train_pool))]
        held_out_meta = [int(s) for s in pooled_meta.get(
            "held_out", list(pool.held_out))]
    else:
        trained_seeds = [int(s) for s in pool.train_pool]
        held_out_meta = [int(s) for s in pool.held_out]
    if p.seeds is not None:
        seeds = [int(x) for x in p.seeds.split(",") if x.strip()]
    else:
        groups = {g.strip() for g in p.seed_groups.split(",") if g.strip()}
        chosen: list[int] = []
        if "train-pool" in groups:
            chosen.extend(trained_seeds)
        if "held-out" in groups:
            chosen.extend(held_out_meta)
        # de-dup, preserve order
        seeds = list(dict.fromkeys(chosen))
    train_pool_set = set(trained_seeds)
    held_out_set = set(held_out_meta)
    out_dir = ensure_dir(run_dir / p.out_subdir)

    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    log.info("device=%s dtype=%s seeds=%s", device, dtype, seeds)

    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)

    # Attach LoRA shell using the latest checkpoint's config (rank/targets).
    latest_ckpt = torch.load(ckpts[-1][1], map_location="cpu",
                             weights_only=False)
    cfg_dict = latest_ckpt.get("config", {})
    rank = int(cfg_dict.get("lora", {}).get("rank", 8))
    alpha = int(cfg_dict.get("lora", {}).get("alpha", rank))
    target_modules = tuple(cfg_dict.get("lora", {}).get(
        "target_modules", ("attn2.to_q", "attn2.to_k", "attn2.to_v")))
    adapter_name = str(cfg_dict.get("lora", {}).get("adapter_name", "lora"))
    log.info("LoRA shell: rank=%d alpha=%d targets=%s", rank, alpha,
             target_modules)
    from peft import LoraConfig
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=list(target_modules), init_lora_weights=True,
    )
    unet = models["unet"]
    unet.add_adapter(lora_cfg, adapter_name=adapter_name)
    with torch.no_grad():
        for name, prm in unet.named_parameters():
            if "lora_" in name:
                prm.data = prm.data.to(torch.float32)
            prm.requires_grad_(False)
    unet.eval()

    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt
    class _CfgShim:
        cell = _PromptShim()
    embeddings = encode_all_prompts(_CfgShim(), models, device, dtype)

    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT
    train_pair_slug = str(cfg_dict.get("cell", {}).get("pair_slug", pool.pair_slug))
    pair_slug = str(p.pair_slug_override) if p.pair_slug_override else train_pair_slug
    if pair_slug != train_pair_slug:
        log.info("held-out-pair render: train_pair=%s eval_pair=%s",
                 train_pair_slug, pair_slug)

    init_by_seed: dict[int, torch.Tensor] = {}
    for s in seeds:
        cell = CellPath.from_root(pair_slug, int(s), cache_root=cache_root)
        init_by_seed[int(s)] = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(p.euler_sigma),
        )

    def sample(lam: float, seed: int, out_png: Path) -> None:
        with torch.inference_mode():
            out = run_lora_residual_inject(
                init_latents=init_by_seed[seed],
                models=models, scheduler=scheduler,
                seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
                seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
                seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
                seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
                lambda_value=float(lam),
                lora_adapter_name=adapter_name,
                record_eps_path=None,
            )
        write_decoded_image(out.image, out_png)

    # Reference columns come from the training cache (already rendered):
    #   <cache_root>/heldout/<pair>/seed_NN/{poe,mono}.png
    poe_ref: dict[int, Path] = {}
    mono_ref: dict[int, Path] = {}
    for s in seeds:
        cell = CellPath.from_root(pair_slug, int(s), cache_root=cache_root)
        poe_ref[int(s)] = cell.root / "poe.png"
        mono_ref[int(s)] = cell.root / "mono.png"
        if not poe_ref[int(s)].exists():
            log.warning("missing cached PoE ref for seed %d: %s",
                        s, poe_ref[int(s)])
        if not mono_ref[int(s)].exists():
            log.warning("missing cached Mono ref for seed %d: %s",
                        s, mono_ref[int(s)])

    manifest: dict = {
        "run_dir": str(run_dir),
        "pair_slug": pair_slug,
        "train_pair_slug": train_pair_slug,
        "prompt_a": p.prompt_a,
        "prompt_b": p.prompt_b,
        "joint_prompt": p.joint_prompt,
        "seeds": seeds,
        "train_pool": sorted(train_pool_set),
        "held_out": sorted(held_out_set),
        "references": {
            str(s): {"poe": str(poe_ref[s]), "mono": str(mono_ref[s])}
            for s in seeds
        },
        "checkpoints": [],
    }

    # Lambda=1 sweep across checkpoints.
    for step, ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        log.info("== checkpoint step=%d epoch=%d (%s)", step, epoch,
                 ckpt_path.name)
        lora_trainer.load_lora_state(unet, ckpt["lora_state"])
        sub = ensure_dir(out_dir / f"epoch_{epoch:04d}_step_{step:06d}")
        pngs: list[Path] = []
        for s in seeds:
            png = sub / f"sample_seed_{int(s):02d}.png"
            sample(1.0, int(s), png)
            pngs.append(png)
        manifest["checkpoints"].append({
            "step": step, "epoch": epoch, "ckpt": str(ckpt_path),
            "out_dir": str(sub),
            "pngs_by_seed": {str(int(s)): str(p)
                             for s, p in zip(seeds, pngs)},
        })

    write_json(out_dir / "manifest.json", manifest)

    # --- Per-seed strips: [PoE | epoch_0 | ... | Mono] ---
    strips_dir = ensure_dir(out_dir / "strips")
    epoch_labels = [f"ep {entry['epoch']:>3d}"
                    for entry in manifest["checkpoints"]]
    col_labels_strip = ["PoE"] + epoch_labels + ["Mono"]
    for s in seeds:
        group = ("train-pool" if int(s) in train_pool_set
                 else "held-out" if int(s) in held_out_set
                 else "other")
        row_pngs = [poe_ref[int(s)]] + [
            Path(entry["pngs_by_seed"][str(int(s))])
            for entry in manifest["checkpoints"]
        ] + [mono_ref[int(s)]]
        strip = compose_grid(
            rows=[(f"seed {int(s)} ({group})", row_pngs)],
            col_labels=col_labels_strip,
            thumb=int(p.thumb_size),
        )
        strip_path = strips_dir / f"strip_seed_{int(s):02d}.png"
        strip.save(strip_path)
        log.info("wrote strip → %s", strip_path)

    # --- Global contact sheet: rows = epoch (+ refs), cols = seed ---
    rows: list[tuple[str, list[Path]]] = [
        ("PoE",  [poe_ref[int(s)] for s in seeds]),
    ]
    for entry in manifest["checkpoints"]:
        rows.append((
            f"epoch {entry['epoch']:>3d}",
            [Path(entry["pngs_by_seed"][str(int(s))]) for s in seeds],
        ))
    rows.append(("Mono", [mono_ref[int(s)] for s in seeds]))
    col_labels = [
        f"seed {int(s)} {'(train)' if int(s) in train_pool_set else '(held)'}"
        for s in seeds
    ]
    grid = compose_grid(rows, col_labels, thumb=int(p.thumb_size))
    grid_path = out_dir / "contact_sheet.png"
    grid.save(grid_path)
    log.info("wrote contact sheet → %s", grid_path)
    log.info("manifest → %s", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
