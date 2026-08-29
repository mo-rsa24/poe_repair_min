"""Render the adapter's side of a joint-prompt failure cell.

The training cache holds cells where the joint prompt itself fails (the
mono render misses one of the two animals). This script loads one pooled
LoRA checkpoint, renders that cell at lambda=1 (adapter-corrected) and
lambda=0 (canary: must reproduce the cached poe.png, which validates the
harness), and composes a labelled strip beside the cached mono and PoE
references. A sidecar JSON records checkpoint step, sampler settings and
every input path.

    co3_bw python scripts/render_joint_failure_repair_cell.py \
        --pair a_cat__x__a_dog --seed 1 --ckpt-step 100000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.methods._sampling import run_lora_residual_inject, write_decoded_image
from poe_repair.runtime import (
    ensure_dir, infer_device, infer_dtype,
    load_ddim_scheduler, load_sdxl_models, write_json,
)
from poe_repair.training_cache import CellPath

log = logging.getLogger("render_joint_failure_repair_cell")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k"
DEFAULT_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
DEFAULT_OUT = REPO / "artifacts/results/does-the-fix-reach-unseen-pairs/joint-failure-repair"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="render_joint_failure_repair_cell")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--ckpt-step", type=int, default=100000)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--skip-canary", action="store_true")
    return ap


def compose_strip(panels: list[tuple[str, Path]], out_png: Path, thumb: int = 340) -> None:
    pad, label_h = 10, 26
    W = len(panels) * thumb + (len(panels) + 1) * pad
    H = label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for i, (label, p) in enumerate(panels):
        x = pad + i * (thumb + pad)
        draw.text((x, 6), label, fill="black", font=font)
        im = Image.open(p).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(im, (x, label_h + pad))
    canvas.save(out_png)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = build_argparser().parse_args(argv)

    ckpt_path = p.run_dir / "checkpoints" / f"lora_step_{p.ckpt_step:06d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    rank = int(cfg.get("lora", {}).get("rank", 8))
    alpha = int(cfg.get("lora", {}).get("alpha", rank))
    targets = list(cfg.get("lora", {}).get(
        "target_modules", ["attn2.to_q", "attn2.to_k", "attn2.to_v"]))
    adapter_name = str(cfg.get("lora", {}).get("adapter_name", "lora"))
    log.info("checkpoint %s: rank=%d alpha=%d targets=%s",
             ckpt_path.name, rank, alpha, targets)

    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)

    from peft import LoraConfig
    unet = models["unet"]
    unet.add_adapter(
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                   target_modules=targets, init_lora_weights=True),
        adapter_name=adapter_name)
    with torch.no_grad():
        for name, prm in unet.named_parameters():
            if "lora_" in name:
                prm.data = prm.data.to(torch.float32)
            prm.requires_grad_(False)
    unet.eval()
    lora_trainer.load_lora_state(unet, ckpt["lora_state"])

    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt
    class _CfgShim:
        cell = _PromptShim()
    emb = encode_all_prompts(_CfgShim(), models, device, dtype)

    cell = CellPath.from_root(p.pair, int(p.seed), cache_root=p.cache_root)
    init = load_pinned_init_latents(cell, device=device, dtype=dtype,
                                    euler_init_noise_sigma=float(p.euler_sigma))
    out_dir = ensure_dir(p.out_root / f"{p.pair}__seed{p.seed:02d}__step{p.ckpt_step:06d}")

    def render(lam: float, out_png: Path) -> None:
        with torch.inference_mode():
            out = run_lora_residual_inject(
                init_latents=init, models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_j=emb["seq_j"], pool_j=emb["pool_j"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
                lambda_value=float(lam), lora_adapter_name=adapter_name,
                record_eps_path=None)
        write_decoded_image(out.image, out_png)
        log.info("wrote %s", out_png)

    adapter_png = out_dir / "adapter_lambda1.png"
    render(1.0, adapter_png)
    canary_png = out_dir / "canary_lambda0.png"
    if not p.skip_canary:
        render(0.0, canary_png)

    mono, poe = cell.root / "mono.png", cell.root / "poe.png"
    strip = out_dir / "strip.png"
    compose_strip([("joint prompt (the target)", mono),
                   ("plain PoE", poe),
                   (f"adapter-corrected ({p.ckpt_step // 1000}k)", adapter_png)], strip)
    write_json(out_dir / "sidecar.json", {
        "pair": p.pair, "seed": int(p.seed), "split": cell.split,
        "checkpoint": str(ckpt_path), "checkpoint_step": int(p.ckpt_step),
        "lora": {"rank": rank, "alpha": alpha, "targets": targets},
        "sampler": {"scheduler": "ddim",
                    "num_inference_steps": int(p.num_inference_steps),
                    "guidance_scale": float(p.guidance_scale),
                    "euler_init_noise_sigma": float(p.euler_sigma),
                    "height": int(p.height), "width": int(p.width)},
        "lambda": 1.0,
        "references": {"joint": str(mono), "poe": str(poe)},
        "outputs": {"adapter": str(adapter_png),
                    "canary_lambda0": str(canary_png) if not p.skip_canary else None,
                    "strip": str(strip)},
        "note": ("mono/poe are the cache cell's own renders; adapter is this "
                 "script's render from the loaded checkpoint at lambda 1. "
                 "canary_lambda0 must visually match poe.png or the harness "
                 "is broken."),
    })
    log.info("done: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
