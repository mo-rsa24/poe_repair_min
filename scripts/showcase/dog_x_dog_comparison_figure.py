#!/usr/bin/env python
"""One labeled strip: every Mono candidate, PoE(A,A) uncorrected, and the LoRA-corrected render.

Complements dog_x_dog_probe.py's numeric per-step diffs with the thing a number can't show:
what the image actually looks like. Five panels, same seed, same x_T, side by side:

    Mono (same guidance) | Mono (double guidance) | Mono (joint prompt) | PoE(A,A) | LoRA-corrected

The first three let you eyeball which Mono definition actually looks like the PoE(A,A) render
(the identity check's real question, worth seeing and not just measuring). The last two are the
project's usual triptych pair (referenced in environment/hpc/execution-protocol.md as logged to
W&B for every experiment): the uncorrected agreeing-pair render next to what the trained LoRA
does to it at lambda=1 over the pre-registered window (steps 0-10).

Usage:
    python scripts/showcase/dog_x_dog_comparison_figure.py --seed 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_cfg,
    run_cfg_poe,
    run_lora_residual_inject_masked,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl

from dog_x_dog_probe import (  # noqa: E402  (path inserted above)
    GUIDANCE_SCALE,
    IDENTITY_MONO_GUIDANCE_SCALE,
    LORA_ADAPTER_NAME,
    NUM_INFERENCE_STEPS,
    OUT_ROOT,
    PROMPT,
    WINDOW_ON_STEPS,
    _attach_and_load_lora,
    _cell,
)


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.detach().float().clamp(0.0, 1.0)
    arr = (arr * 255.0).round().to(torch.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    return Image.fromarray(arr.cpu().numpy())


def _label(img: Image.Image, text: str, thumb: int) -> Image.Image:
    bar_h = 40
    canvas = Image.new("RGB", (thumb, thumb + bar_h), "white")
    canvas.paste(img.resize((thumb, thumb)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, thumb + 8), text, fill="black", font=font)
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--thumb", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    cell = _cell(args.seed)
    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_a, pool_a = encode_prompt_sdxl(PROMPT, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    joint = PROMPT + " and " + PROMPT
    seq_j, pool_j = encode_prompt_sdxl(joint, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_e=seq_e, pool_e=pool_e,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )

    panels: list[tuple[str, Image.Image]] = []

    out = run_cfg(seq_cond=seq_a, pool_cond=pool_a, guidance_scale=GUIDANCE_SCALE, **common)
    panels.append((f"Mono, same guidance (w={GUIDANCE_SCALE})", _tensor_to_pil(out.image)))

    out = run_cfg(seq_cond=seq_a, pool_cond=pool_a, guidance_scale=IDENTITY_MONO_GUIDANCE_SCALE, **common)
    panels.append((f"Mono, double guidance (w={IDENTITY_MONO_GUIDANCE_SCALE})", _tensor_to_pil(out.image)))

    out = run_cfg(seq_cond=seq_j, pool_cond=pool_j, guidance_scale=GUIDANCE_SCALE, **common)
    panels.append((f'Mono, joint prompt ("{joint}")', _tensor_to_pil(out.image)))

    out_poe = run_cfg_poe(
        seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a,
        guidance_scale=GUIDANCE_SCALE, **common,
    )
    panels.append(("PoE(A,A), LoRA off", _tensor_to_pil(out_poe.image)))

    _attach_and_load_lora(ctx.models["unet"])
    mask = [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)
    out_lora = run_lora_residual_inject_masked(
        seq_a=seq_a, pool_a=pool_a, seq_b=seq_a, pool_b=pool_a,
        guidance_scale=GUIDANCE_SCALE, cfg_mask=mask, composition_mode="with_prompt",
        lambda_value=1.0, lora_adapter_name=LORA_ADAPTER_NAME, **common,
    )
    panels.append(("PoE(A,A) + LoRA, lambda=1, steps 0-10", _tensor_to_pil(out_lora.image)))

    thumb = args.thumb
    labeled = [_label(img, text, thumb) for text, img in panels]
    strip = Image.new("RGB", (thumb * len(labeled), thumb + 40), "white")
    for i, panel in enumerate(labeled):
        strip.paste(panel, (i * thumb, 0))

    out_path = Path(args.out) if args.out else OUT_ROOT / f"comparison_strip_seed{args.seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(str(out_path))
    print(f"[dog_x_dog_comparison_figure] wrote {out_path} ({len(panels)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
