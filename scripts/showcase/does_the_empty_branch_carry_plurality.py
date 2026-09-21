"""Does the adapter's empty branch carry plurality on its own?

The V6 objective is ``z - (a + b - u)``. The empty branch ``u`` is the model's answer to the
empty prompt, so it is the SAME tensor for every pair in the pool: cat x dog, typewriter x
cactus, all of them. If the adapter has pushed "there should be two separate things here" into
``u`` rather than into the two concept branches, then one shared tensor is carrying plurality,
and it should work with the concept branches left un-adapted.

This script renders one cell five ways at a given checkpoint and puts them in one strip:

    1. plain PoE            a_bar + b_bar - u_bar        adapter off everywhere (the chimera)
    2. V6, everything on    a + b - u                    what the run actually trains
    3. the probe            a_bar + b_bar - u_lora       ONLY the empty branch adapted
    4. joint prompt, LoRA   "a cat and a dog", adapter on, plain CFG
    5. joint prompt, base   "a cat and a dog", adapter off, plain CFG (the mono reference)

Read it as: if 3 composes and 1 does not, plurality lives in the empty branch and is shared
across pairs. If 3 looks like 1, the adapter put plurality in the concept branches instead, and
the empty branch is doing something else.

    python scripts/showcase/does_the_empty_branch_carry_plurality.py \
        --checkpoint <path to lora_step_XXXXXX.pt> --rank 16 --seeds 9 10 11 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_cfg,
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl

LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"


def attach_and_load(unet, ckpt_path: Path, rank: int, alpha: int) -> dict:
    cfg = LoRAConfig(rank=rank, alpha=alpha, dropout=0.0,
                     target_modules=LORA_TARGET_MODULES, init="gaussian",
                     adapter_name=LORA_ADAPTER_NAME)
    info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=cfg))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state")
    if state is None:
        raise KeyError(f"{ckpt_path} has no 'lora_state' (found {list(ckpt.keys())})")
    lora_trainer.load_lora_state(unet, state)
    info["n_loaded"] = len(state)
    return info


def adapters(unet, enable: bool) -> None:
    try:
        if enable and hasattr(unet, "enable_adapters"):
            unet.enable_adapters()
        elif (not enable) and hasattr(unet, "disable_adapters"):
            unet.disable_adapters()
    except ValueError:
        pass


def strip(paths: list[tuple[str, Path]], title: str, thumb: int = 320) -> Image.Image:
    pad, label_h, title_h = 8, 34, 26
    W = len(paths) * thumb + (len(paths) + 1) * pad
    H = title_h + label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(255, 255, 255))
    for i, (label, p) in enumerate(paths):
        x = pad + i * (thumb + pad)
        for j, line in enumerate(label.split("\n")):
            draw.text((x, title_h + j * 15), line,
                      fill=(235, 200, 120) if j == 0 else (170, 170, 170))
        try:
            im = Image.open(p).convert("RGB"); im.thumbnail((thumb, thumb))
            canvas.paste(im, (x + (thumb - im.width) // 2, title_h + label_h))
        except Exception:
            draw.rectangle([x, title_h + label_h, x + thumb, title_h + label_h + thumb],
                           outline=(90, 90, 90))
    return canvas


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=1024,
                    help="render height and width. MethodCtx does not carry them, so "
                         "they are named here rather than read off the context.")
    ap.add_argument("--out", default="artifacts/results/designing-the-correction-loss/"
                                     "does-the-empty-branch-carry-plurality")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else args.rank

    ckpt = Path(args.checkpoint)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    ctx = make_ctx(num_inference_steps=args.steps, guidance_scale=args.guidance)
    unet = ctx.models["unet"]
    info = attach_and_load(unet, ckpt, args.rank, alpha)
    print(f"LoRA attached: n_matched={info['n_matched']} n_loaded={info['n_loaded']} "
          f"from {ckpt.name}")

    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_a, pool_a = encode_prompt_sdxl(args.prompt_a, models=ctx.models, device=ctx.device,
                                       dtype=ctx.dtype)
    seq_b, pool_b = encode_prompt_sdxl(args.prompt_b, models=ctx.models, device=ctx.device,
                                       dtype=ctx.dtype)
    seq_j, pool_j = encode_prompt_sdxl(args.joint_prompt, models=ctx.models, device=ctx.device,
                                       dtype=ctx.dtype)

    for seed in args.seeds:
        cell = cell_for(args.prompt_a, args.prompt_b, seed)
        init, sigma = initial_latents_for_pair(cell=cell, models=ctx.models,
                                               device=ctx.device, dtype=ctx.dtype)
        d = out_root / f"seed_{seed:02d}"; d.mkdir(parents=True, exist_ok=True)
        common = dict(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                      guidance_scale=args.guidance, num_inference_steps=args.steps,
                      height=args.size, width=args.size, euler_init_noise_sigma=sigma,
                      device=ctx.device, dtype=ctx.dtype)
        poe_args = dict(seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                        seq_j=seq_j, pool_j=pool_j, seq_e=seq_e, pool_e=pool_e, **common)

        # 1. plain PoE: lambda 0 means the step is eps_PoE_frozen, the adapter contributing nothing
        p1 = d / "1_poe_plain.png"
        write_decoded_image(run_lora_residual_inject(lambda_value=0.0, **poe_args).image, p1)
        # 2. V6 as trained: all three branches adapted
        p2 = d / "2_v6_all_adapted.png"
        write_decoded_image(run_lora_residual_inject(lambda_value=1.0, **poe_args).image, p2)
        # 3. THE PROBE: concept branches un-adapted, empty branch adapted
        p3 = d / "3_only_empty_branch_adapted.png"
        write_decoded_image(
            run_lora_residual_inject(lambda_value=1.0, adapt_null_only=True, **poe_args).image, p3)
        # 4 and 5. the joint prompt on its own, adapter on then off
        cfg_args = dict(seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e, **common)
        p4 = d / "4_joint_prompt_lora_on.png"
        adapters(unet, True)
        write_decoded_image(run_cfg(**cfg_args).image, p4)
        p5 = d / "5_joint_prompt_base.png"
        adapters(unet, False)
        write_decoded_image(run_cfg(**cfg_args).image, p5)
        adapters(unet, True)

        panels = [
            ("plain PoE\na_bar + b_bar - u_bar", p1),
            ("V6 as trained\na + b - u", p2),
            ("only the empty branch adapted\na_bar + b_bar - u_lora", p3),
            (f'"{args.joint_prompt}"\nadapter ON', p4),
            (f'"{args.joint_prompt}"\nadapter OFF', p5),
        ]
        s = strip(panels, f"{args.prompt_a} x {args.prompt_b}  seed {seed:02d}   {ckpt.name}")
        sp = out_root / f"strip_seed_{seed:02d}.png"
        s.save(sp)
        print(f"  seed {seed}: {sp}")
    print(f"done -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
