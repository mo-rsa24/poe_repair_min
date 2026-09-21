#!/usr/bin/env python
"""Basins by hand: nudge one cached state by 1% of its norm, finish DDIM three ways, compare.

Plan: plans/05-when-does-the-outcome-lock-in/plans/reading/01-basins-by-hand.md

Loads one pair-and-seed cell's cached mid-run state at each of the given steps (from its
``residuals/step_NNN.pt``), builds a nudged-down and a nudged-up copy (a random unit direction,
scaled to 1% of the state's norm), and finishes all three to step 50 with fresh U-Net calls
using the same plain-PoE composition the cache was made with (``poe_eps`` of the two
CFG-guided per-concept epsilons). Saves the three decoded endings per step, a JSON of the two
relative latent distances (down-vs-mid, mid-vs-up), and one 3x3 contact sheet across all
tested steps.

    co3 python scripts/commitment/perturb_finish.py --cell <cell-dir> --steps 5 25 40 \\
        --nudge 0.01 --out <out-dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.runtime import (  # noqa: E402
    ddim_prev_from_x0_eps,
    decode_latents,
    encode_prompt_sdxl,
    guided_eps,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    poe_eps,
    tweedie_mean,
)
from poe_repair.methods._sampling import add_time_ids, write_decoded_image  # noqa: E402

# The agreement threshold. Set before any run, per the plan's Description; not adjustable
# after seeing the answer.
REL_ENDING_DIST_MAX = 0.05

DISK_LIMIT_PCT = 90
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
HEIGHT = WIDTH = 1024


def _disk_guard(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    pct = 100 * usage.used / usage.total
    print(f"disk: {out_dir} at {pct:.0f}% used")
    if pct >= DISK_LIMIT_PCT:
        raise SystemExit(f"ERROR: over {DISK_LIMIT_PCT}% full, aborting.")


def _load_cell(cell_dir: Path) -> dict:
    summary_path = cell_dir / f"summary_{cell_dir.name}.json"
    if not summary_path.exists():
        raise SystemExit(f"no summary at {summary_path}")
    summary = json.loads(summary_path.read_text())
    return {
        "pair_slug": cell_dir.parent.parent.name,
        "seed": cell_dir.parent.name,
        "prompt_a": summary["pair"][0],
        "prompt_b": summary["pair"][1],
        "guidance_scale": float(summary["guidance_scale"]),
        "num_inference_steps": int(summary["num_inference_steps"]),
    }


def _load_step_state(cell_dir: Path, step_index: int, device, dtype) -> dict:
    step_path = cell_dir / "residuals" / f"step_{step_index:03d}.pt"
    if not step_path.exists():
        raise SystemExit(f"no cached step at {step_path}")
    d = torch.load(step_path, map_location="cpu", weights_only=False)
    if int(d["step_index"]) != step_index:
        raise SystemExit(
            f"step index mismatch: asked for {step_index}, file holds {d['step_index']}"
        )
    return {
        "x_t": d["x_t"].to(device=device, dtype=dtype),
        "seq_a": d["seq_a"].to(device=device, dtype=dtype),
        "pool_a": d["pool_a"].to(device=device, dtype=dtype),
        "seq_b": d["seq_b"].to(device=device, dtype=dtype),
        "pool_b": d["pool_b"].to(device=device, dtype=dtype),
        "timestep": int(d["timestep"]),
    }


@torch.no_grad()
def finish_ddim(
    latents: torch.Tensor,
    step_index: int,
    *,
    pe: torch.Tensor,
    pool: torch.Tensor,
    guidance_scale: float,
    scheduler,
    unet,
    device,
    dtype,
) -> torch.Tensor:
    """Finish the remaining DDIM steps from ``step_index``, plain PoE, fresh model calls."""
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=HEIGHT, width=WIDTH, batch_size=3, device=device, dtype=dtype
        ),
    }
    for si in range(step_index, len(scheduler.timesteps)):
        timestep = scheduler.timesteps[si]
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=si, x0=x0, eps=eps_p,
        )
    return latents


def _rel_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def _build_contact_sheet(out_dir: Path, steps: list[int], thumb: int = 256) -> Path:
    cols = ["down", "mid", "up"]
    sheet = Image.new("RGB", (thumb * 3, thumb * len(steps) + 24 * len(steps)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, step in enumerate(steps):
        y0 = row * (thumb + 24)
        draw.text((4, y0), f"step {step}", fill="black")
        for col, tag in enumerate(cols):
            img = Image.open(out_dir / f"step{step:03d}_{tag}.png").resize((thumb, thumb))
            sheet.paste(img, (col * thumb, y0 + 24))
    path = out_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, type=Path,
                    help="path to a method-dir cell, e.g. .../seed_4/teacher_residual_const_lam000")
    ap.add_argument("--steps", nargs="+", type=int, required=True,
                    help="denoise step indices to test, e.g. 5 25 40")
    ap.add_argument("--nudge", type=float, default=0.01,
                    help="nudge size as a fraction of the state's norm")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--nudge-seed", type=int, default=0,
                    help="RNG seed for the random nudge direction, one draw per tested step")
    args = ap.parse_args(argv)

    _disk_guard(args.out)

    cell_dir = args.cell
    meta = _load_cell(cell_dir)
    print(f"cell: {cell_dir}")
    print(f"pair: {meta['prompt_a']!r} x {meta['prompt_b']!r}  seed {meta['seed']}  "
          f"guidance {meta['guidance_scale']}  steps {meta['num_inference_steps']}")

    device = infer_device(args.device)
    dtype = infer_dtype("float16", device)
    print(f"device: {device}  dtype: {dtype}")

    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    scheduler.set_timesteps(meta["num_inference_steps"])
    unet = models["unet"]
    unet.eval()
    torch.set_grad_enabled(False)

    seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)

    results = []
    for step in sorted(args.steps):
        state = _load_step_state(cell_dir, step, device, dtype)
        pe = torch.cat([state["seq_a"], state["seq_b"], seq_e], dim=0)
        pool = torch.cat([state["pool_a"], state["pool_b"], pool_e], dim=0)

        torch.manual_seed(args.nudge_seed + step)
        u = torch.randn_like(state["x_t"])
        u = u / u.norm()
        d = args.nudge * state["x_t"].norm() * u

        endings = {}
        for delta, tag in ((-1, "down"), (0, "mid"), (1, "up")):
            z = state["x_t"] + delta * d
            ending_latents = finish_ddim(
                z, step, pe=pe, pool=pool, guidance_scale=meta["guidance_scale"],
                scheduler=scheduler, unet=unet, device=device, dtype=dtype,
            )
            endings[tag] = ending_latents
            image = decode_latents(models, ending_latents).cpu()
            write_decoded_image(image, args.out / f"step{step:03d}_{tag}.png")
            print(f"  step {step} {tag}: wrote step{step:03d}_{tag}.png")

        rel_down_mid = _rel_dist(endings["down"], endings["mid"])
        rel_mid_up = _rel_dist(endings["up"], endings["mid"])
        agree = rel_down_mid < REL_ENDING_DIST_MAX and rel_mid_up < REL_ENDING_DIST_MAX
        record = {
            "step": step,
            "timestep": state["timestep"],
            "nudge": args.nudge,
            "rel_dist_down_mid": rel_down_mid,
            "rel_dist_mid_up": rel_mid_up,
            "threshold": REL_ENDING_DIST_MAX,
            "agree": agree,
        }
        results.append(record)
        (args.out / f"step{step:03d}.json").write_text(json.dumps(record, indent=2))
        print(f"  step {step}: rel_dist(down,mid)={rel_down_mid:.4f} "
              f"rel_dist(mid,up)={rel_mid_up:.4f} agree={agree}")

    sheet_path = _build_contact_sheet(args.out, sorted(args.steps))
    print(f"wrote {sheet_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
