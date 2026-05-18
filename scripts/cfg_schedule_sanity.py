"""Sanity checks for the CFG-masked sampler.

A. all-on (mask=[True]*N) must match run_cfg on the same prompt and
   guidance scale up to fp tolerance.
B. all-off (mask=[False]*N) must match run_cfg with guidance_scale=0.0
   (which analytically collapses to ε_uncond).

Writes outputs/cfg_schedule_ablation_no_lora/seed_<n>/sanity/report.json
plus the four reference PNGs used in the comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from poe_repair.composers._helpers import init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._sampling import (
    run_cfg,
    run_cfg_masked,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl, ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cfg_schedule_no_lora.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "cfg_schedule_ablation_no_lora"
LATENT_TOL = 1e-4
PIXEL_TOL_UINT8 = 1


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _png_l_inf_uint8(img_a: torch.Tensor, img_b: torch.Tensor) -> int:
    """Pixel L-inf distance after uint8 quantisation (matches saved PNG)."""
    def _q(x: torch.Tensor) -> torch.Tensor:
        a = x.detach().float().clamp(0.0, 1.0)
        return (a * 255.0).round().to(torch.uint8)
    return int((_q(img_a).int() - _q(img_b).int()).abs().max().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompt = str(cfg["prompt"])
    seed = int(cfg["seed"])
    num_steps = int(cfg["num_inference_steps"])
    guidance_scale = float(cfg["guidance_scale"])
    height = int(cfg["height"])
    width = int(cfg["width"])
    model_id = str(cfg["model_id"])
    init_pair = cfg["init_latents_pair"]

    ctx = make_ctx(
        output_root=args.out_root,
        model_id=model_id,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    pair_cell = cell_for(
        str(init_pair["prompt_a"]),
        str(init_pair["prompt_b"]),
        seed,
        height=height,
        width=width,
    )
    init_latents, euler_sigma = init_latents_for_cell(pair_cell, ctx)

    seq_cond, pool_cond = encode_prompt_sdxl(
        prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_e, pool_e = encode_prompt_sdxl(
        "", models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )

    common = dict(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_e=seq_e, pool_e=pool_e,
        num_inference_steps=num_steps,
        height=height, width=width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )

    sanity_dir = ensure_dir(args.out_root / f"seed_{seed}" / "sanity")

    # ------------------------------------------------------------------
    # A. all-on vs run_cfg at the same guidance scale.
    # ------------------------------------------------------------------
    print("[sanity A] all-on vs run_cfg ...")
    out_masked_on = run_cfg_masked(
        seq_cond=seq_cond, pool_cond=pool_cond,
        guidance_scale=guidance_scale,
        cfg_mask=[True] * num_steps,
        **common,
    )
    out_ref_on = run_cfg(
        seq_cond=seq_cond, pool_cond=pool_cond,
        guidance_scale=guidance_scale,
        **common,
    )
    latent_dA = _max_abs_diff(out_masked_on.latents, out_ref_on.latents)
    pixel_dA = _png_l_inf_uint8(out_masked_on.image, out_ref_on.image)
    passA = (latent_dA < LATENT_TOL) and (pixel_dA <= PIXEL_TOL_UINT8)
    write_decoded_image(out_masked_on.image, sanity_dir / "all_on_masked.png")
    write_decoded_image(out_ref_on.image, sanity_dir / "all_on_reference.png")
    print(f"  latent max|Δ| = {latent_dA:.2e}   pixel L∞ (uint8) = {pixel_dA}   "
          f"{'PASS' if passA else 'FAIL'}")

    # ------------------------------------------------------------------
    # B. all-off vs run_cfg at guidance_scale=0.0
    # ------------------------------------------------------------------
    print("[sanity B] all-off vs run_cfg(guidance=0) ...")
    out_masked_off = run_cfg_masked(
        seq_cond=seq_cond, pool_cond=pool_cond,
        guidance_scale=guidance_scale,
        cfg_mask=[False] * num_steps,
        **common,
    )
    out_ref_off = run_cfg(
        seq_cond=seq_cond, pool_cond=pool_cond,
        guidance_scale=0.0,
        **common,
    )
    latent_dB = _max_abs_diff(out_masked_off.latents, out_ref_off.latents)
    pixel_dB = _png_l_inf_uint8(out_masked_off.image, out_ref_off.image)
    passB = (latent_dB < LATENT_TOL) and (pixel_dB <= PIXEL_TOL_UINT8)
    write_decoded_image(out_masked_off.image, sanity_dir / "all_off_masked.png")
    write_decoded_image(out_ref_off.image, sanity_dir / "all_off_reference.png")
    print(f"  latent max|Δ| = {latent_dB:.2e}   pixel L∞ (uint8) = {pixel_dB}   "
          f"{'PASS' if passB else 'FAIL'}")

    report = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "latent_tol": LATENT_TOL,
        "pixel_tol_uint8": PIXEL_TOL_UINT8,
        "checks": {
            "all_on_vs_cfg": {
                "latent_max_abs_diff": latent_dA,
                "pixel_l_inf_uint8": pixel_dA,
                "pass": passA,
                "masked_png": str((sanity_dir / "all_on_masked.png").relative_to(REPO_ROOT)),
                "reference_png": str((sanity_dir / "all_on_reference.png").relative_to(REPO_ROOT)),
            },
            "all_off_vs_cfg_guidance0": {
                "latent_max_abs_diff": latent_dB,
                "pixel_l_inf_uint8": pixel_dB,
                "pass": passB,
                "masked_png": str((sanity_dir / "all_off_masked.png").relative_to(REPO_ROOT)),
                "reference_png": str((sanity_dir / "all_off_reference.png").relative_to(REPO_ROOT)),
            },
        },
    }
    write_json(sanity_dir / "report.json", report)
    print(f"\nReport: {sanity_dir / 'report.json'}")
    return 0 if (passA and passB) else 1


if __name__ == "__main__":
    raise SystemExit(main())
