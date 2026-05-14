"""Run PoE on cat × dog seed 42 with (a) baseline, (b) ULA corrector,
(c) UHA corrector, (d) tempered composition. One image per variant under
``outputs/mcmc_corrector/<variant>/...``.

Honesty note: the Metropolis-corrected variants from Du et al. (MALA,
CHA) need an energy and SDXL is score-parameterised — those are not
ported. ULA and UHA are unadjusted, so they sample a smoothed
approximation of the PoE posterior, not the exact one. If they don't
flip the basin, the dismissal is "score-only MCMC correctors on the
PoE-composed SDXL score do not fix seed 42", not "Du et al. doesn't
work."

Usage:
    python scripts/run_mcmc_corrector_seed42.py \\
        --variants baseline ula uha tempered \\
        --corrector-steps 5 --step-size-base 1e-3 \\
        --corrector-window 5 25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._mcmc import MCMCCorrectorConfig
from poe_repair.methods._sampling import (
    run_poe_mcmc_corrector,
    run_poe_tempered,
    run_vanilla_poe,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent


def linear_ramp(num_steps: int, t_lo: int, t_hi: int) -> list[float]:
    """β = 0 for step < t_lo, linear ramp 0→1 over [t_lo, t_hi), then β = 1."""
    out = []
    for i in range(num_steps):
        if i < t_lo:
            out.append(0.0)
        elif i >= t_hi:
            out.append(1.0)
        else:
            out.append((i - t_lo) / max(1, t_hi - t_lo))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "ula", "uha", "tempered"],
                    choices=["baseline", "ula", "uha", "tempered"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--corrector-steps", type=int, default=5,
                    help="K per denoising timestep")
    ap.add_argument("--step-size-base", type=float, default=1.0e-3,
                    help="η_t = step_size_base · σ_t² (Song & Ermon recipe)")
    ap.add_argument("--corrector-window", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Active step-index range (half-open). Default = all.")
    ap.add_argument("--uha-damping", type=float, default=0.5)
    ap.add_argument("--uha-leapfrog", type=int, default=3)
    ap.add_argument("--tempered-ramp", type=int, nargs=2, default=[5, 25],
                    metavar=("LO", "HI"),
                    help="Linear β ramp 0→1 over [LO, HI) in step index.")
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "outputs" / "mcmc_corrector")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(output_root=args.output_root)
    cell = cell_for(args.prompt_a, args.prompt_b, args.seed)
    out_root = ensure_dir(
        args.output_root / cell.pair_slug / f"seed_{args.seed}",
    )

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    window = tuple(args.corrector_window) if args.corrector_window else None

    summary: dict = {
        "pair_slug": cell.pair_slug,
        "seed": args.seed,
        "num_inference_steps": ctx.num_inference_steps,
        "guidance_scale": ctx.guidance_scale,
        "variants": {},
    }

    for variant in args.variants:
        image_path = out_root / f"{variant}.png"
        if image_path.exists() and not args.overwrite:
            print(f"[skip] {variant} already at {image_path}")
            summary["variants"][variant] = {"image": str(image_path), "cached": True}
            continue

        t0 = time.time()
        if variant == "baseline":
            out = run_vanilla_poe(**common)
            extras = {}
        elif variant == "ula":
            cfg = MCMCCorrectorConfig(
                method="ula",
                num_corrector_steps=args.corrector_steps,
                step_size_base=args.step_size_base,
                window=window,
            )
            out = run_poe_mcmc_corrector(corrector=cfg, **common)
            extras = out.extras
        elif variant == "uha":
            cfg = MCMCCorrectorConfig(
                method="uha",
                num_corrector_steps=args.corrector_steps,
                step_size_base=args.step_size_base,
                window=window,
                damping=args.uha_damping,
                num_leapfrog=args.uha_leapfrog,
            )
            out = run_poe_mcmc_corrector(corrector=cfg, **common)
            extras = out.extras
        elif variant == "tempered":
            beta = linear_ramp(
                ctx.num_inference_steps, args.tempered_ramp[0], args.tempered_ramp[1],
            )
            out = run_poe_tempered(beta_schedule=beta, **common)
            extras = out.extras
        else:
            raise ValueError(f"unknown variant {variant!r}")

        write_decoded_image(out.image, image_path)
        elapsed = time.time() - t0
        summary["variants"][variant] = {
            "image": str(image_path),
            "elapsed_s": elapsed,
            "extras": extras,
            "cached": False,
        }
        print(f"[done] {variant} in {elapsed:.1f}s -> {image_path}")

    write_json(out_root / "summary.json", summary)
    print(f"\nSummary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
