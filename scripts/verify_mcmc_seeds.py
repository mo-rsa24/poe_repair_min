"""Run baseline + a single ULA config on cat × dog for multiple seeds.

Addresses the cross-seed caveat from the first MCMC corrector run:
seed 42's ULA result was "chimera → pure cat". If ULA on seeds {4, 42, 123}
all collapse to single-concept basins (any of pure-cat / pure-dog / pure-
chimera), then "single-concept modes are the PoE distribution's actual
modes" is locked in and no amount of sampler-side correction will reach
co-occurrence from this prompt pair.

Per seed: outputs/mcmc_verify/<seed>/baseline.png, ula.png + summary.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._mcmc import MCMCCorrectorConfig
from poe_repair.methods._sampling import (
    run_poe_mcmc_corrector,
    run_vanilla_poe,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 42, 123])
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--corrector-steps", type=int, default=5)
    ap.add_argument("--step-size-base", type=float, default=1.0e-3)
    ap.add_argument("--corrector-window", type=int, nargs=2, default=[5, 25])
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "outputs" / "mcmc_verify")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(output_root=args.output_root)

    summary: dict = {
        "prompt_a": args.prompt_a, "prompt_b": args.prompt_b,
        "corrector_steps": args.corrector_steps,
        "step_size_base": args.step_size_base,
        "corrector_window": list(args.corrector_window),
        "num_inference_steps": ctx.num_inference_steps,
        "guidance_scale": ctx.guidance_scale,
        "seeds": {},
    }

    window = tuple(args.corrector_window)

    for seed in args.seeds:
        cell = cell_for(args.prompt_a, args.prompt_b, seed)
        seed_dir = ensure_dir(
            args.output_root / cell.pair_slug / f"seed_{seed}",
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

        per_seed = {}
        for variant in ("baseline", "ula"):
            image_path = seed_dir / f"{variant}.png"
            if image_path.exists() and not args.overwrite:
                print(f"[skip] seed {seed} {variant} cached")
                per_seed[variant] = {"image": str(image_path), "cached": True}
                continue

            t0 = time.time()
            if variant == "baseline":
                out = run_vanilla_poe(**common)
                extras = {}
            else:
                cfg = MCMCCorrectorConfig(
                    method="ula",
                    num_corrector_steps=args.corrector_steps,
                    step_size_base=args.step_size_base,
                    window=window,
                )
                out = run_poe_mcmc_corrector(corrector=cfg, **common)
                extras = out.extras

            write_decoded_image(out.image, image_path)
            elapsed = time.time() - t0
            per_seed[variant] = {
                "image": str(image_path),
                "elapsed_s": elapsed,
                "extras": extras,
                "cached": False,
            }
            print(f"[done] seed {seed} {variant} in {elapsed:.1f}s")

        summary["seeds"][str(seed)] = per_seed

    write_json(args.output_root / "verify_summary.json", summary)
    print(f"\nSummary: {args.output_root / 'verify_summary.json'}")


if __name__ == "__main__":
    main()
