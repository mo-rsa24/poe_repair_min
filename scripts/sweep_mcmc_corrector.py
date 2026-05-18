"""Sweep ULA corrector hyperparameters on cat × dog seed 42.

Addresses three caveats from the first-pass MCMC corrector run:
  (1) step_size_base was a blind guess (1e-3); sweep ∈ {1e-4 .. 1e-2}
  (2) K (corrector steps per timestep) was 5; sweep ∈ {3, 10, 25}
  (3) window was hard-coded to [5, 25]; also try [0, 25], [0, 50]

If no (ss_base, K, window) configuration produces two-animal co-occurrence,
the "sampler-side fix is enough" hypothesis is locked-in failed for this seed.

Output per config: outputs/mcmc_sweep/<config_slug>.png + summary.json
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._mcmc import MCMCCorrectorConfig
from poe_repair.methods._sampling import (
    run_poe_mcmc_corrector,
    run_cfg_poe,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SS_BASE = [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3, 1.0e-2]
DEFAULT_K = [3, 10, 25]
DEFAULT_WINDOWS = [(0, 5), (5, 25), (0, 25), (0, 50)]


def config_slug(ss: float, k: int, window: tuple[int, int]) -> str:
    return f"ula_ss{ss:.0e}_K{k:02d}_w{window[0]:02d}-{window[1]:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--ss-base", type=float, nargs="+", default=DEFAULT_SS_BASE)
    ap.add_argument("--corrector-steps", type=int, nargs="+", default=DEFAULT_K)
    ap.add_argument(
        "--windows", type=int, nargs="+", default=None,
        help="Flat list of (lo hi) pairs. Default uses DEFAULT_WINDOWS.",
    )
    ap.add_argument("--include-baseline", action="store_true",
                    help="Also render plain PoE for reference.")
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "outputs" / "mcmc_sweep")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.windows is None:
        windows = DEFAULT_WINDOWS
    else:
        if len(args.windows) % 2 != 0:
            raise SystemExit("--windows must contain an even number of ints")
        windows = [
            (args.windows[2 * i], args.windows[2 * i + 1])
            for i in range(len(args.windows) // 2)
        ]

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

    summary: dict = {
        "pair_slug": cell.pair_slug,
        "seed": args.seed,
        "num_inference_steps": ctx.num_inference_steps,
        "guidance_scale": ctx.guidance_scale,
        "ss_base_grid": list(args.ss_base),
        "corrector_steps_grid": list(args.corrector_steps),
        "windows_grid": [list(w) for w in windows],
        "runs": {},
    }

    if args.include_baseline:
        image_path = out_root / "baseline.png"
        if image_path.exists() and not args.overwrite:
            print(f"[skip] baseline cached at {image_path}")
            summary["runs"]["baseline"] = {"image": str(image_path), "cached": True}
        else:
            t0 = time.time()
            out = run_cfg_poe(**common)
            write_decoded_image(out.image, image_path)
            summary["runs"]["baseline"] = {
                "image": str(image_path),
                "elapsed_s": time.time() - t0, "cached": False,
            }
            print(f"[done] baseline in {time.time() - t0:.1f}s")

    total = len(args.ss_base) * len(args.corrector_steps) * len(windows)
    print(f"sweeping {total} ULA configurations on {cell.pair_slug} seed {args.seed}")

    for i, (ss, k, window) in enumerate(
        itertools.product(args.ss_base, args.corrector_steps, windows), start=1,
    ):
        slug = config_slug(ss, k, window)
        image_path = out_root / f"{slug}.png"
        if image_path.exists() and not args.overwrite:
            print(f"[skip {i}/{total}] {slug} cached")
            summary["runs"][slug] = {"image": str(image_path), "cached": True}
            continue

        cfg = MCMCCorrectorConfig(
            method="ula",
            num_corrector_steps=k,
            step_size_base=ss,
            window=window,
        )
        t0 = time.time()
        out = run_poe_mcmc_corrector(corrector=cfg, **common)
        write_decoded_image(out.image, image_path)
        elapsed = time.time() - t0
        summary["runs"][slug] = {
            "image": str(image_path),
            "elapsed_s": elapsed,
            "extras": out.extras,
            "cached": False,
        }
        print(f"[done {i}/{total}] {slug} in {elapsed:.1f}s")

    write_json(out_root / "sweep_summary.json", summary)
    print(f"\nSummary: {out_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
