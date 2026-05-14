"""Run several β schedules for tempered PoE on cat × dog seed 42.

The first-pass `ramp_up_5_25` schedule (β ramping 0→1 over [5,25]) ran
the trajectory unconditionally early, leading to a wallpaper-collapse
basin. That run was uninformative about tempering itself.

This script tests schedules that actually probe the "PoE chimera is an
over-sharpness artifact" hypothesis:

  - const_1.00 : β=1 everywhere (= vanilla PoE; sanity control)
  - const_0.75 : uniform mild tempering
  - const_0.50 : uniform stronger tempering
  - const_0.25 : uniform aggressive tempering
  - commit_only_0.50 : β=0.5 inside [5,25], β=1 outside
  - anti_commit_0.50 : β=1 inside [5,25], β=0.5 outside
  - ramp_down_25_45 : β=1 → 0.25 over [25, 45] (softens the late stage)
  - ramp_up_5_25 : the original failure case, included as reference

Per-schedule: outputs/tempered_schedules/<name>.png + summary.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from poe_repair.composers._helpers import encode_pair, init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._sampling import (
    run_poe_tempered,
    run_vanilla_poe,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_schedules(num_steps: int) -> dict[str, list[float]]:
    def const(v: float) -> list[float]:
        return [v] * num_steps

    def windowed(inside: float, outside: float, lo: int, hi: int) -> list[float]:
        return [inside if lo <= i < hi else outside for i in range(num_steps)]

    def linear_ramp(v_lo: float, v_hi: float, lo: int, hi: int) -> list[float]:
        # v_lo before lo, ramp to v_hi over [lo, hi), then v_hi after.
        out = []
        for i in range(num_steps):
            if i < lo:
                out.append(v_lo)
            elif i >= hi:
                out.append(v_hi)
            else:
                frac = (i - lo) / max(1, hi - lo)
                out.append(v_lo + frac * (v_hi - v_lo))
        return out

    return {
        "const_1.00": const(1.0),
        "const_0.75": const(0.75),
        "const_0.50": const(0.50),
        "const_0.25": const(0.25),
        "commit_only_0.50": windowed(inside=0.5, outside=1.0, lo=5, hi=25),
        "anti_commit_0.50": windowed(inside=1.0, outside=0.5, lo=5, hi=25),
        "ramp_down_25_45": linear_ramp(1.0, 0.25, 25, 45),
        "ramp_up_5_25": linear_ramp(0.0, 1.0, 5, 25),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--include-baseline", action="store_true",
                    help="Also render plain PoE for reference.")
    ap.add_argument("--schedules", nargs="+", default=None,
                    help="Subset of schedule names to run; default = all.")
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "outputs" / "tempered_schedules")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(output_root=args.output_root)
    cell = cell_for(args.prompt_a, args.prompt_b, args.seed)
    out_root = ensure_dir(
        args.output_root / cell.pair_slug / f"seed_{args.seed}",
    )

    schedules = make_schedules(ctx.num_inference_steps)
    if args.schedules is not None:
        unknown = sorted(set(args.schedules) - set(schedules))
        if unknown:
            raise SystemExit(f"unknown schedule(s): {unknown}")
        schedules = {k: schedules[k] for k in args.schedules}

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
        "runs": {},
    }

    if args.include_baseline:
        image_path = out_root / "baseline.png"
        if image_path.exists() and not args.overwrite:
            summary["runs"]["baseline"] = {"image": str(image_path), "cached": True}
        else:
            t0 = time.time()
            out = run_vanilla_poe(**common)
            write_decoded_image(out.image, image_path)
            summary["runs"]["baseline"] = {
                "image": str(image_path),
                "elapsed_s": time.time() - t0, "cached": False,
            }
            print(f"[done] baseline in {time.time() - t0:.1f}s")

    for name, beta in schedules.items():
        image_path = out_root / f"{name}.png"
        if image_path.exists() and not args.overwrite:
            print(f"[skip] {name} cached")
            summary["runs"][name] = {"image": str(image_path), "cached": True}
            continue

        t0 = time.time()
        out = run_poe_tempered(beta_schedule=beta, **common)
        write_decoded_image(out.image, image_path)
        elapsed = time.time() - t0
        summary["runs"][name] = {
            "image": str(image_path),
            "elapsed_s": elapsed,
            "beta_schedule": beta,
            "cached": False,
        }
        print(f"[done] {name} in {elapsed:.1f}s")

    write_json(out_root / "schedules_summary.json", summary)
    print(f"\nSummary: {out_root / 'schedules_summary.json'}")


if __name__ == "__main__":
    main()
