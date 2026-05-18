"""CFG-schedule ablation for "cat and dog" / seed 42 on clean SDXL base.

For each schedule (mask[i] = i < k), renders one PNG via the per-step
CFG-masked sampler and writes a per-cell summary.json plus an
inspector_manifest.json under
``outputs/cfg_schedule_ablation_no_lora/seed_<n>/``.

This is the no-residual baseline against which the LoRA's marginal
effect will be measured. See configs/cfg_schedule_no_lora.yaml for the
schedule grammar and prompt.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from poe_repair.composers._helpers import init_latents_for_cell
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._sampling import (
    run_cfg_masked,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import encode_prompt_sdxl, ensure_dir, write_json


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cfg_schedule_no_lora.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "cfg_schedule_ablation_no_lora"


def mask_for_k(k: int, num_steps: int) -> list[bool]:
    """mask[i] = True for i in [0, k)."""
    return [i < int(k) for i in range(int(num_steps))]


def mask_string(mask: list[bool]) -> str:
    return "".join("1" if b else "0" for b in mask)


def render_schedule(
    *,
    ctx,
    sched_id: str,
    k: int,
    out_dir: Path,
    common: dict,
    num_steps: int,
    prompt: str,
    seed: int,
    model_id: str,
    overwrite: bool,
) -> dict:
    image_path = out_dir / "image.png"
    summary_path = out_dir / "summary.json"
    mask = mask_for_k(k, num_steps)

    if image_path.exists() and summary_path.exists() and not overwrite:
        print(f"[skip] {sched_id} cached")
        return {
            "schedule_id": sched_id,
            "k": int(k),
            "mask": mask_string(mask),
            "num_on": sum(mask),
            "image_path": str(image_path.relative_to(REPO_ROOT)),
            "cached": True,
        }

    ensure_dir(out_dir)
    t0 = time.time()
    out = run_cfg_masked(cfg_mask=mask, **common)
    write_decoded_image(out.image, image_path)
    elapsed = time.time() - t0

    summary = {
        "schedule_id": sched_id,
        "k": int(k),
        "mask": mask_string(mask),
        "num_on": int(out.extras["num_on"]),
        "prompt": prompt,
        "seed": int(seed),
        "guidance_scale": float(ctx.guidance_scale),
        "num_inference_steps": int(num_steps),
        "model_id": model_id,
        "elapsed_s": elapsed,
        "image_path": str(image_path.relative_to(REPO_ROOT)),
    }
    write_json(summary_path, summary)
    print(f"[done] {sched_id} (k={k}) in {elapsed:.1f}s")
    return {**summary, "cached": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-sanity", action="store_true",
                    help="Also render schedules from sanity_schedules section.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Subset of schedule ids to run.")
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

    # Share x_T with the existing pair cell (same seed, same SDXL shape).
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
        seq_cond=seq_cond, pool_cond=pool_cond,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        height=height, width=width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )

    out_root = ensure_dir(args.out_root / f"seed_{seed}")

    only = set(args.only) if args.only else None
    schedules = list(cfg.get("schedules") or [])
    if args.include_sanity:
        for entry in (cfg.get("sanity_schedules") or []):
            schedules.append({**dict(entry), "_sanity": True})

    if only:
        unknown = only - {str(s["id"]) for s in schedules}
        if unknown:
            raise SystemExit(f"unknown schedule id(s): {sorted(unknown)}")
        schedules = [s for s in schedules if str(s["id"]) in only]

    manifest_entries: list[dict] = []
    for entry in schedules:
        sched_id = str(entry["id"])
        k = int(entry["k"])
        is_sanity = bool(entry.get("_sanity"))
        if is_sanity:
            cell_dir = out_root / "sanity" / sched_id
        else:
            cell_dir = out_root / sched_id
        info = render_schedule(
            ctx=ctx,
            sched_id=sched_id,
            k=k,
            out_dir=cell_dir,
            common=common,
            num_steps=num_steps,
            prompt=prompt,
            seed=seed,
            model_id=model_id,
            overwrite=args.overwrite,
        )
        info["sanity"] = is_sanity
        manifest_entries.append(info)

    manifest_entries.sort(key=lambda e: (e["sanity"], e["k"]))
    manifest = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "model_id": model_id,
        "schedules": [
            {
                "schedule_id": e["schedule_id"],
                "k": e["k"],
                "mask": e["mask"],
                "num_on": e["num_on"],
                "image_path": e["image_path"],
                "sanity": e["sanity"],
            }
            for e in manifest_entries
        ],
    }
    write_json(out_root / "inspector_manifest.json", manifest)
    print(f"\nManifest: {out_root / 'inspector_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
