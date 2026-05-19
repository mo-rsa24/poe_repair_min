"""Step 0 — Inference-time mono-average pre-screen.

Tests whether the time-only seed-average residual r̄_t, when injected
into a held-out seed's PoE trajectory, already produces images close
to mono on that seed. Costs zero LoRA training — r_t is read directly
from the cache via the closed-form Δ_t expression.

For each held-out seed s* the script samples four conditions:
    1. PoE         — uncorrected baseline (run_cfg_poe)
    2. r̄_t-inject  — train-pool average residual injected (constant per t)
    3. r_t^(s*)    — held-out seed's own oracle residual (upper bound for
                      injection-style methods on this seed)
    4. mono        — diffusers-style CFG on joint prompt (ceiling)

Decodes each, lays them out side by side, writes a contact-sheet PNG
and per-seed metadata.

Usage::

    python -m poe_repair.experiments.cross_seed_lora_pooling.step0_prescreen \
        --out-dir outputs/cross_seed_lora_pooling/step0_prescreen
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from PIL import Image

from poe_repair.experiments.cross_seed_lora_pooling.seed_pool import load_seed_pool
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_cfg,
    run_cfg_poe,
    run_constant_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import (
    DEFAULT_CACHE_ROOT,
    CellPath,
    iter_cell_deltas,
    mean_delta_per_step,
    resolve_cells,
)


log = logging.getLogger(__name__)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="step0_prescreen")
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/cross_seed_lora_pooling/step0_prescreen")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--seed-pool-path", default=None)
    ap.add_argument("--skip-conditions", default="",
                    help="comma-separated subset of {poe,rbar,roracle,mono} to skip")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds to sample; default = seed_pool.held_out. "
                         "Useful for ad-hoc probes (e.g. seed 42).")
    return ap


def _residual_dict_from_cell(cell: CellPath) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for entry in iter_cell_deltas(cell):
        out[int(entry["step_index"])] = entry["delta"]
    return out


def _residual_dict_from_list(
    deltas: list[torch.Tensor], step_indices: list[int],
) -> dict[int, torch.Tensor]:
    return {int(si): d for si, d in zip(step_indices, deltas)}


def _hstack_pngs(paths: list[Path], out_path: Path, *, gap: int = 6) -> Path:
    """Horizontally concatenate PNGs with a small gap; output PNG."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    x = 0
    for im in imgs:
        out.paste(im, (x, (h - im.height) // 2))
        x += im.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def _vstack_pngs(paths: list[Path], out_path: Path, *, gap: int = 6) -> Path:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    y = 0
    for im in imgs:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_SEED_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)
    pool = load_seed_pool(p.seed_pool_path)

    out_root = Path(p.out_dir) if p.out_dir else (
        Path(__file__).resolve().parents[3] / "outputs"
        / "cross_seed_lora_pooling" / "step0_prescreen"
    )
    ensure_dir(out_root)
    pool.persist_alongside(out_root)

    skip = {s.strip() for s in p.skip_conditions.split(",") if s.strip()}
    cond_names = [c for c in ("poe", "rbar", "roracle", "mono") if c not in skip]
    log.info("conditions to run: %s", cond_names)

    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT

    # Build r̄_t from the train pool. Cache-only, no UNet.
    log.info("building r̄_t from train pool seeds=%s ...", list(pool.train_pool))
    train_cells = resolve_cells(pool.pair_slug, list(pool.train_pool),
                                cache_root=cache_root)
    r_bar_list, step_indices, timesteps = mean_delta_per_step(train_cells)
    r_bar = _residual_dict_from_list(r_bar_list, step_indices)
    r_bar_norms = [float(t.float().norm().item()) for t in r_bar_list]
    log.info("r̄_t: %d steps, mean ‖r̄_t‖=%.3f, max=%.3f, min=%.3f",
             len(r_bar_norms), sum(r_bar_norms) / len(r_bar_norms),
             max(r_bar_norms), min(r_bar_norms))
    write_json(out_root / "r_bar_meta.json", {
        "train_pool": list(pool.train_pool),
        "n_steps": len(r_bar_norms),
        "step_indices": step_indices,
        "timesteps": timesteps,
        "norm_per_step": r_bar_norms,
    })

    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    log.info("device=%s dtype=%s", device, dtype)
    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)

    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt
    class _CfgShim:
        cell = _PromptShim()
    emb = encode_all_prompts(_CfgShim(), models, device, dtype)

    contact_rows: list[Path] = []
    per_seed_summary: list[dict] = []

    seeds_to_run = (
        [int(x) for x in p.seeds.split(",") if x.strip()]
        if p.seeds is not None else list(pool.held_out)
    )
    log.info("sampling seeds: %s", seeds_to_run)
    for s in seeds_to_run:
        log.info("===== held-out seed %d =====", int(s))
        cell = CellPath.from_root(pool.pair_slug, int(s), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(p.euler_sigma),
        )
        seed_dir = ensure_dir(out_root / f"seed_{int(s):02d}")
        per_seed_norms: dict = {}
        col_paths: dict[str, Path] = {}

        if "poe" in cond_names:
            png = seed_dir / "poe.png"
            log.info("sampling: PoE baseline")
            o = run_cfg_poe(
                init_latents=init, models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
            )
            write_decoded_image(o.image, png)
            col_paths["poe"] = png

        if "rbar" in cond_names:
            png = seed_dir / "rbar.png"
            log.info("sampling: r̄_t-injected (train-pool average)")
            o = run_constant_residual_inject(
                init_latents=init, models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
                residual_by_step=r_bar,
                lambda_value=1.0,
            )
            write_decoded_image(o.image, png)
            col_paths["rbar"] = png
            per_seed_norms["rbar_inject_norm_per_step"] = o.extras["inject_norm_per_step"]

        if "roracle" in cond_names:
            png = seed_dir / "roracle.png"
            log.info("sampling: r_t^(s*) per-seed oracle injection")
            r_oracle = _residual_dict_from_cell(cell)
            o = run_constant_residual_inject(
                init_latents=init, models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
                residual_by_step=r_oracle,
                lambda_value=1.0,
            )
            write_decoded_image(o.image, png)
            col_paths["roracle"] = png
            per_seed_norms["roracle_inject_norm_per_step"] = o.extras["inject_norm_per_step"]

        if "mono" in cond_names:
            png = seed_dir / "mono.png"
            log.info("sampling: mono ceiling (CFG on joint prompt)")
            o = run_cfg(
                init_latents=init, models=models, scheduler=scheduler,
                seq_cond=emb["seq_j"], pool_cond=emb["pool_j"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=float(p.guidance_scale),
                num_inference_steps=int(p.num_inference_steps),
                height=int(p.height), width=int(p.width),
                euler_init_noise_sigma=float(p.euler_sigma),
                device=device, dtype=dtype,
            )
            write_decoded_image(o.image, png)
            col_paths["mono"] = png

        ordered = [col_paths[c] for c in cond_names if c in col_paths]
        row_png = _hstack_pngs(ordered, seed_dir / "row.png")
        contact_rows.append(row_png)
        per_seed_summary.append({
            "seed": int(s),
            "columns": cond_names,
            "row_png": str(row_png),
            **per_seed_norms,
        })

    contact_sheet = _vstack_pngs(contact_rows, out_root / "contact_sheet.png")
    log.info("contact sheet → %s", contact_sheet)
    write_json(out_root / "summary.json", {
        "pair_slug": pool.pair_slug,
        "train_pool": list(pool.train_pool),
        "held_out": list(pool.held_out),
        "conditions": cond_names,
        "contact_sheet": str(contact_sheet),
        "per_seed": per_seed_summary,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
