#!/usr/bin/env python
"""Build the SuperDiff trajectory cache (plan 08, task 1.2).

Mirrors ``scripts/build_training_cache.py``'s layout so the existing pooled trainer reads it
unchanged: ``<root>/<split>/<pair_slug>/seed_<n>/meta.json`` plus
``residuals/step_{000..199}.pt``, each step holding the PoE cache's key names with SuperDiff's
meaning behind them:

    x_t          the UNet input at that step (SuperDiff's latent divided by sqrt(sigma_t^2 + 1)),
                 so the trainer's forward reproduces the cached raw predictions exactly
    eps_a_raw    the UNet's prediction on prompt 1        (SuperDiff's noise_pred_1)
    eps_b_raw    the UNet's prediction on prompt 2        (noise_pred_2)
    eps_j_raw    the UNet's prediction on the joint prompt
    eps_uncond   the UNet's prediction on the zero (negative) embedding
    timestep     int(round(t * 1000)), the value SuperDiff feeds the UNet; 1000, 995, ..., 5
    step_index, sigma_t, kappa

The trajectory is SuperDiff's own: 200 Euler-Maruyama steps at kappa fixed to 0.5, unclamped,
noise injected every step except the last three, exactly ``poe_repair/composers/superdiff.py``.
The residual a student learns from this cache is ``eps_J~ - eps_M(kappa)``, formed by the
trainer's ``compose="superdiff"`` branch, not the PoE ``delta_t_from_raw``.

Usage (one shard of the pool, one device):
    build_superdiff_cache.py --shard 0 --nshards 2 --device cuda:0
Idempotent: a cell with 200 step files and a meta.json is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers import superdiff as sd  # noqa: E402

DEFAULT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/cache")
DEFAULT_PAIR_PROMPTS = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/pair_prompts.json")
DEFAULT_PAIR_POOL = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/pair_pool.json")
DISK_LIMIT_PCT = 90


def _disk_guard(path: Path) -> None:
    import shutil
    path.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(path)
    pct = 100 * u.used / u.total
    print(f"disk: {path} at {pct:.0f}% used", flush=True)
    if pct >= DISK_LIMIT_PCT:
        raise SystemExit(f"ERROR: over {DISK_LIMIT_PCT}% full, aborting.")


@torch.no_grad()
def build_cell(models, *, prompt_a: str, prompt_b: str, joint_prompt: str, seed: int,
               steps: int, kappa: float, guidance_scale: float, height: int, width: int,
               device, dtype, cell_dir: Path) -> dict:
    """Run SuperDiff at fixed kappa with the joint row, saving every step's raw predictions."""
    res_dir = cell_dir / "residuals"
    res_dir.mkdir(parents=True, exist_ok=True)
    unet = models["unet"]
    generator = torch.cuda.manual_seed(seed)
    latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8),
                          generator=generator, dtype=dtype, device=device)
    prompt_embeds, cond = sd._prepare_prompt_input(models, prompt_a, prompt_b, height, width, device)
    joint_embeds, joint_cond = sd._encode_joint_prompt(models, joint_prompt, height, width, device)
    embeds = torch.cat([prompt_embeds, joint_embeds], dim=0)
    cond4 = {"text_embeds": torch.cat([cond["text_embeds"], joint_cond["text_embeds"]], dim=0),
             "time_ids": torch.cat([cond["time_ids"], joint_cond["time_ids"]], dim=0)}

    t = torch.tensor(1.0)
    dt = 1.0 / steps
    latents = latents * (sd._sigma(t) ** 2 + 1) ** 0.5
    timesteps, r_t_norms, eps_m_norms = [], [], []
    t0 = time.perf_counter()
    for i in range(steps):
        sigma_t = sd._sigma(t)
        dsigma = sd._sigma(t - dt) - sigma_t
        x_in = latents / (sigma_t ** 2 + 1) ** 0.5
        ts_val = int(round(float(t) * 1000))
        out = unet(torch.cat([x_in] * 4), t * 1000, encoder_hidden_states=embeds,
                   added_cond_kwargs=cond4, return_dict=False)[0]
        eu, e1, e2, ej = out.chunk(4)

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma_t) * torch.empty_like(latents).normal_(generator=generator)
        eps_m = eu + guidance_scale * ((e2 - eu) + kappa * (e1 - e2))
        eps_j = eu + guidance_scale * (ej - eu)
        r_t_norms.append((eps_j - eps_m).float().norm().item())
        eps_m_norms.append(eps_m.float().norm().item())

        torch.save({
            "step_index": i, "timestep": ts_val, "sigma_t": float(sigma_t), "kappa": kappa,
            "x_t": x_in.detach().to("cpu", torch.float16),
            "eps_a_raw": e1.detach().to("cpu", torch.float16),
            "eps_b_raw": e2.detach().to("cpu", torch.float16),
            "eps_j_raw": ej.detach().to("cpu", torch.float16),
            "eps_uncond": eu.detach().to("cpu", torch.float16),
        }, res_dir / f"step_{i:03d}.pt")
        timesteps.append(ts_val)

        if i < steps - 3:
            latents = latents + 2 * dsigma * eps_m + noise
        else:
            latents = latents + dsigma * eps_m
        t = t - dt

    meta = {
        "pair": [prompt_a, prompt_b], "seed": seed, "joint_prompt": joint_prompt,
        "guidance_scale": guidance_scale, "num_inference_steps": steps,
        "height": height, "width": width, "kappa": kappa,
        "sampler": "superdiff euler-maruyama, kappa fixed, unclamped, noise on all but the last 3 steps",
        "composer": "poe_repair/composers/superdiff.py (loop mirrored in scripts/build_superdiff_cache.py)",
        "branch_order": ["a", "b", "j", "uncond"], "timesteps": timesteps,
        "r_t_sd_norms": r_t_norms, "eps_m_norms": eps_m_norms,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pair-prompts", type=Path, default=DEFAULT_PAIR_PROMPTS)
    ap.add_argument("--pair-pool", type=Path, default=DEFAULT_PAIR_POOL)
    ap.add_argument("--pairs", nargs="*", default=None, help="pair slugs; default: the pool's train list")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    from poe_repair.runtime import infer_device, infer_dtype, write_json
    prompts = json.loads(args.pair_prompts.read_text())
    pairs = args.pairs or json.loads(args.pair_pool.read_text())["train"]
    cells = [(p, s) for p in pairs for s in args.seeds]
    cells = cells[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(cells)} cells of {len(pairs)} pairs x {len(args.seeds)} seeds", flush=True)

    _disk_guard(args.root)
    device = infer_device(args.device)
    dtype = infer_dtype("fp16", device)
    models = sd._load_superdiff_models(device, dtype)

    done = skipped = 0
    t_all = time.perf_counter()
    for n, (pair, seed) in enumerate(cells, 1):
        cell_dir = args.root / args.split / pair / f"seed_{seed}"
        if not args.overwrite and (cell_dir / "meta.json").exists() and \
                len(list((cell_dir / "residuals").glob("step_*.pt"))) == args.steps:
            skipped += 1
            print(f"[{n}/{len(cells)}] {pair} seed={seed}: cached, skip", flush=True)
            continue
        pp = prompts[pair]
        meta = build_cell(models, prompt_a=pp["prompt_a"], prompt_b=pp["prompt_b"],
                          joint_prompt=pp["joint_prompt"], seed=seed, steps=args.steps,
                          kappa=args.kappa, guidance_scale=args.guidance_scale,
                          height=args.height, width=args.width, device=device, dtype=dtype,
                          cell_dir=cell_dir)
        meta.update({"pair_slug": pair, "split": args.split})
        write_json(cell_dir / "meta.json", meta)
        done += 1
        print(f"[{n}/{len(cells)}] {pair} seed={seed}: {meta['elapsed_seconds']}s "
              f"elapsed={(time.perf_counter()-t_all)/60:.1f}min", flush=True)
    print(f"DONE shard {args.shard}: built {done}, skipped {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
