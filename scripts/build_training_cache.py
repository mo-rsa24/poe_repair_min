"""Build a training cache for Methods 2a / 2b.

For every surviving curated cell under ``dataset/cells/<pair>/<seed>/``,
runs a 4-branch DDIM sampler (A, B, J, ∅) along the PoE trajectory
(λ=0) for ``num_inference_steps`` and writes:

  outputs/training_cache/manifest.json
  outputs/training_cache/<split>/<pair_slug>/seed_<N>/
    meta.json
    embeddings.pt           # (seq, pool) for A, B, J, ∅ + init_latents + euler_sigma
    residuals/step_NNN.pt   # x_t, t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond
    mono.png                # regenerated joint-prompt image (verification)
    poe.png                 # regenerated PoE image (verification)

Cells under the held-out pair (default: a_cat × a_dog) are written under
``<split>=heldout``; everything else under ``<split>=train``. The trainer
filters by split.

Cohorts (predefined, balanced groups of pairs for parallelising across
nodes — see ``COHORTS`` below)::

    A   13 train cells: a_lion__x__a_dog, a_tiger__x__a_dog, a_dog__x__a_horse
    B   13 train cells: a_cat__x__a_horse, a_cat__x__a_lion,
                        a_wolf__x__a_horse, a_wolf__x__a_husky
    H   13 heldout cells: a_cat__x__a_dog

Run two parallel jobs (one per node), each on a different CUDA device::

    # mscluster106:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.build_training_cache --cohort A

    # mscluster107:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.build_training_cache --cohort B

``CUDA_VISIBLE_DEVICES=1`` makes torch see only physical GPU 1 and expose
it as ``cuda:0``; the script then runs without modification.

Other modes::

    python -m scripts.build_training_cache              # all cells (single GPU)
    python -m scripts.build_training_cache --pairs a_dog__x__a_horse a_lion__x__a_dog
    python -m scripts.build_training_cache --cohort H   # cache held-out cat×dog
    python -m scripts.build_training_cache --list-cohorts
    python -m scripts.build_training_cache --overwrite  # ignore caches

Idempotent: skips a cell if its ``residuals/step_049.pt`` already exists,
unless ``--overwrite`` is passed. Two nodes writing under the same
``outputs/training_cache/`` are safe as long as their cohorts are disjoint
— each cell is written by exactly one node, in its own directory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.config import joint_prompt
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._sampling import (
    add_time_ids,
    write_decoded_image,
)
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import (
    PairSeedCell,
    decode_latents,
    ddim_prev_from_x0_eps,
    ensure_dir,
    guided_eps,
    poe_eps,
    tweedie_mean,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CELLS = REPO_ROOT / "dataset" / "cells"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "training_cache"
DEFAULT_HELD_OUT_PAIRS: tuple[str, ...] = ("a_cat__x__a_dog",)


# Predefined cohorts for splitting work across nodes.
# Pair set after horse-pair swap: a_cat__x__a_horse and a_wolf__x__a_horse
# replaced by a_dog__x__a_duck and a_bear__x__a_salmon (5 seeds each).
# Expected counts: A=18 cells, B=14 cells, H=13 cells.
# Discovery skips slugs not present on disk, so partial promotes still work.
COHORTS: dict[str, tuple[str, ...]] = {
    "A": (
        "a_lion__x__a_dog",        # 5 cells
        "a_tiger__x__a_dog",       # 6 cells
        "a_dog__x__a_horse",       # 2 cells
        "a_dog__x__a_duck",        # 5 cells (new)
    ),
    "B": (
        "a_cat__x__a_lion",        # 5 cells
        "a_wolf__x__a_husky",      # 4 cells
        "a_bear__x__a_salmon",     # 5 cells (new)
    ),
    "H": (
        "a_cat__x__a_dog",         # 13 cells (held-out)
    ),
}


@dataclass
class CellSpec:
    pair_slug: str
    prompt_a: str
    prompt_b: str
    seed: int
    split: str  # "train" | "heldout"


def _slug_to_prompts(slug: str) -> tuple[str, str]:
    """Inverse of slugify for our naming convention.

    Slug format: ``<a_underscored>__x__<b_underscored>``. Underscores
    inside a prompt become spaces. Works for our 8-pair set; fragile if
    a prompt contains a literal "x" surrounded by underscores.
    """
    if "__x__" not in slug:
        raise ValueError(f"slug {slug!r} does not contain '__x__'")
    a, b = slug.split("__x__", 1)
    return a.replace("_", " "), b.replace("_", " ")


def discover_cells(
    cells_root: Path,
    *,
    pair_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    held_out_pairs: tuple[str, ...] = DEFAULT_HELD_OUT_PAIRS,
) -> list[CellSpec]:
    specs: list[CellSpec] = []
    if not cells_root.exists():
        raise FileNotFoundError(f"dataset/cells/ not found at {cells_root}")
    held_set = {p.lower() for p in held_out_pairs}
    pair_filter_set = {p.lower() for p in pair_filter} if pair_filter else None
    seed_filter_set = {int(s) for s in seed_filter} if seed_filter else None
    for pair_dir in sorted(cells_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        slug = pair_dir.name
        if pair_filter_set is not None and slug.lower() not in pair_filter_set:
            continue
        try:
            prompt_a, prompt_b = _slug_to_prompts(slug)
        except ValueError:
            print(f"[skip] cannot parse slug: {slug}")
            continue
        split = "heldout" if slug.lower() in held_set else "train"
        for seed_dir in sorted(pair_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if seed_filter_set is not None and seed not in seed_filter_set:
                continue
            # Cell must have at least the curated mono.png + poe.png.
            if not (seed_dir / "mono.png").exists() or not (seed_dir / "poe.png").exists():
                print(f"[skip] {slug}/seed_{seed}: missing mono.png or poe.png")
                continue
            specs.append(CellSpec(
                pair_slug=slug, prompt_a=prompt_a, prompt_b=prompt_b,
                seed=seed, split=split,
            ))
    return specs


def _decoded_diff(curated_path: Path, regenerated_path: Path) -> float | None:
    """Mean absolute pixel diff (uint8 scale, 0–255). Returns None on mismatch shapes."""
    try:
        a = np.asarray(Image.open(curated_path).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(regenerated_path).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            return None
        return float(np.mean(np.abs(a - b)))
    except Exception:
        return None


@torch.no_grad()
def run_cell(
    spec: CellSpec,
    ctx: MethodCtx,
    *,
    out_root: Path,
    overwrite: bool,
) -> dict:
    cell_dir = out_root / spec.split / spec.pair_slug / f"seed_{spec.seed}"
    residuals_dir = cell_dir / "residuals"
    final_step = ctx.num_inference_steps - 1
    sentinel = residuals_dir / f"step_{final_step:03d}.pt"
    if sentinel.exists() and not overwrite:
        return {
            "pair_slug": spec.pair_slug, "seed": spec.seed, "split": spec.split,
            "status": "cached", "cell_dir": str(cell_dir),
        }

    ensure_dir(residuals_dir)
    cell = cell_for(spec.prompt_a, spec.prompt_b, spec.seed)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="literal")
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)

    seq_a, pool_a = emb["seq_a"], emb["pool_a"]
    seq_b, pool_b = emb["seq_b"], emb["pool_b"]
    seq_e, pool_e = emb["seq_e"], emb["pool_e"]   # e_∅

    # 4-branch concat order: A, B, J, ∅. Save this order in meta.json so
    # downstream loaders never have to guess.
    branch_order = ["a", "b", "j", "uncond"]
    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=cell.height, width=cell.width, batch_size=4,
            device=ctx.device, dtype=ctx.dtype,
        ),
    }

    # Save per-cell embeddings + initial latent once (constant across steps).
    save_dtype = torch.float16
    embeddings_payload = {
        "seq_a": seq_a.detach().to(save_dtype).cpu(),
        "pool_a": pool_a.detach().to(save_dtype).cpu(),
        "seq_b": seq_b.detach().to(save_dtype).cpu(),
        "pool_b": pool_b.detach().to(save_dtype).cpu(),
        "seq_j": seq_j.detach().to(save_dtype).cpu(),
        "pool_j": pool_j.detach().to(save_dtype).cpu(),
        "seq_uncond": seq_e.detach().to(save_dtype).cpu(),
        "pool_uncond": pool_e.detach().to(save_dtype).cpu(),
        "init_latents": init_latents.detach().to(save_dtype).cpu(),
        "euler_init_noise_sigma": float(euler_sigma),
    }
    torch.save(embeddings_payload, cell_dir / "embeddings.pt")

    scheduler = ctx.scheduler
    scheduler.set_timesteps(ctx.num_inference_steps)
    latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
    unet = ctx.models["unet"]

    # Track separate Mono trajectory in parallel for verification.
    # PoE walks at the eps_poe direction; Mono walks at the eps_j direction.
    mono_latents = latents.clone()
    timesteps_recorded: list[int] = []

    t0 = time.time()
    for step_index, timestep in enumerate(scheduler.timesteps):
        # ---- 4-branch UNet on PoE-trajectory x_t ----
        latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)

        # Compute the PoE step on the PoE trajectory.
        eps_a_g = guided_eps(eps_a_raw, eps_uncond, ctx.guidance_scale)
        eps_b_g = guided_eps(eps_b_raw, eps_uncond, ctx.guidance_scale)
        eps_p = poe_eps(eps_a_g, eps_b_g, eps_uncond)

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=ctx.device, dtype=ctx.dtype,
        )
        x0_poe = tweedie_mean(latents, alpha_bar_t, eps_p)

        # ---- Save the per-step training tensors ----
        payload = {
            "x_t": latents.detach().to(save_dtype).cpu(),
            "timestep": int(timestep.item()),
            "step_index": int(step_index),
            "eps_a_raw": eps_a_raw.detach().to(save_dtype).cpu(),
            "eps_b_raw": eps_b_raw.detach().to(save_dtype).cpu(),
            "eps_j_raw": eps_j_raw.detach().to(save_dtype).cpu(),
            "eps_uncond": eps_uncond.detach().to(save_dtype).cpu(),
        }
        torch.save(payload, residuals_dir / f"step_{step_index:03d}.pt")
        timesteps_recorded.append(int(timestep.item()))

        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0_poe, eps=eps_p,
        )

        # Mono branch advances along its own trajectory using J's guided eps.
        # Re-run the UNet with mono_latents as x_t, to keep the Mono path
        # numerically faithful (we don't reuse eps_j_raw above because it
        # was evaluated at the PoE-trajectory x_t, not Mono's).
        mono_input = scheduler.scale_model_input(
            mono_latents.repeat(2, 1, 1, 1), timestep,
        )
        mono_pe = torch.cat([seq_j, seq_e], dim=0)
        mono_pool = torch.cat([pool_j, pool_e], dim=0)
        mono_cond = {
            "text_embeds": mono_pool,
            "time_ids": add_time_ids(
                height=cell.height, width=cell.width, batch_size=2,
                device=ctx.device, dtype=ctx.dtype,
            ),
        }
        mono_noise = unet(
            mono_input, timestep, encoder_hidden_states=mono_pe,
            added_cond_kwargs=mono_cond, timestep_cond=None,
        ).sample
        eps_j_mono_raw, eps_uncond_mono = mono_noise.chunk(2)
        eps_j_mono = guided_eps(eps_j_mono_raw, eps_uncond_mono, ctx.guidance_scale)
        x0_mono = tweedie_mean(mono_latents, alpha_bar_t, eps_j_mono)
        mono_latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0_mono, eps=eps_j_mono,
        )

    elapsed = time.time() - t0

    # Decode and write final images.
    poe_image = decode_latents(ctx.models, latents).cpu()
    mono_image = decode_latents(ctx.models, mono_latents).cpu()
    poe_path = cell_dir / "poe.png"
    mono_path = cell_dir / "mono.png"
    write_decoded_image(poe_image, poe_path)
    write_decoded_image(mono_image, mono_path)

    # Verify against curated images (mean-abs pixel diff in uint8).
    curated_dir = DATASET_CELLS / spec.pair_slug / f"seed_{spec.seed}"
    poe_diff = _decoded_diff(curated_dir / "poe.png", poe_path)
    mono_diff = _decoded_diff(curated_dir / "mono.png", mono_path)

    meta = {
        "pair": [spec.prompt_a, spec.prompt_b],
        "pair_slug": spec.pair_slug,
        "seed": spec.seed,
        "split": spec.split,
        "joint_prompt": joint_prompt(spec.prompt_a, spec.prompt_b),
        "guidance_scale": float(ctx.guidance_scale),
        "num_inference_steps": int(ctx.num_inference_steps),
        "height": int(cell.height),
        "width": int(cell.width),
        "branch_order": branch_order,
        "timesteps": timesteps_recorded,
        "euler_init_noise_sigma": float(euler_sigma),
        "verification": {
            "regenerated_vs_curated_poe_mean_abs_pixel_diff": poe_diff,
            "regenerated_vs_curated_mono_mean_abs_pixel_diff": mono_diff,
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json(cell_dir / "meta.json", meta)

    return {
        "pair_slug": spec.pair_slug, "seed": spec.seed, "split": spec.split,
        "status": "built", "cell_dir": str(cell_dir),
        "elapsed_seconds": round(elapsed, 2),
        "poe_diff": poe_diff, "mono_diff": mono_diff,
    }


_BRANCH_ORDER = ["a", "b", "j", "uncond"]
_STEP_KEYS = [
    "x_t", "timestep", "step_index",
    "eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond",
]
_EMB_KEYS = [
    "seq_a", "pool_a", "seq_b", "pool_b",
    "seq_j", "pool_j", "seq_uncond", "pool_uncond",
    "init_latents", "euler_init_noise_sigma",
]


def _manifest_filename(cohort: str | None) -> str:
    """Cohort-scoped filename so two nodes don't race on a shared file."""
    return f"manifest_{cohort}.json" if cohort else "manifest_all.json"


def write_manifest(
    out_root: Path, results: list[dict], *, cohort: str | None,
) -> Path:
    manifest = {
        "version": 1,
        "cohort": cohort,
        "branch_order": _BRANCH_ORDER,
        "step_payload_keys": _STEP_KEYS,
        "embeddings_payload_keys": _EMB_KEYS,
        "splits": {
            "train": [r for r in results if r["split"] == "train"],
            "heldout": [r for r in results if r["split"] == "heldout"],
        },
        "totals": {
            "train": sum(1 for r in results if r["split"] == "train"),
            "heldout": sum(1 for r in results if r["split"] == "heldout"),
        },
    }
    return write_json(out_root / _manifest_filename(cohort), manifest)


def _cell_exists_on_disk(out_root: Path, split: str, pair_slug: str, seed: int) -> bool:
    """A cell counts as on-disk only if both meta.json and the final step
    file exist — i.e. a complete build, not a half-finished one."""
    cell_dir = out_root / split / pair_slug / f"seed_{seed}"
    if not (cell_dir / "meta.json").exists():
        return False
    # Final step file (whatever num_steps was) — read meta to find it.
    try:
        meta = json.loads((cell_dir / "meta.json").read_text())
        n = int(meta.get("num_inference_steps", 50))
    except Exception:
        n = 50
    return (cell_dir / "residuals" / f"step_{n - 1:03d}.pt").exists()


def consolidate_manifests(out_root: Path) -> Path:
    """Merge every ``manifest_*.json`` under ``out_root`` into ``manifest.json``.

    De-duplicates on (split, pair_slug, seed). **Drops entries whose cell
    directory no longer exists on disk** — so deleted/replaced cells are
    not carried forward by stale cohort manifests. Cohort files themselves
    are left in place; the next cohort run rewrites them.
    """
    parts = sorted(out_root.glob("manifest_*.json"))
    if not parts:
        raise FileNotFoundError(f"no manifest_*.json files found in {out_root}")

    seen: dict[tuple[str, str, int], dict] = {}
    dropped_stale: list[tuple[str, str, int]] = []
    contributing_cohorts: list[str | None] = []
    for path in parts:
        data = json.loads(path.read_text())
        contributing_cohorts.append(data.get("cohort"))
        for split_results in data.get("splits", {}).values():
            for r in split_results:
                key = (r["split"], r["pair_slug"], int(r["seed"]))
                if not _cell_exists_on_disk(out_root, *key):
                    dropped_stale.append(key)
                    continue
                seen[key] = r

    merged = list(seen.values())
    out = {
        "version": 1,
        "consolidated_from": [str(p.name) for p in parts],
        "contributing_cohorts": contributing_cohorts,
        "dropped_stale_count": len(dropped_stale),
        "branch_order": _BRANCH_ORDER,
        "step_payload_keys": _STEP_KEYS,
        "embeddings_payload_keys": _EMB_KEYS,
        "splits": {
            "train": [r for r in merged if r["split"] == "train"],
            "heldout": [r for r in merged if r["split"] == "heldout"],
        },
        "totals": {
            "train": sum(1 for r in merged if r["split"] == "train"),
            "heldout": sum(1 for r in merged if r["split"] == "heldout"),
        },
    }
    path = write_json(out_root / "manifest.json", out)
    if dropped_stale:
        sample = sorted(dropped_stale)[:3]
        print(
            f"  Dropped {len(dropped_stale)} stale entries "
            f"(no longer on disk), e.g. {sample}"
        )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cohort", type=str, default=None, choices=sorted(COHORTS),
                    help="Predefined cohort to run (A, B, or H). "
                         "Mutually exclusive with --pairs.")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Restrict to these pair slugs (e.g. a_dog__x__a_horse). "
                         "Mutually exclusive with --cohort.")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Restrict to these seeds.")
    ap.add_argument("--held-out-pairs", nargs="*",
                    default=list(DEFAULT_HELD_OUT_PAIRS),
                    help="Pair slugs to mark held-out. Default: a_cat__x__a_dog.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="Override scheduler step count (default: project config).")
    ap.add_argument("--list-only", action="store_true",
                    help="Discover and print cells, then exit (no SDXL load).")
    ap.add_argument("--list-cohorts", action="store_true",
                    help="Print cohort definitions and exit.")
    ap.add_argument("--consolidate", action="store_true",
                    help="Merge all manifest_*.json under --output-root into "
                         "a single manifest.json. Run once after all nodes finish.")
    args = ap.parse_args()

    if args.list_cohorts:
        print("Predefined cohorts:")
        for name, pairs in COHORTS.items():
            print(f"  {name}: {len(pairs)} pair(s)")
            for p in pairs:
                print(f"      - {p}")
        return

    if args.consolidate:
        out = consolidate_manifests(args.output_root)
        print(f"Wrote consolidated manifest: {out}")
        return

    if args.cohort and args.pairs:
        ap.error("--cohort and --pairs are mutually exclusive; pass only one.")

    pair_filter: list[str] | None
    if args.cohort:
        pair_filter = list(COHORTS[args.cohort])
        print(f"Cohort {args.cohort}: {pair_filter}")
    else:
        pair_filter = args.pairs

    specs = discover_cells(
        DATASET_CELLS,
        pair_filter=pair_filter,
        seed_filter=args.seeds,
        held_out_pairs=tuple(args.held_out_pairs),
    )
    if not specs:
        print("No cells matched filters.")
        return

    print(f"Discovered {len(specs)} cells:")
    by_pair: dict[str, list[CellSpec]] = {}
    for s in specs:
        by_pair.setdefault(s.pair_slug, []).append(s)
    for slug in sorted(by_pair):
        items = by_pair[slug]
        seeds = ", ".join(str(s.seed) for s in items)
        split = items[0].split
        print(f"  [{split:7s}] {slug:30s} {len(items):2d} cells: {seeds}")

    if args.list_only:
        return

    print()
    print("Loading SDXL context...")
    ctx_kwargs = {"output_root": args.output_root.parent}
    if args.num_inference_steps is not None:
        ctx_kwargs["num_inference_steps"] = args.num_inference_steps
    ctx = make_ctx(**ctx_kwargs)
    print(f"  device={ctx.device} dtype={ctx.dtype} steps={ctx.num_inference_steps} "
          f"guidance={ctx.guidance_scale}")

    ensure_dir(args.output_root)
    results: list[dict] = []
    t_total = time.time()
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.split}/{spec.pair_slug}/seed_{spec.seed} ...",
              flush=True)
        try:
            result = run_cell(spec, ctx, out_root=args.output_root, overwrite=args.overwrite)
        except Exception as exc:
            result = {
                "pair_slug": spec.pair_slug, "seed": spec.seed, "split": spec.split,
                "status": "error", "error": str(exc),
            }
            print(f"  ERROR: {exc}")
        else:
            print(
                f"  {result['status']}"
                + (f"  ({result.get('elapsed_seconds', 0):.1f}s, "
                   f"poe_diff={result.get('poe_diff')}, "
                   f"mono_diff={result.get('mono_diff')})"
                   if result['status'] == 'built' else "")
            )
        results.append(result)

    manifest_path = write_manifest(args.output_root, results, cohort=args.cohort)
    print()
    print(f"Total elapsed: {time.time() - t_total:.1f}s")
    print(f"Manifest: {manifest_path}")
    if args.cohort:
        print("(cohort-scoped manifest; run --consolidate once all cohorts finish)")


if __name__ == "__main__":
    main()
