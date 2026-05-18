"""Render Mono+PoE for candidate seeds so you can pick replacements.

Use when you want to swap a "bad" seed in ``dataset/cells/<pair>/`` for a
new one whose Mono is clean and whose PoE clearly fails. Generates only
the final images — fast to inspect, no trajectory cache.

Workflow::

    # 1. Generate previews for candidate seeds:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.preview_seeds \\
        --pair a_cat__x__a_horse --seeds 5 6 7 8 9 10

    # 2. Inspect outputs/seed_preview/a_cat__x__a_horse/seed_*/{mono,poe}.png
    #    Pick the seeds you like (where Mono is correct and PoE fails).

    # 3. Promote approved seeds into dataset/cells/ (writes sources.json):
    python -m scripts.preview_seeds \\
        --pair a_cat__x__a_horse --seeds 5 6 7 8 9 10 \\
        --promote 7 9

    # 4. Manually delete the old bad seeds (and their stale training cache):
    rm -rf dataset/cells/a_cat__x__a_horse/seed_1 \\
           dataset/cells/a_cat__x__a_horse/seed_2 \\
           outputs/training_cache/train/a_cat__x__a_horse/seed_1 \\
           outputs/training_cache/train/a_cat__x__a_horse/seed_2

    # 5. Re-run the trajectory cache for the new seeds:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.build_training_cache \\
        --pairs a_cat__x__a_horse

Output layout::

    outputs/seed_preview/<pair_slug>/seed_<N>/
      mono.png
      poe.png

When ``--promote`` is given, also writes::

    dataset/cells/<pair_slug>/seed_<N>/{mono.png, poe.png, sources.json}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from poe_repair.composers._helpers import (
    encode_pair,
    get_joint_embeds,
    init_latents_for_cell,
)
from poe_repair.experiments._eval_common import cell_for
from poe_repair.methods._sampling import (
    run_cfg,
    run_cfg_poe,
    write_decoded_image,
)
from poe_repair.run import make_ctx
from poe_repair.runtime import ensure_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CELLS = REPO_ROOT / "dataset" / "cells"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "seed_preview"


def _slug_to_prompts(slug: str) -> tuple[str, str]:
    if "__x__" not in slug:
        raise ValueError(f"slug {slug!r} does not contain '__x__'")
    a, b = slug.split("__x__", 1)
    return a.replace("_", " "), b.replace("_", " ")


def render_one(cell, ctx, out_dir: Path, *, overwrite: bool) -> tuple[Path, Path]:
    mono_path = out_dir / "mono.png"
    poe_path = out_dir / "poe.png"
    if mono_path.exists() and poe_path.exists() and not overwrite:
        return mono_path, poe_path

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j, _ = get_joint_embeds(cell, ctx, anchor_source="literal")
    seq_e, pool_e = emb["seq_e"], emb["pool_e"]

    # Mono (single guided branch on the joint embedding).
    mono_out = run_cfg(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_cond=seq_j, pool_cond=pool_j, seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    write_decoded_image(mono_out.image, mono_path)

    # Vanilla PoE.
    poe_out = run_cfg_poe(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )
    write_decoded_image(poe_out.image, poe_path)
    return mono_path, poe_path


def promote_seed(pair_slug: str, seed: int, preview_root: Path) -> Path | None:
    src = preview_root / pair_slug / f"seed_{seed}"
    if not (src / "mono.png").exists() or not (src / "poe.png").exists():
        print(f"  [promote] seed {seed}: preview missing at {src}; skip.")
        return None
    dst = DATASET_CELLS / pair_slug / f"seed_{seed}"
    if dst.exists():
        print(
            f"  [promote] seed {seed}: dataset/cells/{pair_slug}/seed_{seed} "
            f"already exists; not overwriting. Delete it first if you want to replace."
        )
        return None
    ensure_dir(dst)
    shutil.copy2(src / "mono.png", dst / "mono.png")
    shutil.copy2(src / "poe.png", dst / "poe.png")
    (dst / "sources.json").write_text(
        json.dumps(
            {"poe": str((dst / "poe.png").resolve()),
             "mono": str((dst / "mono.png").resolve())},
            indent=2,
        )
    )
    print(f"  [promote] seed {seed} -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pair", required=True,
                    help="Pair slug, e.g. a_cat__x__a_horse")
    ap.add_argument("--seeds", nargs="+", type=int, required=True,
                    help="Candidate seeds to render.")
    ap.add_argument("--preview-root", type=Path, default=DEFAULT_PREVIEW_ROOT)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render even if preview already exists.")
    ap.add_argument("--promote", nargs="+", type=int, default=None,
                    help="After rendering, copy these seeds into "
                         "dataset/cells/<pair>/seed_<N>/. Must be a subset of --seeds.")
    args = ap.parse_args()

    try:
        prompt_a, prompt_b = _slug_to_prompts(args.pair)
    except ValueError as e:
        ap.error(str(e))

    if args.promote:
        invalid = set(args.promote) - set(args.seeds)
        if invalid:
            ap.error(
                f"--promote seeds {sorted(invalid)} not in --seeds. "
                "Re-render first or include them in --seeds."
            )

    print(f"Pair: '{prompt_a}' x '{prompt_b}'  (slug: {args.pair})")
    print(f"Seeds: {args.seeds}")
    if args.promote:
        print(f"Promote: {args.promote}")

    print("Loading SDXL context...")
    ctx = make_ctx(output_root=args.preview_root)
    print(f"  device={ctx.device} dtype={ctx.dtype} steps={ctx.num_inference_steps}")

    for seed in args.seeds:
        out_dir = ensure_dir(args.preview_root / args.pair / f"seed_{seed}")
        cell = cell_for(prompt_a, prompt_b, seed)
        print(f"[seed {seed}] rendering...", flush=True)
        try:
            mono_path, poe_path = render_one(
                cell, ctx, out_dir, overwrite=args.overwrite,
            )
            print(f"  mono: {mono_path}")
            print(f"  poe : {poe_path}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if args.promote:
        print()
        print("Promoting approved seeds into dataset/cells/:")
        for seed in args.promote:
            promote_seed(args.pair, seed, args.preview_root)


if __name__ == "__main__":
    main()
