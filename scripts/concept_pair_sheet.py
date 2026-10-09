#!/usr/bin/env python
"""Two concepts alone, then three ways of drawing them together, seeds down the page.

Columns are concept A alone, concept B alone, a dotted rule, then the joint prompt "{a} and {b}",
plain product-of-experts (PoE) and SuperDiff. Rows are seeds. The two single-concept columns, the
joint prompt and PoE start a row from the same noise: the cached starting latent of
`a_cat__x__a_dog` at that seed, which the cache shares across pairs. SuperDiff draws its own noise
from the seed number and samples with its own stochastic sampler, so its tile is not
pixel-comparable with the rest of the row.

PoE is the repo's plain rule (poe_repair/methods/_sampling.py, run_cfg_poe) with no adapter
attached at all, so nothing can leak into it from a trained correction.

    python scripts/concept_pair_sheet.py --column single    --pairs "a dog|rain" --seeds 1 2 4 9 10
    python scripts/concept_pair_sheet.py --column mono      --pairs "a dog|rain" --seeds 1 2 4 9 10
    python scripts/concept_pair_sheet.py --column poe       --pairs "a dog|rain" --seeds 1 2 4 9 10
    python scripts/concept_pair_sheet.py --column superdiff --pairs "a dog|rain" --seeds 1 2 4 9 10
    python scripts/concept_pair_sheet.py --sheet <tiles_dir> <out_dir> --pairs "a dog|rain" --seeds 1 2 4 9 10

Tiles land as <OUT>/single/<concept>/seedNN.png and <OUT>/<pair>/seedNN/<column>.png, each with a
JSON sidecar; a tile that already exists is skipped.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

OUT = Path(os.environ.get("CONCEPT_PAIR_OUT",
                          "/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet"))
NOISE_PAIR = "a_cat__x__a_dog"          # whose cached starting latent every seed borrows
COLUMNS = ("single", "mono", "poe", "superdiff")


def parse_pair(s: str) -> tuple[str, str]:
    a, b = (p.strip() for p in s.split("|"))
    from disallowed_subjects import check
    check(a)
    check(b)
    return a, b


def concept_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def single_tile(root: Path, concept: str, seed: int) -> Path:
    return root / "single" / concept_slug(concept) / f"seed{seed:02d}.png"


def pair_tile(root: Path, slug: str, seed: int, column: str) -> Path:
    return root / slug / f"seed{seed:02d}" / f"{column}.png"


def write_sidecar(png: Path, record: dict) -> None:
    png.with_suffix(".json").write_text(json.dumps(record, indent=1))


def render(column: str, pairs: list[tuple[str, str]], seeds: list[int]) -> None:
    import torch

    from poe_repair.composers._helpers import encode_pair, get_joint_embeds, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image
    from poe_repair.run import MethodCtx, make_ctx

    if column == "superdiff":
        from poe_repair.composers import superdiff
        from poe_repair.config import RunConfig
        from poe_repair.runtime import infer_device, infer_dtype

        cfg = RunConfig()
        device = infer_device(None)
        # SuperDiff loads its own SDXL; the context only carries steps, guidance and the output root.
        ctx = MethodCtx(models={}, scheduler=None, output_root=OUT / "_superdiff_raw", device=device,
                        dtype=infer_dtype(cfg.dtype, device), guidance_scale=cfg.guidance,
                        num_inference_steps=cfg.num_inference_steps, joint_template=cfg.joint_template)
        for a, b in pairs:
            for seed in seeds:
                cell = cell_for(a, b, seed)
                out = pair_tile(OUT, cell.pair_slug, seed, "superdiff")
                if out.exists():
                    print(f"{cell.pair_slug} seed {seed} superdiff: already rendered", flush=True)
                    continue
                raw = superdiff.run(cell, ctx, exp_name="concept_pair_sheet")
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(raw, out)
                side = json.loads(raw.with_suffix(".json").read_text())
                write_sidecar(out, {"pair": cell.pair_slug, "prompt_a": a, "prompt_b": b, "seed": seed,
                                    "column": "superdiff", "noise_from": "own, cuda seed",
                                    "steps": ctx.num_inference_steps, "guidance": ctx.guidance_scale,
                                    "kappa_used": side.get("kappa_used"), "raw": str(raw)})
                print(f"{cell.pair_slug} seed {seed} superdiff: wrote {out}", flush=True)
        return

    ctx = make_ctx()
    common = dict(models=ctx.models, scheduler=ctx.scheduler, guidance_scale=ctx.guidance_scale,
                  num_inference_steps=ctx.num_inference_steps, device=ctx.device, dtype=ctx.dtype)
    for a, b in pairs:
        for seed in seeds:
            cell = cell_for(a, b, seed)
            init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
            emb = encode_pair(cell, ctx)
            jobs = []
            if column == "single":
                jobs = [(single_tile(OUT, a, seed), a, emb["seq_a"], emb["pool_a"]),
                        (single_tile(OUT, b, seed), b, emb["seq_b"], emb["pool_b"])]
            elif column == "mono":
                seq_j, pool_j = get_joint_embeds(cell, ctx)
                jobs = [(pair_tile(OUT, cell.pair_slug, seed, "mono"),
                         ctx.joint_template.format(a=a, b=b), seq_j, pool_j)]
            elif column == "poe":
                jobs = [(pair_tile(OUT, cell.pair_slug, seed, "poe"), None, None, None)]
            for out, prompt, seq, pool in jobs:
                if out.exists():
                    print(f"{out}: already rendered", flush=True)
                    continue
                if column == "poe":
                    res = run_cfg_poe(init_latents=init, seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                                      seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                                      seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                                      height=cell.height, width=cell.width,
                                      euler_init_noise_sigma=sigma, **common)
                else:
                    res = run_cfg(init_latents=init, seq_cond=seq, pool_cond=pool,
                                  seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                                  height=cell.height, width=cell.width,
                                  euler_init_noise_sigma=sigma, **common)
                out.parent.mkdir(parents=True, exist_ok=True)
                write_decoded_image(res.image, out)
                write_sidecar(out, {"pair": cell.pair_slug, "prompt_a": a, "prompt_b": b, "seed": seed,
                                    "column": column, "prompt": prompt, "noise_from": NOISE_PAIR,
                                    "steps": ctx.num_inference_steps, "guidance": ctx.guidance_scale,
                                    "size": [cell.width, cell.height]})
                del res
                gc.collect()
                torch.cuda.empty_cache()
                print(f"wrote {out}", flush=True)


def sheet(tiles: Path, out_dir: Path, pairs: list[tuple[str, str]], seeds: list[int]) -> None:
    from main_study_grids import grid

    out_dir.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        # The repo's slug rule (poe_repair/experiments/_eval_common.py), inlined so the sheet builds without torch.
        slug = f"{concept_slug(a)}__x__{concept_slug(b)}"

        def resolver(seed: int):
            def resolve(key: str):
                if key == "a":
                    return single_tile(tiles, a, seed)
                if key == "b":
                    return single_tile(tiles, b, seed)
                return pair_tile(tiles, slug, seed, key)
            return resolve

        columns = [("a", [f'"{a}"', "alone"]),
                   ("b", [f'"{b}"', "alone"]),
                   ("mono", ["Joint prompt", f'"{a} and {b}"']),
                   ("poe", ["PoE", "one prompt per concept"]),
                   ("superdiff", ["SuperDiff", "own noise, own sampler"])]
        rows = [(f"seed {s}", resolver(s)) for s in seeds]
        out = out_dir / f"{slug.replace('__x__', '-and-').replace('_', '-')}.png"
        record = grid(rows, columns, out, f'SDXL: "{a}" and "{b}", {len(seeds)} seeds',
                      "Rows are seeds. The first four columns start a row from the same cached noise, so "
                      "a difference along a row is the method. SuperDiff draws its own noise from the seed "
                      "number with its own stochastic sampler. 50 steps, guidance 7.5.",
                      rule_before=2)
        out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))
        print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", choices=COLUMNS)
    ap.add_argument("--pairs", nargs="+", required=True, help='each as "a dog|rain"')
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--sheet", nargs=2, type=Path, metavar=("TILES", "OUT_DIR"))
    args = ap.parse_args()
    pairs = [parse_pair(p) for p in args.pairs]
    if args.sheet:
        return sheet(*args.sheet, pairs, args.seeds)
    if not args.column:
        ap.error("give --column or --sheet")
    render(args.column, pairs, args.seeds)


if __name__ == "__main__":
    main()
