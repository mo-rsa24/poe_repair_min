"""Generate the three reference anchors per validation pair (compose-scorer plan 01).

Per pair the scorer measures each output against three anchors:
  A-alone, B-alone, and the joint "A and B" through the good composer (Mono).
"Mono" = single-prompt CFG on the joint prompt, the same good-composer definition
that produced the existing cat×dog ``mono.png``. So every anchor here is a plain
``run_cfg`` call, and A-alone / B-alone / joint all share the SAME pinned init
latent per pair (fair distance comparison).

cat×dog: joint already exists on disk as ``mono.png`` (seed 42). We regenerate
cat-alone and dog-alone on the same pinned seed-42 latent, and re-emit the joint
into the anchors dir for a self-contained set.
wolf×husky: all three generated fresh on the pinned seed-9 latent.

Run (co3 env, local GPU):
  PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
  $PY -m poe_repair.experiments.compose_scorer_validation.gen_anchors
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from poe_repair.runtime import (
    load_sdxl_models,
    load_ddim_scheduler,
    infer_device,
    infer_dtype,
)
from poe_repair._sdxl.runtime import encode_prompt_sdxl
from poe_repair.methods._sampling import run_cfg, write_decoded_image
from poe_repair.training_cache import CellPath
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents

log = logging.getLogger("compose_scorer.gen_anchors")

REPO = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(
    "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
)
OUT_ROOT = REPO / "outputs" / "compose_scorer" / "anchors"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
GUIDANCE = 7.5
STEPS = 50
H = W = 1024
EULER_SIGMA = 1.0

# Each validation pair: which pinned cache cell (subset/pair/seed) to draw the
# shared init latent from, and the three anchor prompts.
PAIRS = [
    {
        "slug": "a_cat__x__a_dog",
        "cache_subset": "heldout",  # where seed_42 has step files
        "seed": 42,
        "prompt_a": "a cat",
        "prompt_b": "a dog",
        "joint_prompt": "a cat and a dog",
    },
    {
        "slug": "a_wolf__x__a_husky",
        "cache_subset": "heldout",
        "seed": 9,
        "prompt_a": "a wolf",
        "prompt_b": "a husky",
        "joint_prompt": "a wolf and a husky",
    },
]


def _cell_for(subset: str, slug: str, seed: int) -> CellPath:
    """from_root itself searches <root>/heldout/... and <root>/train/...,
    so we pass the base cache root; ``subset`` is documentation only."""
    return CellPath.from_root(slug, int(seed), cache_root=CACHE_ROOT)


def _gen_one(prompt, init, models, scheduler, device, dtype) -> torch.Tensor:
    seq_c, pool_c = encode_prompt_sdxl(prompt, models=models, device=device, dtype=dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
    out = run_cfg(
        init_latents=init, models=models, scheduler=scheduler,
        seq_cond=seq_c, pool_cond=pool_c, seq_e=seq_e, pool_e=pool_e,
        guidance_scale=GUIDANCE, num_inference_steps=STEPS,
        height=H, width=W, euler_init_noise_sigma=EULER_SIGMA,
        device=device, dtype=dtype,
    )
    return out.image


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    log.info("device=%s dtype=%s", device, dtype)
    models = load_sdxl_models(model_id=MODEL_ID, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(MODEL_ID)

    import json

    for pair in PAIRS:
        slug = pair["slug"]
        out_dir = OUT_ROOT / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        cell = _cell_for(pair["cache_subset"], slug, pair["seed"])
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype, euler_init_noise_sigma=EULER_SIGMA
        )
        log.info("pair=%s seed=%d init=%s", slug, pair["seed"], tuple(init.shape))

        anchors = {
            "a_alone": pair["prompt_a"],
            "b_alone": pair["prompt_b"],
            "joint": pair["joint_prompt"],
        }
        written = {}
        for key, prompt in anchors.items():
            img = _gen_one(prompt, init, models, scheduler, device, dtype)
            path = out_dir / f"anchor_{key}.png"
            write_decoded_image(img, path)
            written[key] = str(path)
            log.info("  wrote %s ('%s')", path.name, prompt)

        manifest = {
            "pair_slug": slug,
            "seed": pair["seed"],
            "cache_cell": str(cell.root if hasattr(cell, "root") else cell),
            "prompts": anchors,
            "anchors": written,
            "good_composer": "mono (single-prompt CFG on joint prompt)",
            "guidance_scale": GUIDANCE,
            "num_inference_steps": STEPS,
        }
        (out_dir / "anchors_manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("pair=%s anchors complete → %s", slug, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
