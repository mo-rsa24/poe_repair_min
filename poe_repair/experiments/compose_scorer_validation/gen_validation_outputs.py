"""Generate the non-circular cat×dog COMPOSE-positive validation outputs.

The cat×dog joint anchor is the Mono compose at seed 42. Using that same image as
the compose test output would be circular (it IS the anchor). So we render fresh
cat×dog Mono composes at OTHER seeds (9, 10, 11) as the compose-positive outputs.

The rest of the validation set is already on disk:
  - cat×dog BLEND-negative: the vanilla-PoE poe.png (a fused cat-dog).
  - wolf×husky BLEND-negatives: the 4 LoRA-corrected sample_seed_09..12.png
    (the case that fools the eye — one animal wearing both coats).

Run (co3 env, GPU):
  $PY -m poe_repair.experiments.compose_scorer_validation.gen_validation_outputs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from poe_repair.runtime import (
    load_sdxl_models, load_ddim_scheduler, infer_device, infer_dtype,
)
from poe_repair._sdxl.runtime import encode_prompt_sdxl
from poe_repair.methods._sampling import run_cfg, write_decoded_image
from poe_repair.training_cache import CellPath
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair import paths

log = logging.getLogger("compose_scorer.gen_validation_outputs")

REPO = Path(__file__).resolve().parents[3]
CACHE_ROOT = paths.resolve(paths.TRAINING_CACHE)
OUT_DIR = REPO / "outputs" / "compose_scorer" / "validation_outputs" / "a_cat__x__a_dog_compose"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
GUIDANCE, STEPS, H, W, EULER_SIGMA = 7.5, 50, 1024, 1024, 1.0
JOINT_PROMPT = "a cat and a dog"
COMPOSE_SEEDS = [9, 10, 11]  # != anchor seed 42


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=MODEL_ID, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(MODEL_ID)
    seq_c, pool_c = encode_prompt_sdxl(JOINT_PROMPT, models=models, device=device, dtype=dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)

    written = []
    for s in COMPOSE_SEEDS:
        cell = CellPath.from_root("a_cat__x__a_dog", int(s), cache_root=CACHE_ROOT)
        init = load_pinned_init_latents(cell, device=device, dtype=dtype, euler_init_noise_sigma=EULER_SIGMA)
        out = run_cfg(
            init_latents=init, models=models, scheduler=scheduler,
            seq_cond=seq_c, pool_cond=pool_c, seq_e=seq_e, pool_e=pool_e,
            guidance_scale=GUIDANCE, num_inference_steps=STEPS,
            height=H, width=W, euler_init_noise_sigma=EULER_SIGMA,
            device=device, dtype=dtype,
        )
        path = OUT_DIR / f"compose_seed_{s:02d}.png"
        write_decoded_image(out.image, path)
        written.append({"seed": s, "png": str(path)})
        log.info("wrote %s ('%s' Mono)", path.name, JOINT_PROMPT)

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "pair_slug": "a_cat__x__a_dog", "prompt": JOINT_PROMPT,
        "good_composer": "mono (single-prompt CFG on joint)", "seeds": COMPOSE_SEEDS,
        "role": "compose-positive (non-circular: distinct from anchor seed 42)",
        "samples": written,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
