"""CO3 composer — direct call into ``composition/debottam_co3/composers/Co3``.

CO3 (Dutta et al., ICLR 2026, arXiv 2509.25940) is SDXL-native and ships
a self-contained class that loads its own SDXL pipeline, scheduler, and
attention machinery. We do NOT port CO3 — we call their class with the
inputs they expect:

    prompt_orig = "a cat and a dog"        # the joint prompt
    prompt      = "a cat+a dog"            # per-concept, '+' separated
    seeds       = [seed]
    output_path = <our output dir>

Their `run_sampling()` does seeding, denoising, decoding, and returns a
PIL image. We save that image at our composer's output path. Configuration
defaults are inherited from `Co3Config`: ``corrector_algo='co3-hybrid'``,
``num_ts_to_correct=1``, ``num_resampling_steps=4``,
``num_latent_corrector_steps=10``, ``use_cfgpp=True``,
``guidance_scale=0.8``, ``lmda=0.8``.

CO3 loads SDXL itself (it does not accept an external UNet), so this
composer instantiates a fresh CO3 pipeline per cell. Memory cost is
~7 GB on top of our existing SDXL load. Cached after first call within
a single `tier_eval_*` invocation if the caller passes a shared object.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from poe_repair.composers._helpers import (
    AnchorSource,
    cell_output_dir,
    joint_text_for,
)
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell


METHOD_NAME_BY_SOURCE = {
    "literal": "co3_literal",
    # CO3's pipeline encodes the joint prompt itself. To run with ê_J we
    # would have to monkey-patch their `prepare_embeds`. Defer that until
    # the synthesizer-improvement phase.
    "synth": "co3_synth_NOT_YET_SUPPORTED",
}


# Singleton cache: CO3 loads SDXL itself, so we keep one instance alive across
# many cells inside a single experiment run.
_CO3_INSTANCE = None
_CO3_LOADED_PROMPT_KEY: tuple | None = None


def _ensure_composers_path():
    co3_root = Path(__file__).resolve().parent.parent.parent / "composition" / "debottam_co3"
    co3_str = str(co3_root)
    if co3_str not in sys.path:
        sys.path.insert(0, co3_str)


def _get_co3_class():
    _ensure_composers_path()
    from composers.Co3 import Co3 as Co3Class
    from composers.config import Co3Config
    from composers.utils_custom import seed_everything
    return Co3Class, Co3Config, seed_everything


def _build_co3_config(
    *, prompt_orig: str, prompt_concat: str, seed: int, output_dir: Path,
    overrides: dict | None = None,
):
    """Build a Co3Config matching their published defaults, with optional overrides
    for any algorithmic knob (corrector_algo, num_latent_corrector_steps,
    num_resampling_steps, num_ts_to_correct, lmda, use_cfgpp, guidance_scale, ...).
    """
    _, Co3Config, _ = _get_co3_class()
    cfg = Co3Config(
        prompt=prompt_concat,
        prompt_orig=prompt_orig,
        concept="",
        negative_prompt="",
        seeds=[int(seed)],
        device="cuda:0",
        output_path=str(output_dir),
        output_path_all=str(output_dir),
    )
    if overrides:
        cfg.update(**overrides)
    return cfg


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    anchor_source: AnchorSource = "literal",
    co3_overrides: dict | None = None,
    method_name_suffix: str = "",
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    if anchor_source != "literal":
        raise NotImplementedError(
            "CO3 + ê_J is not yet wired (their pipeline encodes the joint "
            "prompt internally; injecting ê_J requires monkey-patching their "
            "prepare_embeds). Defer until synthesiser-improvement phase."
        )
    method_name = METHOD_NAME_BY_SOURCE[anchor_source] + method_name_suffix
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    Co3Class, _, seed_everything = _get_co3_class()
    prompt_orig = joint_text_for(cell, ctx)
    prompt_concat = f"{cell.prompt_a}+{cell.prompt_b}"
    config = _build_co3_config(
        prompt_orig=prompt_orig,
        prompt_concat=prompt_concat,
        seed=cell.seed,
        output_dir=out_dir,
        overrides=co3_overrides,
    )

    # CO3 owns its own SDXL pipeline. Build it once per (prompt, prompt_concat)
    # — the embeddings are baked in at __init__, so we rebuild only when
    # prompts change. ALL OTHER fields of `config` (overrides like
    # guidance_scale, use_cfgpp, num_ts_to_correct, lmda, ...) must be
    # re-applied every call, otherwise overrides silently get dropped.
    global _CO3_INSTANCE, _CO3_LOADED_PROMPT_KEY
    key = (prompt_orig, prompt_concat)

    # Carry over the dynamic `latent_corrector_ts` attribute (CO3 sets it
    # in __init__ but it isn't declared on Co3Config, so a fresh config
    # lacks it). Build it from scratch if no instance exists yet.
    def _ensure_latent_corrector_ts(cfg, scheduler):
        if hasattr(cfg, "latent_corrector_ts"):
            return
        cfg.latent_corrector_ts = []
        if getattr(cfg, "perform_latent_correction", False):
            if getattr(cfg, "latent_corrector_type", None) == "Co3Corrector":
                t_cond = getattr(cfg, "num_ts_to_correct", 0)
                cfg.latent_corrector_ts = (
                    scheduler.timesteps[:t_cond] if t_cond >= 0 else []
                )

    if _CO3_INSTANCE is None:
        _CO3_INSTANCE = Co3Class(config)
        _CO3_LOADED_PROMPT_KEY = key
    else:
        existing_lct = getattr(_CO3_INSTANCE.config, "latent_corrector_ts", None)
        if existing_lct is not None:
            config.latent_corrector_ts = existing_lct
        else:
            _ensure_latent_corrector_ts(config, _CO3_INSTANCE.scheduler)
        # ALWAYS apply the new config so per-call overrides take effect.
        _CO3_INSTANCE.config = config
        if _CO3_LOADED_PROMPT_KEY != key:
            _CO3_INSTANCE.prepare_prompts(config)
            _CO3_INSTANCE.prepare_embeds()
            _CO3_LOADED_PROMPT_KEY = key
    _CO3_INSTANCE.config.seed = int(cell.seed)
    _CO3_INSTANCE.config.seeds = [int(cell.seed)]

    seed_everything(int(cell.seed))
    _CO3_INSTANCE.config.seed = int(cell.seed)
    images = _CO3_INSTANCE.run_sampling()  # returns list of PIL images
    images[0].save(str(image_path))
    return image_path
