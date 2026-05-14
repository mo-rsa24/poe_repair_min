"""Method 2 composer — PoE plus learned residual-prompt correction.

At inference, the trained ``p*`` (loaded from a separate checkpoint than
the Method 1 ê_J) is fed as a fourth UNet branch. Its raw output is
added to the guided PoE score under a λ_t schedule:

    ε_t = ε̃_PoE + λ_t · ε(x_t, t, p*)

The trained ``p*`` lives at ``checkpoints/residual_prompt/<output_name>/best.pt``,
distinct from Method 1's synthesizer at ``checkpoints/synthesizer_distilled/...``.
Pass the path via ``pstar_ckpt`` or set ``POE_REPAIR_PSTAR_CKPT``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from poe_repair.composers._helpers import (
    cell_output_dir,
    encode_pair,
    init_latents_for_cell,
)
from poe_repair.embeddings.infer import load_synthesizer, synthesize_joint
from poe_repair.methods._sampling import (
    run_residual_prompt_inject,
    write_decoded_image,
)
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, write_json


METHOD_NAME = "residual_prompt"


def build_rect_schedule(
    num_steps: int, window_frac: float = 0.4, lambda_max: float = 1.0,
) -> torch.Tensor:
    """Phase-11 rectangular schedule, mirroring sched-M2."""
    sched = torch.zeros(num_steps)
    sched[: int(round(window_frac * num_steps))] = float(lambda_max)
    return sched


@lru_cache(maxsize=4)
def _cached_synth(ckpt_path: str, device_str: str):
    return load_synthesizer(
        Path(ckpt_path),
        device=torch.device(device_str),
        dtype=torch.float32,
    )


def _resolve_pstar_ckpt(pstar_ckpt: str | Path | None) -> Path:
    if pstar_ckpt is not None:
        return Path(pstar_ckpt)
    env = os.environ.get("POE_REPAIR_PSTAR_CKPT")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "checkpoints" / "residual_prompt" / "residual_mlp_pstar_v2" / "best.pt"


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    window_frac: float = 0.4,
    lambda_max: float = 1.0,
    pstar_ckpt: str | Path | None = None,
    correction_max_rel_norm: float | None = None,
    exp_name: str = "tmp",
    overwrite: bool = False,
) -> Path:
    out_dir = cell_output_dir(ctx, exp_name, METHOD_NAME, cell)
    image_path = out_dir / f"{METHOD_NAME}.png"
    summary_path = out_dir / f"summary_{METHOD_NAME}.json"
    if image_path.exists() and not overwrite:
        return image_path

    ckpt_path = _resolve_pstar_ckpt(pstar_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"residual_prompt checkpoint not found at {ckpt_path}. "
            "Train it first with `python -m poe_repair.embeddings.train_residual_prompt`."
        )
    synth = _cached_synth(str(ckpt_path), str(ctx.device))

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)

    pstar = synthesize_joint(
        synth,
        prompt_a=cell.prompt_a, prompt_b=cell.prompt_b,
        models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )
    seq_p = pstar.seq.to(ctx.device, ctx.dtype)
    pool_p = pstar.pooled.to(ctx.device, ctx.dtype)

    schedule = build_rect_schedule(ctx.num_inference_steps, window_frac, lambda_max)

    out = run_residual_prompt_inject(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_p=seq_p, pool_p=pool_p,
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        lambda_schedule=schedule,
        correction_max_rel_norm=correction_max_rel_norm,
    )
    write_decoded_image(out.image, image_path)
    write_json(
        summary_path,
        {
            "method": METHOD_NAME,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "image_path": str(image_path),
            "pair": [cell.prompt_a, cell.prompt_b],
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "window_frac": float(window_frac),
            "lambda_max": float(lambda_max),
            "pstar_ckpt": str(ckpt_path),
            "correction_max_rel_norm": (
                None if correction_max_rel_norm is None
                else float(correction_max_rel_norm)
            ),
            **out.extras,
        },
    )
    return image_path
