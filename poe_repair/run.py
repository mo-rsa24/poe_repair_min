"""Single dispatcher for all sampling methods.

Each method maps to one of the legacy samplers in `methods/_sampling.py`
plus the right embeddings. `run_method(name, cell, ctx)` encapsulates the
per-cell boilerplate: encode prompts, load latents, sample, write PNG +
summary JSON. Idempotent — returns the existing PNG if already present.

Recognised methods:
    "solo_a"     — single CFG branch on cell.prompt_a
    "solo_b"     — single CFG branch on cell.prompt_b
    "poe"        — vanilla PoE
    "mono"       — single CFG branch on the literal joint embedding e_J
    "c_poe"      — conflict-angle damped PoE (gamma)
    "m2_replace" — single CFG branch on the synthesised joint ê_J
    "m2_c_poe"   — combined (alpha, lambda_j)

`solo_a`, `solo_b`, `mono`, and `m2_replace` all reuse `run_m2_replace`
under the hood — they are single-branch CFG with different conditioning
embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from poe_repair.config import RunConfig, joint_prompt
from poe_repair.methods._sampling import (
    initial_latents_for_pair,
    run_c_poe,
    run_m2_c_poe,
    run_m2_replace,
    run_vanilla_poe,
    write_decoded_image,
)
from poe_repair.runtime import (
    PairSeedCell,
    encode_prompt_sdxl,
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)


@dataclass
class MethodCtx:
    """Shared context across many `run_method` calls.

    Holds SDXL models + scheduler so they are loaded once across cells and
    methods. Lazily loads the M2 synthesizer when first requested.
    """

    models: dict
    scheduler: object
    output_root: Path
    device: torch.device
    dtype: torch.dtype
    guidance_scale: float
    num_inference_steps: int
    joint_template: str = "{a} and {b}"
    synth: object | None = None

    def get_synth(self):
        if self.synth is None:
            from poe_repair.embeddings.infer import load_synthesizer

            cfg = RunConfig()
            self.synth = load_synthesizer(
                cfg.paths.synthesizer_checkpoint,
                device=self.device,
                dtype=torch.float32,
            )
        return self.synth


def make_ctx(
    *,
    output_root: Path | None = None,
    model_id: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    joint_template: str | None = None,
) -> MethodCtx:
    """Build a `MethodCtx` from `RunConfig` defaults; loads SDXL once."""
    cfg = RunConfig()
    device = device or infer_device(None)
    dtype = dtype or infer_dtype(cfg.dtype, device)
    model_id = model_id or cfg.model_id
    return MethodCtx(
        models=load_sdxl_models(model_id=model_id, device=device, dtype=dtype),
        scheduler=load_ddim_scheduler(model_id),
        output_root=output_root or cfg.paths.output_root,
        device=device,
        dtype=dtype,
        guidance_scale=guidance_scale or cfg.guidance,
        num_inference_steps=num_inference_steps or cfg.num_inference_steps,
        joint_template=joint_template or cfg.synth.joint_template,
    )


def _encode(prompt: str, ctx: MethodCtx):
    return encode_prompt_sdxl(prompt, models=ctx.models, device=ctx.device, dtype=ctx.dtype)


def run_method(
    name: str,
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    gamma: float = 2.0,
    lambda_j: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Run `name` on `cell`, save PNG + summary, return image path."""
    cell_dir = ensure_dir(
        ctx.output_root / name / "pairs" / cell.pair_slug / f"seed_{cell.seed}"
    )
    image_path = cell_dir / f"{name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    init_latents, euler_sigma = initial_latents_for_pair(
        cell=cell, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    seq_e, pool_e = _encode("", ctx)
    common = dict(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_e=seq_e,
        pool_e=pool_e,
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height,
        width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    t0 = time.time()
    if name == "solo_a":
        seq, pool = _encode(cell.prompt_a, ctx)
        out = run_m2_replace(seq_j=seq, pool_j=pool, **common)
    elif name == "solo_b":
        seq, pool = _encode(cell.prompt_b, ctx)
        out = run_m2_replace(seq_j=seq, pool_j=pool, **common)
    elif name == "poe":
        seq_a, pool_a = _encode(cell.prompt_a, ctx)
        seq_b, pool_b = _encode(cell.prompt_b, ctx)
        out = run_vanilla_poe(
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, **common
        )
    elif name == "mono":
        joint = joint_prompt(cell.prompt_a, cell.prompt_b, template=ctx.joint_template)
        seq_j, pool_j = _encode(joint, ctx)
        out = run_m2_replace(seq_j=seq_j, pool_j=pool_j, **common)
    elif name == "c_poe":
        seq_a, pool_a = _encode(cell.prompt_a, ctx)
        seq_b, pool_b = _encode(cell.prompt_b, ctx)
        out = run_c_poe(
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
            gamma=gamma, **common,
        )
    elif name == "m2_replace":
        from poe_repair.embeddings.infer import synthesize_joint

        sout = synthesize_joint(
            ctx.get_synth(),
            prompt_a=cell.prompt_a, prompt_b=cell.prompt_b,
            models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        out = run_m2_replace(
            seq_j=sout.seq.to(ctx.device, ctx.dtype),
            pool_j=sout.pooled.to(ctx.device, ctx.dtype),
            **common,
        )
    elif name == "m2_c_poe":
        from poe_repair.embeddings.infer import synthesize_joint

        seq_a, pool_a = _encode(cell.prompt_a, ctx)
        seq_b, pool_b = _encode(cell.prompt_b, ctx)
        sout = synthesize_joint(
            ctx.get_synth(),
            prompt_a=cell.prompt_a, prompt_b=cell.prompt_b,
            models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        out = run_m2_c_poe(
            seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
            seq_j=sout.seq.to(ctx.device, ctx.dtype),
            pool_j=sout.pooled.to(ctx.device, ctx.dtype),
            gamma=gamma, lambda_j=lambda_j, **common,
        )
    else:
        raise ValueError(
            f"unknown method {name!r}; expected one of "
            "{'solo_a','solo_b','poe','mono','c_poe','m2_replace','m2_c_poe'}"
        )

    write_decoded_image(out.image, image_path)
    write_json(
        cell_dir / f"summary_{name}.json",
        {
            "method": name,
            "pair_slug": cell.pair_slug,
            "seed": cell.seed,
            "image_path": str(image_path),
            "elapsed_s": time.time() - t0,
            "guidance_scale": ctx.guidance_scale,
            "num_inference_steps": ctx.num_inference_steps,
            "pair": [cell.prompt_a, cell.prompt_b],
            **out.extras,
        },
    )
    print(f"[{name}] {cell.pair_slug} seed={cell.seed} -> {image_path.name}")
    return image_path
