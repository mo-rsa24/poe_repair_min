"""Shared composer helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from poe_repair.config import joint_prompt
from poe_repair.run import MethodCtx
from poe_repair.runtime import (
    PairSeedCell,
    encode_prompt_sdxl,
    ensure_dir,
    load_shared_latents,
)


def encode_pair(cell: PairSeedCell, ctx: MethodCtx) -> dict[str, torch.Tensor]:
    """Encode (e_A, e_B, e_∅) once for a cell."""
    enc = lambda p: encode_prompt_sdxl(
        p, models=ctx.models, device=ctx.device, dtype=ctx.dtype
    )
    seq_e, pool_e = enc("")
    seq_a, pool_a = enc(cell.prompt_a)
    seq_b, pool_b = enc(cell.prompt_b)
    return {
        "seq_a": seq_a, "pool_a": pool_a,
        "seq_b": seq_b, "pool_b": pool_b,
        "seq_e": seq_e, "pool_e": pool_e,
    }


def joint_text_for(cell: PairSeedCell, ctx: MethodCtx) -> str:
    return joint_prompt(cell.prompt_a, cell.prompt_b, template=ctx.joint_template)


def get_joint_embeds(
    cell: PairSeedCell,
    ctx: MethodCtx,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(seq_J, pool_J)`` for the literal joint embedding.

    encode_prompt_sdxl(joint_prompt(...)) — what SDXL would compute for
    "a cat and a dog" directly.
    """
    return encode_prompt_sdxl(
        joint_text_for(cell, ctx),
        models=ctx.models, device=ctx.device, dtype=ctx.dtype,
    )


def init_latents_for_cell(
    cell: PairSeedCell, ctx: MethodCtx,
) -> tuple[torch.Tensor, float]:
    """Resolve x_T from disk if available, else from-seed fallback."""
    return load_shared_latents(
        pair_dir=cell.pair_dir, seed=cell.seed,
        device=ctx.device, dtype=ctx.dtype, models=ctx.models,
        height=cell.height, width=cell.width,
    )


def cell_output_dir(
    ctx: MethodCtx, exp_name: str, method_name: str, cell: PairSeedCell,
) -> Path:
    return ensure_dir(
        ctx.output_root / exp_name / "pairs" / cell.pair_slug
        / f"seed_{cell.seed}" / method_name
    )


def compute_token_indices(
    prompt_a: str,
    prompt_b: str,
    tokenizer,
    template: str = "{a} and {b}",
) -> tuple[int, int, int]:
    """Find subject token positions in the joint encoding + the real-token count.

    Returns (token_index_a, token_index_b, text_token_count). Falls back to
    ``(2, 5, 8)`` for the standard "a X and a Y" pattern if anything fails.
    """
    joint = template.format(a=prompt_a, b=prompt_b)
    try:
        joint_ids = tokenizer(joint, add_special_tokens=True)["input_ids"]
        a_ids = tokenizer(prompt_a, add_special_tokens=False)["input_ids"]
        b_ids = tokenizer(prompt_b, add_special_tokens=False)["input_ids"]

        def find_run(haystack, needle):
            for i in range(len(haystack) - len(needle) + 1):
                if haystack[i: i + len(needle)] == needle:
                    return i
            return -1

        i_a = find_run(joint_ids, a_ids)
        i_b = find_run(joint_ids, b_ids)
        if i_a < 0 or i_b < 0:
            return 2, 5, 8

        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            text_count = len(joint_ids)
        else:
            try:
                text_count = joint_ids.index(eos) + 1
            except ValueError:
                text_count = len(joint_ids)
        return i_a + len(a_ids) - 1, i_b + len(b_ids) - 1, text_count
    except Exception:
        return 2, 5, 8
