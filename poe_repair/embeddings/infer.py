"""Inference helpers for the trained embedding synthesizer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from poe_repair.config import RunConfig
from poe_repair.embeddings.synthesizer import SynthesizerOutputs, build_synthesizer
from poe_repair.runtime import encode_prompt_sdxl


def load_synthesizer(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    arch_override: Optional[str] = None,
) -> torch.nn.Module:
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    arch = arch_override or state.get("arch") or "residual_mlp"
    cfg = RunConfig().synth
    synth = build_synthesizer(
        arch,
        seq_dim=cfg.seq_dim,
        pooled_dim=cfg.pooled_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device=device, dtype=dtype)
    # Trainers saved the weights under different top-level keys over time:
    # the original synthesizer trainer used "state_dict"; the residual-prompt
    # trainer (Method 2a) uses "synth". Accept either; fall back to a bare
    # state_dict if neither key is present.
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "synth" in state:
        sd = state["synth"]
    else:
        sd = state
    synth.load_state_dict(sd)
    synth.eval()
    return synth


@torch.no_grad()
def synthesize_joint(
    synth: torch.nn.Module,
    *,
    prompt_a: str,
    prompt_b: str,
    models: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> SynthesizerOutputs:
    """Run the synthesizer end-to-end to produce ê_J from (c_1, c_2)."""
    seq_a, pool_a = encode_prompt_sdxl(prompt_a, models=models, device=device, dtype=dtype)
    seq_b, pool_b = encode_prompt_sdxl(prompt_b, models=models, device=device, dtype=dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
    synth_dtype = next(synth.parameters()).dtype
    out = synth(
        seq_a=seq_a.to(device=device, dtype=synth_dtype),
        seq_b=seq_b.to(device=device, dtype=synth_dtype),
        seq_e=seq_e.to(device=device, dtype=synth_dtype),
        pool_a=pool_a.to(device=device, dtype=synth_dtype),
        pool_b=pool_b.to(device=device, dtype=synth_dtype),
        pool_e=pool_e.to(device=device, dtype=synth_dtype),
    )
    return SynthesizerOutputs(
        seq=out.seq.to(device=device, dtype=dtype),
        pooled=out.pooled.to(device=device, dtype=dtype),
    )
