"""What shape of adapter a checkpoint holds: which projections it covers and at what rank.

Every runner that attaches a saved adapter has to build the same config the training run used. When
a runner hardcodes it, a checkpoint of a different shape still loads without error and renders as
though the adapter were half absent: a cross-and-self checkpoint loaded into a cross-only attachment
matches 210 of its 420 modules and the other 210 are dropped in silence.

The checkpoint states its own shape. Each `lora_A` is `(rank, d_in)` and its key names the module it
belongs to, so both can be read off the file.
"""
from __future__ import annotations

from pathlib import Path

CROSS = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
SELF = ("attn1.to_q", "attn1.to_k", "attn1.to_v")


def from_state(state: dict) -> dict:
    """``target_modules``, ``rank`` and any per-module ``rank_pattern`` this state needs."""
    targets = tuple(t for t in CROSS + SELF if any(f".{t}." in k for k in state))
    ranks = {t: int(v.shape[0]) for t in targets
             for k, v in state.items() if f".{t}." in k and "lora_A" in k}
    if not ranks:
        raise ValueError("no lora_A tensors in this state: not an adapter checkpoint")
    base = max(ranks.values())
    pattern = {t: r for t, r in ranks.items() if r != base}
    return {"target_modules": targets, "rank": base, "alpha": base,
            "rank_pattern": pattern or None, "n_tensors": len(state)}


def attach(unet, checkpoint: Path, *, adapter_name: str = "lora", key: str = "lora_state",
           disable: bool = False) -> dict:
    """Attach an adapter of the shape this checkpoint holds and load it. Returns what it did."""
    import torch
    from types import SimpleNamespace
    from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get(key)
    if state is None:
        raise KeyError(f"{checkpoint} has no {key!r} (found: {list(ckpt)})")
    shape = from_state(state)
    cfg = LoRAConfig(rank=shape["rank"], alpha=shape["alpha"], dropout=0.0,
                     target_modules=shape["target_modules"], init="gaussian",
                     adapter_name=adapter_name, rank_pattern=shape["rank_pattern"],
                     alpha_pattern=shape["rank_pattern"])
    info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=cfg))
    n_loaded = lora_trainer.load_lora_state(unet, state)
    if disable:
        unet.disable_adapters()
    if int(info["n_matched"]) * 2 != len(state):
        raise RuntimeError(
            f"{checkpoint}: attached {info['n_matched']} modules for {len(state)} saved tensors; "
            "the adapter would render as if it were half absent")
    return {**shape, "n_matched": info["n_matched"], "n_loaded": n_loaded,
            "checkpoint": str(checkpoint), "checkpoint_step": int(ckpt.get("step", -1))}
