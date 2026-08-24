#!/usr/bin/env python
"""Read-only integrity check for the "kept" LoRA artifacts.

Load-tests the final checkpoint of each kept run dir (plus every suspect run)
with torch.load(weights_only=True), confirms LoRA tensor keys are present and
shapes are non-empty. Does NOT open every checkpoint (cost) — one headline
per run dir + all suspects. Prints a machine-readable table to stdout.

Nothing is written or deleted. Safe to re-run.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

REPO = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
DAT = Path("/datasets/mmolefe/poe_repair_min/outputs")

# (label, path, expected_note)  — headline ckpt per kept run + all suspects
TARGETS = [
    # --- lora (repo) ---
    ("lora/cat_dog final",           REPO / "outputs/lora/cat_dog/seed_42/results/checkpoints/lora_step_080000.pt", "flagship"),
    ("lora/cat_dog paper-headline",  REPO / "outputs/lora/cat_dog/seed_42/results/checkpoints/lora_step_062500.pt", "README headline"),
    ("lora/typewriter finished",     REPO / "outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__a_typewriter__x__a_cactus__seed42__r8__lr1e-04__20260520-133632/checkpoints/lora_step_080000.pt", "wag4z592 finished"),
    ("lora/typewriter false-start",  REPO / "outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__a_typewriter__x__a_cactus__seed42__r8__lr1e-04__20260520-131542/checkpoints/lora_step_000000.pt", "SUSPECT lu7g7svh step0"),
    ("lora/camel finished",          REPO / "outputs/lora/a_camel__x__a_desert_landscape/seed_42/lora_sdxl__a_camel__x__a_desert_landscape__seed42__r8__lr1e-04__20260520-131542/checkpoints/lora_step_080000.pt", "8p1spi5b finished"),
    # --- cross_seed (repo) ---
    ("xseed/task_b k01 final",       REPO / "outputs/cross_seed_lora_pooling/task_b_learning_curve/k01_pick1__ep1600/checkpoints/lora_step_080000.pt", "hbpotmnk finished"),
    ("xseed/task_c s9 final",        REPO / "outputs/cross_seed_lora_pooling/task_c_per_seed_ceiling/per_seed_s9__ep1600/checkpoints/lora_step_016510.pt", "SUSPECT d5b2706v sync-fatal"),
    # --- cross_pair (repo) ---
    ("xpair/all_groups main final",  REPO / "outputs/cross_pair_lora_pooling/all_groups/main/checkpoints/lora_step_030000.pt", "SUSPECT 0y9un0o4 died-early"),
    ("xpair/all_groups dryrun",      REPO / "outputs/cross_pair_lora_pooling/all_groups/dryrun/checkpoints/lora_step_000005.pt", "smoke"),
    ("xpair/within g6 final",        REPO / "outputs/cross_pair_lora_pooling/within_group/g6/main/checkpoints/lora_step_030000.pt", "SUSPECT ow1jo0xq train died-early"),
    # --- cross_seed seed banks (datasets root) ---
    ("D:xseed dog_oil final",        DAT / "cross_seed_lora_pooling/a_dog__x__oil_painting_style/task_b_learning_curve/k04__ep2000/checkpoints/lora_step_100000.pt", "seed bank"),
    ("D:xseed dolphin final",        DAT / "cross_seed_lora_pooling/a_dolphin__x__an_ocean_wave/task_b_learning_curve/k04__ep2000/checkpoints/lora_step_100000.pt", "seed bank"),
    ("D:xseed mailbox final",        DAT / "cross_seed_lora_pooling/a_mailbox__x__a_snowfield/task_b_learning_curve/k04__ep2000/checkpoints/lora_step_080000.pt", "seed bank (stops 80k)"),
    ("D:xseed typewriter final",     DAT / "cross_seed_lora_pooling/a_typewriter__x__a_cactus/task_b_learning_curve/k04__ep2000/checkpoints/lora_step_100000.pt", "seed bank"),
    ("D:xseed G6 resumed final",     DAT / "cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed/checkpoints/lora_step_100000.pt", "verdict ok"),
    ("D:xseed k04_ep200 final",      DAT / "cross_seed_lora_pooling/task_b_learning_curve/k04__ep200/checkpoints/lora_step_010000.pt", "short run"),
]


def find_lora_tensors(obj):
    """Return (state_dict_used, lora_key_list, sample_shape) or (None, [], None)."""
    sd = None
    if isinstance(obj, dict):
        for k in ("lora_state", "lora", "state_dict", "model", "lora_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
        if sd is None:
            # maybe obj itself is a flat tensor state_dict
            if all(isinstance(v, torch.Tensor) for v in obj.values()) and obj:
                sd = obj
    if sd is None:
        return None, [], None
    lora_keys = [k for k in sd if "lora" in k.lower()]
    sample = None
    for k in lora_keys:
        t = sd[k]
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            sample = tuple(t.shape)
            break
    return sd, lora_keys, sample


rows = []
for label, path, note in TARGETS:
    r = {"label": label, "path": str(path), "note": note}
    if not path.exists():
        r.update(status="MISSING", detail="file does not exist")
        rows.append(r); continue
    try:
        sz = path.stat().st_size
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd, lk, sample = find_lora_tensors(obj)
        top = list(obj.keys())[:6] if isinstance(obj, dict) else type(obj).__name__
        if sd is None:
            r.update(status="FAIL", detail=f"no state_dict found; top={top}", size_mb=round(sz/1e6, 1))
        elif not lk:
            r.update(status="FAIL", detail=f"loaded but 0 lora keys; {len(sd)} total keys; top={top}", size_mb=round(sz/1e6, 1))
        elif sample is None:
            r.update(status="FAIL", detail=f"{len(lk)} lora keys but all empty/non-tensor", size_mb=round(sz/1e6, 1))
        else:
            r.update(status="PASS", detail=f"{len(lk)} lora keys, sample shape {sample}", size_mb=round(sz/1e6, 1))
    except Exception as e:
        r.update(status="ERROR", detail=f"{type(e).__name__}: {e}")
    rows.append(r)

print(json.dumps(rows, indent=2))
n_pass = sum(1 for r in rows if r.get("status") == "PASS")
print(f"\n# {n_pass}/{len(rows)} PASS", file=sys.stderr)
