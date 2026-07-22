#!/usr/bin/env python
"""Read-only completeness check for training_cache + manifold_cache.

Per-cell contract (verified against manifest_all.json step/embeddings keys):
  <cell>/embeddings.pt, <cell>/meta.json, <cell>/residuals/step_000..049.pt (50)
Counts shards per cell; flags cells that deviate. Spot-opens a sample of
shards with torch.load. Nothing written or deleted.
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
import torch

CACHE = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
MANIFOLD = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/outputs/manifold_cache")
EXPECT_SHARDS = 50

def check_cell(cell: Path):
    problems = []
    if not (cell / "embeddings.pt").exists(): problems.append("no embeddings.pt")
    if not (cell / "meta.json").exists():     problems.append("no meta.json")
    rdir = cell / "residuals"
    n = len(list(rdir.glob("step_*.pt"))) if rdir.is_dir() else 0
    if n != EXPECT_SHARDS: problems.append(f"{n} shards (want {EXPECT_SHARDS})")
    return n, problems

summary = {}
all_cells = []
for split in ("train", "heldout"):
    sdir = CACHE / split
    if not sdir.is_dir():
        summary[split] = {"pairs": 0, "cells": 0, "bad": []}
        continue
    pairs = sorted([p for p in sdir.iterdir() if p.is_dir()])
    cells = []
    bad = []
    for p in pairs:
        for cell in sorted([c for c in p.iterdir() if c.is_dir() and c.name.startswith("seed_")]):
            cells.append(cell); all_cells.append(cell)
            n, probs = check_cell(cell)
            if probs:
                bad.append({"cell": str(cell.relative_to(CACHE)), "issues": probs})
    summary[split] = {"pairs": len(pairs), "cells": len(cells), "bad_count": len(bad), "bad": bad[:30]}

# spot-open a random sample of shards + embeddings
random.seed(0)
sample = random.sample(all_cells, min(12, len(all_cells)))
open_results = []
for cell in sample:
    rec = {"cell": str(cell.relative_to(CACHE))}
    try:
        emb = torch.load(cell / "embeddings.pt", map_location="cpu", weights_only=True)
        rec["embeddings_keys"] = len(emb) if isinstance(emb, dict) else "not-dict"
        shard = cell / "residuals" / "step_025.pt"
        if shard.exists():
            s = torch.load(shard, map_location="cpu", weights_only=True)
            rec["step025_keys"] = sorted(s.keys())[:4] if isinstance(s, dict) else "not-dict"
            rec["status"] = "OK"
        else:
            rec["status"] = "step_025 missing"
    except Exception as e:
        rec["status"] = f"ERROR {type(e).__name__}: {e}"
    open_results.append(rec)

# manifold_cache vs inventory.json
manifold = {"exists": MANIFOLD.is_dir()}
inv = MANIFOLD / "inventory.json"
if inv.exists():
    try:
        d = json.load(open(inv))
        manifold["inventory_top_keys"] = list(d)[:10] if isinstance(d, dict) else type(d).__name__
        manifold["subdirs_on_disk"] = sorted([p.name for p in MANIFOLD.iterdir() if p.is_dir()])
    except Exception as e:
        manifold["inventory_error"] = str(e)

print(json.dumps({"summary": summary, "spot_open": open_results, "manifold": manifold}, indent=2))
tot = sum(summary[s]["cells"] for s in summary)
badtot = sum(summary[s].get("bad_count", 0) for s in summary)
print(f"\n# training_cache: {tot} cells, {badtot} with issues", file=sys.stderr)
