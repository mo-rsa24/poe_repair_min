# 🎛️ Inspector tabs for this scope's results

## Description
Extend the LoRA Inspector with tabs that make this scope's finished figures
driveable: a dose slider over the λ-sweep images, a window slider for both
timing experiments with the curves highlighting the current window, the
manifold walk animated, per-step curves synced to the image being viewed.

## Purpose
The supervisor demo and the paper's interactive supplementary. Hard rule: the
UI consumes only grids this scope's experiments already produced; it never
generates. Serves DoD 11.

## Goal
The enhanced Inspector serving the new tabs off the finished result grids.

## Environment Facts This Plan Depends On
- Inspector serves on 127.0.0.1:5050 via scripts/run_lora_inspector.sh with an
  SSH tunnel from the laptop (the script prints the node to tunnel to).
- Consumes /datasets result grids from plans 03-08; runs in-session.

## Tasks
- [ ] ⚠️ build the tabs  → decomposed: see
      `inspector-interaction-term/MASTER_PLAN.md`

## Recommended skill
▶ custom build inside the sub-scope; `/demonstrate` ✅ as a screen recording of
   each tab driving real grids.

## Engagement Instructions
```bash
bash scripts/run_lora_inspector.sh   # serve; open http://localhost:5050
# expect: dose scrub, window scrub, manifold animation tabs, all reading
# /datasets grids only (no generation code paths in the new tabs)
```
