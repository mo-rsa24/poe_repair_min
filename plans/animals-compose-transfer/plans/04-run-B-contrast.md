# 🅱️ The control pool: was it the animals, or just the amount of data?

Design only. Verdicts live in [../review/04-run-B-contrast.md](../review/04-run-B-contrast.md).

## What this asks, in one line
Run the identical training on a same-size pool of non-animal concepts and evaluate on the same
held-out animal pairs: if the mixed pool does as well, the win was data volume, not the pool.

## Why this plan exists
A transfer win in plan 03 is worth little if any pool of the same size would have won.
This plan rules that out: it runs a same-size mixed pool on the identical held-out
animal pairs, so a win can be attributed to the animals pool specifically, not to how
much data there was.

## Description
Build a mixed pool the same size as the animals pool (~15), swapping the animal pairs
for scene/style/object concepts, then run it and evaluate on the SAME animal held-out
pairs used in (A). Same size kills the "more data" confound: a win means animals help,
not that there was more training signal.

## Purpose
Serves Objective 3 (Contrast B) and Definition-of-Done item 4.

## Goal
A size-matched mixed pool built and run, evaluated on the identical animal held-out
set as (A), with the animals-vs-mixed contrast reported per held-out pair.

## Tasks
- [ ] Build the size-matched mixed `pair_pool.yaml`: equal N to the animals pool,
  animal pairs swapped for scene/style/object concepts, overlap assertion passing.
- [ ] Run the mixed pool through the same training + wired-eval path as (A).
- [ ] Evaluate the mixed-pool LoRA on the SAME animal held-out pairs used in (A),
  two-tier read.
- [ ] Report the animals-vs-mixed contrast per held-out pair (compose-rate +
  direction), on the identical held-out set.

## Engagement Instructions
GATE (unattended pass/fail): the mixed `pair_pool.yaml` loads with the overlap
assertion passing AND its pair count equals the animals pool's; both pools are
evaluated on the identical animal held-out set (same pair slugs). A script asserts
equal N and identical held-out slugs.
STOP: if a size-matched mixed pool cannot be built at equal N → halt (B); (A) still
carries the scope. Per-run: same delivery-null stop as plan 03 (distance-reached at
floor past the commitment window → mark delivery-null, move on).

## Recommended skill
▶ `/run-experiment` ✅: drives the mixed-pool run and the same-held-out-set eval.
   alt: `/debug-config` for the size-matched pair_pool.yaml construction.
