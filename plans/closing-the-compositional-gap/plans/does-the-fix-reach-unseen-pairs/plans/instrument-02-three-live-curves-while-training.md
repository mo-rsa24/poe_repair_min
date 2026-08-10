# 🔌 Three live curves, so a sweep can be read while it runs

Design only. Verdicts live in [../review/instrument-02-three-live-curves-while-training.md](../review/instrument-02-three-live-curves-while-training.md).

## What this asks, in one line
Put the scorer and the two direction measures inside the training loop, so every run reports
compose-rate, direction-cosine, and fraction-of-distance-reached live, and a floor result can be
split into 'the fix never arrived' versus 'it arrived and did not transfer'.

## Why this plan exists
The two axes that diagnose a null (see "Reading a result" in the master plan) have to
be logged live, or a 15-run unattended sweep produces numbers no one can interpret
until it finishes. This plan wires the scorer and both direction metrics into the eval
hook so every run reports compose-rate, direction-cosine, and distance-reached as it
trains. It is the one genuinely new build in the scope; the rest reuses existing paths.

## Description
Import the compose-scorer module and run it inside the training eval hook, so transfer
shows up live on W&B instead of only after a run finishes. Three separate curves per
eval: compose-rate (the scorer), direction-cosine (Task D), and
fraction-of-distance-reached. They stay separate so a floor result can be split into
delivery-null vs no-transfer. This must pass a green 1-epoch smoke before any 15-run
fan-out.

## Background
This is the single point the unattended sweep depends on. If the in-loop scorer is
wrong, the fan-out silently produces garbage compose-rates that look like a real null.
So the smoke gate here is load-bearing, not a formality.

## Purpose
Serves Objective 4 (Diagnose) and Definition-of-Done item 2. Unblocks the whole
unattended sweep (plans 03, 04).

## Goal
A green 1-epoch smoke run showing compose-rate, direction-cosine, and
fraction-of-distance-reached logging as three separate live W&B curves without error.

## Tasks
- [x] Wire the compose-scorer module into the eval hook (reuse the eval-crossbar /
  inline-sampling path in `cross_pair_lora_pooling`), computing a compose/blend label
  per held-out eval output.
- [x] Add the direction-cosine (Task D) computation: cosine of the current
  correction to the pool-mean correction, logged per eval. ✓ verified
  (`_inline_sampling.py::direction_metrics`, `build_pool_mean_cache`; logged per-cell as
  `eval/direction_cosine/{quadrant}/{pair}/seed_{NN}` + an `eval/direction_cosine/mean`
  aggregate from `train_pooled.py::_run_inline_sample`).
- [x] Add the fraction-of-distance-reached metric (toward the PoE→Mono target),
  logged per eval, so the ~40% plateau is visible live.`, logged as
  `eval/frac_distance_reached/{quadrant}/{pair}/seed_{NN}` + mean).
- [ ] Run a 1-epoch smoke and confirm all three metrics appear as separate W&B
  curves (wandb.log/Table hooks in `experiments/lora/main.py` +
  `cross_pair_lora_pooling/train_pooled.py`). Code is wired and import-clean; the smoke
  itself needs a GPU node, not run in this session.

## Engagement Instructions
GATE (unattended pass/fail): a 1-epoch smoke run completes without error (exit 0) and
compose-rate, direction-cosine, and fraction-of-distance-reached all appear as
separate logged series in the W&B run. A script asserts the run finished and the three
metric keys are present and non-empty.
STOP: if the eval hook errors or stalls on the smoke (import failure, scorer crash,
metric not logging) → HALT before the 15-run sweep. Do NOT start the fan-out until
this smoke is green. This is the single unattended-safety gate for the whole scope.

## Next

1. `/run-experiment` the 1-epoch smoke. Cost: one epoch on a GPU node. Buys: the review file's
   open question, and the green light without which the 15-run sweep may not start.
2. Answer the review question with the three W&B series named.

## Recommended skill
▶ `/run-experiment` ✅: drives the 1-epoch smoke and confirms the three curves. The
   wiring itself is custom code (in-loop scorer import); no skill authors it.
