# 🧬 A clean pool, behind the scorer gate

Design only. Verdicts live in [../review/instrument-01-the-clean-pair-pool.md](../review/instrument-01-the-clean-pair-pool.md).

## What this asks, in one line
Build the set of animal pairs the whole scope runs on: no animal word repeated anywhere (so a
transfer win cannot be memorisation), every training pair proven to fail by default (so there is
something to fix), and none of it starts unless the validated scorer exists.

## Why this plan exists
Leave-one-pair-out is only a fair transfer test if no animal word repeats across
pairs. If "wolf" appeared in a training pair and the held-out pair, the model would
have seen the concept, so a compose would prove nothing. This plan builds a
token-disjoint pool and confirms each pair actually fails by default, so the test in
plan 03 starts from clean ground.

## Description
The pool the scope runs on, behind the gate that lets the scope start at all. First,
check the cross-scope precondition: the sibling compose-scorer must have written
`scorer_validated.json` with a passing verdict. Without it there is no trusted scorer,
so the scope halts. Then curate ~15 blend-prone animal×animal pairs where no animal
word repeats across pairs (~15 pairs ≈ ~30 distinct animals). Finalise `pair_pool.yaml`
by confirming each training pair fails by default, measured as a fail-rate over 8 seeds
by the compose-scorer, not by eye. Keep a few compose-by-default pairs as a do-no-harm
control.

## Purpose
Serves Objective 1 (Pool) and Definition-of-Done item 1. Also the unattended
entry-gate: this is where the dependency on compose-scorer is enforced.

## Goal
`pair_pool.yaml` finalised and overlap-clean: ~15 token-disjoint blend-prone animal
pairs + a few controls, each training pair's fails-by-default rate over 8 seeds
recorded, gated behind a passing `scorer_validated.json`.

The pool on disk: `outputs/animals_compose_transfer/{pair_pool.yaml, pair_prompts.yaml}`,
19 pairs (15 blend-prone, cat×dog as the known-failure reference, 3 compose-by-default
controls), 38 distinct animals, no word repeated.

## Tasks
- [x] Precondition check: assert `scorer_validated.json` (from plans/completed/compose-scorer)
  exists and its pass flag is true.
- [x] Curate the first-draft pair list: 15 blend-prone animal×animal pairs,
  token-disjoint. Candidate list + blend rationale recorded in pair_pool.yaml comments.
- [x] Write `pair_pool.yaml` and confirm it loads through
  `pair_pool.py` with its built-in train/held-out overlap assertion passing (exit 0).
  Also passed a stronger animal-token-disjointness + prompt-coverage check.
- [x] Score each pair fails-by-default over 8 seeds with the compose-scorer (fail-
  RATE, not eyeball). Keep pairs above the fail-rate threshold as training pairs;
  set aside a few compose-by-default pairs as the do-no-harm control.
 

## Engagement Instructions
GATE (unattended pass/fail): (a) `scorer_validated.json` exists AND pass flag true;
(b) `pair_pool.yaml` loads and the pair_pool.py overlap assertion passes (exit 0);
(c) each training pair's recorded fail-rate is above threshold and each control pair's
is below. A script asserts all three.
STOP: if `scorer_validated.json` is missing or its verdict is not pass → HALT, the
compose-scorer dependency is unmet. If fewer than 10 token-disjoint blend-prone pairs
survive scoring → halt (pool too small to support leave-one-out). If a pair intended
for training scores compose-by-default → drop it, then re-check the ≥10 floor.

## Recommended skill
▶ `/run-experiment` ✅: drives the fail-rate scoring pass over 8 seeds and confirms
   the pool. alt: `/debug-config` for the pair_pool.yaml load + overlap assertion.
