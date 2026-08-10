# 🅰️ Hold out each pair in turn: transfer as a rate, not an anecdote

Design only. Verdicts live in [../review/03-run-A-leave-one-pair-out.md](../review/03-run-A-leave-one-pair-out.md).

## What this asks, in one line
Train fifteen LoRAs, each missing one pair, and test each on exactly the pair it never saw:
fifteen transfer points give a rate and a degradation curve where one test would give a yes or no.

## Why this plan exists
This is the scope's main question: does the fix transfer to a pair the model never
trained on? Holding out each pair in turn, instead of testing one pair once, turns a
single yes/no into 15 transfer points, so the answer comes with a rate and a
degradation curve rather than one anecdote.

## Description
Train 15 LoRAs, each holding out exactly one pair from the finalised pool, then eval
each held-out pair on its own LoRA. Every pair is held out exactly once, so this yields
15 transfer points: enough for a compose-rate and a degradation curve (rate vs fraction
held out), not a single existence result. Each eval is read on both axes via the wired
scorer (see "Reading a result" in the master plan).

## Purpose
Serves Objective 2 (Transfer A) and Definition-of-Done item 3.

## Goal
15 LoRAs trained, each held-out pair scored on its own LoRA, a cross-run leaderboard
table (one row per held-out pair) and a degradation curve produced.

## Tasks
- [ ] Configure the 15 leave-one-pair-out runs from `pair_pool.yaml` (each run holds
  out one pair, trains on the other ~14), reusing `multi_pair_trainer.py` /
  `train_pooled.py`.
- [ ] Run the 15-run sweep with the wired eval hook (compose-rate + direction-cosine
  + distance-reached live per run), each to its step budget.
- [ ] Eval each held-out pair on its own LoRA; collect compose/blend + direction-
  cosine per held-out pair.
- [ ] Build the cross-run leaderboard table (one row per held-out pair: compose y/n,
  distance-reached, direction-cosine, both embedding spaces) and the degradation curve.

## Engagement Instructions
GATE (unattended pass/fail): all 15 runs reach their step budget and write a verdict;
the leaderboard table is non-empty with one row per held-out pair and both label
columns populated. A script asserts 15 verdicts exist and the table row-count matches.
STOP (per-run, this is what makes 15 unattended runs safe): if a run's
fraction-of-distance-reached stays at floor past the commitment window, mark that run a
delivery-null and move on. Do NOT burn the full step budget on a dead cell. Record it
as delivery-null (not no-transfer) so the two-tier read stays honest.

## Recommended skill
▶ `/run-experiment` ✅: drives the 15-run leave-one-pair-out sweep and collects the
   leaderboard. alt: `/analyze-run` to sweep the runs for verdicts as they land.
