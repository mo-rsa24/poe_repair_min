# 🔬 Re-probe the mechanism beyond seed 9

## Background
The value-channel finding (the LoRA changes what the dropped concept's token
writes, not where words look) rests on one seed of one pair, with one prior
retraction in its result family. It gates the paper's mechanism section.

## Description
Run the value/attention probe on the converged pooled checkpoint
(phase1_r8_100k, lora_step_100000.pt) across the held-out animal pairs and
seeds, and read the verdict against the scope's Goal 6.

## Purpose
Turns a one-cell lead into either the mechanism figure or an honest negative
paragraph, before anything is built on it. Serves DoD 2.

## Goal
A per-seed, per-pair table comparing how much the adapter changes the SPATIAL
PATTERN of each word's painted content against its attention weights, one
qualitative map pair per outcome class, and a recorded verdict: replicates
(mechanism section proceeds) or does not (negative paragraph only).

(Reworded 2026-08-05. The original said "value-direction rotation vs
attention-weight correlation". Those two quantities are not on a common scale,
and comparing them directly reverses the answer. See the note on the table
task below.)

## Illustrations
*(image not yet generated; save under plans/interaction-term/assets/ and replace this placeholder)*

**Prompt for image generation:**
> Generate an image of a flowchart showing this experiment: generalize the
> token map beyond cat and dog, smoke one cell, submit the 64-cell sweep,
> compute the rotation table, record the verdict. Success path green with
> checkmark "Completed" pills. Failure path red on the smoke stage labeled
> "token map missed the words, maps are garbage" with an X icon and a dashed
> "Retry Stage" callout. Downstream stages muted gray with "Skipped" pills.
> Glossy, minimalistic, modern UI/UX dashboard panel, dark background,
> rounded rectangle stage cards in a horizontal row connected by directional
> arrows, clean sans-serif labels, generous spacing, no clutter.

## Environment Facts This Plan Depends On
- co3 python at the absolute path; SDXL inference fits the in-session 3090,
  full sweep goes to biggpu first, else bigbatch.
- Attention/value .pt files accumulate: write to /datasets (disk guard), never
  /home-mscluster.
- Checkpoint: artifacts/scopes/animals-compose-transfer/pooled_lora/
  phase1_r8_100k/checkpoints/lora_step_100000.pt (filed 2026-08-04; the old
  outputs/animals_compose_transfer/... path still resolves via compat symlink).
  Its 420 LoRA tensors are under sd["lora_state"], not at the top level.

## Tasks
- [x] ✅ generalize value_probe.py beyond the hardcoded cat/dog token indices:
      a per-pair token map derived from each pair's prompts
      ✓ verified: token_map.py, checked across all 19 pool pairs, 0 mismatches.
      Caught 3 pairs the old index-2 hardcode would have read as fragments:
      walrus (wal|rus), chimpanzee (chim|pan|zee), porpoise. a_seal__x__a_walrus
      is IN this sweep, so it would have probed "wal" with no error.
- [x] ✅ one-cell smoke in-session (one held-out pair, one seed): maps look
      sane, token map verified against the tokenizer
      ✓ verified: an_eagle__x__a_hawk seed 9 (+ frog/toad, seal/walrus,
      cat/dog). Maps render as a bird/frog head in profile, not garbage.
      Figure: docs/evidence/mechanism-reprobe/smoke_eagle_hawk.png
- [ ] ⚠️ full sweep: 8 held-out pairs × seeds 9-16, adapter OFF vs ON at
      matched steps. RUNNING on mscluster109 (not a Slurm job: biggpu allows
      one job per user and an interactive session holds the slot).
      run_sweep_on_this_node.sh, ~4 min/cell, resumable.
- [ ] ⚠️ compute the table, per cell. NOTE the measure changed: this said
      "value-direction rotation vs attention-map correlation", but the obvious
      version of that comparison gives the WRONG answer. Weight maps are
      row-stochastic and content maps are not, and the adapter dims the weights
      ~25% overall, so ||on-off||/||off|| says weight moves 1.70x more (against
      the hypothesis) while the scale-free pattern change says content moves
      1.5-2x more (for it). scripts/mechanism_study/reprobe_table.py compares
      the PATTERN term. See docs/evidence/mechanism-reprobe/measure-fairness.md
- [ ] ⚠️ /pair-figure decision: per-seed points vs pair-level means as the
      figure's statistical entity
- [ ] ⚠️ record the verdict against Goal 6 (replicates or negative paragraph).
      The bar is PRE-REGISTERED in reprobe_table.py, written before the sweep
      ran: median ratio >= 1.2x AND >= 75% of rows above 1. Do not loosen it to
      rescue a negative; the negative paragraph is a result the plan provides
      for.

## Success/Failure Outcomes
- **one-cell smoke**
  - Success: cat/dog-style weight and content maps render, token map indexes
    the right words for that pair.
  - Failure: garbage or empty maps, meaning the token indices missed the
    words. Fix the token map before any sweep.
- **full sweep**
  - Success: 64 cells captured, table computes without NaNs.
  - Failure: OOM (move to biggpu) or missing capture files for named cells
    (rerun those cells only; the runner skips finished ones).
- **the verdict**
  - Replicates: median pattern ratio >= 1.2x AND >= 75% of rows above 1.
  - Does not: the mechanism section shrinks to the negative paragraph. Early
    signal is mixed. The four smoke cells ran 1.27-2.03, but the first real
    sweep cell (a_leopard__x__a_jaguar seed 9) reads 1.15, below the bar. This
    may land close either way.

## Recommended skill
▶ `/run-experiment` ✅ for the sweep; `/pair-figure` ✅ before plotting.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.mechanism_study.value_probe \
  --checkpoint <lora_step_100000.pt> --pair-slug an_eagle__x__a_hawk --seed 9 \
  --steps 10,25,40                              # smoke: expect 3 step files
ls -d /datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe/*/seed_* | wc -l
                                                # expect 64 cells
$PY scripts/mechanism_study/reprobe_table.py    # table + verdict.json
```
