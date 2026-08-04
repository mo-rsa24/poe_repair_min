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
A per-seed, per-pair table of value-direction rotation vs attention-weight
correlation, one qualitative map pair per outcome class, and a recorded
verdict: replicates (mechanism section proceeds) or does not (negative
paragraph only).

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
- [ ] ⚠️ generalize value_probe.py beyond the hardcoded cat/dog token indices:
      a per-pair token map derived from each pair's prompts
- [ ] ⚠️ one-cell smoke in-session (one held-out pair, one seed): maps look
      sane, token map verified against the tokenizer
- [ ] ⚠️ full sweep as a job: 8 held-out pairs × seeds 9-16, adapter OFF vs ON
      at matched steps
- [ ] ⚠️ compute the table: value-direction rotation per token, attention-map
      correlation, per cell
- [ ] ⚠️ /pair-figure decision: per-seed points vs pair-level means as the
      figure's statistical entity
- [ ] ⚠️ record the verdict against Goal 6 (replicates or negative paragraph)

## Success/Failure Outcomes
- **one-cell smoke**
  - Success: cat/dog-style weight and content maps render, token map indexes
    the right words for that pair.
  - Failure: garbage or empty maps, meaning the token indices missed the
    words. Fix the token map before any sweep.
- **full sweep job**
  - Success: 64 cells captured, table computes without NaNs.
  - Failure: OOM (move to biggpu) or missing capture files for named cells
    (rerun those cells only).

## Recommended skill
▶ `/run-experiment` ✅ for the sweep; `/pair-figure` ✅ before plotting.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.mechanism_study.value_probe \
  --checkpoint <lora_step_100000.pt> --pair-slug an_eagle__x__a_hawk --seed 9 \
  --steps 10,25,40                              # smoke: expect 3 step files
ls /datasets/.../interaction_term/reprobe/ | wc -l   # expect 64 cell dirs
cat /datasets/.../interaction_term/reprobe/verdict.json  # verdict recorded
```
