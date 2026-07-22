# 🏔️ Scale — one LoRA spans the taxonomy, held out on both axes (or the catalogue fallback)

## Description
Train a single rank-8 LoRA on 5 representative pairs across 8 seeds (40 training cells). Then test it
on a 2×2 grid: pair seen or unseen, crossed with seed seen or unseen. The hardest cell is the one
where both the pair and the seed are new (`out_out`). Folds phase files 11 / 14 / 15. The LoRA is
trained, but the grid test was never run.

## Purpose
Serves Objective 5 (Scale) and Definition-of-Done item 5. It is the strongest version of the
shippable claim (one LoRA for the whole taxonomy), or the evidence that says ship a per-group set
instead.

## Goal
Run and read the 2×2 grid test; classify the both-unseen (`out_out`) cell per group; and decide what
to ship: one LoRA, or a per-group set.

## Tasks
- [x] ✅ Write the cross-pair package (per-step `multi_pair_trainer`, `sample_crossbar`, `task_d_bridge`). → Plan 15 caught that the "thin wrapper" assumption was false (the base trainer binds one pair's embeddings per epoch).  ✓ verified (3 modules present)
- [x] ✅ Train `all_groups/main` on 40 cells. → G11: reached `lora_step_030000.pt` (`artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/`). Later run `0y9un0o4` died early.  ✓ verified (ckpt loads; no samples/cells.jsonl)
- [ ] ⚠️ **[publishable-bar]** Finish training (resume from `lora_step_030000.pt`; restart only if the pool spec changed), then run the crossbar — the `out_out` quadrant was never sampled.
  Prompt (`/run-experiment`): `$PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar --checkpoint artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/checkpoints/lora_step_030000.pt --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml --pair-prompts outputs/cross_pair_lora_pooling/pair_prompts.yaml --out-dir artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/samples` (all four quadrants, not just in_in,out_in).
- [ ] ⚠️ **[publishable-bar]** Task D (Δ̄_t bridge across pair,seed) + four-quadrant contact sheets; the held-pair × held-seed sheet is the paper figure.
- [ ] ⚠️ **[publishable-bar]** Classify per quadrant with the two-tier bar (image composes = deployable; Task D cosine to Δ̄_t = scientific, labelled separately); pick the landing (Good = one LoRA covers the taxonomy / Mixed = easy groups only / Bad = per-group catalogue).
- [ ] ⚠️ **[optional]** Stop the groups from interfering (graft: orthogonal adaptation, ACROSS groups only): in the one-LoRA-for-everything run, keep each group's update in its own direction so the animal-splitting fix (G6) doesn't leak into the object-separating fix (G4). Do NOT do this inside a group — there the fixes should line up, not stay apart. Compare against the plain all-groups LoRA.
  How: add a cross-group orthogonality penalty to `multi_pair_trainer` (idea from arxiv 2312.02432), retrain `all_groups`, compare the both-unseen (`out_out`) cell against the plain version.

## Recommended skill
▶ `/run-experiment` ✅ — resumes training + runs `sample_crossbar`.
   alt: `/analyze-run 2em6frqv` for the pooled learning curve; `training-analyst` to sweep new saves into tracking.

## Engagement Instructions
```
$ ls artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/samples/cells.jsonl   # currently ABSENT — the gap this rung closes
# Done when cells.jsonl exists with out_out rows and contact_sheet_out_out.png is rendered.
```
