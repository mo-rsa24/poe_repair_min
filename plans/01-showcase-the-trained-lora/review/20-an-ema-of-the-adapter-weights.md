# 🧪 Review: does an averaged copy of the adapter's weights render nearer the joint-prompt image than the raw weights?

**The run is in flight as Slurm job 50343 on mscluster75; the first probe is possible at 30k, about 14 hours in.** This file judges [the design](../plans/experiments/20-an-ema-of-the-adapter-weights.md). Its answer decides which weights the showcase wall renders from.

## Recommended prompt (when the run lands)

```
/analyze-run phase1_r32_wd0.1_ema0.999_40k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/20-an-ema-of-the-adapter-weights.md) | the flag, the launcher, the probe key, the bars and why 0.999 |
| **this file** | **the verdict against those bars** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Raw weights**: `lora_state`, the optimizer's. **EMA**: `lora_state_ema`, the per-epoch exponential moving average at per-step decay 0.999 (per-epoch 0.951), a shadow copy that never feeds training.
- **Drift**: DINOv2 drift as the decay finding reads it, mean of the 8 held-out cat × dog seeds at λ 1 over all 50 steps, from `summary.full."1.0".mean_dino_drift` in a probe folder's `results.json`. Negative is nearer the joint-prompt image.
- **Experiment D**: `phase1_r32_wd0.1_100k`, the run this one replicates for its first 40k steps.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: one training-side intervention that leaves the trajectory untouched). A null closes the plan and hands the fidelity question to plan 21; support changes the weights the wall renders from and turns the flag on for every later pooled run.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Where | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| 2026-09-06 03:01 | `bigbatch`, job 50335 | | the launcher | 1 second | died on `mscluster65`, whose GPU is dead; exclusion list extended |
| 2026-09-06 03:05 | `mscluster53` (RTX 3090, 24 GB), Slurm job 50339, W&B `oawqrx2u` | `phase1_r32_wd0.1_ema0.999_40k` | the same flags without gradient checkpointing | 2 minutes | out of memory in the first epoch (the forward needs about 25 GB); run folder moved to `phase1_r32_wd0.1_ema0.999_40k_oom_attempt_50339` |
| 2026-09-06 03:14 | `mscluster75` (RTX 3090, 24 GB), Slurm job 50343, log `logs/r32_wd_ema-50343.err` | `phase1_r32_wd0.1_ema0.999_40k` | rank 32, alpha 32, lr 1e-4, weight decay 0.1, EMA 0.999, kill criterion off, gradient checkpointing, 40k steps | | in flight |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 30k, is the EMA's 8-seed drift more negative than the raw weights' by at least 0.03, with the compose count within one seed?** The bar, in `scripts/showcase/ema_verdict.py` as `DRIFT_MARGIN = 0.03` and `MAX_COMPOSE_LOSS = 1`: **support** if `drift_ema − drift_raw ≤ −0.03` and `compose_ema ≥ compose_raw − 1`; **null** if the gap is inside ±0.03 (the average renders the same) or the EMA is further away or loses two or more seeds. The 40k pair is read the same way; the plan is null only if both steps are null.
      What would surprise: the EMA composing more seeds than the raw weights, which would say the raw checkpoint's seed 14 failure is weight noise rather than a basin.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Do the raw weights replicate experiment D at 30k?** `REPLICATE_MARGIN = 0.03` on drift and one seed on the count, against `figure_r32_wd0.1_030000` when experiment D's task 1.3 has run. The two runs share every flag and the torch seed; a gap beyond the margin is batch-shape nondeterminism on different cards (A6000 against 3090) and is recorded in `artifacts/notes/batch-shape-nondeterminism/`, not judged.
- [ ] ⚠️ **How far does the EMA norm lag the raw norm?** From W&B `train/lora_weight_norm` and `train/lora_weight_norm_ema` at 30k and 40k. A lag under 1% says the average is not doing anything the raw weights are not; the verdict then reads null for a known reason.
- [ ] ⚠️ **Does the run fit a 24 GB card?** Without gradient checkpointing, no: job 50339 died in its first epoch needing about 25 GB. With it, the first `epoch=` line of job 50343 is the read, and the step time beside it is the cost of the recomputation.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions that arise from the grids go here, marked as post-hoc, with the number and the file.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- **The EMA was loaded as the raw weights, or the reverse.** The probe records `attach_info.lora_key` in `render_run.json`; check both folders name different keys before reading a number.
- **Different cards for the two probes.** Both probe folders of one step are rendered by the same launcher on `bigbatch`; if they land on different nodes the fp16 band (about 2 grey levels) is far below the 0.03 drift margin, but the note is made.
- **The EMA at 30k still remembers the start.** With a 1,000-step memory and the loss floor at 10k, the 30k average covers steps 29k to 30k only; the design's "why 0.999" paragraph is the check.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- Which weights the showcase wall renders from, with the drift gap and the count.
- One line in the decay finding's `Still open`: the EMA remedy, tested, with the verdict.
- The replicate check as a sentence in the batch-shape nondeterminism note.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] One decay (0.999) and one weight-decay setting (0.1). Whether the EMA helps the baseline run at weight decay 0 is not tested; experiment D and this run share the 0.1 setting so the two answers stack.
- [ ] The read stops at 40k. Whether the EMA delays the late haze past 70k is a resume of this run, not this plan.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

At 30k: the two probes (task 1.3), the grids side by side (instruction 2.2), then the verdict script. Then `/sync-plan-tree plans/01-showcase-the-trained-lora/`.
