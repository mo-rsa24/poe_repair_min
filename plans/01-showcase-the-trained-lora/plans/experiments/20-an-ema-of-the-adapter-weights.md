# 🧪 An EMA of the adapter weights: does an averaged copy render nearer the joint-prompt image than the raw weights?

**This plan asks one question: at the same training step of the same run, does an exponential moving average of the LoRA weights give held-out renders nearer the joint-prompt image than the raw weights, with the two animals kept?**

**Step 62 in the root running order. Waits on nothing to launch; its verdict is read beside [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md), whose run it replicates with one shadow copy added.**

## Recommended prompt (after this plan completes)

```
/analyze-run phase1_r32_wd0.1_ema0.999_40k
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 57 (sibling) | [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) | the rank-32 run again with weight decay 0.1; the trajectory this run replicates |
| 61 (sibling) | [19-does-a-stochastic-sampler-sharpen-the-corrected-render](19-does-a-stochastic-sampler-sharpen-the-corrected-render.md) | the sampler-side fidelity fix on the existing checkpoint |
| **62 (current)** | **20: the EMA** | the same run as experiment D to 40k with an averaged copy of the weights saved beside the raw ones |
| 63 (next) | [21-give-each-branch-its-partners-embedding](21-give-each-branch-its-partners-embedding.md) | the architecture-side fix, trained on top of whichever regularisation this plan and plan 15 support |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The raw weights**: the LoRA factors the optimizer updates, saved as `lora_state` in every checkpoint, the only weights any earlier run has.
- **The EMA**: an exponential moving average of those factors, updated once per epoch (50 steps) with per-epoch decay `0.999^50 = 0.951`, so its memory reaches back about 1,000 steps. Saved as `lora_state_ema` beside `lora_state` in every checkpoint of this run. It is a shadow copy: it never feeds the optimizer, so the raw trajectory is exactly experiment D's up to hardware nondeterminism.
- **Experiment D**: `phase1_r32_wd0.1_100k`, rank 32, weight decay 0.1, the kill criterion off, launched by the sibling plan on 2026-09-06.
- **Drift**: DINOv2 drift as the decay finding reads it: the corrected render's distance to the joint-prompt render minus its distance to the plain-PoE render, mean of the 8 held-out cat × dog seeds at λ 1 over all 50 steps, from `summary.full."1.0".mean_dino_drift` in a probe folder's `results.json`. Negative is nearer the joint-prompt image. The baseline rank-32 run reads −0.091 at 30,050.
- **A probe folder**: what `scripts/showcase/lambda_boundary_probe.py` writes for one checkpoint and one weight key (`--lora-key lora_state` or `lora_state_ema`): the 8-seed grid, `results.json`.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The loss stops falling near 10k and the weights keep moving; the late correction on a held-out pair is a state-specific vector with no cross-seed structure, so the last few thousand steps of raw weights carry noise the renders inherit. An average over the last thousand steps is the standard remedy in diffusion fine-tuning, and no run here has ever saved one. Experiment D holds the norm down with weight decay; this run replicates it and saves the average beside the raw weights, so the two are read at the same step of the same trajectory.

**The hypothesis.** At 30k the EMA weights render nearer the joint-prompt image than the raw weights by more than the grids can tell apart, with the compose count within one seed.

**If true.** The showcase checkpoint becomes an EMA checkpoint, and every future pooled run trains with the flag on.

**If false.** The average changes nothing at the step the wall uses, and the fidelity gap is not weight noise; plan 21 and the on-policy arm in the pressure-test route carry it.

**Dataset.** The same 11 training pairs and 88 cells as every pooled run; held-out cat × dog seeds 9 to 16 for the read.

**Associated materials.**
- Review questions: [the review file](../../review/20-an-ema-of-the-adapter-weights.md)
- The decay this answers: [the decay finding](../../../../report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md)
- The remedies in order, from the literature: failure (c) in [the pressure-test route, part 5](../../../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** 40,000 steps at rank 32. The baseline stepped at 1.144 s on a Quadro RTX 8000; an RTX 3090 with gradient checkpointing is expected at 1.5 to 2 s, so 17 to 22 hours plus the tracking-set renders every 10k. Read the real rate from the log's `ep_t=` line. Two probe folders at 30k and two at 40k, about 15 minutes each. One Slurm job on `bigbatch`.

**Buys.** The one review question, a replicate of experiment D's first 40k steps (a free nondeterminism read), and the `--ema-decay` flag for every later pooled run.

**Why 0.999 and not 0.9999.** Per-step 0.9999 remembers 10,000 steps, a quarter of this run, so its 30k average would still be weighted toward the 20k weights. 0.999 remembers 1,000 steps: past the loss floor at 10k, every checkpoint's EMA is an average over a settled stretch.

**Why 40k and not 100k.** The showcase checkpoint is at 30k and the decay question is experiment D's. This run reads 30k and 40k and stops.

**Memory.** The plain rank-32 forward needs about 25 GB (experiment D reads 24.8 GB on its A6000; job 50339 ran out of memory on the 24 GB card in its first epoch). The launcher passes `--gradient-checkpointing`, which recomputes the UNet's activations in the backward pass: the same arithmetic, about a third slower, and the run fits. The replicate check against experiment D is unaffected, since checkpointing changes no number.

**W&B project:** `prime_lab/poe-repair-animals-compose`, run id `phase1_r32_wd0.1_ema0.999_40k`. Run root: `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_wd0.1_ema0.999_40k/`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Every `biggpu` device is busy, so the training runs as a Slurm job on `bigbatch` (RTX 3090, 24 GB, 3-day limit) with `co3` python; the seven `bigbatch` nodes with a dead or unreachable GPU are excluded on the `#SBATCH --exclude` line ([nodes](../../../../environment/hpc/nodes.md)).
- Checkpoints to `/datasets` only; the launcher's disk guard reads `/datasets` ([storage](../../../../environment/storage.md)).
- `torch.cuda.is_available()` asserted before training, because a faulted card runs on the CPU with no error ([poe-launch-002](../../../../environment/known-failures.md)).
- The kill criterion aborted the fresh rank-32 baseline at 6,500 steps; experiment D and this run pass `--kill-halve-after-steps 1000000000` to disable it.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**An averaged copy of the adapter's weights renders held-out cat × dog nearer the joint-prompt image than the raw weights at the same step, with the two animals kept.**

**Why this matters right now.** It is the cheapest training-side lever left: no new data, no new loss, one flag, and it composes with whatever experiment D finds about weight decay.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Every checkpoint ever probed is a single point on a noisy weight trajectory, and the decay finding reads fidelity from those points alone.

**The solution.** Save the average beside the point and probe both with the same script, so the comparison differs on nothing but the averaging.

**Key insights.**
1. The EMA is a shadow copy, so this run is also experiment D's replicate; the raw 30k probe against D's 30k probe is a free read of batch-shape nondeterminism.
2. The bar is the same 0.03 drift margin experiment D uses, so the two plans' answers are on one scale.
3. Reading the EMA needs a `--lora-key` on the probe, which is the only code change beyond the trainer flag.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
training step:     0 ──── 10k ──── 20k ──── 30k ──── 40k
raw weights        ●───────●────────●────────●────────●    lora_state      (the optimizer's)
EMA (0.999/step)   ○···············○········○········○    lora_state_ema  (shadow copy, 1k-step memory)
                                              │        │
                                   probe both ┘        └ probe both
                                   8-seed grid, λ 1, full window, drift and compose count
                                   raw vs EMA at the same step: the one question
                                   raw vs experiment D at 30k: the replicate check
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The flag**: `--ema-decay` on the pooled trainer, an average of every `lora_` parameter updated once per epoch, saved as `lora_state_ema` in every checkpoint and restored on resume. `train/lora_weight_norm` and `train/lora_weight_norm_ema` logged at every log interval.
2. **The launcher**: `scripts/showcase/regularised_r32.sbatch`, experiment D's flags plus `--ema-decay 0.999`, 800 epochs.
3. **The probe key**: `--lora-key lora_state_ema` on `scripts/showcase/lambda_boundary_probe.py`.
4. **The verdict script**: `scripts/showcase/ema_verdict.py`, the bars as constants, reading the probe folders.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4. Checkable outcomes:

1. The run reaches 40k with 8 checkpoints, each carrying both weight keys, on `/datasets`.
2. Four probe folders exist with `results.json`; `verdict.json` written by the verdict script; the review file's question answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/20-an-ema-of-the-adapter-weights.md`. Done when the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--launch-probe-and-judge)**.

### 1. 🧪 Launch, probe and judge

- [x] **1.1 Add the flag, the probe key, the launcher and the verdict script** (the four files under Description).
- [x] **1.2 Launch and record.** `sbatch scripts/showcase/regularised_r32.sbatch` from the repo root. The first attempt (job 50339, `mscluster53`, 03:05 UTC) ran out of memory in its first epoch: the plain rank-32 forward needs about 25 GB and the card has 24. The launcher now passes `--gradient-checkpointing` (same arithmetic, activations recomputed in the backward pass) and the run is job 50343 on `mscluster75` since 03:14 UTC. Done when `logs/r32_wd_ema-50343.err` prints `gradient checkpointing enabled`, `EMA on: per-step decay 0.99900, per-epoch decay 0.95121` and then an `epoch=` line without an out-of-memory error.
- [ ] **1.3 Probe the 30k checkpoint with both keys** as soon as `checkpoints/lora_step_030000.pt` lands (about 10 hours in). Two `bigbatch` jobs, or one after the other on a free device:
  - `sbatch --partition=bigbatch --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83 --time=2:00:00 --job-name=probe_raw --output=logs/probe_raw_30k-%j.out --wrap "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python scripts/showcase/lambda_boundary_probe.py --checkpoint /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_wd0.1_ema0.999_40k/checkpoints/lora_step_030000.pt --rank 32 --lora-key lora_state --out-root /datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_wd0.1_ema_030000_raw --windows full --lambdas 1.0"`
  - the same with `--lora-key lora_state_ema` and `--out-root .../figure_r32_wd0.1_ema_030000_ema`.
  - Done when both folders hold `results.json` and `grid_full_window.png`.
- [ ] **1.4 Probe the 40k checkpoint with both keys** at completion, the same two commands with `lora_step_040000.pt` and `_040000_raw` / `_040000_ema`.
- [ ] **1.5 Run the verdict script**: `python scripts/showcase/ema_verdict.py --raw-30k .../figure_r32_wd0.1_ema_030000_raw --ema-30k .../figure_r32_wd0.1_ema_030000_ema --raw-40k .../figure_r32_wd0.1_ema_040000_raw --ema-40k .../figure_r32_wd0.1_ema_040000_ema --d-30k .../figure_r32_wd0.1_030000 --out .../phase1_r32_wd0.1_ema0.999_40k/verdict.json` (the `--d-30k` folder is experiment D's task 1.3; omit it if that probe has not run). Paste the printed lines into the review file.

▶ **Next: [instruction 2.1](#2--watch-and-judge)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 2. 📈 Watch and judge

◀ **Needs: [task 1.2](#1--launch-probe-and-judge)** done, so the run is live.

2.1 **Mid-run health check** (once, a few hours in): open `https://wandb.ai/prime_lab/poe-repair-animals-compose`, find `phase1_r32_wd0.1_ema0.999_40k`, Charts tab, search `train/lora_weight_norm`. ✅ two curves, `train/lora_weight_norm` and `train/lora_weight_norm_ema`, the EMA one smoother and lagging; ❌ one curve only means the flag did not take, stop the job and read the log.

2.2 **Look at the two 30k grids side by side**: `figure_r32_wd0.1_ema_030000_raw/grid_full_window.png` and `..._ema/grid_full_window.png`. Same seed, same row. Write one word per seed (crisper, softer, same, animals lost) into the review file before reading the number.

2.3 **Judge against the bars** with `verdict.json`; write the verdict into the [review file](../../review/20-an-ema-of-the-adapter-weights.md).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

- The launcher's guards: `/datasets` under 90%, `torch.cuda.is_available()` true, no existing run folder of the same id.
- The first checkpoint (`lora_step_005000.pt`) carries both `lora_state` and `lora_state_ema` with the same keys.
- The bars in `ema_verdict.py` unchanged since this plan was written (diff the file).

**Fail criteria:** an out-of-memory error on the 24 GB card in the first epoch; then the run moves to the shared-device path on a `biggpu` node and the review file records the move.

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure lands in `artifacts/results/does-an-ema-of-the-adapter-weights-render-nearer-the-joint-prompt/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| raw against EMA at 30k | two 8-seed grids, one tile per held-out cat × dog seed 9 to 16 at λ 1, full window, the scorer's count under each tile; raw above, EMA below | whether the average is visibly nearer the joint-prompt tiles | anything about 90k, which is experiment D's | `grid-30k-raw.png`, `grid-30k-ema.png` |
| the same at 40k | same layout | whether the gap grows with training | | `grid-40k-raw.png`, `grid-40k-ema.png` |
| weight norm against step, raw and EMA | x training step, y Frobenius norm; two lines from W&B `train/lora_weight_norm*` | how far the average lags the raw weights | | `weight-norm-raw-vs-ema.png` |
| the verdict | one row per step: raw drift and count, EMA drift and count, the gap, the verdict; the replicate check against experiment D | which weights the wall should use | | `verdict.json` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

Verdict into the review file first, then one line into the decay finding's `Still open` (the EMA remedy, tested), then the row in the scope table and the root running order.

### 3. 🧹 Close out

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

- `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`: `--ema-decay`, `_ema_init`, `_ema_update`, `_lora_norm`, the `lora_state_ema` payload in `_dump_checkpoint`
- `scripts/showcase/regularised_r32.sbatch`: the launcher
- `scripts/showcase/lambda_boundary_probe.py`: `--lora-key`
- `scripts/showcase/ema_verdict.py`: the bars

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

`/analyze-run` when the run lands; `/ingest-error-pattern --from-run-log` if it dies.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Support: the showcase wall re-renders from the EMA weights of the best regularised checkpoint, and [21-give-each-branch-its-partners-embedding](21-give-each-branch-its-partners-embedding.md) trains with the flag on. Null: plan 21 trains on experiment D's setting alone, and the fidelity gap is written as the adapter's direction error rather than its weight noise.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` in the first epoch | rank-32 training does not fit a 24 GB card at 1024 px with the 3-wide forward | relaunch on a `biggpu` device over SSH with experiment D's launcher shape plus `--ema-decay 0.999` |
| `No devices were found` in the first second | the job landed on a `bigbatch` node with a dead GPU not yet on the exclusion list | add it to `#SBATCH --exclude` in both launchers and to `environment/hpc/nodes.md`, resubmit |
| the probe raises `has no 'lora_state_ema' key` | the checkpoint is from a run without `--ema-decay` | only this run's checkpoints carry the key |
| `kill: commit-bucket loss ... did not halve` | the kill criterion; the launcher disables it | check the launcher's command |
