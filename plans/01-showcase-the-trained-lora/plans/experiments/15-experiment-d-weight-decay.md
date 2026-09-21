# 🧪 Experiment D: the rank-32 run again with weight decay on

**This plan asks one question: does AdamW weight decay stop the rank-32 adapter's late loss of fidelity, and does it change the best checkpoint at all?**

**Step 57 in the root running order. Waits on nothing: the baseline it compares against (`phase1_r32_100k`) is finished and its 30k and 90k grids are on disk. Next: nothing in this scope; the verdict feeds [the decay finding](../../../../report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md).**

## Recommended prompt (after run completes)

```
/analyze-run phase1_r32_wd0.1_100k
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 53 (previous) | [14-correct-early-then-clean-up](14-correct-early-then-clean-up.md) | The sampling-side fidelity fix |
| **57 (current)** | **15: experiment-d-weight-decay** | The training-side fidelity fix, one axis |
| next | none in this scope | the verdict lands in the decay finding |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
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

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis:** *the adapter's late loss of fidelity is caused by its weights growing without bound, so a counter-force on the norm keeps the 90k renders as close to the joint-prompt image as the 30k ones.*

> Fidelity here is DINOv2 drift: the corrected render's distance to the joint-prompt render
> minus its distance to the plain product-of-experts render, mean of the 8 held-out cat × dog
> seeds at λ 1 over all 50 steps. Negative means nearer the joint-prompt image. The baseline
> rank-32 run reads −0.091 at 30k, −0.060 at 60k, −0.003 at 70k and −0.013 at 90k while its
> LoRA weight norm goes 58.9 → 68.8 → 79.0 → 86.9 → 89.3 over 10k, 30k, 60k, 90k, 100k.

**The design:** the rank-32 pooled run again, every flag copied from the run that made
`phase1_r32_100k`, with AdamW `weight_decay` 0.1 instead of 0.0. Nothing else changes: same 11
training pairs, 88 cells, learning rate 1e-4 constant, alpha 32, 100k steps, checkpoints every
5k, the tracking set rendered every 10k.

**Why 0.1 and not 0.01.** AdamW's decay is decoupled, so each step shrinks the weights by the
factor `1 − lr × wd`. At lr 1e-4 that is 1e-5 per step for wd 0.1, a pull of e⁻¹ ≈ 0.37 over
100k steps absent any gradient, against a baseline norm that grew 52% over the same run. At wd
0.01 the pull is e⁻⁰·¹ ≈ 0.90, too weak to tell apart from no decay on a curve that noisy. One
arm, at the top of the range the LoRA literature uses.

**The bars, fixed before launch** and held as constants in `scripts/showcase/experiment_d_verdict.py`:

- **Question 1, the one that moves the plan.** At 90k, drift ≤ −0.060 (the baseline's 60k value)
  is support: the haze did not arrive. Drift ≥ −0.030 is a null: the decay happened with a
  smaller adapter, so norm growth was not its cause and exposure bias (training on plain-PoE
  states, running on corrected ones) moves to the front. Between the two is inconclusive. The
  read is only valid if the 90k checkpoint still composes at least 6 of 8 seeds and its weight
  norm sits at or below 80% of the baseline's 86.9; otherwise the decay was too weak or crushed
  the adapter, and the question stays open.
- **Question 2, the one the showcase cares about.** At 30k, drift more negative than −0.121
  (better by 0.03, the smallest gap the baseline grids told apart between neighbouring
  checkpoints) with 7 of 8 composing means the best checkpoint itself got crisper. Inside
  ±0.03 means weight decay changes nothing at the best checkpoint and only matters past it.

**Associated materials:**
- **Review questions:** [the review file](../../review/15-experiment-d-weight-decay.md)
- **What motivated it:** [the decay finding](../../../../report/does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md) and failure (c) in [the pressure-test route](../../../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** the baseline took 1.144 s per step on a Quadro RTX 8000 (32 hours for
100k). On an RTX A6000 the measured ratio from experiment A's estimates is about 0.6, so roughly
19 to 20 hours, plus about an hour of tracking-set renders. Measure it from the log's ETA line.

**Launch shape:** the shared-device path over SSH on a pinned device of an allocated `biggpu`
node, invisible to `squeue`, per [the execution protocol](../../../../environment/hpc/execution-protocol.md).
The launcher's memory guard is 4096 MiB rather than the usual 1024, because on 2026-09-06 the
only idle device held another user's 2.9 GB process at 0% utilisation; the utilisation guard is
what protects the run's speed.

**Prerequisites:** the training cache under `/datasets`; the two baseline probe folders
`figure_r32_030050` and `figure_r32_090050` under the showcase outputs.

**W&B project:** `prime_lab/poe-repair-animals-compose`, run id `phase1_r32_wd0.1_100k`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- `biggpu` allows one Slurm job per user, so a second training goes over SSH to a pinned free device on an allocated node ([execution protocol](../../../../environment/hpc/execution-protocol.md)).
- `co3` python on the A6000 and RTX 8000 nodes (106, 108, 109); `co3_bw` on the Blackwell nodes (110 to 112) ([nodes](../../../../environment/hpc/nodes.md)).
- Checkpoints to `/datasets` only; the launcher's disk guard reads `/datasets`.
- `torch.cuda.is_available()` asserted after pinning the device, because mscluster111 runs on the CPU with no error.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The cause of the late decay, settled by one run whose bars were fixed before launch.** The
decay finding infers its cause from three things that rise together; this run moves one of them
and leaves the other two alone. Either answer is a finding.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** The rank-32 adapter's held-out renders are best at 30k and go grey and soft
from 70k, and the paper has to say why. The finding names norm growth under weight decay 0 as
the likely cause and says plainly that no run has varied it.

**The solution.** The same run with the norm held down. If the 90k render stays as good as the
30k one, the cause is settled and training longer becomes safe. If it decays anyway, the cause
is elsewhere and the on-policy training arm in the pressure-test route moves up.

**What this cannot fix.** The 30k checkpoint is already at its best; weight decay may leave it
exactly where it is. Question 2 exists so that outcome is a recorded answer rather than a
disappointment.

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The flag**: `--weight-decay` on the pooled trainer, wired into `cfg.optim.weight_decay`,
   which `make_optimizer` already passes to AdamW. Default 0.0, so every existing launcher is unchanged.
2. **The launcher**: `scripts/showcase/experiment_d_weight_decay.sh <wd> <device> [python]`, a
   copy of the rank-32 launcher with the decay flag and the kill criterion disabled (it aborted
   the fresh baseline at 6,500 steps; the resumed baseline ran without it).
3. **The verdict script**: `scripts/showcase/experiment_d_verdict.py`, the bars as constants,
   reading the new run's 30k and 90k probe folders and the 90k checkpoint's weight norm.
4. **Two probe folders**: `figure_r32_wd0.1_030000` and `figure_r32_wd0.1_090000`, made by the
   [8-seed grid recipe](../../../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#1-render-the-8-seed-held-out-grid-for-one-checkpoint).

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4. Checkable outcomes:

1. The run reaches 100k with 20 checkpoints and the tracking renders on `/datasets`.
2. The two probe folders exist with `results.json`; `verdict.json` written by the verdict script; the review file's two questions answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/experiments/15-experiment-d-weight-decay.md`**

▶ **Next: [task 1.1](#1--launch-and-harvest)**.

### 1. 🧪 Launch and harvest

- [x] **1.1 Add the flag and write the launcher and the verdict script** (the three files under Description).
- [x] **1.2 Launch on a free device and record.** Launched 2026-09-06 02:56 UTC on mscluster109 device 0, PID 1862739, W&B `x1p36f9j`; the guards printed `used=2916MiB util=0%, /datasets=34%, cuda ok` and the saved config reads `weight_decay 0.1`. Completion is observable: the log prints `LoRA attached` then a step line; the W&B run shows `optim.weight_decay: 0.1` in its config.
- [ ] **1.3 Probe the 30k checkpoint as soon as it lands** (about 6 hours in): the grid recipe with `--checkpoint .../phase1_r32_wd0.1_100k/checkpoints/lora_step_030000.pt --out-root .../figure_r32_wd0.1_030000 --windows full --lambdas 1.0`. Question 2 is readable from this alone.
- [ ] **1.4 Probe the 90k checkpoint at completion**, same recipe, `lora_step_090000.pt`, `figure_r32_wd0.1_090000`.
- [ ] **1.5 Run the verdict script** with the two probe folders and the 90k checkpoint, `--out .../phase1_r32_wd0.1_100k/verdict.json`, and paste its two verdict lines into the review file.

▶ **Next: [instruction 2.1](#2--watch-and-judge)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 📈 Watch and judge

◀ **Needs: [task 1.2](#1--launch-and-harvest)** done, so the run is live.

2.1 **Mid-run health check** (once, a few hours in): open `https://wandb.ai/prime_lab/poe-repair-animals-compose`, find `phase1_r32_wd0.1_100k`, Charts tab, search `train/loss` and `eval/tracking/embedding_drift/dino/out_out`. ✅ both advancing; ❌ frozen or the process gone from `pgrep -af train_pooled` on the node: harvest the log, `/ingest-error-pattern --from-run-log`.

2.2 **Look at the 30k grid beside the baseline's.** Open `figure_r32_wd0.1_030000/grid.png` and `figure_r32_030050/grid.png` side by side. Same seed, same row. Write one line per seed that differs by eye (crisper, softer, same) into the review file under question 2, before reading the number.

2.3 **Look at the 90k grid beside the baseline's** the same way (`figure_r32_wd0.1_090000` against `figure_r32_090050`). The baseline's seed 9 has a malformed dog head and drained colour at 90k; say whether the new run's seed 9 does.

2.4 **Judge against the bars** with `verdict.json`; **write both verdicts** into the [review file](../../review/15-experiment-d-weight-decay.md), then carry the answer into the decay finding's `What this cannot tell you` (its "cause is inferred" paragraph becomes a measured line).

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

- The launcher's guards: device under 4096 MiB and under 5% utilisation, `/datasets` under 90%, `torch.cuda.is_available()` true, no existing run folder of the same id.
- The bars in `experiment_d_verdict.py` unchanged since this plan was written (diff the file).

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

| Figure | What is on it | Where it comes from |
|---|---|---|
| Two 8-seed grids at 30k, baseline above, weight decay below | one tile per held-out cat × dog seed 9 to 16 at λ 1, full window; the scorer's verdict on each tile | `figure_r32_030050/grid.png`, `figure_r32_wd0.1_030000/grid.png` |
| Two 8-seed grids at 90k, same layout | the decay question by eye | `figure_r32_090050/grid.png`, `figure_r32_wd0.1_090000/grid.png` |
| Weight norm against training step, both runs | x training step, y Frobenius norm of `lora_state`; one line per run | every 5k checkpoint of both runs, via the norm function in the verdict script |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

Verdict goes into the review file first, then one line into the decay finding, then the row in
the scope table and the root running order flips from ◑ to the verdict mark.

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

- `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`: the `--weight-decay` flag
- `poe_repair/experiments/one_pair_one_seed/trainer.py`: `make_optimizer`, where AdamW receives it
- `scripts/showcase/experiment_d_weight_decay.sh`: the launcher
- `scripts/showcase/experiment_d_verdict.py`: the bars
- `scripts/showcase/lambda_boundary_probe.py`: the 8-seed grid and `results.json`

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

`/analyze-run` when the run lands; `/ingest-error-pattern --from-run-log` if it dies.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

If question 1 is a null, the on-policy training arm (half the batch from the adapter's own
corrected trajectory, targets from a joint forward at those states) in the pressure-test route
becomes the next training-side plan. If it is support, resume the baseline rank-32 run past
100k with decay on and read whether fidelity keeps improving.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

| Symptom | Cause | Fix |
|---|---|---|
| `ABORT: device N at X% utilisation` | someone started computing on the device between the check and the launch | pick another device |
| log stops after `LoRA attached`, never prints a step | hardware-faulted device (mscluster111 style) | `kill -9` the PID, relaunch elsewhere |
| `kill: commit-bucket loss ... did not halve` | the kill criterion; the launcher disables it with `--kill-halve-after-steps 1000000000`, so this means the flag was dropped | check the launcher's command |
| verdict script raises `no (full, 1.0) cell` | the probe was run with different `--windows` or `--lambdas` | rerun the probe with `--windows full --lambdas 1.0` |
