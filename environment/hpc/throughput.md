# HPC: how long a pooled-LoRA training run actually takes

Navigation: ⬅️ [Nodes](nodes.md) | 📋 [Index](../00-INDEX.md) | [Execution protocol](execution-protocol.md)

Real per-step, per-checkpoint, and per-eval-pass timing for
`poe_repair.experiments.cross_pair_lora_pooling.train_pooled`, measured before launching
experiments A and B (`01-showcase-the-trained-lora`, plans 08 and 09), so their wall-clock
estimates come from real numbers rather than a guess. Every number below is either **measured**
(a real run, this file says which one and when) or **projected** (measured rate x the real run's
step count) — never a bare estimate with no source.

## Table of contents

- [Methodology](#methodology)
- [Raw measurements](#raw-measurements)
- [Projected durations for experiments A and B](#projected-durations-for-experiments-a-and-b)
- [What this does not tell you](#what-this-does-not-tell-you)

## Methodology

Three smoke runs, 2026-09-01, on `biggpu`'s shared-device path (no idle node was available;
`squeue -u mmolefe` was empty and `sinfo -p biggpu` showed 6 nodes `alloc`, 1 `down`). All three
used the reduced tracking scope plan 06 settled on for cost reasons (one in-pair
`a_wolf__x__a_husky`, one out-pair `a_cat__x__a_dog`, 2 seeds each, 50 DDIM steps) — this is an
open assumption, not yet confirmed against real A/B launch scripts, which don't exist yet. See
"What this does not tell you" below.

- **Rank 8, 200 epochs (10,000 steps)** — the one run sized to hit the real checkpoint (every
  100 epochs / 5,000 steps) and eval-pass (every 200 epochs / 10,000 steps) cadence, so those
  numbers are real, not extrapolated from a shorter run.
- **Rank 16 and rank 32, 20 epochs (1,000 steps) each** — short, parallel, comparing per-step
  rate only; too short to reach a checkpoint or eval pass (both fire past epoch 100).

**A hardware fault interrupted the first attempt.** `mscluster111` device 0 (RTX PRO 6000
Blackwell, confirmed free by the standard memory guard) turned out to be in a fault state
(`nvidia-smi` reported `[GPU requires reset]` on its utilization and temperature columns) — the
process hung producing zero progress for 9+ minutes while identical launches on two RTX 8000
nodes both reached their first epoch in under a minute. Killed and relaunched on
`mscluster109` (RTX A6000) instead. Recorded as `poe-launch-002` in
[known-failures.md](../known-failures.md); the pre-launch guard in
[execution-protocol.md](execution-protocol.md) and both `scripts/showcase/*_smoke.sh` launchers
now check `utilization.gpu`/`temperature.gpu` for `N/A`/`reset`, not just memory used, so this
should not repeat silently.

**A second Blackwell, `mscluster110`, was later found healthy and idle** (2026-09-01, evening),
and experiment A's real launch (below) confirms real Blackwell numbers, superseding the
"RTX A6000 is the fastest confirmed hardware" line that stood earlier in this file.

**A batch-size comparison was attempted and its result must be discarded.** `timing_smoke.sh`
accepted a `--train-batch-size` argument on the command line but did not forward it to the
trainer — every "batch=4" run in that comparison, across all three ranks, silently trained at
the default of 1. The "batch size does not change step time" conclusion drawn from that
comparison is void; whether a larger batch size would speed up training is genuinely unknown.
The flag is now wired correctly, but no valid batch>1 measurement has been taken since the fix,
and none is currently planned — the real A/B launches (below) all use batch=1 for reasons
independent of speed (matching `phase1_r8_100k`'s recipe, keeping the length/capacity grid
single-axis-clean), so the answer was never load-bearing for the decision actually made.

## Raw measurements

| Metric | Value | Unit | Source | Status |
|---|---|---|---|---|
| Rank 8 step time, steady state | 0.706 | s/step | `mscluster109`, RTX A6000, `timing_smoke_r8`, epochs 5-20 of 200 | measured, 2026-09-01 |
| Rank 16 step time, steady state | 1.236 | s/step | `mscluster106`, RTX 8000, `timing_smoke_r16`, epochs 5-19 of 20 | measured, 2026-09-01 |
| Rank 32 step time, steady state | 1.144 | s/step | `mscluster108`, RTX 8000, `timing_smoke_r32`, epochs 5-20 of 20 | measured, 2026-09-01 |
| Rank 32 total wall time, 1,000 steps (incl. model load ~65s) | 1,213 | s | same run, launch to `DONE` | measured, 2026-09-01 |
| Rank 16 total wall time, 1,000 steps (incl. model load ~65s) | 1,256 | s | same run, launch to `DONE` | measured, 2026-09-01 |
| Checkpoint save time | 0.7 | s | `timing_smoke_r8` (RTX A6000), both checkpoints (epoch 100, epoch 200) | measured, 2026-09-01 |
| Tracking-set eval-pass time (4 cells, 50 DDIM steps each) | 163.1 | s (40.8 s/cell) | `timing_smoke_r8` (RTX A6000), epoch 200 | measured, 2026-09-01 |
| Rank 8 training loop only, 10,000 steps (2 checkpoints, 1 eval pass, no model load) | 7,275 | s (2h1m) | `timing_smoke_r8`, "training plan" line to `DONE` | measured, 2026-09-01 |
| Rank 8 total wall time, 10,000 steps (incl. model load, prompt encoding, wandb sync) | 7,583 | s (2h6m) | `timing_smoke_r8`, launch to `DONE` | measured, 2026-09-01 |
| Rank 8 step time, steady state, **Blackwell** | 0.236 | s/step | `mscluster110`, RTX PRO 6000 Blackwell, real experiment A launch (`phase1_r8_200k`, run `8vl2uzak`) | measured, 2026-09-01 |
| Checkpoint save time, Blackwell | 0.3 | s | same run, first checkpoint at step 105,000 | measured, 2026-09-01/02 |
| Tracking-set eval-pass time, Blackwell (4 cells, 50 DDIM steps each) | 64.1 | s (16.0 s/cell), settling to ~45s (~11.2s/cell) on later passes | same run, first eval pass at step 110,000, repeated each 10k steps since | measured, 2026-09-01/02 |
| Tracking-set eval-pass time, **RTX 8000** (4 cells, 50 DDIM steps each) | 206.0 | s (51.5 s/cell) | `mscluster106`, real experiment B rank 16 (`phase1_r16_100k`, run `jii2mtb0`), first eval pass at step 10,000 | measured, 2026-09-02 |

**Rank does not meaningfully change step time.** Rank 32 trained *faster* per step than rank 16
(1.144s vs 1.236s) — the opposite of the naive expectation from more trainable parameters. Both
ran on shared devices with another user's process on the sibling GPU on each node, so this ~7%
gap reads as device co-tenancy noise, not a real rank effect. The trainable LoRA layers are a
small fraction of the frozen SDXL UNet's compute at any of 8/16/32.

## Projected durations for experiments A and B

Every row here is measured rate x the real run's step count, plus the real checkpoint/eval-pass
cost at production cadence (checkpoint every 5k steps = 20 saves per 100k, eval pass every 10k
steps = 10 passes per 100k) — a projection, not a second measurement. Experiment A resumes an
existing checkpoint; resuming doesn't change per-step compute, so the rank-8 rate above stands
in directly.

| Run | Training | +20 checkpoint saves | +10 eval passes | Projected total |
|---|---|---|---|---|
| **Experiment A, actually running** (resume `phase1_r8_100k`, 100k → 200k), Blackwell | 23,600s (6.6h) | +6s (negligible) | +10.7min (641s) | **~6.7 hours** |
| Experiment A, on RTX A6000 (not used; kept for comparison) | 19.6h (70,600s) | +14s | +27.2min (1,631s) | ~20.1 hours |
| Experiment A, on RTX 8000 (not used; kept for comparison) | 33.1h (119,000s, ~1.19 s/step avg) | +14s | +34.3min (2,060s, now measured — see below) | ~33.7 hours |
| **Experiment B rank 16, actually running** (fresh, 100k steps), RTX 8000 | 34.3h (123,600s) | +14s | +34.3min (2,060s) | **~34.9 hours** |
| **Experiment B rank 32, actually running** (fresh, 100k steps), RTX 8000 | 31.8h (114,400s) | +14s | +34.3min (2,060s) | **~32.4 hours** |
| Experiment B, both ranks in parallel on separate devices | — | — | — | **~35.1 hours wall-clock** (the slower of the two, run concurrently, not summed) — matches how B was actually launched: one Slurm job (48965), two devices on `mscluster106` |

Checkpoint-save cost is real but small enough to ignore in practice (0.3s x 20 = 6s across a
100k-step run on Blackwell). The eval-pass addition is now measured on both hardware types, not
scaled: **~10.7 minutes** on Blackwell, **~34.3 minutes** on RTX 8000, across a full 100k-step
run — this is the number that justified cutting the tracking set from 152 cells to 4 (plan 06's
review file: the 152-cell scope projected to ~20 *hours* added, not minutes, at the same
cadence).

**Bold rows are the actual real launches**, not smoke tests: experiment A is run `8vl2uzak`
(W&B), B rank 16 is run `jii2mtb0`, B rank 32 is run `le2tp2ti`, all in
`prime_lab/poe-repair-animals-compose`, launched 2026-09-01 evening.

**Rank 32 aborted once, at step 6,500** (2026-09-02, 01:48 UTC): a kill criterion meant to catch
stalled runs (`commit-bucket loss must fall below half its initial value by a fixed step`) fired
on a value that had rounded-display-equal to exactly half, almost certainly a noisy fluctuation
crossing a strict inequality rather than a real stall — rank 16, running the same setup
concurrently, was unaffected. Resumed from the checkpoint the abort path itself saves
(`lora_step_006500.pt`) as a new run, `6xc2l8ix`; this also structurally avoids the same
criterion recurring, since `initial_commit_loss` isn't checkpointed and only ever gets set at
exactly optimizer step 200, which a resumed run never revisits.

## What this does not tell you

- **Whether a larger batch size would speed anything up is genuinely unknown**, not "tested and
  found not to help" as an earlier version of this file claimed. That comparison's script had a
  bug (the `--train-batch-size` flag was accepted but never forwarded to the trainer), so every
  "batch=4" run in it silently trained at batch=1. The bug is fixed, but no valid batch>1
  measurement has been taken since, and none is currently planned — real A and B both use
  batch=1 for reasons independent of speed (matching `phase1_r8_100k`'s recipe, keeping the
  length/capacity grid single-axis-clean).
- **The RTX 8000 eval-pass number (206.0s, 51.5s/cell) is one measurement from one pass** on
  rank 16's real run. It's real, not scaled, but hasn't been repeated enough times to know its
  run-to-run spread the way the Blackwell number has (three passes: 64.1s, 44.9s, 45.1s).
- **The tracking-set scope this was measured under (one in-pair, one out-pair) is now confirmed
  as what the real A/B launch scripts use** — `experiment_a_resume.sh` and
  `experiment_b_rank.sbatch` both adopt it. This resolves what was previously an open assumption.
- **These are training-only projections.** They do not include time for queueing behind another
  user on a shared device, or the harvest/scoring step after each run completes
  (`experiment_a_verdict_inputs.json`, `rank_ablation.json`).
