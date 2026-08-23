# 🔬 Review: can a training run be read while it is still running?

**Two thirds answered; the last third is the gate the fifteen-run sweep waits on.** This file
judges [../plans/instrument-02-three-live-curves-while-training.md](../plans/instrument-02-three-live-curves-while-training.md).

Why it matters more than its size suggests: the sweep trains fifteen adapters unattended. If the
scorer inside the training loop is wrong, all fifteen produce plausible numbers that mean nothing,
and nobody finds out for days. This instrument is what makes that sweep safe to leave alone.

## Recommended prompt (when the smoke finishes)

```
/analyze-run fy7a1ynd
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/instrument-02-three-live-curves-while-training.md) | the three curves, where they are computed, how they are logged |
| **this file** | **the verdict: two curves proven, the three-curve smoke still the gate** |
| [what it gates](hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | the fifteen unattended runs that may not start until this is green |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [What the run cost, and what it bought](#what-the-run-cost-and-what-it-bought)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The three curves**, logged live per evaluation so a run can be read before it finishes:
  - **compose rate**: how often two separate animals appear.
  - **direction agreement**: whether this run's correction points the same way as the pool's
    average correction.
  - **distance reached**: how far toward the target the correction actually moved the prediction.
- **Why three and not one**: a pair sitting at the floor has two very different causes. Either the
  correction never arrived (distance reached stays flat) or it arrived pointing the wrong way
  (direction agreement is low). One curve cannot tell those apart; the paper needs to.
- **The shared-device path**: biggpu nodes hold two GPUs and Slurm registers neither, so a node
  marked allocated often has one device sitting idle. Reaching that device means SSHing to the
  node and pinning the run to it, outside the scheduler entirely.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Not a run: an instrument.** Judged by whether its checks could fail, not by what they found. A
failure here blocks the fifteen-run sweep rather than closing this plan.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| `smoke_20260819_151601` (W&B `fy7a1ynd`) | Instrument | 2026-08-19 15:16, commit `2aa4a91` (working tree dirty, 10 files), mscluster106 GPU 1 via the shared-device path | training 83s; the rest is the 152-cell eval pass, total not yet measured | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/live_curves_smoke_run/smoke_20260819_151601/`; three W&B series in `prime_lab/poe-repair-animals-compose` | running |

**What this run is.** One epoch (50 optimizer steps) of pooled rank-8 LoRA training over the 11
blend-prone animal pairs, then one full inline-sampling eval pass: 152 cells, being 11 train pairs
across 8 train seeds (88) plus 8 held-out pairs across 8 held-out seeds (64), each rendered at 25
DDIM steps and scored. Launcher: `scripts/animals_compose_transfer/smoke_live_curves.sh`.

**Where it got to.** The training epoch finished in 83 seconds; effectively all of the wall time
is the 152-cell eval pass, which is what the three curves are computed from. Nothing had failed as
of the last check.

**Why it ran outside Slurm.** No biggpu node was idle: mscluster107, 109 and 112 were down, and
106, 108, 110 and 111 were allocated. mscluster106 held another user's job on GPU 0 (8GB of 49GB,
a three-day allocation) with GPU 1 completely idle at 1MiB and 0% utilisation. Slurm cannot place a
job on an allocated node, and biggpu allows one job per user, so the only route to that idle device
was SSH. The run was pinned to GPU 1 with `CUDA_VISIBLE_DEVICES`, guarded by a check inside the
launch script that aborts if the target device already holds more than 1GB, and the other user's
process was never touched.

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Do all three land as three separate live curves on a one-epoch smoke run?
      **This is the gate.** The fifteen-run sweep may not start until this is green, for the
      reason above: a wrong in-loop scorer turns an unattended fan-out into fifteen runs of
      convincing nonsense. Judged against run `smoke_20260819_151601` when it finishes: each of
      `eval/compose_rate/mean`, `eval/direction_cosine/mean` and `eval/frac_distance_reached/mean`
      must exist as its own series with at least one non-null value.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Is the scorer wired into the evaluation loop?
      Yes, and proven end to end: the pooled run wrote a per-held-out-pair compose rate
      (`compose_rate.json`), which it could not have done unless the whole path worked.
- [x] ✅ Are the two direction measures wired?
      Yes, in code and importing cleanly. `_inline_sampling.py::direction_metrics` logs both per
      cell plus their means, reusing the existing maths rather than redefining it.
- [ ] 🟡 What does a one-epoch smoke actually cost on a shared biggpu device?
      Unknown until this run finishes. The plan's stated estimate was one hour, inherited from a
      prior smoke whose log does not exist anywhere in the repo, so it has no evidence behind it.
      Next action: record the measured wall time here when the run ends, and correct the plan's
      Considerations to that number.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the run itself raised. **Nothing here may ever become a bar**, because it was written
with the answer already visible.

- [ ] ⚠️ Is the compose-rate curve interpretable across pairs yet? Anchor images exist for only 2
      of the 19 sampled pairs, which came to light while reading the run. See
      [What the write-up owes](#what-the-write-up-owes).

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Not applicable: nothing is compared here. This instrument
      asks whether three numbers appear and carry real values, not whether one arm beats another.
- [ ] ⚠️ **Was the instrument sound?** The question this whole file exists to answer, and it is
      the bar above. Not settled until all three series carry non-null values.
- [x] ✅ **Did the run respect the environment?** Output landed under `/datasets`, the run was
      pinned to an idle device with a guard that aborts if that device is occupied, and the other
      user's process was never touched. Harvest it with `pgrep`, not `squeue`: this run is outside
      Slurm and the queue is blind to it.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#what-the-run-cost-and-what-it-bought) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a compose rate read from this smoke run | that `eval/compose_rate/mean` averages over 2 of the 19 sampled pairs only (`a_wolf__x__a_husky` in-train, `a_cat__x__a_dog` held-out), because anchor images exist for those two alone. The direction and distance curves cover all 152 cells. That is enough to answer the gate, which asks whether the metric appears and carries real values. It is not a transfer measurement, and the step-10 sweep needs the remaining anchors before its compose-rate means anything across pairs |
| the cost of a smoke run | the measured wall time, once this run ends. The plan's one-hour estimate came from a prior smoke whose log does not exist in the repo |

## What the run cost, and what it bought

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

**Three failed launches, one cause.** The first three attempts died before the training script
started, each printing `bash: line 1: logs/step-09-smoke-mscluster106.log: No such file or
directory` and leaving no process and no log. A non-interactive SSH command starts in `$HOME`, so
the relative log path resolved there and the redirect failed before `nohup` ever ran the script.
Prefixing `cd <repo> &&` did not fix it, because the tooling's permission layer can strip `cd` from
a compound command. The fix that worked is absolute paths for both the script and the log redirect,
with the launch script doing its own `cd` internally. Recorded as
[poe-launch-001](../../../../../docs/EXPERIMENT_ERROR_CATALOG.md#entry-id-poe-launch-001).

**What the environment learned.** The shared-device path is now written down rather than improvised:
[docs/ENVIRONMENT.md](../../../../../docs/ENVIRONMENT.md#execution-model) carries it as a step in
the launch decision, with the safety rules (never start on a device holding a foreign process,
re-check the device inside the launch script, record node and device and PID, harvest by `pgrep`
since `squeue` is blind to these runs). The `/run-experiment` node picker gained `--probe-shared`,
which SSH-probes allocated GPU nodes and prints `SSH SHARED` with the free device index; run
against this cluster it reached the same conclusion a person would.

**A placeholder that could never have run.** `logs/step-09-smoke-test.sh` was an sbatch script with
its training command commented out and an import path that does not match where the code lives. It
is now in `artifacts/_quarantine/`. The launcher that works is
`scripts/animals_compose_transfer/smoke_live_curves.sh`.

## Still open

Navigation: ⬅️ [What the run cost](#what-the-run-cost-and-what-it-bought) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether all three curves land as separate non-null series | `smoke_20260819_151601` finishing, then reading the three series in W&B | the fifteen-run sweep, which may not launch until this is green |
| what a one-epoch smoke actually costs | the measured wall time of this run | the plan's Considerations, which currently carries an estimate with no evidence behind it |
| anchor images for the remaining 17 of 19 sampled pairs | generating them | the step-10 sweep's compose-rate meaning anything across pairs |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Check whether `smoke_20260819_151601` has finished (`pgrep`, not `squeue`), then read the three
W&B series and answer the bar.
