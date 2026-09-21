# Refine walk: preparing and running V0a, V4 and the crispness variation

**Compiled 2026-09-19.** Every piece below is written into the plans. The walk is closed.

## What it changed

| File | What changed |
|---|---|
| `plans/09-designing-the-correction-loss/MASTER_PLAN.md` | five variation rows corrected against runs on disk, one new row for variation `08`, the promotion rule, the rank paragraph, the plans table, and `## Do this next` |
| `plans/.../plans/hypothesis/04-freezing-the-empty-branch.md` | tasks 2.1 and 2.3 became confirmations, 2.3b fixes `--no-dial`, task 3 launches `00a` and `04` instead of relaunching `01`, one error-matrix entry |
| `plans/.../plans/hypothesis/06-fitting-the-whole-path-and-handing-back-the-tail.md` | new |
| `plans/.../review/06-fitting-the-whole-path-and-handing-back-the-tail.md` | new, bars fixed before any render |
| `MASTER_PLAN.md` | step 76 |
| `artifacts/results/designing-the-correction-loss/why-the-renders-are-not-crisp/` | three sheets, their scripts, and the card |

## What the walk found, in order

**V1 had already run.** `v1_freeze_null_r16_s0_25`, rank 16, 30,000 steps, W&B `kdzx03ji`. V3, V6 and V6a too. Every rank-32 attempt died within five minutes of the others on 2026-09-17. V0a, the baseline the scope reads everything against, has never been launched.

**The `--no-dial` switch does not do what its help text says.** The branch at `trainer.py:532` composes with the adapted null and never reads `freeze_null`, so `--no-dial --freeze-null` is not V1 up to a constant, and the sampler would detach a branch training kept.

**The haze is not the training step range.** `pool43-all50` against `pool43-early25`, rank 32, same pool, one axis apart: neither is blurry and both carry dense detail.

**The rendering style is carried by the seed.** Two cells of the same pair, same run, same weights, opposite look at every checkpoint.

**Compose rate cannot see the failure.** It reads 1.000 from step 31,250 onward in both runs, including the step where a cell collapses into one body with two heads.

**The record already holds the working lever.** Handing the tail to the frozen model after step 29 moved renders 0.038 nearer the target against a 0.05 bar, the largest movement any lever produced. Plan 06 sweeps the step it was fixed at.

## Routed onward

- To `sync-plan-tree`: plan 01's task 2.1 is satisfied by `v1_freeze_null_r16_s0_25` (43 cells all `split: train`, `loss_undialled` across 30,830 rows, `train_step_range [0, 25]` in config, checkpoints beside renders).
- To `sync-plan-tree`: `plans/01-showcase-the-trained-lora/review/19-does-a-stochastic-sampler-sharpen-the-corrected-render.md` says a sweep is in flight as job 50338. Nothing is running.

## Left open, on purpose

- Why the three rank-32 runs died. Nothing has read those logs.
- What measure separates two well-formed animals from two animal-shaped things on one body. Carried in plan 06's review file under `Still open`.
