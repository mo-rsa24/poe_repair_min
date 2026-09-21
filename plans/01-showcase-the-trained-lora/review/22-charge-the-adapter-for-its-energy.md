# 🧪 Review: does a running cost on the correction give two animals with the haze gone?

**Null. All three arms and the readout have run (W&B `2cfdtdnl`): the price cut the adapter's on-policy energy by 27% with the fit unchanged, and the renders moved away from the joint-prompt image, not toward it. The haze is not excess energy.** This file judges [the design](../plans/experiments/22-charge-the-adapter-for-its-energy.md). Its answer decides whether the priced adapter replaces the shipped checkpoint on the showcase wall, and which fix runs next if it does not.

## Recommended prompt (when the run lands)

```
/analyze-run energy_penalty_readout
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/22-charge-the-adapter-for-its-energy.md) | the price, the three arms, the reads, the code, the constants |
| **this file** | **the verdict: does the price remove the haze, and what did the adapter give up for it** |

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

- **The shipped checkpoint**: the rank-32 pooled adapter `phase1_r32_100k` at step 30,050; 7 of 8 held-out cat × dog seeds compose at λ 1, DINOv2 drift −0.091, contrast 37.5.
- **β, the price**: the multiplier on the control energy term added to the trainer's loss. β 0 is the trainer as it was and is the control arm. Arms at 0.01 and 0.05.
- **Control energy**: `Σ_k ½ γ_k ‖r_k‖²` in nats, `γ_k = (1 − ᾱ_k/ᾱ_prev)/(1 − ᾱ_k)`, the Girsanov price of a correction `r` read as a control drift on the reverse SDE. **Mismatch**: the same sum on `r̂ − r_true` along the adapter's own trajectory.
- **Drift**: the corrected render's DINOv2 cosine distance to the joint-prompt render minus its distance to the plain-PoE render, mean of 8 seeds at λ 1; negative is nearer the joint-prompt image. **Compose count**: seeds of 8 with two or more detected animals. **Contrast**: grey-level standard deviation of the 1024 px render.
- **The map from the monograph to this repository** (Tang, arXiv 2603.18992): the reference process Q is the plain-PoE reverse run; the target bridge is the joint-prompt run; the control `σ_t u` is the true correction `r_t`; the h-function of section 4.4 is the pointwise mutual information whose gradient `r_t` is; the running cost `½‖u‖²` of definition 3.1 is the term this plan adds; the value function of proposition 3.7 is steep before the commit step and flat after, which is why the energy spent late buys no composition.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: one training axis moved, the price, against a control arm identical in every other respect). The β 0.01 arm is the dose read (group 8, a monotonicity check, reported whole). The free read is descriptive and carries no bar. A missed bar closes the plan; the fidelity caveat is then written as a bounded sentence and the follow-on named in **Still open** opens.

**Varied axis:** `energy_penalty` ∈ {0, 0.01, 0.05}, resumed from one checkpoint, one state sequence, one launch shape.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the free read: control energy of the true correction from the cache | descriptive, CPU | 2026-09-06, session node mscluster85, uncommitted working tree over `5b422c0` | 1 minute | `artifacts/results/does-charging-the-adapter-for-its-energy-sharpen-the-fix/control-energy-over-steps.{png,json}` | done |
| smoke: one epoch from 30,050 at β 0.05, W&B off | smoke | 2026-09-06 03:22, Slurm job 50347 (`energy_smoke`), bigbatch; job 50345 before it died at the parser, two sessions had added a `--gradient-checkpointing` flag to the same trainer and the duplicate was removed | about 10 minutes | the log under `showcase/logs/energy_penalty_energy_smoke_50347.log`; the run folder is removed; writes `energy_penalty/smoke.json` with the price-to-fit ratio and peak memory, which the arm jobs gate on | done 03:39 UTC on mscluster55 (RTX 3090): 50 steps in 77 s (about 1.5 s per step), mean `loss_fit` 0.00098, mean `loss_energy` 0.01536, ratio 0.781, peak memory 11.2 GB. It ran the job script as submitted, before the script wrote `smoke.json` itself, so the file was transcribed from this log |
| arm β 0 (control) | tests the claim | Slurm job 50353, started 04:08 UTC on mscluster55, W&B `4cv3mtk1` (job 50348 before it stopped at the calibration gate, exit 8, because `smoke.json` did not exist yet) | about 5.5 hours | `phase1_r32_energy0_from30050_40k/`, `figure_r32_energy0_040050/`, `frames_energy0/`, `energy_penalty/on_policy/energy0_040050/` | done 08:37 UTC: 7 of 8 compose, drift −0.054; on-policy spend 4,671 nats, true 8,332, mismatch 2,987, cosine 0.81 |
| arm β 0.01 | dose | Slurm job 50354, started 04:18 UTC on mscluster46, W&B `47z6ydvv` (50349 stopped at the gate the same way) | about 5.5 hours | the same four folders with `energy0.01` | done 08:54 UTC: 6 of 8 compose, drift −0.003; on-policy spend 3,733 nats, true 7,224, mismatch 2,504, cosine 0.82 |
| arm β 0.05 | tests the claim | Slurm job 50355, started 05:42 UTC on mscluster54, W&B `whdcr8wy` (50350 stopped at the gate the same way) | about 5.5 hours | the same four folders with `energy0.05` | done 10:21 UTC: 6 of 8 compose, drift +0.019; on-policy spend 3,420 nats, true 8,365, mismatch 3,034, cosine 0.82 |
| readout: the baseline's 40,050 grid, two baseline on-policy reads, figures, strips, verdict, W&B | figures | Slurm job 50356, `afterany:50353:50354:50355`; job 50351 started 03:39 UTC on mscluster55 after the gate stops and is doing the baseline 40,050 grid and the two baseline on-policy reads, which 50356 will find on disk and skip; the figure and strip stages were exercised on the session node's CPU on 2026-09-06 with the columns that exist (Mono, plain PoE, shipped 30,050) | about 1.5 hours | `figure_r32_040050/`, `energy_penalty/on_policy/baseline_{030050,040050}/`, `energy_penalty/{figures,strips}/`, `verdict.json`, W&B `energy_penalty_readout` | 50351 done 04:09 UTC (baseline 40,050 grid and the two baseline on-policy reads on disk; partial figures on W&B `dvbx2u2q`); 50356 done 10:22 UTC on mscluster54, figures, strips, `verdict.json`, `cell-table.md`, W&B `2cfdtdnl` |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ❌ **At step 40,050, does the β 0.05 arm render the held-out pair nearer the joint-prompt image than the β 0 control, with the same seeds composing?**
      The bar, in source as constants of `scripts/showcase/energy_penalty_readout.py`:
      **support** if `compose_n(β 0.05) ≥ MIN_COMPOSE_COUNT = 6` and `drift(β 0.05) ≤ drift(β 0) − DRIFT_MARGIN`, `DRIFT_MARGIN = 0.03` (the smallest gap the baseline grids told apart between neighbouring checkpoints);
      **null** if `compose_n(β 0.05) < 6` (the price removed the part that composes, and this is said as the cause), or if `|drift(β 0.05) − drift(β 0)| < 0.03` (the price changed nothing the reader can see), or if `drift(β 0.05) ≥ drift(β 0) + 0.03` (the price hurt);
      **inconclusive** only if a probe failed to produce all 8 seeds.
      The validity gate is checked first: the control arm must compose at least 6 of 8 and sit within `CONTROL_VALIDITY_DRIFT_TOL = 0.03` of the baseline's own 40,050 drift, or the comparison to the shipped lineage is not made and the arms are read against the control alone.
      Secondary, same question, the contrast: closure of the gap between the control's mean grey-level standard deviation and the joint prompt's; support at `CONTRAST_SUPPORT_CLOSURE = 0.5`, null at `CONTRAST_NULL_CLOSURE = 0.1` or less.
      Answer: **null, the price hurt.** β 0.05 composes 6 of 8 (above the floor) with drift +0.019 against the control's −0.054, on the wrong side of the margin. Contrast 35.0 against the control's 36.1 and the joint prompt's 47.5: closure −0.10, null. By eye (the sheet, seed 9): the control draws the same crisp cat and dog the shipped checkpoint draws; β 0.01 keeps them but the dog turns to face away; β 0.05 fuses the two into one overlapping body with two heads, the plain-PoE failure returning. From `energy_penalty/verdict.json` and `sheet-all-seeds.png`, readout job 50356, W&B `2cfdtdnl`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] 📖 **Where does the true correction spend its control energy along the run?** A read, no bar; the prediction made before it was computed (in the session of 2026-09-06, from the value-function argument) was "mostly after the commit step".
      Answer: after it. Mean over the 8 seeds of the per-seed total, 11,044 nats (2,499 to 19,127); steps 0 to 10 carry 3.5% of the total and steps 20 to 49 carry 82%; the mean per-step curve peaks at step 28. The correction's norm itself is 10 at step 0, 70 at step 28, 25 at step 49. From `control-energy-over-steps.json`, fields `summary` and `mean_over_seeds`.
- [x] 📖 **The reference numbers the arms are read against** (from readout job 50351, 2026-09-06 04:09 UTC, before any arm finished). The baseline lineage's own 40,050: 7 of 8 compose at λ 1, drift −0.090 (the shipped 30,050 reads −0.091), so the lineage did not move between the two. On-policy, mean of 8 seeds, in nats: the shipped 30,050 spends 3,806 where the true correction at the states it visits would cost 8,181, with a mismatch of 2,733 and a mean cosine of 0.83; the lineage's 40,050 spends 4,189 against 10,317 true, mismatch 3,366, cosine 0.83. Steps 25 to 49 carry 73% of the adapter's spend at both. From `figure_r32_040050/results.json` and `energy_penalty/on_policy/baseline_{030050,040050}/on_policy_energy.json`.
- [x] 🟡 **Does the price remove energy without removing the fit?** On-policy at 40,050, mean of 8 seeds: **support** if `energy_hat(β 0.05) ≤ ENERGY_SUPPORT_RATIO × energy_hat(β 0)`, `ENERGY_SUPPORT_RATIO = 0.7`, and `mismatch(β 0.05) ≤ MISMATCH_TOLERANCE × mismatch(β 0)`, `MISMATCH_TOLERANCE = 1.10`; **null** if `energy_hat(β 0.05) > ENERGY_NULL_RATIO × energy_hat(β 0)`, `ENERGY_NULL_RATIO = 0.9` (the price did not bite at this β); **fit lost** if the energy bar passes and the mismatch bar fails; **inconclusive** between. Prediction: the late-step share of the adapter's energy falls first, because that is where the true correction spends and where the adapter's fit is weakest.
      Answer: **inconclusive by the letter, and the prediction was wrong.** Energy ratio 0.732 (3,420 against the control's 4,671 nats), just above the 0.7 support bar; mismatch ratio 1.016 (3,034 against 2,987), inside the tolerance, so the fit was kept. The late share of the spend went from 74% (control) to 72% (β 0.05): the price shrank the correction at every step alike, not the late part first. What the figure adds: at every checkpoint the adapter spends about half of what the true correction costs at the states the run visits (3,400 to 4,700 against 7,200 to 10,300 nats), so there was no excess to remove; the price took energy the correction needed. From `figures/mismatch-vs-energy.{png,json}`.
- [x] 🟡 **Is the effect monotone in the price?** `energy_hat(β 0) ≥ energy_hat(β 0.01) ≥ energy_hat(β 0.05)` and `drift(β 0) ≥ drift(β 0.01) ≥ drift(β 0.05)` (drift more negative with more price). Both orderings hold: the effect is the price. One fails: reported as the spread, and question 1 is read with that caveat.
      Answer: energy is monotone (4,671, 3,733, 3,420 nats at β 0, 0.01, 0.05). Drift is monotone in the opposite direction to the one written (−0.054, −0.003, +0.019): more price, nearer plain PoE. The compose count drops from 7 to 6 at both priced arms. The effect is the price, and it is a harm.
- [x] ❌ **Did the launch shape change the training?** The control arm's drift within 0.03 of the baseline's own 40,050 and its compose count within one seed; the control's `train/loss_fit` on the baseline's `train/loss` over steps 30,050 to 40,050 within the noise of the curve. If not, the arms are read against the control only.
      Answer: yes, narrowly. The control arm at 40,050 composes 7 of 8 (the lineage's 40,050: 7 of 8) with drift −0.054 against the lineage's −0.090, a gap of 0.036 over the 0.03 tolerance. The arm differs from the lineage in the card (RTX 3090 with gradient checkpointing against an RTX 8000 without), and in the sequence of cached states after the resume (the CPU generator restarts from seed 42 rather than continuing where the lineage's was at 30,050). So the three arms are read against each other, all sharing that launch shape and that state sequence, and no arm is read against the shipped lineage. From `figure_r32_energy0_040050/results.json` (job 50353, 08:34 UTC).
- [x] ✅ **What was the price-to-fit ratio at launch?** From the smoke log: `β · loss_energy / loss_fit` over the first 50 steps at β 0.05, and the peak memory. Written here before the arms launch.
      Answer: 0.781 (mean `loss_fit` 0.00098, mean `loss_energy` 0.01536 before β, 50 steps from the 30,050 checkpoint), inside the 0.2 to 5 band, so β stays at 0.05 and no rescaling happened. Peak memory 11.2 GB on the RTX 3090 with gradient checkpointing, against 23.5 GB and an out-of-memory without it. Step time about 1.5 s. From `logs/energy_penalty_energy_smoke_50347.log` and `energy_penalty/smoke.json`.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [ ] ⚠️ **Is the running cost just a learned λ below 1?** The penalty shrank the correction at every step by about the same fraction, which is what multiplying it by a constant would do, and plan 03's amount series already says λ below 1 composes fewer seeds. A check without a run: the cosine between the β 0.05 arm's correction and the control's at the same cached states; near 1 with a smaller norm ratio says the price found no new direction. Cache only.
- [ ] ⚠️ **Why does the adapter spend half the true correction's energy and still compose?** At the states it visits, the true correction would cost about twice the adapter's spend, yet 7 of 8 seeds compose. Either the composition needs only the early, cheap part (the free read says steps 0 to 10 carry 3.5% of the true energy), or the true correction at a corrected state overshoots. The early-window cells of plan 07 and the on-policy read together can answer this without training.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Yes: `config.json` of the three arms differs in `energy_penalty` alone (checked after the runs); the comparison to the shipped lineage was not made, per the validity gate. Every arm resumes from the same file with the same optimizer moments, the same torch seed and the same CPU generator, so the sequence of cached states is identical; `config.json` differs in `energy_penalty` alone. Every compared cell renders through the same windowed sampler at λ 1.0 from the same cached noise. Mono is a reference column.
- [x] ✅ **Was the measuring tool sound?** Yes, and the eye agrees with the detector on the seed-9 strip. The scorer is the validated instance count; drift and the DINOv2 plane are the landing finding's instruments; the energy weights were printed on the grid before use (mean 1.000). The nats are a price under an SDE reading of a run sampled as an ODE, and every figure says so.
- [x] ✅ **Did the run respect the environment?** Yes: jobs 50353 to 50356 on mscluster55, 46 and 54, guards passed in every log, peak memory 11.2 GB. Output under `/datasets`, the job script's guards passed, node and job id in the log header and here, gradient checkpointing named in `config.json`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the priced adapter on the wall, if supported | β, the weight `w_k`, that it resumed from 30,050 for 10,000 steps, the compose count and drift against the control, and that the nats are a Girsanov price under the SDE reading |
| the fidelity caveat, if null | which of the three nulls it was (composition lost, no change, worse), and the on-policy read that says whether the haze was ever the energy |
| the free read, either way | that 82% of the true correction's control energy falls after step 20, with the figure |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| a fresh run with the price from step 0, rather than a resume | one 40k training at the supported β, if question 1 supports | nothing until then |
| on-policy training (states from the adapter's own trajectory, targets from a joint forward, the relative-entropy objective of the monograph's definition 3.13) | its own plan; opens if the on-policy read says mismatch and not energy carries the haze | nothing until the readout lands |
| the adjoint-matching form with a terminal reward (both-ness) instead of the per-step fit | ranked last in the reading; a plan only if both fixes above fail | nothing |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

The plan closes on the null. The follow-on is on-policy training (the second row of **Still open**): the mismatch, not the energy, is where the adapter falls short of the true correction along its own trajectory, and this run says the haze is not excess energy.
