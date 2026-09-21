# 🧪 Review: does training on the clean-estimate residual give a crisper held-out fix at 30k?

The code is written and smoke-tested; the run is waiting for a free biggpu device (a poller on the session node claims the first one). This file judges [the design](../plans/experiments/16-experiment-e-train-on-the-clean-estimate-residual.md). Its answer decides which adapter the showcase wall carries and what the paper's fidelity caveat says.

## Recommended prompt (when the run lands)

```
/analyze-run phase1_r32_x0loss_40k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/16-experiment-e-train-on-the-clean-estimate-residual.md) | the weight, the cap and why 22, the chain, the figures |
| **this file** | **the verdict: the two bars, the eye's read, what the run could not settle** |

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

- **The baseline**: `phase1_r32_100k` (W&B `6xc2l8ix`), rank 32, trained on the unweighted noise-space error. At 30,050: 7 of 8 held-out cat × dog seeds compose at λ 1, 8-seed DINOv2 drift −0.091, mean contrast of the 1024 px renders 37.5.
- **The new run**: `phase1_r32_x0loss_40k`, the same run with the loss weighted by `(1 − ᾱ_t)/ᾱ_t` clipped at 22 and normalised to mean 1 over the 50-step grid, so the error is measured between clean estimates and steps 0 to 10 carry 65% of the weight.
- **Drift**: the corrected render's DINOv2 cosine distance to the joint-prompt render minus its distance to the plain PoE render, mean over the 8 seeds at λ 1, all 50 steps. Negative means nearer the joint-prompt image. `summary.full.1.0.mean_dino_drift` in a probe folder's `results.json`.
- **Compose count**: seeds of 8 where the detector counts two or more animals.
- **Contrast**: the standard deviation of a render's grey-level pixels (0 to 255). Reference values on the 1024 px renders, 8-seed means: joint prompt 46.4, plain PoE 40.6, baseline adapter 37.5. **Gap closed** is `(new − 37.5) / (46.4 − 37.5)`, computed from the files at verdict time.
- **The bars** live in `scripts/showcase/experiment_e_x0_loss.py`: `DRIFT_SUPPORT = −0.121`, `DRIFT_NULL = −0.061` (the baseline ± the 0.03 margin its own grids told apart), `MIN_COMPOSE_COUNT = 6`, `CONTRAST_SUPPORT_CLOSURE = 0.5`, `CONTRAST_NULL_CLOSURE = 0.1`.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: one training axis varied, two bars fixed before launch). A missed bar closes the arm: the baseline stays on the wall, the fidelity caveat stands, and the uncapped arm in **Still open** runs only on an inconclusive read.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Where | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| 2026-09-06 | session node mscluster85, poller `experiment_e_wait_and_launch.sh`, log `logs/experiment_e_wait.log` | | waiting for a free device among mscluster106, 108, 109 (all six devices held at launch time) | | waiting |
| | | `phase1_r32_x0loss_40k` | rank 32, alpha 32, lr 1e-4, loss space x0 cap 22, 40k steps, then the two grids and the readout | | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 30k, with the clean-estimate loss, is the 8-seed drift at λ 1 at or below −0.121, with at least 6 of 8 seeds composing?** Support means the new adapter's renders are nearer the joint-prompt image than the baseline's by more than the baseline's own grids could tell apart, and the wall's adapter changes. Drift at or above −0.061 is a null: weighting the early steps does not move the held-out fix. Between the two is inconclusive. Fewer than 6 composing makes the read invalid: the re-weighting traded composition away, which is its own null.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **At 30k, does the new adapter close at least half of the contrast gap between the baseline adapter and the joint prompt on the 1024 px renders?** Support at 0.5 or more of the gap closed, null at 0.1 or less, inconclusive between. This is the haze read; drift can improve while the haze stays.
- [ ] ⚠️ **Does 40k read the same as 30k, or better?** The baseline holds from 30k to 60k and decays from 70k. If 40k is already worse than 30k on drift by more than 0.03, the new loss brings the decay forward, and that is written beside the answer.
- [ ] ⚠️ **On the training curves, does the new run's `train/loss_bucket/early` sit below the baseline's at matched steps, and its late bucket above?** That is what the weighting should buy and cost; if the early bucket does not fall, the weight did not bite.
- [ ] ⚠️ **On the per-step frames, does the new adapter's contrast leave the plain band later than the baseline's (which is below it from step 0), and does its both-ness reach the joint-prompt centroid's value by step 15 as the baseline's does?**
- [ ] ⚠️ **By eye, same seed same row: which seeds differ between the third and fourth strip columns, and how (crisper, softer, same, different composition)?** Written before the numbers are read.

## Asked after the result

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was written with the answer already visible.

(none yet)

## Could the answer be an artefact

Navigation: ⬅️ [Asked after](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Diff the two `config.json` files. Exactly three fields may differ: `loss_space` (eps against x0), `loss_weight_cap` (absent or 22.0 against 22.0), and `kill.commit_bucket_halve_after_steps` (5,000 against 1e9; the kill rule aborted the fresh baseline at 6,500 and could not fire on its resume, so the baseline effectively trained without it). `schedule.total_epochs` differs (2000 against 800) and is the run's length, not a training setting.
- [ ] ⚠️ **Different card.** The baseline ran on a Quadro RTX 8000; this run lands on whichever device freed first. Cross-device fp16 pixel differences are about 2 of 255 and do not move drift by 0.03; state which card it was.
- [ ] ⚠️ **Did the checkpoint land at 30,000 and not 30,050?** The baseline's saves sit at 5k + 50 because it resumed from 6,500; the new run saves at exact multiples of 5k. A 50-step offset is inside noise; the probe points at `lora_step_030000.pt`.
- [ ] ⚠️ **Is the λ 0 column the same picture as the baseline probe's λ 0 column?** Same seeds, same cache, no adapter: mean absolute pixel difference at or under 6 of 255 per seed. If not, the sampler or the cache changed and nothing is read.
- [ ] ⚠️ **Did the loss scale stay comparable?** `train/loss_weight_mean` should average 1.0 over the run; `train/loss` and `train/loss_eps` should differ by the weighting alone.
- [ ] ⚠️ **Did the run respect the environment?** Shared-device path over SSH, guards printed in the log header, dry run passed and removed, checkpoints under `/datasets`.

## What the write-up owes

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

A supported read owes the wall a new adapter row and the methods section one sentence naming the weight, the cap and the reason (Li and He 2025, arXiv 2511.13720, on predicting clean data). A null owes the fidelity caveat one sentence saying the softness does not move with the loss's per-step weight. Either way the write-up cites the training-curves figure for what the weighting bought and cost.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] The cap at 22 is one arm. On an inconclusive read, the uncapped arm (the exact clean-estimate error, 86% of the weight on steps 0 to 10) is the next launch, not a re-read of this one.
- [ ] Only cat × dog is read. The transfer read on the unseen pairs (scope 04) is owed before the new adapter replaces the baseline anywhere but the cat × dog rows.
- [ ] The split corrector (the re-weightable part supplied in closed form, the adapter trained on the remainder) named in the correction-composition finding is a different plan, not this arm.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

When the chain's log prints `readout DONE`, read `verdict.json`, fill the Runs table and the questions above, then `/sync-plan-tree plans/01-showcase-the-trained-lora/`.
