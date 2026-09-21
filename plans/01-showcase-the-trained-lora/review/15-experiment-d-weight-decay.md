# 🧪 Review: does weight decay stop the late loss of fidelity?

This file judges [the design](../plans/experiments/15-experiment-d-weight-decay.md).
Run kind: hypothesis (one training axis varied, two bars fixed before launch and held in
`scripts/showcase/experiment_d_verdict.py`).

## Recommended prompt (when the run lands)

```
/analyze-run phase1_r32_wd0.1_100k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/15-experiment-d-weight-decay.md) | the run, the one flag that changes, the two bars and why 0.1 |
| this file | the verdict against those bars |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **Drift**: DINOv2 drift, the corrected render's cosine distance to the joint-prompt render
  minus its distance to the plain product-of-experts render, mean over the 8 held-out cat × dog
  seeds at λ 1 over all 50 steps. Negative means nearer the joint-prompt image. From
  `summary.full.1.0.mean_dino_drift` in a probe folder's `results.json`.
- **The baseline**: `phase1_r32_100k`, rank 32, weight decay 0. Drift −0.091 at 30k and −0.013
  at 90k; weight norm 86.9 at 90k; 7 of 8 seeds composing at both.
- **Weight norm**: the Frobenius norm of every tensor in a checkpoint's `lora_state`.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Where | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| 2026-09-06 | mscluster109 device 0 (RTX A6000), shared-device path, PID 1862739, log `logs/experiment_d_wd0.1.log` | `phase1_r32_wd0.1_100k`, W&B `x1p36f9j` | rank 32, alpha 32, lr 1e-4, weight decay 0.1, 100k steps | | in flight |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 90k, with weight decay 0.1, is the 8-seed drift at or below −0.060?** Support
  means the haze the baseline shows from 70k never arrived, so norm growth was its cause and
  training longer with decay on is safe. Drift at or above −0.030 is a null: the adapter decayed
  with its norm held down, so the cause is exposure bias or something else, and the on-policy
  training arm moves up. Between the two is inconclusive. The read is valid only if the 90k
  checkpoint composes at least 6 of 8 and its weight norm is at or below 69.5 (80% of the
  baseline's); if the norm did not drop, the decay was too weak and the question stays open.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ **At 30k, is the drift more negative than −0.121 with 7 of 8 composing?** That is the
  best checkpoint getting crisper by more than the 0.03 the baseline grids could tell apart.
  Inside ±0.03 of −0.091 means weight decay does nothing at the best checkpoint and only
  matters past it, which is the expected answer.
- [ ] ⚠️ Does the weight norm curve flatten, and at what value? (One point per 5k checkpoint,
  both runs on one axis.)
- [ ] ⚠️ Does the training loss on the 11 training pairs sit above the baseline's at matched
  steps (the cost of the regulariser), and by how much?
- [ ] ⚠️ By eye, same seed same row at 30k and at 90k: which seeds differ, and how (crisper,
  softer, same, different composition)? Written before the number is read.

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Diff the two `config.json` files. Exactly two fields may
  differ: `optim.weight_decay` (0.0 against 0.1) and `kill.commit_bucket_halve_after_steps`
  (5,000 against 1e9). The kill criterion aborted the fresh baseline at 6,500 and could not fire
  on its resume, so the baseline effectively trained without it; say so in the verdict.
- [ ] ⚠️ **Different card.** The baseline ran on a Quadro RTX 8000, this run on an RTX A6000.
  Cross-device fp16 pixel differences are about 2 of 255 and do not move drift by 0.03; state it.
- [ ] ⚠️ **Did the step-30k checkpoint land at 30,000 and not 30,050?** The baseline's saves
  sit at 5k + 50 because it resumed from 6,500; the new run saves at exact multiples of 5k. A
  50-step offset is inside noise, but the file names differ, so the probe must point at
  `lora_step_030000.pt`.
- [ ] ⚠️ **Did the run respect the environment?** Shared-device path over SSH, guards printed in
  the log header, checkpoints under `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

- [ ] Weight decay 0.1 is one arm. If it reads inconclusive because the norm did not drop, a
  wd 1.0 arm is the next launch, not a re-read of this one.
