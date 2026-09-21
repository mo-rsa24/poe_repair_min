# 🎲 Review: Feynman-Kac steering on a detector reward

This file judges [the design](../plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md):
K particles of the plain product-of-experts sampler resampled toward the ones the validated compose
scorer already counts two animals in, read on the model's current guess of the finished image.
Both runs are done and the verdict is null; the bars below were written before the first launch.

## Recommended prompt (when the run lands)

```
/analyze-run the fk-steering run in prime_lab/poe-repair-animals-compose, group fk-steering: the two sheets, compose/* and agreement/*
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/10-feynman-kac-steering-on-a-detector-reward.md) | the sampler, the reward, the potential, the sheet, the bars |
| **this file** | **the verdict** |
| [plan 09's verdict](09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) | the same sampler with a learned weight, inconclusive |

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

- **x0-hat**: the model's current guess of the finished image at a step, decoded so the detector
  can look at it.
- **The reward**: the validated instance-count scorer's count on x0-hat, clipped at 2.
- **The control**: the same K particles with the same noise draws, never weighted or resampled;
  its particle 0 is the plain product-of-experts render at `eta` 1.0 from the seed's cached noise.
- **Compose rate**: the share of the 8 seeds whose finished image the detector counts two or more
  animal instances in. For FK, the picked particle per seed; for the control, particle 0.
- **Agreement at step s**: the share of final particles whose reward at step s (traced back
  through the resampling ancestry) gives the same compose verdict as the detector on their
  finished image.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** A null is a finding about whether the product's proposals contain
composing states at K 16; a reward blind at step 10 is inconclusive, not a null.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Smoke, `GPU=1 bash scripts/fk_steering/run_fk_steering.sh smoke` over SSH on mscluster109 device 1 (`co3`), PID 1856060 | Builds a measuring tool | 2026-09-05 16:57 | 29 s: cat × dog seed 9, K 2, 10 steps, 512², resampling at 0, 2, 4, 6, 8 | `outputs/interaction_term/fk_steering/smoke_K2_s10_20260905-165736/`: `mono_eta0.png`, `poe_eta0.png`, `ctrl_K2/` and `fk_K2/` with both finals and `run.json`, `fk_K2/xhat/step_{00,02,04,06,08}/p{0,1}.png`, the one-row sheet with its sidecar, `summary.json`, `verdict.json`. The log printed `torch sees NVIDIA RTX A6000` and `device cuda`. Images are noise at 10 steps and 512²; the verdict string is exercised, not read. A first attempt on the session node's RTX 3090 was refused by the device guard (another session's process held 6.5 GB at 100%), which is the guard working | ✅ the whole path runs |
| Full, `GPU=1 bash scripts/fk_steering/run_fk_steering.sh full` over SSH on mscluster109 device 1 (RTX A6000, `co3`), PID 1856701, launch commit `39f1964` with 118 dirty paths | Establishes a baseline | 2026-09-05 16:59 | 16 cells × (2 `eta` 0 references + K 16 and K 4 controls + K 16 and K 4 steered runs), 50 steps, 1024²; estimated 6 to 7 h | run dir `outputs/interaction_term/fk_steering/fk_K4-16_s50_20260905-165958/`, log `outputs/interaction_term/fk_steering/logs/full.log`, W&B `prime_lab/poe-repair-animals-compose/runs/czim1n0w` (group `fk-steering`). Finished 2026-09-05 20:04 after 3 h 5 min (about 11.5 min per cell, half the estimate). `verdict.json`: null. Cat × dog compose rate over 8 seeds: Mono 1.0, plain PoE 0.0, control particle 0 at K 16 0.0, best of 16 0.0, FK K 4 0.0, FK K 16 0.0; fraction over all particles 0 of 128 unweighted and 0 of 128 steered at K 16, 0 of 32 and 0 of 32 at K 4. Step-10 reward agrees with the final verdict on 0.99 of the K 16 particles (0.38 at step 0). Informative resamples per run: 0.625 of 5 at K 16. Butterfly × meadow: Mono 0.125, plain PoE 0.0, control 0.0, best of 16 0.625, FK K 4 0.125, FK K 16 0.75. Sheets filed as `fk-steering-cat-dog-sheet.png` and `fk-steering-butterfly-meadow-sheet.png` under `artifacts/results/is-the-gap-the-samplers-or-the-models/` with the summary and sidecars | ✅ done, null |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does resampling the plain PoE particles on the compose scorer's read of x0-hat raise
      the compose rate over the same particles left unweighted, on cat × dog seeds 9 to 16?**
      Read from `verdict.json`. Support if FK at K 16 composes at least 0.25 more often than the
      unweighted control at K 16 (`PASS_MARGIN`). Null if FK is within 0.10 of the control at both
      K 4 and K 16 (`NULL_MARGIN`). Inconclusive if the reward read at step 10 (`REWARD_BLIND_STEP`)
      disagrees with the final detector verdict on more than 0.50 of the final particles
      (`MAX_REWARD_DISAGREEMENT`), because then the reward was blind where the compose decision is
      made and the method was never tested. All four constants sit in
      `poe_repair/experiments/fk_steering/steer.py`. Fixed by the paper: λ 10, the max potential,
      resampling at step indices 0, 10, 20, 30, 40 of 50, DDIM `eta` 1.0, the base sampler as
      proposal, instance count clipped at 2 as the reward, no aesthetic term.
      **Null.** FK at K 16 composes 0 of 8 against 0 of 8 for the control; at K 4 the same. The
      disagreement at step 10 is 0.008, far under the 0.50 that would have made this
      inconclusive, so the reward saw what the judge saw. Every one of the 128 unweighted K 16
      particles on cat × dog is a single animal, so there was nothing to select: the plain
      product's proposals do not contain a composing state at K 16 on these seeds. `verdict.json`
      in the run directory, W&B `czim1n0w`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Did the smoke run produce the two references, a control, a steered run with x0-hat
      reads at every resample step, a sheet, `summary.json` and `verdict.json`, and did the log
      print the device name and `cuda` from torch? Yes, all of it (Runs table, PID 1856060).
- [x] ⚠️ Does the agreement between the reward and the final verdict rise with the read step
      (0, 10, 20, 30, 40)? Yes, as predicted: on cat × dog at K 16 it is 0.38 at step 0, 0.99 at
      10, 0.99 at 20, 0.98 at 30, 1.0 at 40. The x0-hat at step 10 already shows one face with two
      ears (seed 9's `fk_K16/xhat/step_10/p0.png`), so the product has committed to one animal
      by then, which matches the commit step in the landing finding.
- [x] ⚠️ Is the FK K 16 compose rate above best of 16 unweighted? Both are 0.0 on cat × dog; on
      butterfly × meadow FK K 16 is 0.75 against best of 16 at 0.625, but see the next answer for
      what those counts are.
- [x] ⚠️ Does butterfly × meadow keep its compose rate under FK at both K? The scorer cannot read
      this pair: it counts animals, a butterfly over a meadow is one animal, and Mono itself
      scores 0.125. The green tiles in the FK K 16 column are frames with two butterflies (seeds 9,
      11, 12, 13, 15) or a butterfly drawn out of flowers (seed 10), which the detector counts as
      two. No FK tile lost the butterfly or the meadow, so the method did not break the pair; its
      numbers on this pair are butterfly counts, not composition.
- [x] ⚠️ Are particle 0 of the control and particle 0 of the FK run byte-identical at step 0
      before the first resample? They start from the same tensor by construction (one
      `particle_init` call per cell) and the resampler draws from its own generator so the DDIM
      noise streams match; the runs where no resample was informative (3 of 8 cat × dog cells at
      K 16) were checked by md5 against the control's finals and matched
      (`identical_to_control_when_uninformative` in `run.json`).
- [x] ⚠️ How many resamples changed the population? Mean 0.625 of 5 per run at K 16 on cat × dog
      (never more than one), 1.0 on butterfly × meadow. The reward is flat at 1 for almost every
      cat × dog particle at every read, so there was nothing to separate.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] ⚠️ Would a larger K or a reward read at more steps find a composing proposal? Not from
      this run: 128 particles per seed-K-pair, 0 composing, and the reward agreed with the judge
      from step 10 on. (Raised by the all-particle fraction of 0.)
- [ ] 🟡 Would the same steering on top of the rank-32 correction add anything once the proposal
      spans composing states? That is [plan 11](../plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md).

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ⚠️ **Was the comparison fair?** Yes: same K, `eta`, initial noise, fresh noise draws,
      steps, guidance and resolution between the control and the FK run at each K; only the
      weighting differs, and the md5 check above confirms the two coincide when no resample was
      informative. The `eta` 0 reference columns are named on the sheet and not compared
      numerically to the `eta` 1 columns.
- [x] ⚠️ **Was the measuring tool sound?** On cat × dog every tile counts 1 and the eye agrees on
      all 48 sheet tiles: each is one animal, most a cat-dog blend. No count of 2 occurred among
      the 320 cat × dog particles, so the collar-tag failure of plan 09's re-score did not arise.
      On butterfly × meadow the tool measures butterfly count, not composition (answered above).
- [x] ⚠️ **Did the run respect the environment?** Yes: everything under `/datasets`, `co3` on
      mscluster109 device 1, `torch sees NVIDIA RTX A6000` in the log, the device held only a
      786 MiB co-tenant at launch, node, device and PID in the Runs table.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a selection-only sampler guided by the judge composes, or does not | the compose rates per column with K, `eta`, the resample steps, λ and the seed count |
| the reward was the compose scorer itself | that the picked particle is chosen by the scorer that scores it, and the best-of-K column that separates picking from resampling |
| the method follows Singhal et al. | the one departure: systematic instead of multinomial resampling, and why |
| the result bounds the sampler's share | that selection can only reach what the proposal spans, and the in-span share from rung 2 of scope 05's test-07 once it exists |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| How the result reads against the in-span share (selection can only reach what the proposal spans) | rung 2 of [what the correction is made of](../../05-when-does-the-outcome-lock-in/review/07-what-the-correction-is-made-of.md), which writes `artifacts/results/what-the-correction-is-made-of/orthogonal-share-over-steps.json`; not started as of 2026-09-05 17:10 | the sentence in the paper's corrector section |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Read [plan 11's review](13-feynman-kac-steering-on-top-of-the-rank-32-correction.md) when its run
lands: the same steering with composing states put into the proposal on purpose.
