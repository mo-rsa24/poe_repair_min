# 🎲 Review: Feynman-Kac steering on top of the rank-32 correction

This file judges [the design](../plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md):
plan 10's particle sampler with the rank-32 correction in the proposal at λ 0.5 and λ 1.2, read
against the corrected sampler left alone. The run is done and the verdict is support at λ 0.5,
with one caveat that changes how the number reads. The bars below were written before the first
launch.

## Recommended prompt (when the run lands)

```
/analyze-run the fka run in prime_lab/poe-repair-animals-compose, group fk-steering: the three sheets, compose/*_lam* and agreement/*_lam*
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/13-feynman-kac-steering-on-top-of-the-rank-32-correction.md) | the mixed forward, the two doses, the sheet, the bars |
| **this file** | **the verdict** |
| [plan 10's verdict](10-feynman-kac-steering-on-a-detector-reward.md) | the same steering over plain PoE: null |

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

- **The correction at λ**: `ε_frozen + λ · (ε_lora − ε_frozen)`, plain PoE plus λ times what the
  rank-32 adapter (step 30050) adds to it.
- **Adapter alone**: one particle of that sampler at `eta` 0 from the pinned noise, the showcase
  render at that λ.
- **The control**: K 16 particles of the corrected sampler at `eta` 1.0, never weighted; particle 0
  starts from the pinned noise.
- **Compose rate**: share of the 8 seeds whose finished image the detector counts two or more
  animals in; the picked particle for FK, particle 0 for the control.
- **Headroom**: one minus the adapter-alone compose rate at that λ.
- **The fidelity tie-breaker**: ImageReward (Xu et al. 2023, the human-preference reward the FK
  paper steers SDXL on), scored against the joint prompt "a cat and a dog". The reward is
  `min(count, 2) + 0.5 · sigmoid(ImageReward)`, so the count decides the level and fidelity orders
  particles within a level; it can never lift a one-animal particle over a two-animal one. The
  weight is `FIDELITY_WEIGHT` in `steer.py`. The compose rate the bars judge is the count alone.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** A null says selection adds nothing to the correction at either dose;
a reward blind at step 10 is inconclusive, not a null.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Adapter smoke without the fidelity term, `run_fk_steering.sh full --smoke --adapter-checkpoint <r32 step 30050> --adapter-lambdas 0.5` over SSH on mscluster109 device 1 (`co3`), PID 1859746 | Builds a measuring tool | 2026-09-06 02:40 | 19 s: cat × dog seed 9, K 2, 10 steps, 512² | `outputs/interaction_term/fk_steering/smoke_fka_K2_s10_20260906-024051/`: `adapter.json` (420 tensors, checkpoint step 30050), `adapter_eta0_lam0.5.png`, `ctrl_K2_lam0.5/`, `fk_K2_lam0.5/`, the one-row sheet, `verdict.json` | ✅ the adapter path runs |
| ImageReward install and check, session node CPU | Builds a measuring tool | 2026-09-06 02:50 | `pip install image-reward` then `pip install --no-deps openai-clip` into `co3`; torch 2.5.1, transformers 4.44.1, diffusers 0.29.2 unchanged; checkpoint load 82 s on CPU | scores against "a cat and a dog" on plan 10's seed 9: Mono +0.88, plain PoE −1.03, the steered fused particle −2.08, its step-10 x0-hat −1.50 | ✅ orders clean above blended |
| Adapter smoke with the fidelity term, `run_fk_steering.sh adapter --smoke --adapter-lambdas 0.5`, first over SSH on mscluster109 device 1 (refused by the device guard: another session had taken the card, 16.4 GB at 100%), then as Slurm jobs on `bigbatch` RTX 3090 nodes via `scripts/fk_steering/fk_steering.sbatch`: jobs 50307 (mscluster44), 50309 (mscluster45) and 50321 (mscluster65) failed in one second because those idle nodes' GPUs are faulted (`Unable to determine the device handle for GPU0`, now the third form of `poe-launch-002`); job 50312 on mscluster52 passed at 02:53 (so did 50313, 50314, 50317 on mscluster53, 54, 60); jobs 50311, 50315, 50316 collided on one run directory because they started in the same second and the run id was the timestamp alone (the run id now carries host and pid) | Builds a measuring tool | 2026-09-06 02:51 to 02:57 | 1 min per node: cat × dog seed 9, K 2, 10 steps, 512², adapter at λ 0.5, ImageReward on | `outputs/interaction_term/fk_steering/smoke_fka_K2_s10_20260906-025146/` (mscluster52), logs `logs/fk_steer-503{07..30}.out` | ✅ the adapter path with the fidelity term runs; the sheet carries an `IR` score on every tile |
| Full, Slurm job 50332 on mscluster52 (`bigbatch`, RTX 3090, `co3`), `sbatch --nodelist=mscluster52 scripts/fk_steering/fk_steering.sbatch adapter`, launch commit `39f1964` plus uncommitted edits | Establishes a baseline | 2026-09-06 03:24 | 24 cell-λ × (adapter reference + K 16 control + K 16 steered) with ImageReward reads at every x0-hat and final; 10 h 02 min on the node's RTX 3090, about 25 min per cell-λ | run dir `outputs/interaction_term/fk_steering/fka_K16_s50_20260906-025826/` (the node's clock reads 02:58 where the session node reads 03:24), PID 737254 on mscluster52 device 0, log `logs/fk_steer-50332.out`, W&B `prime_lab/poe-repair-animals-compose/runs/dfrceppl` (group `fk-steering`); `adapter.json` 420 tensors, checkpoint step 30050. Finished 2026-09-06 13:00, `verdict.json`: support at λ 0.5. Three sheets, three sidecars, `summary.json` and `verdict.json` filed under `artifacts/results/is-the-gap-the-samplers-or-the-models/` as the `fk-steering-on-adapter-*` set | ✅ done, support |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **With the rank-32 correction at λ 0.5 in the proposal, does resampling on the compose
      scorer's read of x0-hat raise the cat × dog compose rate over the corrected sampler's
      particle 0?** Read from `verdict.json`. Support if FK at K 16 composes at least 0.25 more
      often than the control at λ 0.5 (`PASS_MARGIN`). Null if FK is within 0.10 of the control at
      both λ 0.5 and λ 1.2 (`NULL_MARGIN`, over `ADAPTER_LAMBDAS`). Inconclusive if the step-10
      reward disagrees with the final verdict on more than 0.50 of the final particles at λ 0.5
      (`MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`). Judged at `ADAPTER_JUDGED_LAMBDA` 0.5,
      K `ADAPTER_K` 16, adapter `ADAPTER_CHECKPOINT` rank `ADAPTER_RANK` 32, all in
      `poe_repair/experiments/fk_steering/steer.py`. The potential, λ 10, the five resample steps
      and `eta` 1.0 are plan 10's, unchanged. The reward carries the ImageReward tie-breaker
      (`FIDELITY_WEIGHT` 0.5) so the picked two-animal particle is the cleanest one; the compose
      rate is judged on the count and the fidelity is read beside it, never in the bar.
      **Support, by a wide margin, with the compose gain coming from the 16 draws rather than from
      the resampling.** Cat × dog compose rate over the 8 held-out seeds at λ 0.5: steering 8 of 8
      against 2 of 8 for the control's particle 0, a gap of 0.75 against a 0.25 bar; the adapter
      alone at `eta` 0 is 3 of 8. Disagreement at step 10 is 0.219, under the 0.50 that would have
      made this inconclusive. At λ 1.2 steering is 8 of 8 and the control is 8 of 8, so the gap
      there is 0 with no headroom, as the design said it would be.
      **The best of 16 unweighted particles is also 8 of 8 at λ 0.5**, so resampling did not raise
      the compose rate over drawing 16 and keeping the best. What resampling changed is which
      composing particle survives: mean ImageReward 0.754 for the steered pick against 0.452 for
      best of 16 and −0.768 for the adapter alone, on the same 8 seeds. `verdict.json` in the run
      directory, W&B `dfrceppl`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Did the smoke attach 420 tensors from checkpoint step 30050 and produce the adapter-alone
      reference, a control, a steered run, a sheet and `verdict.json`? Yes, on mscluster109
      without the fidelity term and on mscluster52 with it (Runs table).
- [x] ⚠️ What is the adapter-alone compose rate at λ 0.5 and at λ 1.2 on cat × dog? 3 of 8 at
      λ 0.5 and 7 of 8 at λ 1.2. The λ 1.2 figure matches the showcase grids, so the adapter
      attached as it does there, and λ 0.5 landed part way as the design predicted, which is what
      made it the informative dose to judge at.
- [x] ⚠️ Is FK at K 16 above best of 16 at λ 0.5? No, both are 8 of 8. Picking did the composing
      work, not resampling. The two differ on picture quality instead: mean ImageReward 0.754
      steered against 0.452 for best of 16. This is the main qualifier on the support verdict and
      it belongs in every caption built on this sheet.
- [x] ⚠️ Does the control's particle 0 (`eta` 1) compose at about the adapter-alone (`eta` 0)
      rate at each λ? Close at both: 2 of 8 against 3 of 8 at λ 0.5, and 8 of 8 against 7 of 8 at
      λ 1.2. Switching the sampler to `eta` 1 does not by itself change what the correction
      delivers on this pair.
- [x] ⚠️ Does agreement between the step-10 reward and the final verdict stay above 0.5 at λ 0.5?
      Yes, 0.781, so the disagreement is 0.219 against a 0.50 bar. Across the read steps it runs
      0.812, 0.781, 0.969, 0.961, 0.969, so the reward is informative early here, unlike the
      plain-product run whose step 0 sat at 0.38.
- [x] ⚠️ How many of the five resamples were informative per run at each λ? 4.875 of 5 at λ 0.5
      and 5.0 of 5 at λ 1.2, against 0.625 in the plain-product run. The prediction that λ 1.2
      would rarely separate particles was wrong: the fidelity term keeps separating them after the
      count saturates, which is what it was added to do.
- [x] ⚠️ Does the butterfly × meadow sheet keep a clear butterfly over a meadow in every FK tile?
      Yes, no steered tile lost either concept, and the steered column has the highest mean
      ImageReward of the six (1.675 against 1.240 for the adapter alone). Its compose numbers are
      butterfly counts and are not read as composition.
- [x] ⚠️ Is the picked FK particle's ImageReward above the adapter-alone tile's and the control
      particle 0's, on average over the 8 seeds at each λ? Yes at both doses and on both pairs. At
      λ 0.5: 0.754 steered against −0.768 adapter alone and −1.468 control. At λ 1.2: 1.162
      against 0.049 and 0.007. On butterfly × meadow: 1.675 against 1.240 and 1.323. The steered
      pick also outscores the joint prompt's own render (0.237) on cat × dog at both doses.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] ⚠️ Does the count scorer's verdict survive an identity check, given that it counts a generic
      animal and never checks that one is a cat and the other a dog? Partly. By eye at λ 0.5, all
      8 steered tiles show two separate animals but only 5 show one clear cat beside one clear
      dog: seed 14 is two dogs, and seeds 11 and 13 pair a small animal with one that keeps cat
      ears and a dog muzzle. At λ 1.2 the same read gives 7 of 8, with seed 12 showing two cats. A
      per-concept detector pass cannot arbitrate: querying "a cat" and "a dog" separately returns
      both on every tile of every column, including single fused animals, so identity rests on the
      eye. (Raised by the sheet, checked with `scripts/fk_steering/identity_check.py`.)
- [ ] ⚠️ Is the compose rate per unit of compute better than the single corrected render's? Sixteen
      particles with two network passes each cost about 25 minutes per seed against seconds for one
      render, and nothing here measures that trade. (Raised by the cost line in the Runs table.)
- [ ] 🟡 Would selection help at ranks 8 and 16, or on a pair the adapter transfers to less well?

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ⚠️ **Was the comparison fair?** Yes. Same K, `eta`, noise draws, steps, guidance,
      resolution, adapter and λ between the control and the steered run; only the weighting
      differs. The `eta` 0 adapter-alone column sits on the sheet as a reference, and the verdict
      compares the steered column against the `eta` 1 control rather than against it.
- [x] ⚠️ **Was the measuring tool sound?** For counting bodies, yes: every tile counting 2 was
      opened or read on the sheet and shows two separate animals, with none of the collar-tag
      errors plan 09's re-score found. For the question a reader will actually ask, no: the tool
      counts animals and cannot tell a cat beside a dog from two dogs, and the identity read above
      says that bites on 3 of 8 tiles at λ 0.5. The compose numbers stay as the instrument reports
      them, with the eye read printed beside them.
- [x] ⚠️ **Did the run respect the environment?** Yes. Everything under `/datasets`, `co3` on
      mscluster52, the log printed `torch sees NVIDIA GeForce RTX 3090`, the job held the node's
      only card alone, and `adapter.json` reports 420 tensors from checkpoint step 30050. Node,
      device and PID are in the Runs table.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| selection on top of the correction adds, or does not | the adapter-alone rate at each λ beside the FK rate, so the λ 1.2 gap is read against its headroom |
| the guide is the judge | the best-of-16 column, as in plan 10 |
| this is a combination result | that it neither changes plan 10's null nor sizes the sampler's share |
| the control pair | that the detector counts butterflies there, so its rates are not composition |
| the picked image is cleaner | that ImageReward was in the reward as a tie-breaker, its weight, and the mean score per column |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| Whether an instrument can tell a cat beside a dog from two dogs | a scorer that reads identity per instance, which neither the instance count nor a per-concept query does | every identity claim in a caption built on these sheets |
| Whether selection helps at ranks 8 and 16, or on a pair that transfers less well | one more run of the same script with a different checkpoint and pair | how general the combination sentence can be |
| The compose rate per unit of compute against the single corrected render | timing both to a fixed budget | whether the paper recommends this or only reports it |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

The finding is written:
[does steering the corrected sampler produce clean pairs](../../../report/is-the-gap-the-samplers-or-the-models/does-steering-the-corrected-sampler-produce-clean-pairs.md).
Next is an instrument that reads identity per instance, since every claim about a cat beside a dog
currently rests on the eye.
