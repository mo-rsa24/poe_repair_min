# 🧪 Review: does CFG++ re-noising, the APG projection or keeping PoE's norm make the corrected two-animal render crisp?

**Both stages have run; every cell is scored and stripped (W&B run `2bbd7npp`). No fix makes the render crisp.** This file judges [the design](../plans/experiments/23-three-inference-time-fixes-for-the-soft-corrected-render.md). Its answer fills the showcase wall's "fix" row if a cell is supported, and otherwise says the softness is the correction's direction and not how it enters the step.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/23-three-inference-time-fixes-for-the-soft-corrected-render.md) | the three fixes, the identity check, the code, the constants |
| **this file** | **the verdict: which fix, if any, keeps the animals and returns the sharpness** |

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

- **Full window**: the rank-32 step-30050 correction at λ 1.2 on all 50 steps rendered through this plan's loop with the hook off (`full_1.2_sameloop`), the column every fix is judged against. Plan 14's render of the same configuration is the identity check's reference.
- **CFG++**: the corrected prediction forms the running estimate; the DDIM re-noising uses the plain PoE prediction.
- **APG**: the part of the correction parallel to the PoE prediction is dropped (`APG_ETA` 0), the orthogonal part kept.
- **Norm kept**: the corrected prediction rescaled to the PoE prediction's norm each step.
- **Sharpness**: Laplacian variance at 1024 px, compared per seed against the same seed's full-window render. **Sharper-than-full count**: seeds of 8 where the fix's sharpness exceeds the full window's.
- **Compose count**: seeds of 8 with two or more animal instances by the validated detector. **Kept count**: the full window's composing seeds that the fix also composes.
- **d(Mono)**: DINOv2 cosine distance to the same seed's joint-prompt render; lower is nearer.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: an intervention on how the correction enters the sampler step, no training). A missed bar on all three fixes closes the plan and says the softness is in the correction's direction; the training-side plans (15, 16, 20) then carry the fidelity question alone.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| `STAGE=all`: reference copy, identity check, 24 fix renders, score, strips | tests the claim | 2026-09-06 03:26, mscluster108 device 1 (Quadro RTX 8000), bash PID 321031, python PID 321116, `co3`, `nohup` over SSH; log `/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/crisp_fix_all.log` | about 30 GPU-minutes | `/datasets/mmolefe/poe_repair_min/outputs/showcase/crisp_fix_strip/{identity_check.json,results.json,renders/,strips/}`, W&B run `crisp_fix_strip_r32_030050`, id `2bbd7npp` | done 03:47; the work completed, but the launcher file was edited while this instance was running and bash read a shifted line after the last python call, so `DONE stage=all` was never printed |
| `STAGE=baseline`: the same-loop full window on 8 seeds, re-score, re-strip | tests the claim | queued 2026-09-06 03:33 behind PID 321031 by a waiter on mscluster108, device 1; log `.../logs/crisp_fix_baseline.log` | about 10 GPU-minutes | the `full_1.2_sameloop` renders, `results.json` with `reference_full_window` and `fp16_floor_grey_levels`, strips | done 03:58 |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does any fix keep the compose count within one seed of the same-loop full window and come out sharper than it on at least 6 of the 8 seeds, paired per seed, on cat × dog?**
      The bar, in source as `SUPPORT_MAX_COMPOSE_LOSS = 1`, `NULL_MIN_COMPOSE_LOSS = 2`, `SUPPORT_MIN_SHARPER_SEEDS = 6`, `NULL_MAX_SHARPER_SEEDS = 4` in `verdict_for_fix` of `scripts/showcase/crisp_fix_strip.py`:
      **support** if `compose_n(fix) ≥ compose_n(full) − 1` and `sharper_than_full_n ≥ 6`;
      **null** if `compose_n(fix) ≤ compose_n(full) − 2` (the animals go with the fix) or `sharper_than_full_n ≤ 4` (nothing gained);
      **inconclusive** otherwise (the animals stay and the fix is sharper on exactly 5 seeds).
      Predictions, written before the run: the norm-kept cell is sharper on most seeds because the correction finding says the reachable part of the correction shortens the prediction; CFG++ keeps every full-window seed because the running estimate is unchanged and only the re-noise differs; APG at η 0 is the one most likely to lose seeds, because the parallel part it drops is where the damping lives and the damping may be part of what composes.
      Answer: **⚪ null for all three, with one cell that passes the bar's letter and fails in substance.** Against the same-loop full window (compose 6 of 8, seeds 11 and 14 failing; sharpness per seed 38, 52, 94, 36, 12, 43, 170, 28):
      **CFG++** compose 6 of 8 but keeps only 4 of the full window's 6 seeds, sharper on 8 of 8, so `verdict_for_fix` prints `support`. The strips say otherwise: seed 9 is a guitar and flags with no animal (detector count 0), seeds 11 and 16 are flower posters (counts 3 and 3, all false positives), seed 14 is a building. Its d(Mono) is 0.893 mean, further from the joint-prompt image than plain PoE's 0.643, and its both-ness 0.209 equals plain PoE's 0.200. The sharpness gain is the Laplacian counting the edges of a poster. The rule as written has no fidelity guard, so its letter is met; the cell is a null. The port itself is the likely cause: CFG++ re-noises with the unconditional prediction under a guidance scale in [0, 1], and re-noising a PoE chain with a prediction that differs from the one that formed x0 by a vector of norm 13 to 24 per step drives the latent off the prompt within a few steps.
      **APG at η 0** compose 8 of 8, keeps 6 of 6, sharper on 2 of 8, mean sharpness 48.9 against 59.4: null on the sharpness bar. **Norm kept** compose 8 of 8, keeps 6 of 6, sharper on 2 of 8, mean 50.3: null. The two tiles are near-indistinguishable from the full window by eye on every seed; on seeds 11 and 14 a second small animal appears where the full window has one, and both-ness rises from 0.406 to 0.456 and 0.457, while d(Mono) rises from 0.490 to 0.537 and 0.539 (further from Mono on 5 of 8 seeds). From `crisp_fix_strip/results.json` (`summary`) and the eight strips.
      **What the null says.** Changing how the correction enters the DDIM step does not touch the softness. The training-side plans (15 weight decay, 20 EMA, 16 clean-estimate loss) and the re-synthesis routes (plan 14's re-noise sweep, plan 19's stochastic sampler) are the remaining ways to it.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] 🟡 **Did the identity check pass?** The hook set to off on seed 9 against plan 14's full-window render of seed 9: mean absolute pixel difference at or under 6.0 grey levels (`IDENTITY_MAX_MEAN_ABS_DIFF_CROSS_DEVICE`) for the chain to continue; 1.0 recorded as the same-device figure. If it fails the 6.0 bar, nothing below is read.
      Answer: **2.899 grey levels** on mscluster108 device 1, the card plan 14 also rendered on: under the 6.0 bar, over the 1.0 figure. The two loops are the same code except that this one combines the two predictions in fp32 and casts back, and plan 14 combined them in fp16; over 50 steps that rounding difference is worth about 3 grey levels, the size of the cross-device fp16 drift seen on the rank-8 grid. So plan 14's render cannot be the bar's reference without that floor sitting inside every paired sharpness read, and the reference is the same configuration rendered through this loop (`full_1.2_sameloop`), decided before any fix render was scored. The per-seed floor between the two is recorded in `results.json` under `fp16_floor_grey_levels`. From `crisp_fix_strip/identity_check.json`.
- [x] 🟡 **Does the correction shorten the PoE prediction on the live trajectory?** The parallel coefficient `⟨correction, PoE⟩ / ‖PoE‖²` per step, mean over seeds, from the diagnostics files. Negative on steps 0 to 9 with the norm ratio of the corrected prediction below 1 says yes, and matches the correction finding's in-span damping read off the cache. Positive or near zero says the damping the cache showed is not what the adapter does at inference.
      Answer: **yes in sign, no in size.** The coefficient is negative on every step and every seed, from about −0.005 at step 0 to −0.04 at step 33, back to −0.02 at the end; the norm of the corrected prediction sits at 0.994 of PoE's on steps 0 to 9 and bottoms at 0.97 around step 32. The parallel share of the correction is 0.13 early and 0.17 over the run. So the adapter's correction is 85 percent orthogonal to PoE's prediction and shortens it by 1 to 3 percent, far from the cache read's re-weighting of each expert from 7.5 to 1 to 3. The norm-kept cell restores that 1 to 3 percent and nothing changes; the norm is not where the softness lives. From `correction-parallel-share-and-norm-per-step.png` and the per-seed diagnostics files.
- [x] ✅ **Do the fixes move d(Mono) the same way they move sharpness?** `nearer_mono_than_full_n` per fix beside `sharper_than_full_n`. Descriptive, not part of the bar: a fix that is sharper and further from Mono has traded fidelity for edges.
      Answer: **no, and that is what exposed the CFG++ cell.** CFG++: sharper on 8 of 8, nearer Mono on 0 of 8. APG and norm kept: sharper on 2 of 8, nearer Mono on 3 of 8. The two reads disagree exactly where the eye disagrees with the Laplacian.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was written with the answer already visible.

- [x] ✅ **Should the bar have carried a fidelity guard?** Yes. A cell whose d(Mono) is worse than plain PoE's has left the prompt, and no sharpness count should be read on it. The rule in source stays as written; the next plan that pairs a sharpness count with a compose count adds `d_mono_mean(cell) ≤ d_mono_mean(poe)` as a third condition, written before its run.
- [x] 🟡 **How large is the fp16 floor per seed, and does it move the compose count?** Plan 14's full window against the same configuration through this loop, mean absolute pixel difference per seed: 2.9, 8.2, 1.5, 2.6, 1.5, 5.4, 3.0, 3.3 grey levels (seeds 9 to 16), mean 3.55. Seed 10 sits above the 6.0 cross-device bar the identity check uses, so a one-seed identity check can pass on a seed whose neighbour would fail. The compose count agrees on 8 of 8 seeds and the same-loop render is sharper on 3 of 8, so the floor moves pixels and not verdicts here. From `results.json`, `fp16_floor_grey_levels`.
- [ ] ⚠️ **Is the CFG++ collapse the port or the idea?** CFG++ re-noises with the unconditional prediction at a scale in [0, 1] and derives that from an inverse-problem view of text guidance; here the re-noise prediction is the PoE prediction and the scale is 1.2. A faithful port would re-noise with `eps_∅` and restate λ on their scale. Untested.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Every fix runs through the same loop from the same cached noise with the same adapter and λ; only the hook setting differs. The bar's reference is the same loop with the hook off; plan 14's renders are the identity check (2.9 grey levels on seed 9) and the Mono and plain PoE columns. 8 seeds in every cell.
- [x] 🟡 **Was the measuring tool sound?** The scorer is the validated instance count, and on the CFG++ tiles it counted flowers as animals (3 on seeds 11 and 16); the count was validated on animal renders and has no off-prompt case. Sharpness is plan 07's function, read per seed so the sketch seeds cannot set a band, and it counts a poster's edges as sharpness. d(Mono) is the DINOv2 distance plan 19 and the landing finding use, and it was the read that held.
- [x] ✅ **Did the run respect the environment?** Output under `/datasets`, the launcher's guards passed (disk 34 percent, device 1 at 1 MiB), node, device and PID in the log header and here, `torch.cuda.is_available()` true on the pinned device.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a "fix" row on the wall, if supported | which fix, its one-line rule in the sampler, the compose count against the full window, and that sharpness is a paired per-seed Laplacian read on one pair |
| the fidelity caveat, if null | that three sampler-side fixes were tried and which count each lost, so the caveat points at the correction's direction and not at the sampler |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| APG's rescaling and momentum terms are not tried; only the projection at η 0 is | one more cell per term | nothing; the projection is the part APG's ablation credits |
| no control pair | a butterfly × meadow column through the same hook | nothing now: no cell is supported |
| a faithful CFG++ port (re-noise with `eps_∅`, λ on their [0, 1] scale) | one more cell, 8 renders | nothing; the direction of the correction, not its entry into the step, is where the other two cells point |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Task 3.2 in the design: `/sync-plan-tree plans/01-showcase-the-trained-lora/`, so the null propagates to the wall plan and the fidelity caveat is written as a bounded sentence: three sampler-side entries of the correction were tried and none sharpened the render.
