# 🧪 Review: does a correction confined to the early steps, followed by a plain-PoE tail, give two animals at plain-PoE sharpness?

**Every stage has run and every cell is scored (W&B run `kvtuv0q7`). No cell clears the bar, because the premise it rests on fails: under Laplacian variance the full-window run is not softer than plain product-of-experts.** This file judges [the design](../plans/experiments/14-correct-early-then-clean-up.md). Its answer fills the showcase wall's "correct early" row if a cell is supported, and otherwise bounds the paper's fidelity caveat.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/14-correct-early-then-clean-up.md) | the schedules, the tail, the re-noise cell, the code, the constants |
| **this file** | **the verdict: which cell, if any, keeps the animals and returns the sharpness** |

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

- **λ**: the multiplier on the rank-32 adapter's change to the product-of-experts noise prediction; 0 is plain PoE, 1.2 the shipped setting. A **schedule** is λ as a function of the denoising timestep.
- **Full window**: λ 1.2 on all 50 steps, the adapter-alone column.
- **Hard cut**: λ 1.2 on steps 0 to 10 (timesteps 981 down to 781), then 0. **Decay**: the same, then a straight line to 0 at step 20 (timestep 581).
- **Sharpness**: Laplacian variance of the greyscale image, higher is crisper, the function plan 07 used. On the 1024 px renders for the cells; on the 256 px saved frames for the per-step read. The two are not on one scale.
- **The plain-PoE band**: the lowest to highest sharpness over the 8 plain-PoE seeds of a pair.
- **Compose count**: seeds of 8 where the validated detector counts two or more animal instances.
- **Both-ness**: projection of the DINOv2 embedding toward the "a cat and a dog" cloud on the axes already fitted for the landing finding; cat × dog only.
- **The re-noise level**: the timestep the committed image is noised back to before plain PoE finishes the run; `RENOISE_LEVELS_T = (481, 381, 281, 181)`, steps 25, 30, 35 and 40 on the 50-step grid. Higher means more noise added and more texture re-synthesised by the tail. **The re-noise source** is which committed image is noised: `x0s20`, the decay run's running estimate at step 20 (the plan's cell), or `final`, the decay run's finished render. Cells are named `decay_10_20_renoise_<source>_t<level>`.
- **Butterfly present**: on the control pair, whether the detector finds a "butterfly" box at confidence 0.30 or more (`BUTTERFLY_PRESENT_CONF`). The validated compose rule counts animals and a meadow is not one, so on this pair the rule cannot apply; this is the descriptive presence read the control uses instead.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: an intervention on when the correction acts, no training). A missed bar closes the plan; the fidelity caveat in the paper is then written as a bounded sentence and no follow-on opens unless the re-noise cell also fails, in which case the softness is the correction's own and that becomes the finding.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| task 1: sharpness of the saved frames against step | tests the claim, CPU | 2026-09-05 16:49, session node mscluster85 | 2 minutes | `artifacts/results/does-correcting-early-then-cleaning-up-restore-sharpness/sharpness-over-denoising-steps.{png,json}` | done |
| tasks 2 and 3.1 (`STAGE=all`, first half): detachment proof and the 80-render schedule grid | tests the claim | 2026-09-05 16:51 to 18:22, mscluster108 device 1 (Quadro RTX 8000), bash PID 293034, python PID 294004, `co3`, `nohup` over SSH; log `.../logs/correct_early_all_part1.log` | 90 minutes; a corrected 50-step render costs 62 s on this card (two UNet passes per step) | `.../correct_early_then_clean_up/detach_check.json`, `renders/<pair>/<condition>/seed_<n>.png` (80 of 80) | done; the chain then died at scoring: the DINOv2 embedder on CPU hit the CUDA-only xformers kernel (poe-mem-002 in a new guise), fixed by embedding on the GPU |
| tasks 3.2 to 4.3 (`STAGE=rest`): score, 200-step tail, re-noise, score | tests the claim | 2026-09-06 02:30, mscluster106 device 1 (Quadro RTX 8000), bash PID 2509686, python PID 2509777, `co3`; log `.../logs/correct_early_rest.log` (mscluster108 device 1 had been taken by the corrector window sweep of another session in the meantime, and the launcher's guard refused it) | about 2.5 GPU-hours, the 200-step tail being most of it | `results.json`, `sheets/`, 272 renders and 26 sheets under the same root | done 2026-09-06; a later session extended the re-noise stage past this plan's design, adding four re-noise levels (t 181, 281, 381, 481) and a variant that re-noises the step-20 running estimate rather than the latent, so the folder holds 16 cells where the design named 6 |
| task 4.5: per-seed strips of the cells rendered so far (Mono, plain PoE, full window, decay) | descriptive, CPU | 2026-09-06 02:55, session node mscluster85 | 3 minutes | `strips/strip-<pair>-seed_<n>.png`, W&B `strips/` images and the `strips_by_seed` table on run `kvtuv0q7` | done |
| task 4.4 (`STAGE=sweep`): the re-noise level sweep (2 sources × 4 levels × 8 seeds × 2 pairs, 128 tails behind 16 head renders), score, strips | tests the claim | queued 2026-09-06 02:58 on mscluster106 device 1 behind bash PID 2509686 (a waiter starts the launcher when that PID exits; every other device on the biggpu nodes and the session node's card carried a process); log `.../logs/correct_early_sweep.log` | about 45 GPU-minutes | the sweep renders under `renders/<pair>/decay_10_20_renoise_<source>_t<level>/`, `results.json`, sheets and strips, W&B | queued |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] 🟡 **Does any cell keep the compose count within one seed of the full-window run and return its mean sharpness to the plain-PoE band, on cat × dog?**
      The bar, in source as `SUPPORT_MAX_COMPOSE_LOSS = 1` and the band rule in `verdict_for_cell` of `scripts/showcase/correct_early_then_clean_up.py`:
      **support** if `compose_n(cell) ≥ compose_n(full window) − 1` and `sharpness_mean(cell) ≥ plain band lower edge`;
      **null** if `compose_n(cell) ≤ compose_n(full window) − 2` (the animals go with the correction) or `sharpness_mean(cell) ≤ sharpness_mean(full window)` (nothing gained);
      **inconclusive** otherwise (the animals stay, sharpness moves up, and does not reach the band).
      The premise is checked first: if the full-window mean sharpness is already at or above the plain band's lower edge at 1024 px, there is no softness to fix on this pair at this checkpoint, and every cell reads 🟡 on that fact.
      Answer over all 16 cells: **🟡 on the premise.** Full-window mean sharpness 59.9 against a plain band of 12.3 to 499.5, so the gate fires and both schedules read inconclusive under the rule. The rule's own arithmetic, read past the gate, would pass both: hard cut 5 of 8 and decay 5 of 8 against the full window's 6, mean sharpness 89.0 and 90.3, both inside the band and above the full window's 59.9. Per seed, the decay is sharper than the full window on 7 of 8 seeds and the hard cut on 6 of 8, while both stay below plain PoE's mean of 152, which the three sketch seeds set. The tail and re-noise cells do not change this: the best compose count in the folder is 7 of 8 (re-noising the step-20 running estimate to t 381) at mean sharpness 44.9, and the best mean sharpness among cells that keep six seeds is 90.5 (re-noising the finished render to t 181). From `results.json` (W&B run `kvtuv0q7`), 272 renders and 26 sheets under `/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up/`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ **When does the corrected run's sharpness leave the plain run's band, on the saved frames?** Constants `EARLY_STEP = 20`, `LATE_STEP = 50`; the band is the plain-PoE seeds' min to max at that step. **Late** if the λ 1.2 mean is inside the band at step 20 and below its lower edge at step 50: a clean tail can reach the softness. **Committed** if the mean is already below the lower edge at step 20: only a re-noise can. **Inconclusive** if neither (inside the band at both, or above it).
      Answer: **committed**. Laplacian variance of the 256 px running estimate, mean over 8 seeds: λ 1.2 run 11.9 at step 20 against a plain-PoE band of 25.9 to 342.4 (plain mean 132.1); at step 50 the λ 1.2 run reads 60.4 inside a plain band of 20.2 to 126.9 (plain mean 54.7). The corrected running estimate is smoother than every plain seed at step 20 and has caught up by step 50. From `sharpness-over-denoising-steps.json`, field `read`. What the rule did not anticipate: the plain band is wide because three seeds (11, 14, 15) render as line drawings with high edge density, so the band's lower edge, not its mean, is the part of the read that holds.
- [x] 🟡 **Does the hard cut lose seeds that the decay keeps?** The commit step of the corrected run reaches 35 on some seeds; if the cut at step 10 composes fewer seeds than the decay, those seeds needed the correction past step 10.
      Answer: no, by the detector; the two schedules compose the same five seeds (9, 12, 13, 15, 16) and both lose seed 10, which the full window keeps (its count here is 6 of 8: seeds 11 and 14 fail; the earlier filed render of the same configuration counted seed 11 as two, so seed 11 sits on the detector's edge). Where the two schedules differ is in both-ness: decay 0.424 against hard cut 0.367 (full window 0.415, plain PoE 0.200), and in the eye's read of the sketch seeds: on seed 11 the decay draws a kitten beside a second animal and the hard cut a single cat; on seed 15 the hard cut draws one lying cat that the detector counts as two. The detector and the eye disagree on three of the 24 corrected tiles, all sketch-style seeds, so the seed-count question is 🟡 on this pair. From `results.json`, `summary.a_cat__x__a_dog`, and the two sheets `sheet-a_cat__x__a_dog-{hardcut_10,decay_10_20}.png`.
- [x] ✅ **Does any cell break the control pair?** On butterfly × flower meadow the butterfly is present on 8 of 8 seeds in all 16 cells, and no cell's mean sharpness falls below that pair's plain band floor of 18.0. Means: plain PoE 60.3, hard cut 124.9, decay 156.0, full window 519.9, the re-noise cells 51.1 to 163.1. Control intact everywhere.
- [x] ⚪ **Does the 200-step tail sharpen more than the 50-step tail at the same switch-off noise level?** Compared against plain PoE at 200 steps, since a longer run may move the band itself.
      Answer: no. The decay schedule at 200 steps reads mean sharpness 92.4 against 90.3 at 50 steps, a 2% move, while plain PoE's own mean rises from 152.2 to 225.2 over the same change. The longer run sharpens the plain baseline and leaves the corrected cell where it was, so the tail was not short of steps. Compose count is 5 of 8 at both lengths, both-ness 0.388 at 200 steps against 0.424 at 50. The switch-off lands at the same noise level: the 200-step timestep grid contains both t 781 and t 581, checked in the log before the renders.
- [x] 🟡 **Does the re-noise cell reach the band where the tails did not?** If the per-step read says committed and the re-noise cell is supported, the softness is in the structure and re-synthesis fixes it.
      Answer: it recovers composition but not sharpness, and the gate below makes every reading inconclusive. Re-noising the schedule's latent at step 20 back to t 281 and finishing with plain PoE composes 6 of 8, matching the full window, at mean sharpness 62.8 against the full window's 59.9. A later session swept the re-noise level and the re-noise target: taking the step-20 running estimate instead of the latent reaches 7 of 8 at t 381, the highest count of any cell in the folder, at mean sharpness 44.9, the lowest. Re-noising the finished render instead reads 5 to 6 of 8 at mean sharpness 89 to 97. Nothing in the sweep both keeps six seeds and leaves the full window's sharpness behind by more than the schedules already did.
- [x] ✅ **Did the detachment proof pass?** Mean absolute pixel difference between plain PoE rendered before the adapter was attached and plain PoE rendered after a windowed run in the same process, at or under `DETACH_MAX_MEAN_ABS_DIFF = 1.0` grey levels; and the λ 0 render against the cached `poe.png` at or under `IDENTITY_MAX_MEAN_ABS_DIFF = 6.0` (cross-device fp16 drift was about 2 of 255 on the rank-8 grid; the contaminated case read 11 to 31).
      Answer: yes. Cat × dog seed 9 on mscluster108 device 1: before against after **0.0** grey levels (identical), before against λ 0 through the scheduled sampler 0.0, either against the cached `poe.png` 1.32, and the hard-cut render against plain 40.6, so the adapter acted inside its window and nowhere after. From `correct_early_then_clean_up/detach_check.json`.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [x] ❌ **Is the premise (the full-window run is softer than plain PoE) true under Laplacian variance at all?** No, and this is the result that governs every cell. (raised by task 1's step-50 read and by a pre-launch read of the existing rank-32 renders under `figure_r32_030050/renders/full/`: plain PoE per seed 16, 28, 369, 12, 13, 270, 517, 22, mean 156, band 12 to 517; λ 1.2 per seed 38, 48, 91, 38, 12, 35, 173, 27, mean 58.) The full-window mean sits inside the plain band because the band's floor is 12, so the pre-registered premise check reads "no softness to fix" and every cell will read 🟡 on it. Per seed the picture is split: seeds 11, 14 and 15 (the line-drawing seeds) lose most of their edge density under the correction, seeds 9, 10 and 12 gain some. The measure counts edges, so a sketch scores as sharper than a photograph. The bar stays where it was written; a per-seed paired read is reported beside it as post-hoc.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Every compared cell runs through the same sampler, from the same cached noise, differing only in λ(t) and, for the tail cell, the step count. Mono is a reference column, not a compared cell. Every cell carries 8 seeds (`n` in each `summary` block). The 200-step cells are compared against plain PoE at 200 steps, not at 50. The control pair's seeds 13 to 16 take their initial noise from the cat × dog cache, since the two pairs share one noise per seed and the control's own cache stops at seed 12.
- [x] ⚠️ **Was the measuring tool sound?** The scorer is the validated instance count on cat × dog; on the control pair it is a presence read and says so. Sharpness is the same function plan 07 used, and it does respond to blur: blurring one render at sigma 2 takes it from 16.4 to 0.2. It does not isolate blur. On the eight plain-PoE renders it reads 12 to 517, and the top three are the seeds that draw as line art, so across seeds it measures drawn edge density and not focus. The detector and the eye also disagree on three of the 24 corrected tiles, all sketch seeds. Treat every sharpness number here as within-seed evidence only.
- [x] ✅ **Did the run respect the environment?** Output under `/datasets` only, the launcher's disk, python, memory, fault and CUDA guards passed on both nodes, node, device and PID in the log header and in the Runs table above, `torch.cuda.is_available()` true on each pinned device. The guard did its job once: when mscluster108 device 1 was claimed by another session's corrector sweep between stages, the relaunch refused it rather than starting on a foreign process, and the work moved to mscluster106 device 1.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a "correct early" row on the wall, if supported | the schedule in timesteps, the compose count against the full window, and that sharpness is Laplacian variance on one pair |
| the fidelity caveat, if null | which window the softness sits in (from the per-step read), so the caveat is bounded rather than vague |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| the refiner cell: the SDXL refiner is not on this cluster's Hugging Face cache and is not downloaded here | a deliberate download to `/datasets` and one cell | nothing; the base-tail cells answer the question without it |
| the premise gate: Laplacian variance counts edges, so three sketch-style seeds (11, 14, 15) set a plain band floor of 12.3 and a mean of 152.2, and the rule can never fire on this pair | a sharpness measure that separates blur from drawn line density, or a paired per-seed bar written before the next run | any future fidelity bar on cat × dog, including the corrector and clean-tail finding's 10% band |
| no finding file: this result is not in `report/` | the finding written into `report/is-the-gap-the-samplers-or-the-models/`, beside the corrector and clean-tail finding whose `Still open` names this plan | a reader asking whether an early-only schedule was tried |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Task 1.1 in the design: the per-step sharpness read on the session node.
