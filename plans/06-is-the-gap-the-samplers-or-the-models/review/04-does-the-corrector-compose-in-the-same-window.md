# 🧪 Review: does the corrector's compose rate peak in the same window the injected correction does?

**Everything has run. The corrector composes in no window (0 of 4 in every column). On the
eight-seed sheet it is plain PoE's compose rate exactly (0 of 8) and changes the style. On the
adapter's tail it is a null (sharpness +8%, composition held). The clean tail, the adapter early
and the frozen model after, is a null at the 0.05 bar too, with the hand-off alone the one lever
that moves the right way (DINOv2 distance to the joint render 0.472 to 0.434) while the corrector
on the frozen score moves the wrong way and adds grain. W&B run
`prime_lab/poe-repair-animals-compose/s61hldbc`.** Every question below was written before any corrector existed. This file
judges [the corrector window design](../plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md).
It is the only thing this scope can say about the compose-decisive early window, because
[the residual measurement at step 26](03-what-is-left-once-the-chain-settles.md) cannot
attribute there by construction.

## Recommended prompt (when the run lands)

```
/analyze-run the corrector window sweep, 4 seeds by 10 window positions on a_cat__x__a_dog
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md) | the nine positions, the layout it must match, and where the figure files |
| **this file** | **the verdict: not yet run** |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | the measurement this waits on, though not strictly, and the `k` these renders run at |
| [the timing verdict](../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md) | the injected-correction figure this is compared against, render for render |

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

- **The window**: the stretch of denoising steps during which the corrector is allowed to act. Ten
  steps wide, slid to nine positions across the 50, plus an all-50 condition.
- **The injected correction**: the cached `r_t` added back into plain product-of-experts, which is
  the mechanism the existing set of window renders used. The corrector is a different mechanism
  aimed at the same failure.
- **Composed**: the detector scored the picture as two separate animals rather than one blended
  one. The green border on a render encodes exactly this.
- **The peak window**: the column with the most seeds composed out of four.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** There is no pass or fail here. Same window and different window are both
results, and both are reported. What would make the run worthless is a layout that does not match
the figure it is compared against, since the comparison is the entire read.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The corrector across ten window columns, 4 seeds × 10 columns | Tests the claim | 2026-09-05 23:18 to 2026-09-06 00:56, `nohup` on mscluster108 device 1 (RTX 8000, `co3`), pid 304312, commit 0150704, `c = 3`, `k = 20` | 40 renders, 1.9 min per window column and 7.6 min per all-50 column on that card, 1.6 h in all, then scored | `corrector/window_curves_mcmc.json`, renders under `corrector/window/pairs/a_cat__x__a_dog/seed_<n>/poe_langevin_k020_c3000[_w<s>-<e>]/`, figure `mcmc/samples-as-a-ten-step-corrector-window-slides.png` with sidecar | ✅ done: 0 of 4 composed in every column |
| The eight-seed sheet, 2 pairs × 8 seeds × (joint prompt, plain PoE, corrector on all 50 steps) | Tests the claim | 2026-09-06 00:56 to 03:14, `nohup` on mscluster108 device 1 (RTX 8000, `co3`), pid 306543, commit 0150704, `c = 3`, `k = 20` | 48 renders, 8.3 min per cell (two references plus one corrector render), 2.3 h | `corrector/sheet_scores.json`, references under `corrector/sheet/references/`, corrector renders under `corrector/sheet/pairs/`, sheets `corrector-<pair>-eight-seed-sheet.png` | ✅ done: cat × dog joint 8, PoE 0, corrector 0 of 8; control pair 8, 8, 8 |
| The clean tail, 2 pairs × 8 seeds × cutoff `{20, 30}` × `k ∈ {0, 5, 20}`, adapter early then the frozen model, corrector on the frozen score | Tests the claim | 2026-09-06 02:42 to 05:05, `nohup` on mscluster85 device 0 (RTX 3090, `co3`), pid 355987, commit 87d6cb2, `c = 3` | 96 renders, 34 s at `k = 0`, about 60 s at `k = 5`, 150 s at `k = 20` on a 3090 (one call per Langevin step on the frozen score), about 3.2 h on one card | `corrector/clean_tail.json`, renders under `corrector/clean_tail/<pair>/seed_<n>/`, sheets `corrector-clean-tail-<pair>-eight-seed-sheet.png` | ✅ done: branch **null**, best gain +0.038 at cutoff 30, `k = 0` |
| The W&B log of every sheet and sidecar | records | 2026-09-06 05:20, from mscluster85 | one run | `prime_lab/poe-repair-animals-compose/s61hldbc`: six sheets and the two figures as images, every sidecar and verdict as the artifact `scope06-corrector-sidecars` | ✅ |
| The tail condition, 2 pairs × 8 seeds × `k ∈ {0, 5, 20}` on the rank-32 λ 1.2 run, corrector on steps 35 to 49 | Tests the claim | 2026-09-06 00:02 to 02:01, `nohup` on mscluster85 device 0 (RTX 3090, `co3`), pid 321793, commit 0150704, `c = 3`, adapter `lora_step_030050.pt` (210 modules matched, 420 tensors loaded) | 48 renders: 42 s at `k = 0`, 105 s at `k = 5`, 293 s at `k = 20` per render, 2 h in all | `corrector/tail_fidelity.json`, renders under `corrector/tail/<pair>/seed_<n>/`, sheet `corrector-on-adapter-tail-<pair>-eight-seed-sheet.png` | ✅ done: branch **null**, mean sharpness +8% from `k = 0` to `k = 20`, composed 7, 8, 7 of 8 |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ❌ **Does the corrector's compose rate peak in the same window the injected correction
      does?** It has no peak: 0 of 4 seeds compose in every column, the nine ten-step windows and
      the all-50 column alike, where the injected correction composes 3 of 4 at steps 0 to 10 and
      2 of 4 at 5 to 15 on the same seeds (`window_curves.json`). The detector counts one animal
      in all 40 renders (`n_instances` 1 everywhere, cat and dog both detected at confidence
      0.88 to 0.95 in every tile, which is the one-fused-animal signature), and the eye agrees on
      all 40. So a corrector that only settles the latent into the product's own distribution
      does not do what adding the missing term does, at this compute budget (`k = 20`, 60 extra
      UNet evaluations per level inside the window). What the early windows do instead is change
      the style: seed 9 at steps 5 to 15 is a cartoon, seed 11 at 5 to 25 a line drawing, and
      windows from step 20 on leave plain PoE's picture almost unchanged column to column, the
      same late-window inertness the injected correction showed. Run against an inconclusive
      step 26 (its residual curve fired no licensed branch at any step size), which this file
      says here as the plan asked. Same window means two different mechanisms acting at the same
      moment. A different window is the stronger result and needs its own paragraph. This question
      is answered even if the measurement at step 26 returns a null, because a flat residual curve
      does not imply a flat [compose rate](../../../context/world/compose-rate.md). The corrector can
      relocate the trajectory without shrinking `‖r_t‖`. If it is run against a flat curve at step
      26, this file says so.

      > A null at step 26 means the correction's size came out the same with the corrector running
      > as without it.

- [x] ⚪ **Does a corrector on the last fifteen steps sharpen the corrected run without costing
      composition?** The condition is the rank-32 adapter at λ 1.2 on all 50 steps, plus
      `k ∈ {0, 5, 20}` Langevin steps on the corrected prediction inside steps 35 to 49, eight
      held-out seeds of cat × dog, with butterfly × meadow as the control. Sharpness is the
      Laplacian variance of the greyscale render; composed is the validated instance count.
      Thresholds in `scripts/corrector_window_sweep.py`: `FIDELITY_MIN_SHARPNESS_RISE = 0.10`,
      `FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1`.
      **Support** if the mean sharpness over the 8 seeds at `k=20` is at least 10% above `k=0`,
      `k=5` sits at or above `k=0`, and the composed count at `k=20` is within one seed of `k=0`.
      **Null** if the mean sharpness at `k=20` is within 10% of `k=0` either way and composition
      holds within one seed. **Composition breaks** if the composed count falls by two or more
      seeds, whatever sharpness did. **Inconclusive** if the control pair loses two or more of
      its composed seeds at `k=20`, since a corrector that breaks what already works licenses no
      fidelity reading, or if sharpness moves non-monotonically (a rise past the band at `k=20`
      with `k=5` below `k=0`). The sampler-share read at step 26 is untouched by this question.
      **Answer: ⚪ null**, from `tail_fidelity.json` at `c = 3`. Cat × dog, mean Laplacian
      variance over the 8 seeds: 57.1 at `k = 0`, 61.1 at `k = 5`, 61.7 at `k = 20`, a rise of
      +8.0% from `k = 0` to `k = 20`, inside the 10% band; medians 37.2, 43.9, 45.4. Per seed,
      `k = 20` is sharper than `k = 0` on 4 of 8 and `k = 5` on 5 of 8. Composed seeds by the
      validated instance count: 7, 8, 7 of 8 (seed 11 counts one animal at `k = 0` and three at
      `k = 20`; seed 10 counts two at `k = 0` and one at `k = 20`), within the one-seed bar. The
      control pair keeps 8 of 8 at every `k` by its both-concepts read, and its sharpness rises
      more (506, 568, 726: +44%), which is the meadow's fine texture returning as the chain
      settles. The chain moved inside the window (median relative displacement 0.36 at `k = 5`
      and 0.67 at `k = 20` on cat × dog seed 9), so this is a null and not a stalled instrument.
      Read: corrector steps on the corrected score at low noise neither sharpen the adapter's
      output past the band nor cost it composition at this budget; they redraw fine detail seed
      by seed in both directions. **The eye read** (instruction 4, Claude, veto after) on
      `corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png`: row by row, `k = 20` against
      `k = 0` is the same picture with fur and edges redrawn, crisper on seeds 9, 12 and 13, the
      same on 14 and 15, softer on 16, and on seed 10 the cat dissolves into the dog's flank (the
      count drops from 2 to 1); on seed 11 the adapter's own render already shows two animals the
      detector counted as one, so its `k = 0` red frame is instrument error. No tile is the clean
      photograph the joint prompt gives; the softness lives in the adapter's tail and the
      corrector on that same score keeps it. On the control sheet every tile keeps its
      butterfly. The number and the eye agree: null.
- [x] ⚪ **Does handing the tail to the frozen model, with or without a corrector on its score,
      return the adapter's renders to plain-PoE sharpness without losing the composition?** The
      conditions: the rank-32 adapter at λ 1.2 on steps `[0, cutoff)` for cutoff 20 and 30, the
      frozen model's plain PoE step after, and `k ∈ {0, 5, 20}` Langevin steps on the frozen
      score inside steps 35 to 49; both pairs, seeds 9 to 16. Baseline: the adapter alone on all
      50 steps (7 of 8 composed, from the tail run). The primary read is the DINOv2 ViT-S/14
      cosine distance between a render and the seed's joint-prompt render, the compose
      scorer's own embedder, lower being nearer the clean image the adapter is meant to reach;
      Laplacian-variance sharpness is reported beside it as a secondary read only, because on
      the plain-PoE references it is heavy-tailed (cat × dog seeds 11, 14 and 15 render grainy
      at 275 to 499 against about 15 elsewhere, mean 152 ± 183 over the 8 seeds, measured at
      02:40 before this grid ran), so a band built from it cannot separate conditions.
      Thresholds in `scripts/corrector_window_sweep.py`: `CLEAN_MIN_MONO_GAIN = 0.05` (the
      8-seed mean distance must fall by at least 0.05 against the adapter-alone run, the size
      the training-longer finding read as a real move between checkpoints) and
      `CLEAN_MAX_COMPOSE_LOSS_SEEDS = 1`. **Support** if at least one condition moves at least
      0.05 nearer the joint render with its composed count within one seed of the baseline and
      the control pair's within one seed of its own. **Composition breaks** if a condition moves
      nearer only by losing two or more composed seeds: the softness is the price of
      composition and the hand-off is too early. **Null** if no condition moves 0.05 nearer
      while holding composition. **Inconclusive** if the control pair loses two or more
      composed seeds in any condition. Written 2026-09-06 02:50, before the grid ran.

      **Answer: ⚪ null**, from `clean_tail.json`. Cat × dog, 8-seed mean DINOv2 distance to the
      joint render (adapter alone 0.472, plain PoE 0.643): adapter on steps 0 to 19 then the
      frozen tail 0.462, 0.467, 0.518 at `k = 0, 5, 20`; adapter on 0 to 29 then the frozen tail
      0.434, 0.469, 0.490. The best gain is +0.038 at cutoff 30 with no corrector, under the 0.05
      bar; the corrector on the frozen score moves away from the joint render at `k = 20` (−0.046
      and −0.018) and raises Laplacian variance from 57 to 112 and 94, which the eye reads as
      grain. Composed seeds by the validated count: 5, 6, 7 and 6, 7, 7 of 8 against the adapter's
      7, so `k = 0` at cutoff 20 misses the one-seed bar and the rest hold it. The control pair
      keeps 8 of 8 in every condition. **The eye read** (instruction 4.4, Claude, veto after):
      every hand-off tile shows two animals except seed 14, where a child with a dog and a small
      cat fills the frame in every adapter column; the detector's red frames on seeds 10 and 11 at
      `k = 0` are one animal hidden behind the other and a cat the counter missed, so the eye's
      count for cutoff 20, `k = 0` is 7 of 8, not 5. The cutoff-30 hand-off with no corrector is
      the crispest column on seeds 9, 12, 13 and 16, and the one that looks most like the joint
      prompt's render; the corrector columns are grainier at `k = 20` on every seed. So the
      softness the person sees lives in the adapter's low-noise steps, handing those steps to the
      frozen model removes part of it while keeping the composition, and settling further into
      the frozen model's distribution with Langevin steps does not help. What would move the
      0.038 to a real gain is the lever plan 14 of scope 01 already holds: the λ schedule across
      the hand-off, and the re-noise-and-redenoise tail.
- [x] ✅ **On the eight-seed sheet, what is the compose rate of the corrector on all 50 steps
      against plain PoE and the joint prompt, on both pairs?** Cat × dog, validated instance
      count over seeds 9 to 16: joint prompt 8 of 8, plain PoE 0 of 8, PoE plus the corrector
      (`k = 20`, `c = 3`, every level) 0 of 8; every corrector render counts one animal, with
      cat and dog both detected in it. The control pair by the both-concepts read: 8, 8, 8 of 8,
      so the bar holds (no composed seed lost). Read: on the seeds every parallel session reports
      on, the corrector alone is plain PoE's compose rate exactly, and what it changes is the
      style, with five of eight cat × dog renders leaving photography for a cartoon or a line
      drawing (`corrector-cat-dog-eight-seed-sheet.png`). Detector confidence for the butterfly
      falls on three control seeds (0.94 to 0.73, 0.50 to 0.33, 0.95 to 0.56) while the frame
      stays green, which is the drift toward an illustration read by the detector.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Does the layout match the injected-correction figure on all four facts, meaning the pair,
      the seeds, the nine window positions, and the exact rule the green border encodes? Yes.
      The four facts, read off `poe/samples-over-the-window-map.json` and
      `scripts/window_samples_over_map.py` before rendering: pair `a_cat__x__a_dog`; seeds 9, 10,
      11, 12 as rows; windows (0,10), (5,15), … (40,50), nine at stride 5 from
      `window_grid.windows()`; green frame where `compose == 1` in the scored file, which is the
      validated instance count of at least 2. This figure uses the same four and adds one column,
      the corrector on all 50 steps, labelled as such. The sidecar names the match.
- [x] ✅ How many seeds compose per column, by the detector and by eye, for each of the ten columns?
      Detector: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 of 4. Eye (Claude's read of the strip, veto after):
      the same, 0 of 4 in every column; no render shows two bodies, and the closest is seed 12 at
      steps 0 to 10 (a cat's head on a cushion with a paw that could be read as a second animal's,
      counted one by both). No disagreement to name.
- [x] ✅ Does the all-50 column compose more than the best ten-step window? No: 0 of 4 against
      0 of 4. A corrector on every level is no better than a corrector in any window, and it is
      the column whose images drift furthest from plain PoE's (a different scene on every seed).
- [x] ✅ At which `k` was this run, and where does that `k` sit on step 26's curve? `k = 20` at
      `c = 3`. Step 26's curve licensed no flat part at any `c` (cat × dog's `k = 100` and
      `k = 200` differ by 21% at `c = 3`), so the `k` is a compute budget chosen for a different
      reason, said here: it is the largest count at which the settled samples of both pairs are
      still photographs (from `k = 100` they are flat graphics at this `c`), and it is the count
      the step-size search ran at. A `k` off a falling part would be a different experiment; this
      one is off no part, and the caption says so.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the renders are made.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Across the ten columns only the corrector window varies:
      same pair, same seeds, same cached starting noise, same 50 DDIM steps, guidance 7.5, same
      `k`, `c` and Langevin noise stream, one device. Between this figure and the injected one,
      the mechanism is the only difference the design names; the injected runs used the
      four-branch sampler and these the three-branch one, which changes fp16 rounding and not
      the trajectory (the composer's `k = 0` is byte-identical to `run_cfg_poe`).
- [x] ✅ **Was the measuring tool sound?** The validated instance count, run on the launch node,
      and an eye read of every tile beside it; the two agree on all 40 here, so the usual cat ×
      dog disagreement did not arise.
- [x] ✅ **Did the run respect the environment?** 40 renders counted in
      `window_curves_mcmc.json`, all under `corrector/window/` on `/datasets`, only the figure and
      sidecar in the repo, launched under `nohup` on mscluster108 device 1 with node, device and
      PID in the log header, harvested by `pgrep`. No adapter was attached in this process.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the corrector composes in window X | that it was run at one `k`, which is a compute budget rather than a property of the problem |
| the green borders | the exact rule they encode, quoted from the figure code, and the eye count beside the detector count where the two differ |
| anything comparing the two mechanisms | that this is a behavioural comparison and not an attribution, because attribution is impossible in this window |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| the clean tail's next lever | the λ schedule across the hand-off (1.2 on steps 0 to 19, linear to 0 by 29) and the re-noise-and-redenoise tail, both in [scope 01's plan 14](../../01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md), judged on the same DINOv2 bar with the same baseline; the hand-off alone reached +0.038 of the 0.05 | the fidelity question; this scope's corrector is not the tool for it |
| whether a corrector composes at a larger compute budget or a step size between 0.3 and 3 | the same ten columns at `k = 100`, or at `c = 0.3`; each is 1.6 h on an RTX 8000 or about 40 min on the Blackwell card. The images at `k ≥ 100` are flat graphics on both pairs at `c = 3`, so the larger budget is expected to change style and not count | nothing; the recorded answer bounds what a training-free corrector does at 60 extra evaluations per level |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Nothing left to run here. The fidelity question moves to scope 01's plan 14 with the DINOv2 bar
and the adapter-alone baseline from this file; the sampler-share question at step 26 waits on
seeds 10 to 12.

## Cross-references

- The two findings built from this verdict: [does a corrector alone produce two animals](../../../report/is-the-gap-the-samplers-or-the-models/does-a-corrector-alone-produce-two-animals.md) and [can a corrector or a clean tail sharpen the adapter's renders](../../../report/is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md).
- The recipe that reruns the window sweep, the sheets and the two tails: [running the Langevin corrector](../../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md).
