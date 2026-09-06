# 🧪 Review: does the corrector's compose rate peak in the same window the injected correction does?

**The window sweep has run: the corrector composes in no window. Zero of four seeds in every one of
the ten columns, by the detector and by eye, at `c = 3` and `k = 20`, against an inconclusive
residual curve at step 26. The eight-seed sheets and the tail condition are still rendering.**
Every question below was written before any corrector existed. This file
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
| The eight-seed sheet, 2 pairs × 8 seeds × (joint prompt, plain PoE, corrector on all 50 steps) | Tests the claim | not launched | 48 renders, 16 of them at the corrector's `k` | `corrector/sheet_scores.json`, `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-<pair>-eight-seed-sheet.png` | ⚠️ waiting on step 26's `k` |
| The tail condition, 2 pairs × 8 seeds × `k ∈ {0, 5, 20}` on the rank-32 λ 1.2 run, corrector on steps 35 to 49 | Tests the claim | not launched | 48 renders; a level inside the window costs `6k + 6` UNet evaluations | `corrector/tail_fidelity.json`, `corrector-on-adapter-tail-<pair>-eight-seed-sheet.png` | ⚠️ waiting on step 25's `c` |

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

- [ ] ⚠️ **Does a corrector on the last fifteen steps sharpen the corrected run without costing
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
- [ ] ⚠️ **On the eight-seed sheet, what is the compose rate of the corrector on all 50 steps
      against plain PoE and the joint prompt, on both pairs?** Recorded either way; it is the
      read-out every parallel session reports on the same seeds. The one bar: the control pair's
      corrector column may not lose more than one composed seed against its plain-PoE column,
      which is the same `FIDELITY_MAX_COMPOSE_LOSS_SEEDS` in source.

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
| whether a corrector composes at a larger compute budget or a step size between 0.3 and 3 | the same ten columns at `k = 100`, or at `c = 0.3`; each is 1.6 h on an RTX 8000 or about 40 min on the Blackwell card. The images at `k ≥ 100` are flat graphics on both pairs at `c = 3`, so the larger budget is expected to change style and not count | nothing; the recorded answer bounds what a training-free corrector does at 60 extra evaluations per level |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Harvest the eight-seed sheet (`corrector/sheet_scores.json`) and the tail condition
(`corrector/tail_fidelity.json`), draw the four sheets with `--figures`, log with `--wandb`, and
answer the two questions above that still carry ⚠️.
