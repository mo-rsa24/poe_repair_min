# 🧪 Review: does the corrector's compose rate peak in the same window the injected correction does?

**Nothing has run yet.** Every question below was written before any corrector existed. This file
judges [the corrector window design](../plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md).
It is the only thing this scope can say about the compose-decisive early window, because
[the residual measurement at step 26](hypothesis-02-what-is-left-once-the-chain-settles.md) cannot
attribute there by construction.

## Recommended prompt (when the run lands)

```
/analyze-run the corrector window sweep, 4 seeds by 10 window positions on a_cat__x__a_dog
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md) | the nine positions, the layout it must match, and where the figure files |
| **this file** | **the verdict: not yet run** |
| [the step 26 verdict](hypothesis-02-what-is-left-once-the-chain-settles.md) | the measurement this waits on, though not strictly, and the `k` these renders run at |
| [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md) | the injected-correction figure this is compared against, render for render |

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
| The corrector across ten window columns, 4 seeds × 10 columns | Tests the claim | not launched | 40 renders, each decoded and scored | `corrector/window_curves_mcmc.json` and `mcmc/samples-as-a-ten-step-corrector-window-slides.png` | ⚠️ waiting on the threshold at step 26 |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does the corrector's compose rate peak in the same window the injected correction
      does?** Recorded either way. Same window means two different mechanisms acting at the same
      moment. A different window is the stronger result and needs its own paragraph. This question
      is answered even if the measurement at step 26 returns a null, because a flat residual curve
      does not imply a flat [compose rate](context/world/compose-rate.md). The corrector can
      relocate the trajectory without shrinking `‖r_t‖`. If it is run against a flat curve at step
      26, this file says so.

      > A null at step 26 means the correction's size came out the same with the corrector running
      > as without it.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Does the layout match the injected-correction figure on all four facts, meaning the pair,
      the seeds, the nine window positions, and the exact rule the green border encodes? A
      nearly-matched pair of figures is worse than an obviously different one, because a reader
      compares them anyway.
- [ ] ⚠️ How many seeds compose per column, by the detector and by eye, for each of the ten columns?
      The two counts go side by side, with the disagreements named render by render.
- [ ] ⚠️ Does the all-50 column compose more than the best ten-step window? If a full-run corrector
      is not better than a well-placed short one, that is a statement about when the corrector's
      work actually matters.
- [ ] ⚠️ At which `k` was this run, and where does that `k` sit on step 26's curve? A `k` chosen
      off the flat part is a compute budget; a `k` chosen off the falling part is a different
      experiment.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the renders are made.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only the window position varies across columns, and only the
      mechanism varies between this figure and the injected-correction one. Same pair, same seeds,
      same 50 DDIM steps, same guidance, same scorer.
- [ ] ⚠️ **Was the measuring tool sound?** The scorer is the validated instance-count detector, and
      [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md)
      records that it disagrees with the eye on cat and dog often enough that the eye read is the
      one cited. Both counts are taken here for that reason.
- [ ] ⚠️ **Did the run respect the environment?** All 40 renders present, saved under `/datasets`
      with only the finished figure and its sidecar in the repo, launched under `nohup` outside
      Slurm and harvested by `pgrep` rather than `squeue`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the corrector composes in window X | that it was run at one `k`, which is a compute budget rather than a property of the problem |
| the green borders | the exact rule they encode, quoted from the figure code, and the eye count beside the detector count where the two differ |
| anything comparing the two mechanisms | that this is a behavioural comparison and not an attribution, because attribution is impossible in this window |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Read the existing injected-correction figure and write down the four layout facts this figure has to
match, before anything is rendered.
