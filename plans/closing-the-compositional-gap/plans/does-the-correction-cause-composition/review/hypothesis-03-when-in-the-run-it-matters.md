# 🧪 Review: when in the run is the correction needed?

**Answered: timing decides, and it decides at the very start.** The questions below were written
before the runs, so the answers cannot be chosen after the fact. This file judges
[../plans/hypothesis-03-when-in-the-run-it-matters.md](../plans/hypothesis-03-when-in-the-run-it-matters.md),
and its answers fill register slot **F4**, the figure showing when in the denoising run the
correction does its work.

## Recommended prompt (to write the figure)

```
/design-figure F4 the timing curve, compose rate against window position
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-03-when-in-the-run-it-matters.md) | the sliding window, the grid, the bar |
| **this file** | **the verdict: timing decides, and it decides at the very start** |
| [the independent estimate](hypothesis-04-what-the-cached-runs-already-show.md) | the fork step, step 16, which disagrees with this and is not a contradiction |
| [the register](../../../../../paper/iclr/figures.md) | F4's row, whose layout this result changes |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️
- **The correction**: the step-by-step gap between what the joined prompt predicts and what
  adding the two separate prompts predicts. It is what the broken method leaves out.
- **A window**: a stretch of the 50 denoising steps. Inside it the correction is allowed to act;
  outside it, nothing is injected. Sliding the window from start to finish and scoring each
  position is the whole experiment.
- **The window slides over the correction, not over the prompt.** The prompt stays on at every
  step of every run here. Sliding a window over the prompt itself is a different experiment,
  answering when conditioning is needed at all; it is named in the plan and is not run.
- **The fork step**: step 16, where the broken path and the working path start pulling apart,
  measured in `hypothesis-04-what-the-cached-runs-already-show`. An independent estimate of the
  same moment, from cached data rather than new runs.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️
**Tests the claim.** A failure of the bar below closes the plan and opens one follow-on.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Leak check, `--window off --check-identity`, a_cat×a_dog seed 9, 50 steps | Checks the harness | 2026-08-10, in-session mscluster GPU 1 | 1 cell | stdout only | done, passed |
| Smoke, 3 windows (0-10, 15-25, 40-50), a_cat×a_dog seed 9 | Checks the harness | 2026-08-10, in-session GPU 1 | 3 cells | `window/pairs/a_cat__x__a_dog/seed_9/` | done, passed |
| Timing grid, 9 windows × 8 pairs × 4 seeds = 288 cells | Tests the claim | 2026-08-10 19:00, `run_window_sweep.sh` under nohup, GPU 1 | 288 cells | `/datasets/.../interaction_term/window/pairs/`, scored into `window_curves.json` | done, all 288 present, no missing or skipped windows |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ Does the compose rate peak at some window position rather than staying flat?
      Yes, and more sharply than the bar expected. It is not a peak in the middle of the run but
      a cliff at the start. The earliest window, steps 0 to 10, composes 0.656 of its 32 cells.
      One notch later, steps 5 to 15, it is 0.250. By steps 20 to 30 it is 0.000 and it stays
      there. Timing matters, and it matters at the very beginning.

      One limit, which belongs in the caption. The best window measured is also the earliest
      window measured, so the curve cannot say whether the true best sits earlier still. The
      grid starts at step 0 and a ten-step window cannot begin before it, so answering that
      needs a narrower window near the start, not a longer sweep.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ With the window switched off everywhere, does the output match plain PoE exactly?
      Yes, byte-identical on a_cat×a_dog seed 9 at 50 steps. The comparison is against a full-dose
      window placed past the last step, not against `run_cfg_poe`: both runs then batch four UNet
      branches, so only the window logic can differ. Comparing against the three-branch PoE
      sampler would fail for batch-shape reasons alone and would say nothing about leakage.
- [x] ✅ Does the peak sit where the correction is largest?
      No, and the two are close to opposites. On cat × dog the correction averages 0.48 over
      steps 0 to 14 and 1.33 over steps 20 to 40, so it is about 2.7 times larger late than
      early, and all four seeds peak between steps 21 and 37. The window that composes is steps
      0 to 10, where the correction is at its smallest.

      This is the answer the bar hoped for, arrived at by a different route. The bar assumed size
      would be nearly flat and so could not pick out a moment. Size is not flat, it rises through
      the run, and it still does not pick out the moment: it rises where the compose rate falls.
      Either way size is ruled out as the explanation, and the stronger sentence is now available.
      The correction matters when it lands, not where it is big.

      What follows for the writing: the paper may not argue that the correction matters because
      it is large there, and no figure may put size and timing on one axis without saying they
      disagree. `correction-size-per-step-beside-outcome-per-window` exists to show exactly that.
- [x] ✅ Does the peak land near step 16, the fork step measured from cached data?
      No. The peak is at window centre 5 and the window centred at 15 composes 0.094, seven
      times worse. The two estimates disagree, so neither may be printed as confirming the other.

      The diagnosis is that they were never measuring the same thing. The fork step is where the
      corrected and uncorrected paths visibly separate, which is downstream of the correction
      doing its work. The window sweep asks where the correction has to be applied for the
      outcome to change. A cause acting early and a difference becoming visible later is the
      ordinary relationship between the two, not a contradiction, and the run that measured
      step 16 is not called into question by this.

      What follows for the writing: step 16 may not be described as the moment the correction
      matters, and F4's caption may not draw the fork step as a band behind the timing curve, as
      the register's layout currently says it should. That layout was written when the two
      numbers were expected to agree.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a bar**, because it was written with the timing answer already
visible. These two runs exist to remove that answer's one confound.

The nine-window sweep could not separate two things. The windows differ in when the correction
lands, and also in how much of it lands, because the correction's own size grows through the run.
The early window that composes delivers less correction than the late window that fails, so
"early wins" and "small wins" were tangled together. Two runs untie them, both cat x dog, four
seeds, through the same sampler as the sweep and verified against it.

- [x] ✅ Does the late window still fail when given exactly the correction total that works early?
      Yes, on every seed. Sixteen cells crossing two windows against two doses
      (`--mode swap`, `dose_matched/swap_manifest.json`). On seed 12 the early window at its own
      dose of 3.6 units gives a white cat lying beside a tan dog; the late window at that same
      3.6 units gives one fused animal with cat ears and a dog muzzle. Same seed, same noise,
      same prompt, same delivered correction, and the only difference is when it arrives.

      Two things fell out that were not asked for. Tripling the correction in the late window
      changes almost nothing: the 10.7-unit cell is near-identical to the 3.6-unit one, same
      pose and same background, so late application is inert rather than merely weak. And the
      early window tolerates that 3x overdose without breaking, still composing at lambda 2.96,
      which is well outside anything the dose sweep measured.

      What follows for the writing: the paper may say timing decides rather than dose. The
      verdict is by eye on the images; the detector's count is recorded in
      `dose_matched/swap_scores.json` beside it, and disagrees with the pictures on this pair
      often enough that the eye read is the one cited.

- [x] ✅ Does the whole nine-window curve keep its shape once every window delivers the same
      total? Yes. Thirty-six cells, four per window (`--mode matched`,
      `dose_matched/matched_scores.json`). The matched curve is 0.75 composed at steps 0 to 10,
      0.50 at 5 to 15, 0.25 at 10 to 20, and zero from 15 to 25 onward, against the full-strength
      curve on the same pair of 0.75, 0.75, 0.25, 0.25, then zero. Holding the delivered total
      constant moves two points by one cell each and does not touch the shape.

      This also answers the objection that made the run look not worth doing. The worry was that
      matching the totals forces the late windows to roughly a quarter strength, so a flat tail
      would just be a weak dose. It is not: the windows at steps 5 to 15 and 10 to 20 compose at
      0.52 and 0.37 strength. A fraction-strength correction works early and does nothing late,
      so strength does not explain the tail.

      What follows for the writing: the caption may say the cliff survives dose-matching. It may
      not present these as population rates. Four cells per point on one pair is a rate over four
      runs, and the eight-pair sweep remains the population estimate.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** The nine windows differed in when the correction lands and
      also in how much of it lands, because the correction's own size grows through the run. That
      confound was real and it is removed: the two dose-matched runs under
      [Asked after the result](#asked-after-the-result) hold the delivered total constant and the
      cliff survives. Same seed, same noise, same prompt, same delivered correction, only the
      timing differs.
- [x] ✅ **Was the instrument sound?** The leak check answers it, and it was pre-registered rather
      than added afterwards: with the window switched off everywhere the output is byte-identical
      to plain PoE on a_cat×a_dog seed 9 at 50 steps. See the first question under
      [Written before the run](#written-before-the-run-answered-after) for why the comparison is
      against a full-dose window placed past the last step and not against `run_cfg_poe`.
- [x] ✅ **Did the run respect the environment?** All 288 cells present, no missing or skipped
      windows, output under `/datasets`. The sweep ran under `nohup` on GPU 1 outside Slurm, so
      harvest it by `pgrep` rather than `squeue`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Every row here has its reasoning in one of the answered questions above; the table is the index a
writer reads, not a second copy of the argument.

| What the paper says | What it owes alongside it |
|---|---|
| the correction matters early | that the best window measured is also the earliest window measured, so the curve cannot say whether the true best sits earlier still |
| why the correction matters early | not because it is large there. It is about 2.7 times larger late than early, and it rises where the compose rate falls. No figure may put size and timing on one axis without saying they disagree, which is what `correction-size-per-step-beside-outcome-per-window` exists to show |
| step 16, the fork step | it may not be described as the moment the correction matters, and F4's caption may not draw it as a band behind the timing curve. The register's current layout says it should, and that layout was written when the two numbers were expected to agree |
| timing decides rather than dose | the verdict is by eye on the images. The detector's count is in `dose_matched/swap_scores.json` beside it and disagrees with the pictures on this pair often enough that the eye read is the one cited |
| the cliff survives dose-matching | these are not population rates. Four cells per point on one pair is a rate over four runs; the eight-pair sweep remains the population estimate |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the true best window sits earlier than steps 0 to 10 | a narrower window near the start, not a longer sweep. A ten-step window cannot begin before step 0 | nothing is blocked; F4's caption states the limit instead |
| F4's layout in the register, which draws the fork step as a band behind the timing curve | rewriting that row, since the two estimates disagree and neither confirms the other | building F4, which would otherwise be drawn to a layout the result contradicts |
| why the detector disagrees with the eye on cat × dog | comparing `dose_matched/swap_scores.json` against the images cell by cell | nothing yet, but every claim on this pair currently rests on an eye read rather than a count |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Rewrite F4's row in [the register](../../../../../paper/iclr/figures.md) to drop the fork-step
band, then build F4.
