# 🔬 Review: do the four switches select what their names claim?

Nothing has run. This file judges [the design](../plans/tools/01-the-three-switches-and-the-guidance-interval.md), which builds the three trainer switches, the sampler's guidance interval, the 14-pair pool and the exclusion list. Its answers gate every other plan in this scope: a switch that selects nothing would send the chain of five trainings twelve hours down the wrong road while reporting plausible numbers.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(This plan produces no W&B run. For a failure worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/01-the-three-switches-and-the-guidance-interval.md) | the four switches, the two config files, the checks |
| **this file** | **the verdict: whether each switch has a non-empty target group, and whether the sampler change left the shipped path alone** |

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

- **A silent no-op**: a flag whose target group is empty. The run completes, reports a plausible metric, and describes a different configuration than its name claims. It is the failure this whole file exists to catch.
- **The dry run**: building the training dataset and printing what it contains without training on it.
- **The identity check**: rendering with the new guidance interval set to cover all 50 steps at guidance 7.5, and comparing against the kept original render of the same seeds, in the same process and precision.
- **A grey level**: one step of 255 in an 8-bit channel. Same-loop fp16 drift over 50 steps runs to about 3; the check's tolerance is 6.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds a measuring tool** (an instrument: no result lands on any claim). It is judged by whether it can fail, not by what it found. A failed bar here stops every other plan in the scope until it is fixed; nothing downstream may run against a switch that has not shown it selects something.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the dry run on the parent configuration | builds a measuring tool | | | `improve_r32/` dry-run log | not started |
| the sampler identity check, 8 seeds | builds a measuring tool | | | `improve_r32/identity_check/` | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does every switch select a non-empty group, and does the dry run build the cell count the design predicts?**
      This is the bar because a switch that selects nothing is invisible downstream: the chain runs, the metrics look reasonable, and five runs describe configurations nobody chose. **Pass** if the dry run prints a kept-cell count equal to 108 minus a non-zero excluded count, a step range selecting 24 distinct steps, an orthogonal weight of 3.0, and every pair in the 14-pair pool resolving to at least one cached cell. **Fail** if any of those counts is zero or the arithmetic does not close. There is no inconclusive branch: a count is either there or it is not.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does the guidance interval, set to cover the whole run at 7.5, reproduce today's render?**
      The tolerance is 6 grey levels of mean absolute difference per seed, against the kept original render rather than a freshly made one, so the check cannot pass by both sides moving together. At or under 6 on all eight seeds, the interval is an addition. Above it on any seed, the interval has changed the shipped sampler, and both the baseline and every stacked render in this scope would be measuring a path that no longer matches what produced checkpoint 30050's existing results.

- [ ] ⚠️ **Do the in-span and orthogonal parts recombine to the target's norm?**
      Report either way. If they do, the projection is a decomposition and `--orth-weight` charges a real split. If they do not, it is a truncation wearing a projection's name, and the parent's fourth change means something other than what the design says.

- [ ] ⚠️ **How many cells does the exclusion list actually drop, and out of how many, per pair?**
      Both numbers and the per-pair breakdown, not the fraction. The list is built from a read known to over-flag on visually similar pairs, and **every one of the eleven original training pairs is a look-alike pair**: wolf and husky, turtle and tortoise, rabbit and hare, lion and tiger, horse and zebra, gorilla and chimpanzee, donkey and pony, dolphin and porpoise, crow and raven, crocodile and alligator, cheetah and cougar. The read is therefore working at its hardest on the entire original pool, and a large drop is expected rather than surprising. The per-pair breakdown is what makes the expectation checkable: if the three added pairs (lion and horse, wolf and horse, bear and salmon, all plainly different animals) drop far fewer cells proportionally than the eleven, that is the over-flagging showing itself in the data, and it is what child C1 is later measured against. An unrecorded count leaves C1's answer uninterpretable.

- [ ] ⚠️ **Do the three added pairs contribute the 20 cells the cache holds?**
      Lion × horse and wolf × horse hold seeds 1 to 8; bear × salmon holds only seeds 1 to 4, its fifth cached seed being 42, outside the training seed list. That is 8 plus 8 plus 4, so 20 cells and not the 28 the design's prose claimed. If the resolution pass prints anything else, the cache holds something other than what was counted and the parent's pool is not the pool that was designed.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the results raise. Nothing here may become the question above.

- [ ] ⚠️ **Is 28 surviving cells enough to train a rank-32 adapter for 30,000 steps?** (raised by reading `target_quality.json` before the switches were built)
      The exclusion count was computed ahead of the plan, over the fourteen pool pairs on training seeds 1 to 8. It drops 80 of 108 cells. Four pairs lose every cell (rabbit and hare, horse and zebra, donkey and pony, cheetah and cougar), so the parent's pool is ten pairs rather than fourteen. Loosening the rule to drop only a clear same-animal-twice recovers 6 cells and no pairs, so the filter is not miscalibrated: on look-alike pairs the joint prompt genuinely fails to draw two distinct animals about three quarters of the time.
      This is a measurement, not a verdict, and it changes what the chain is testing rather than whether it should run. It is carried into [plan 04's review](04-the-parent-and-the-four-children.md) because it confounds child C1.

- [ ] ⚠️ **Does the dry run reproduce 28, and if not, which of the two counts is wrong?** (raised by the same reading)
      The number above was computed by hand from the quality file. The dry run computes it from the dataset the trainer actually builds. Agreement confirms the switch selects what this file claims; disagreement means one of the two reads the pair or seed list differently, and that has to be settled before launch rather than explained afterwards.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** The identity check must differ from its reference in exactly one thing: whether the guidance interval code path is taken. Same seeds, same precision, same process, and the reference file kept from before rather than regenerated.
- [ ] ⚠️ **Was the measuring tool sound?** The dry run must report counts computed from the dataset it actually built, not from the configuration it was handed. A count echoed back from the config would pass this bar while proving nothing.
- [ ] ⚠️ **Did the run respect the environment?** Every flag selected a non-empty group, the outputs landed on `/datasets` rather than `/home-mscluster`, and the pool yaml sits under the hard-coded config path the loader reads.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any sentence describing the trained adapter's data as filtered | the number of cells dropped and the total, and that the filter is a read known to over-flag on visually similar pairs |
| any sentence describing the loss as weighting the orthogonal part | the factor, and that the split is a projection onto the two experts' own span rather than an arbitrary subspace |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the automatic target check can be cited anywhere, given that [the target sheets](../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/README.md) show it passing two of eight on `a_lion__x__a_tiger` where several failures plainly show both animals, and one of eight on `a_mailbox__x__a_snowfield` where every tile is correct | the filled `eye_verdict` column of [`eye-verdicts.csv`](../../../artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/eye-verdicts.csv), and the agreement table task 3.2 emits from it | the exclusion list, and any sentence in the paper that leans on this check |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's task 1.1](../plans/tools/01-the-three-switches-and-the-guidance-interval.md#1--write-the-three-trainer-switches). Nothing else in this scope may start first.
