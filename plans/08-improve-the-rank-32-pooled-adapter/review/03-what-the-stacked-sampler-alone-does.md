# 🧪 Review: does changing how checkpoint 30050 is run already give more clean seeds?

Nothing has run. This file judges [the design](../plans/baselines/03-what-the-stacked-sampler-alone-does.md), which renders checkpoint 30050 under both samplers and labels all 25 seeds blind. Its answer decides which column every training run in this scope is compared against, so it is written before any training result is opened.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/03-what-the-stacked-sampler-alone-does.md) | the two samplers, the 25 seeds, the bar in source |
| **this file** | **the verdict: the clean-seed count under each sampler, and which one is the baseline from here on** |
| [procedure](../procedures/tools-02-run-the-blind-label-pass.md) | the steps for taking the labels |

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

- **The baseline, B**: checkpoint 30050 of the rank-32 pooled adapter. No training happens in this plan.
- **Shipped**: the adapter on all 50 denoising steps, deterministic sampling, guidance 7.5 throughout. How the checkpoint is run today.
- **Stacked**: the adapter on steps 0 to 24, the frozen model finishing, fresh randomness at every step, guidance 7.5 on steps 5 to 35 and 1.0 outside.
- **A clean seed**: one labelled **clean** in the blind pass, meaning both named animals present, each clearly itself, nothing unnatural on inspection.
- **Better, under the same rule**: more clean seeds, and none the other side had clean falling out of that label. Both halves are required.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** Frozen the moment it lands: no re-render with better settings once a training run's number is known, and no tuning of the stacked sampler's span or eta after the labels are in. A failed bar does not stop the scope; it fixes which column the training runs are read against.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| checkpoint 30050, shipped sampler, 25 seeds | establishes a baseline | | | `improve_r32/baseline/renders/shipped/` | not started |
| checkpoint 30050, stacked sampler, 25 seeds | establishes a baseline | | | `improve_r32/baseline/renders/stacked/` | not started |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Under the stacked sampler, does checkpoint 30050 give more clean cat × dog seeds than it does under the shipped sampler, and lose none it had clean?**
      This is the bar because it decides what every training run in the scope is measured against. The thresholds are constants in the scoring script, `MIN_CLEAN_GAIN = 1` and `MAX_CLEAN_LOST = 0`, so they cannot be moved after the labels are in without appearing in a diff. **Support**: at least one more clean seed and none lost, so the stacked column becomes the baseline and this is reported before any training run is read. **Null**: no gain, or a gain paid for with a loss, so the shipped column stays the baseline. **The other direction**: the stacked sampler loses clean seeds, which is reported as a finding of its own, since it would say the frozen tail or the weak late guidance costs composition.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does elephant × penguin move the same way as cat × dog?**
      Report either way. Both pairs moving together says the sampler change is general. Cat × dog gaining while elephant × penguin loses says the stacked settings suit one pair, which bounds every later claim to the pair it was tuned on and is the kind of thing that is invisible unless a second pair is carried.

- [ ] ⚠️ **Which seeds change label between the samplers, and in which direction?**
      The per-seed list, not the totals. A sampler that turns three **not two** into **unclear** and nothing into **clean** has changed the pictures without improving them, and the totals alone would hide that.

- [ ] ⚠️ **How many of the 17 cat × dog seeds have a joint target that itself shows the wrong animals?**
      Counted and named here, once, because these are the seeds the adapter was trained toward incorrectly and every later reading of them inherits the problem. Three of the eight originally examined seeds were like this; the number over all 17 is not yet known.

- [ ] ⚠️ **Do the two secondary reads separate the two samplers at all?**
      Report either way. If the count read gives the same number for both columns while the labels differ, that is direct evidence for why the scorer is not the read, and it is worth having stated on a baseline rather than argued from the earlier examples.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

*(none yet: nothing has run)*

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** One axis differs between the columns: the sampler. Same checkpoint, same seeds, same cached initial noise per seed, and the counts confirm it rather than the configuration claiming it. Three things move together inside the stacked sampler (the adapter's window, the randomness, the guidance span), which is stated openly here rather than treated as one axis it is not.
- [ ] ⚠️ **Was the measuring tool sound?** The labels came through the blind pass whose known-example smoke has passed, over the tiles this plan wrote and no others, with the mapping untouched during labelling.
- [ ] ⚠️ **Did the run respect the environment?** The renders landed on `/datasets`, the references were made with no adapter attached, and both columns came from the same code path with only the sampler settings differing.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any claim that the trained adapter improved | which sampler the comparison used, and what the same checkpoint does under the other one |
| any use of the stacked sampler's result | that three things change at once inside it, and that this plan did not separate them |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| which of the stacked sampler's three changes carries any gain it shows | a three-cell sweep turning each on alone, roughly a GPU-hour | nothing in this scope; it becomes worth running only if the stacked column wins |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's task 1.1](../plans/baselines/03-what-the-stacked-sampler-alone-does.md#1--fix-the-bar-in-source-before-rendering), which puts the rule in source before anything is rendered.
