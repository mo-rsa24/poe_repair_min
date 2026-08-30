# 🔌 Review: does SuperDiff still compose at fifty steps?

**Nothing has run yet.** Every question below was written before the pipeline was wired. This file
judges [the SuperDiff design](../plans/baselines/05-superdiff-at-this-repos-fifty-steps.md). Its
answer decides whether every later comparison against SuperDiff is a fair one, or a comparison
between a working rule and one run outside its intended settings.

## Recommended prompt (when the run lands)

```
/analyze-run the SuperDiff step-count parity check, 200 steps against 50
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/05-superdiff-at-this-repos-fifty-steps.md) | the wiring, the step-count match, and the per-step prediction hook |
| **this file** | **the verdict: not yet run** |
| [the amount-axis verdict](06-three-rules-on-one-amount-axis.md) | where this row is actually compared, and what the parity answer changes in its captions |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | whether this half of the scope is a diagnosis or a baselines table |

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

- **SuperDiff**: a composition rule derived from the continuity equation rather than from adding
  scores. Skreta et al., [arXiv 2412.17762](https://arxiv.org/abs/2412.17762), ICLR 2025 Spotlight.
- **The step count**: how many denoising steps a render takes. SuperDiff's pipeline defaults to
  200; everything measured in this project runs at 50.
- **The per-step prediction, `eps_M`**: what a composition rule predicts at each step. Exposing it
  is what lets `r_t^SD = eps_J - eps_M` be formed, which the shared amount axis needs.
- **The parity check**: the same pair and seed rendered at 200 steps and at 50, changing nothing
  else, to measure what matching this repo's setting cost.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** Missing the threshold means the comparison has no fair starting point.
Every figure comparing against SuperDiff would then compare a working rule against a crippled one.
It does not close the plan; it adds a sentence to every caption that uses this row.

Per this project's run conventions a baseline may not change a claim, and it freezes on landing.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| SuperDiff at 50 against 200 steps | Establishes a baseline | not launched | 2 renders | `corrector/superdiff/parity/`, both renders scored | ⚠️ not run |
| The per-step prediction hook, verified on one render | Establishes a baseline | not launched | 1 render | 50 per-step norms of `r_t^SD` | ⚠️ not run |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Does SuperDiff still compose at 50 steps?** Its default is 200. If it does not, every
      comparison against it is between a working rule and a crippled one, and that sentence goes in
      the caption rather than in a footnote. Answered by the detector and by eye, on the same pair
      and seed at both step counts, with nothing else changed.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Is `eps_M` available at every one of the 50 steps, verified by forming `r_t^SD` and
      printing its per-step norm? A wrapper that returns only a finished image cannot supply the
      shared amount axis, and finding that out inside the grid is expensive.
- [ ] ⚠️ Does the 200-step render compose at all on the tested pair? If neither step count composes,
      the pair is wrong for this check rather than the method being at fault, and one more pair is
      tried before anything is recorded.
- [ ] ⚠️ Did the checkpoint download land under `/datasets` rather than `/home-mscluster`? The home
      filesystem has hit 100% once here and silently killed checkpointing.
- [ ] ⚠️ Which branch did
      [the measurement at step 26](03-what-is-left-once-the-chain-settles.md) fire? The
      runs in this plan are the same either way, though the sentences around them are not.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the parity check runs.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only the step count differs between the two parity renders.
      Same pair, same seed, same model, same guidance, same prompt handling.
- [ ] ⚠️ **Was the measuring tool sound?** The scorer is the validated instance-count detector, read
      over these two renders and no others, with the eye verdict recorded beside it.
- [ ] ⚠️ **Did the run respect the environment?** Renders and weights under `/datasets` with the
      disk guard on the filesystem actually written to, and no relative path in a launch line onto
      a node this session is not on.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any comparison against SuperDiff | that it was run at 50 steps rather than its own default of 200, and whether that changed whether it composes |
| SuperDiff as a baseline | that the parity check was on one pair and one seed, which bounds how much it can say |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Wire the pipeline and verify the per-step prediction hook by printing 50 norms, before either
parity render is made. A wrapper that cannot supply `eps_M` changes what this plan can deliver.
