# 🔬 Review: does the corrector change nothing when switched off, and what step size does it run at?

**Nothing has run yet.** Every question below was written before any corrector existed. This file
judges [the corrector and step-size design](../plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md).
Steps 26, 27 and 29 all measure something on the path this composer produces, so a failure recorded
here blocks all three rather than bounding any of them.

## Recommended prompt (when the run lands)

```
/analyze-run the step-size search, five c values at k=20 on a_cat__x__a_dog seed 9
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md) | the composer, the two leak checks, the search and its bounds |
| **this file** | **the verdict: not yet run** |
| [the gate's verdict](hypothesis-02-what-is-left-once-the-chain-settles.md) | what this instrument is built to measure |
| [the questions this file was split from](../source/the-questions-pre-registered-against-it.md) | the whole pre-registered set, written before the scope existed |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [The step-size search](#the-step-size-search)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The corrector count, `k`**: how many Langevin steps run at each of the 50 noise levels before
  the reverse step. `k=0` is plain product-of-experts.
- **The step-size multiplier, `c`**: the corrector's one free parameter, where the step size at
  noise level `t` is `δ_t = c·β_t`.
- **The relative displacement**: how far the chain actually moved,
  `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖`. It is the only thing that tells a corrector doing nothing apart
  from a corrector that was never switched on.
- **A leak check**: a run where the corrector is switched off in some way, whose output must be
  byte-identical to plain product-of-experts. A single differing byte means the composer changes
  something it should not.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Builds an instrument.** A missed bar blocks every plan downstream of it: steps 26, 27 and 29 all
run this composer. It does not bound a claim, it decides whether any claim can be measured.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Leak check, `k=0` byte-identical to plain product-of-experts | Builds an instrument | not launched | 1 cell | stdout only | ⚠️ not run |
| Leak check, `k=200` with the window past the last step | Builds an instrument | not launched | 1 cell | stdout only | ⚠️ not run |
| Step-size search, `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` | Builds an instrument | not launched | ~110 plain-render equivalents | `corrector/step_size_search.json`, and [the search table](#the-step-size-search) | ⚠️ not run |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Is there a step size at which the chain provably moves without diverging?** Pass if at
      least one `c` has a median relative displacement at or above `MIN_CHAIN_DISPLACEMENT = 0.05`
      while its maximum latent norm stays within 1.5× the uncorrected latent's and its ratio does
      not rise monotonically with `k`; the pick is the largest such `c`. A search where every `c`
      fails is a finding and stops the scope here: it says this corrector cannot be run stably on
      this model at these settings. The bound lives in source as a module-level constant, so moving
      it after the answer is visible shows up in a diff.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#the-step-size-search) ➡️

- [ ] ⚠️ With `k=0`, does the composer reproduce plain product-of-experts byte-identical?
- [ ] ⚠️ With a corrector window placed past the last step and `k=200`, is the output still
      byte-identical? This catches a corrector running outside its window, which the first check
      cannot see.
- [ ] ⚠️ Does the picked `c` sit in the middle of the tested range rather than at its edge? A pick
      at the smallest or largest value means the range was wrong, and the fix is to extend the
      sweep rather than accept the edge.
- [ ] ⚠️ Does anything else in the tree already measure a corrected residual? The vendored Du et al.
      code under `composition/reduce_reuse_recycle/` is expected and is a reference, not a
      collision.

## The step-size search

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Filled by task 2. The cost of the grid at step 26 rides on this pick, so the table is here rather
than in a log. `δ_t = c·β_t`, at `k=20`, on `a_cat__x__a_dog` seed 9.

| `c` | Median relative displacement | Max latent norm, × uncorrected | Ratio rises with `k`? | Verdict |
|---|---|---|---|---|
| 0.01 | | | | |
| 0.035 | | | | |
| 0.1 | | | | |
| 0.3 | | | | |
| 1.0 | | | | |

**Picked `c`:** not yet.

**Was the range adequate?** A pick at the smallest or largest tested `c` means it was not, and the
fix is to extend the sweep rather than accept the edge. Recorded by hand under instruction 3.3 of
the design.

## Asked after the result

Navigation: ⬅️ [The step-size search](#the-step-size-search) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a bar**, because anything written here is written with the answer
already visible. Empty until the search runs.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Across the five search rows, only `c` varies. Same pair, same
      seed, same `k`, same 50 DDIM steps, same guidance, same starting latent.
- [ ] ⚠️ **Was the instrument sound?** The leak checks compare against a `k=0` run of the new
      composer rather than against `run_cfg_poe`, so only the corrector logic differs and not the
      batch shape. And a flat displacement column across every `c` means the corrector is not being
      applied at all, which is a build bug wearing the costume of a search result.
- [ ] ⚠️ **Did the run respect the environment?** Output under `/datasets` with the disk guard on
      the filesystem actually written to, norms upcast to fp32 from fp16, and no relative path in
      any launch line onto a node this session is not on.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| anything measured on the corrected path | that the step size was picked by a recorded search rather than inherited, and which value it is |
| a corrector result at any `k` | that the corrector is unadjusted Langevin, with no Metropolis acceptance step, because SDXL is a score model with no energy head |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether one pair and one seed is enough to fix a step size for the whole scope | the search rerun on the composing pair, at the cost of the search again. Deferred until the gate's two panels are seen to agree or disagree | nothing yet. This file records the single pair as a choice rather than an oversight |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run the two leak checks. Neither costs more than one cell, and a failure in either stops the scope
before 670 plain-render equivalents are spent on a composer that is not inert.
