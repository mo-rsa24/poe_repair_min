# 🧪 Review: how much of the correction is the sampler's, and how much is the model's?

This file asks whether the leftover correction the sampler cannot remove is big enough to matter,
and it judges every answer against numbers written down before the corrector existed.

**Not yet run.** Every question below was written before any corrector existed, so no answer here
can be chosen after the fact. It judges [the whole corrector design](the-whole-corrector-design.md),
and its answer decides whether section 7 carries the corrector as a limitation, as an alternative,
or as a two-sentence note.

**These questions split, they do not move.** Each of the scope's seven plans takes the ones that
judge its own run into its own review file under `review/`, keeping the wording verbatim, because
a pre-registered question rewritten after the fact is no longer pre-registered.

## Recommended prompt (to read the result)

```
/analyze-run the corrector residual curve, k sweep against denoising step
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](the-whole-corrector-design.md) | the corrector, the step-size search, the grid, the thresholds |
| **this file** | **the verdict: not yet run** |
| [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md) | the result this plan puts under threat, and the numbers it quotes |
| [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) | cites Soiffer et al. for the claim this plan turns into a number |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [The step-size search](#the-step-size-search)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

The full vocabulary is in the design file's `## Words this plan uses` and is not copied here. The
four that every answer below turns on:

- **Error A**, the share a corrector can remove, which goes to zero as noise goes to zero.
- **Error B**, the share no corrector touches, which does not.
- **`k`**, how many Langevin steps run at each of the 50 noise levels before the reverse step.
- **The read zone**, the last five denoising steps, where error A has vanished by construction and
  anything left is error B. The early steps give a size only. They cannot say which of the two
  errors the size came from, and the design file's `## Where the two errors can be told apart`
  section says why.

> A Langevin step nudges the current latent along the model's score and adds a little fresh noise.
> Repeating it holds the sample on the distribution the model believes in at that noise level,
> without moving the denoising clock forward.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** for tasks 1 to 5. **Baseline** for tasks 6 and 7, which compare published
composition rules and may not change any claim on their own. **Idea** for task 8.

A result that misses the threshold below does not close the plan. It selects which of three paragraphs section 7
carries, and all three are written in the design file's `## Why this plan exists`.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The free bound: cached correction size as noise goes to zero | Tests the claim | not launched | no GPU, reads files on disk | this file's [first threshold](#the-question-written-before-the-run) | ⚠️ not run |
| Leak check, `k=0` byte-identical to plain product-of-experts | Checks the runner | not launched | 1 run | stdout only | ⚠️ not run |
| Leak check, `k=200` with the window past the last step | Checks the runner | not launched | 1 run | stdout only | ⚠️ not run |
| Step-size search, `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` | Checks the runner | not launched | ~110 plain-render equivalents | [the search table](#the-step-size-search) | ⚠️ not run |
| The `k` grid, 2 pairs × 6 `k` × 50 steps | Tests the claim | not launched | ~670 plain-render equivalents | `corrector/residual_curves.json`, 600 rows | ⚠️ not run |
| Sliding the corrector window, 4 seeds × 10 columns | Tests the claim | not launched | 40 runs | `mcmc/samples-as-a-ten-step-corrector-window-slides.png` | ⚠️ waiting on the threshold |
| SuperDiff at 50 against 200 steps | Baseline | not launched | 2 runs | step-count parity check | ⚠️ not run |
| The two grids of rules against correction amount | Baseline | not launched | 16 + 20 runs | `how-much-is-added/across-composition-rules/` | ⚠️ not run |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

The thresholds live in source as constants in `scripts/corrector_residual_curve.py`, so moving one
after the answer is visible shows up in a diff.

- [ ] ⚠️ **Is the correction still bounded away from zero at the last denoising step, on the
      uncorrected path, in the numbers already on disk?** Error A vanishes at zero noise and error
      B does not, so a nonzero size there is a lower limit on error B before any corrector runs.
      The answer states the number, its unit, and which trajectory it was cached along, or it does
      not count.

- [ ] ⚠️ **Does the corrector remove part of the correction and leave part of it?** Support if,
      over the last five steps, the ratio at the `k` where the curve has flattened has fallen by at
      least 0.20 of its `k=0` value and at least 0.20 of it is still there. Null if the change is
      under 0.05 at every step while the chain provably moved. Inconclusive if `k=100` and `k=200`
      still differ by more than 0.05, if the median displacement is under 0.05, or if the composing
      pair behaves like the failing pair.

      **The value at any single `k` describes the compute budget it was given. Only the trend
      across `k` is a result, and only once the curve has flattened.**

- [ ] ⚠️ **Does the corrector's compose rate peak in the same window the injected correction
      does?** Recorded either way. Same window means two different mechanisms acting at the same
      moment. A different window is the stronger result and needs its own paragraph. This question
      is answered even if the check above returns a null, because a flat residual curve does not
      imply a flat [compose rate](context/world/compose-rate.md). The corrector can relocate the
      trajectory without shrinking `‖r_t‖`.

- [ ] ⚠️ **Does SuperDiff still compose at 50 steps?** Its default is 200. If it does not, every
      comparison against it is between a working rule and a crippled one, and that sentence goes in
      the caption rather than in a footnote.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#the-step-size-search) ➡️

- [ ] ⚠️ With `k=0`, does the composer reproduce plain product-of-experts byte-identical?
- [ ] ⚠️ With a corrector window placed past the last step and `k=200`, is the output still
      byte-identical? This catches a corrector running outside its window, which the first check
      cannot see.
- [ ] ⚠️ Did the numerator `‖eps_J - eps_PoE‖` fall, or did the denominator `‖eps_PoE‖` rise? The
      ratio alone cannot say, and both are recorded for this reason.
- [ ] ⚠️ Did the chain actually move? The median relative displacement per `(k, t)` against the
      minimum written in source.
- [ ] ⚠️ Did it equilibrate? `k=100` against `k=200`, curve on curve.
- [ ] ⚠️ Does the pair that composes by default behave differently from the pair that blends? If
      both curves rise with `k`, the rise is the joint branch degrading off-distribution and the
      test is measuring itself.
- [ ] ⚠️ At `λ=1`, which of the four rules reproduce the joint render exactly? The two
      product-of-experts-family rows should. The corrector rows should not, since the chain has
      already left the joint trajectory. A row that fails to converge there is doing something
      beyond combining scores, which makes that column a free classifier.

## The step-size search

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Filled by task 3. The grid's whole cost rides on this pick, so the table is here rather than in a
log. `δ_t = c·β_t`, at `k=20`, on `a_cat__x__a_dog` seed 9.

| `c` | Median relative displacement | Max latent norm, × uncorrected | Ratio rises with `k`? | Verdict |
|---|---|---|---|---|
| 0.01 | | | | |
| 0.035 | | | | |
| 0.1 | | | | |
| 0.3 | | | | |
| 1.0 | | | | |

**Picked `c`:** not yet.

**Was the range adequate?** A pick at the smallest or largest tested `c` means it was not, and the
fix is to widen the range of `c` rather than accept the edge. Recorded by hand under instruction 9.3.

## Asked after the result

Navigation: ⬅️ [The step-size search](#the-step-size-search) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a threshold**, because anything written here is written with the
answer already visible. Empty until the grid runs.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only `k` varies across the curves. Same pair, same seed, same
      50 DDIM steps, same guidance, same step size, same starting latent.
- [ ] ⚠️ **Was the measuring tool sound?** The two leak checks, the displacement column, and the point
      where the curve flattens in `k`. Any one of them failing voids the reading.
- [ ] ⚠️ **Is the quantity what the caption says it is?** The residual norm at the settled point
      stands in for the distributional gap without being that gap. The corrector does not change
      the function `eps_J - eps_PoE`, it changes where that function is evaluated. And at high
      noise the smallest value it can reach is the non-commutation gap plus the model gap, so only
      the last steps say which error is which.
      Both sentences belong in the caption and neither may be dropped for space.
- [ ] ⚠️ **Did the run respect the environment?** All 600 rows present, output under `/datasets`,
      norms upcast to fp32 from fp16 before they were taken, launched under `nohup` outside Slurm
      and harvested by `pgrep` rather than `squeue`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Filled as the questions above are answered. Three rows are owed whatever the result, because they
are limits of the design rather than properties of the answer.

| What the paper says | What it owes alongside it |
|---|---|
| the size of the model's share | that it is measured at the last five denoising steps only, because that is the one place the two errors separate |
| anything about the early window | that steps 0 to 10 are where the compose rate is decided and where this measurement cannot attribute, so the timing result is not explained by this plan, only bounded by it |
| the corrected curve at any `k` | that `k=5` and `k=20` are two different trajectories rather than one point wiggled twice, and that `eps_J` is evaluated where the joint model would never go |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the early window's correction is sampler or model error | nothing in this plan. The two errors are inseparable at high noise by construction. A different design would be needed, and none is currently known | the strongest version of the timing paragraph. The weaker version, which this plan supports, is that a corrector does or does not reproduce the timing behaviour |
| whether one seed is enough for the curve the threshold is read from | a second seed, at the cost of the whole grid again. Deferred until the two pairs are seen to agree or disagree | nothing yet. The review records the single seed as a choice |
| whether Feynman-Kac correctors are built or cited | a full read of arXiv 2503.02819, which waits on this plan's tasks 4 and 7 | nothing. The default is cited |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Answer the first pre-registered question, the free bound. It needs no GPU and it reads files
already on disk, and its number changes how much the rest of the plan is worth running.
