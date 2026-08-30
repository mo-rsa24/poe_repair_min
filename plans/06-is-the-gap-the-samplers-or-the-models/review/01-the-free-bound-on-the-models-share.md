# 🧪 Review: is the correction still bounded away from zero at the end of the run?

**Nothing has run yet.** Every question below was written before any number was looked at. This
file judges [the free-bound design](../plans/hypothesis/01-the-free-bound-on-the-models-share.md),
and its answer says how much the rest of this scope is worth. A lower bound above zero means the
model's share is already known to be nonzero before a single corrector step runs.

## Recommended prompt (when the run lands)

```
/analyze-run the correction size as the run approaches zero noise, from paper/iclr/figures/correction-size-over-the-denoising-run.json
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/01-the-free-bound-on-the-models-share.md) | which files to read, the three caveats, and what the number does to the scope |
| **this file** | **the verdict: not yet run** |
| [the step 26 verdict](03-what-is-left-once-the-chain-settles.md) | the measurement this lower bound is read against |
| [the questions this file was split from](../source/the-questions-pre-registered-against-it.md) | the whole pre-registered set, written before the scope existed |

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

- **The correction, `r_t`**: the per-step gap between what the joined prompt predicts and what
  adding the two separate prompts predicts, both evaluated at the same latent.
- **The sampler's share**: the part a Markov-chain corrector can take away. It goes to zero as the
  noise goes to zero, which is the whole reason this read works.
- **The model's share**: the part no corrector touches, because multiplying two distributions asks
  for one thing that is both animals. It does not vanish at zero noise.
- **The uncorrected path**: the trajectory plain product-of-experts actually follows, which is the
  path every cached number in this project was written along.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** Missing the threshold here does not close anything, because the threshold is on
the honesty of the read rather than on its value. What it changes is how much weight step 26's
answer carries. Without this lower bound, that answer stands alone rather than being read against a
number that cost nothing.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The free bound: cached correction size as noise goes to zero | Tests the claim | not launched | no GPU, reads files already on disk | this file's [the question written before the run](#the-question-written-before-the-run) | ⚠️ not run |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **Is the correction still bounded away from zero at the last denoising step, on the
      uncorrected path, in the numbers already on disk?** The sampler's share vanishes at zero
      noise and the model's does not, so a nonzero size there is a lower bound on the model's share
      before any corrector runs. The answer states four things or it does not count: the number
      with its unit, the seeds it came from, which trajectory it was cached along, and the
      unnormalised `‖r_t‖` beside the ratio.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ What is the unnormalised `‖r_t‖` at the last denoising steps, read from the raw `delta`
      tensors rather than from the ratio? The JSON stores only `‖r_t‖/‖eps_PoE‖`, and a ratio can
      fall because its denominator rose, so the numerator and the denominator are reported
      separately.
- [ ] ⚠️ Do the numerator and the ratio tell the same story at the low-noise end? A numerator
      bounded away from zero while the ratio falls means the denominator grew, which is a fact
      about `eps_PoE` late in the run and not about the model's share.
- [ ] ⚠️ Which trajectory were the residuals cached along? At λ=0 that is the plain
      product-of-experts path, and the answer is about that path only.
- [ ] ⚠️ Were the joint prediction and the product-of-experts prediction evaluated at the same
      latent at each step? If they were not, the late-step values carry accumulated path difference
      as well as rule difference and cannot be read at all.
- [ ] ⚠️ Do the three numbers this scope quotes from the timing verdict still say what they said?
      0.656 at steps 0 to 10, 0.000 from steps 20 to 30 onward, and the correction about 2.7 times
      larger late than early. If any has moved, the threat this scope answers has changed shape.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the read happens.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Not applicable in the usual sense. Nothing is compared here;
      one curve is read at one end. What stands in for it is the "same latent at each step"
      question above, which is the condition under which the late-step values mean anything.
- [ ] ⚠️ **Was the measuring tool sound?** Does
      [correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) at
      line 101 compute what this read assumes it computes, over the renders the JSON claims, upcast
      to fp32 from the fp16 cache?
- [ ] ⚠️ **Did the run respect the environment?** Nothing is written and no GPU is used, so the
      check reduces to whether the three files read were the ones named in the design rather than
      similarly-named neighbours.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the lower bound on the model's share | that it is read at the low-noise end of the uncorrected path only, from a cache written for a different question |
| that lower bound as a number | whether it is the unnormalised size or a ratio, since the two mean different things and only one of them answers the question |
| this lower bound beside the remainder measured at step 26 | that the two come from different seeds. The cached curve is seeds 4, 42 and 123; step 26 runs seed 9, on the same two pairs. Putting them side by side is a cross-seed comparison and the caption says so |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Answer the pre-registered question. It needs no GPU and reads files already on disk, and its number
changes how much the rest of the scope is worth running.
