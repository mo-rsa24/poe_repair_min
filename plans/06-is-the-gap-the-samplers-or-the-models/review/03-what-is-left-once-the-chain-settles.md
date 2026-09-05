# 🧪 Review: does the corrector remove part of the correction and leave part of it?

**The grid has run once, at the step size the search picked (`c = 30`), and fired the inconclusive
branch on every guard at once: the chain has not settled between `k = 100` and `k = 200`, its
latent norm passes 1.5× at the largest `k`, and the composing pair's residual rises with `k`. The
grid is being repeated at `c = 3` and `c = 0.3`.** Every question below was written before any
corrector existed, so no answer here can be chosen after the fact. This file judges
[the design for this measurement](../plans/hypothesis/03-what-is-left-once-the-chain-settles.md),
and its answer
decides whether section 7 of the paper carries the corrector as a limitation, as an alternative, or
as a two-sentence note. Step 21 of the running order cannot be written honestly until it is
answered.

## Recommended prompt (when the run lands)

```
/analyze-run the corrector residual curve, k sweep against denoising step
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis/03-what-is-left-once-the-chain-settles.md) | the grid, the thresholds in source, and the section on where the two errors separate |
| **this file** | **the verdict: not yet run** |
| [the corrector's verdict](02-the-corrector-and-the-step-size-it-runs-at.md) | the composer and the step size this grid runs at |
| [the free bound's verdict](01-the-free-bound-on-the-models-share.md) | the lower bound this grid's answer is read against |
| [the timing verdict](../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md) | the result this grid puts under threat, and the numbers it quotes |
| [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/checks/09-two-literature-checks-before-print.md) | cites Soiffer et al. for the claim this grid turns into a number |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [Where the two errors can be told apart, and where they cannot](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

The full vocabulary is in the design file's `## Words this plan uses` and is not copied here. The
four that every answer below turns on:

- **The sampler's share**, the part a corrector can remove, which goes to zero as noise goes to
  zero.
- **The model's share**, the part no corrector touches, which does not.
- **`k`**, how many Langevin steps run at each of the 50 noise levels before the reverse step.
- **The read zone**, the last five denoising steps, where the sampler's share has vanished by
  construction and anything left is the model's. The early steps give a size rather than an
  attribution, and [the section below](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
  says why.

> A Langevin corrector is a small repeated random walk that nudges the latent along the score and
> adds a little noise each time. Run for long enough at a fixed noise level it forgets where it
> started and settles wherever the score says the probability actually is.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** Missing the threshold below does not close the plan. It selects which of three
paragraphs section 7 carries, and all three are written in the design file's
`## Why this plan exists`.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The first short run, one pair, one seed, `k ∈ {0,1}` and then `k=5` | Tests the claim | 2026-09-05 16:52 to 17:02, in-session on mscluster85 device 0 (RTX 3090), pids 258072 and 260090, commit 72b8826, at `c=0.035` for timing only | 8.5 min in all | `corrector/logs/smoke.log`; wall time per noise level 1.5 s (`k=0`, 5 UNet evaluations), 2.3 s (`k=1`, 8), 5.3 s (`k=5`, 20) | ✅ within the bar: measured ratios 1.5× and 3.5× against the cost table's 1.6× and 4×, so the branch count is right; one Langevin step costs about 0.76 s on the 3090, so `k=200` on one pair is about 2.1 h there and the whole two-pair grid about 7 h on one 3090 |
| The `k` grid at `c = 30`, 2 pairs × 6 `k` × 50 steps | Tests the claim | 2026-09-05 19:04 to 20:11, `nohup` on mscluster110 device 0 (RTX PRO 6000 Blackwell, `co3_bw`), pids 433699 and 434481, commit 0150704; the composing pair first, then cat × dog, one process each | 67 min for both pairs on that card (`k = 200` takes 16 to 26 min per pair) | `corrector/curves/*__c30__k*.json` (12 cells), aggregated into `corrector/residual_curves.json` with `c = 30` on every row; `corrector/verdict_c30.txt`; figure `mcmc/how-much-of-the-correction-a-corrector-removes-c30.png` | ❓ inconclusive on four guards, see the question below; the composer's final images at this `c` are texture noise from `k = 1` |
| The `k` grid at `c = 3` and `c = 0.3`, same cells | Tests the claim | 2026-09-05 19:12 on mscluster108 device 1 (the composing pair at `c = 0.3`, `co3`, pid 300147) and from 20:12 on mscluster110 device 0 (the queue `corrector/queue_110_b.sh`: composing pair at 3, cat × dog at 3, cat × dog at 0.3) | ~1 h on the Blackwell card per pair and `c`; ~3 h on the RTX 8000 | the same files keyed by `c`; `verdict_c3.txt`, `verdict_c0p3.txt` | ◑ running |

## Where the two errors can be told apart, and where they cannot

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

**This is why the measurement is read at the end of the run rather than across it, and it is
repeated here in full rather than compressed into a caveat line.** Getting it wrong would put a
number in the paper that means something other than what the caption says.

**What the corrector converges to.** Langevin driven by the product-of-experts score at noise level
`t` has stationary distribution `q_t = p_t(cat)·p_t(dog)`, the product of the two diffused
marginals. That is the distribution the summed score is the exact score of. At `t = 0` the noise is
gone and `q_0 = p(cat)·p(dog)`, which is the product of experts the method asked for in the first
place. So a converged corrector delivers the product exactly, and that is the whole of Du et al.'s
claim.

**What is left at the end of the run is the model's share, cleanly.** At the last steps the latent
is drawn from `p(cat)·p(dog)`. The residual `eps_J - eps_PoE` is then the joint model disagreeing
with the product on the product's own support. There is no sampler error left to contaminate it,
because that error is defined to vanish there. This number is the paper's estimate of the model's
share.

**What is left at the start of the run is still both.** At `t > 0` the corrector has settled the
latent into `q_t`, the product of the diffused marginals. The thing the reverse process would need
is the diffusion of the product, and those two differ precisely because noising and multiplying do
not commute. So what remains at high noise is the non-commutation gap plus the diffused model gap,
and no amount of `k` separates them. **The early part of the curve gives a size rather than an
attribution.**

**Which lands on the timing result.** The compose-decisive window is steps 0 to 10, the high-noise
end, which is exactly the region where this measurement cannot attribute. That limit belongs in the
caption and in this file, and it is never argued away. What this scope can say about the early
window is behavioural rather than attributional, and it is
[step 27](04-does-the-corrector-compose-in-the-same-window.md).

**Why the residual norm is a proxy and not the gap itself.** The corrector does not change the
function `eps_J - eps_PoE`; it changes where that function is evaluated. A falling curve says the
settled latents sit where the two networks agree better, which is evidence about the distributional
gap rather than a measurement of it.

**What the composing pair controls.** On `a_butterfly__x__a_flower_meadow`, which composes under
plain product-of-experts, `q_t` is already close to the joint and `eps_J` is evaluated somewhere it
has seen. Its curve should be low at `k=0` and should not rise with `k`. If it rises the same way
the failing pair does, the rise is this measurement walking the joint branch off-distribution and no
reading of either curve is licensed. `an_elephant__x__a_penguin` is not used for this, because
whether it composes by default is
[an open question in another review file](../../04-does-the-fix-reach-unseen-pairs/review/01-the-clean-pair-pool.md),
and a control whose own behaviour is unsettled controls nothing.

## The question written before the run

Navigation: ⬅️ [Where the two errors can be told apart](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

The thresholds live in source as constants in `scripts/corrector_residual_curve.py`, so moving one
after the answer is visible shows up in a diff.

- [ ] ⚠️ **Does the corrector remove part of the correction and leave part of it?** Support if,
      over the last five steps, the ratio at the `k` where the curve has flattened has fallen by at
      least 0.20 of its `k=0` value and at least 0.20 of it is still there. Null if the change is
      under 0.05 at every step while the chain provably moved. Inconclusive if `k=100` and `k=200`
      still differ by more than 0.05, if the median displacement is under 0.05, or if the composing
      pair behaves like the failing pair.

      > A null here means the correction's size came out the same with the corrector running as
      > without it.

      **The value at any single `k` is a statement about the compute budget rather than about the
      problem. Only the trend across `k` is a result, and only once the curve has flattened.** That
      rule is enforced by `MAX_K_INSTABILITY` in source, which refuses to license a reading, rather
      than by a caption asking the reader to remember it.

      **At `c = 30`, the step size the search picked: ❓ inconclusive**, from `verdict_c30.txt`.
      Read-zone means over the last five steps, one seed, one trajectory per `k`. Cat × dog:
      0.1685 at `k = 0`, then 0.1137, 0.1209, 0.1111, 0.1205, 0.1124 at `k = 1, 5, 20, 100, 200`;
      the numerator fell from 37.9 to 27.9 while the denominator rose from 225 to 278. The
      composing pair: 0.0896 at `k = 0`, then 0.1071, 0.1342, 0.1692, 0.1343, 0.1110. Four guards
      failed: `k = 100` against `k = 200` differ by 0.067 (cat × dog) and 0.173 (composing pair) of
      the `k = 100` value against the 0.05 bar; the latent norm reached 1.52× and 1.66× its start
      at `k = 200` against the 1.5× bar; and the composing pair's read-zone ratio rose by 24%
      against the 5% drift bar. The chain moved (median displacement 1.26 on both pairs). The
      images behind the rows say why: at this step size every `k ≥ 1` render is texture noise
      (`corrector/search_renders/butterfly_c30_by_k.png`), so the joint branch is evaluated far
      off the data and its disagreement with the product means nothing. Per the branch's own
      instruction the step size goes back to step 25, and the grid is repeated at `c = 3` and
      `c = 0.3`; the verdict below this line is written when those land.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Did the numerator `‖eps_J - eps_PoE‖` fall, or did the denominator `‖eps_PoE‖` rise? The
      ratio alone cannot say, and both are recorded for this reason.
- [ ] ⚠️ Did the chain actually move? The median relative displacement per `(k, t)` against the
      minimum written in source.
- [ ] ⚠️ Did it equilibrate? `k=100` against `k=200`, curve on curve.
- [ ] ⚠️ Does the pair that composes by default behave differently from the pair that blends? If
      both curves rise with `k`, the rise is the joint branch degrading off-distribution and the
      measurement is measuring itself.
- [ ] ⚠️ How does the measured remainder at the last steps compare with
      [the free bound](01-the-free-bound-on-the-models-share.md) read off the cached
      renders? Two independent routes to the same quantity that disagree is a finding about one of
      the two measuring tools.
- [x] ✅ Was the first short run's measured wall time per noise level within 2× of the cost table?
      Yes. Per noise level on the 3090: 1.5 s at `k=0`, 2.3 s at `k=1`, 5.3 s at `k=5`, against
      5, 8 and 20 UNet evaluations. The measured ratios to `k=0` are 1.5× and 3.5× where the table
      predicts 1.6× and 4×, so the evaluation count per level is right and the grid was not
      re-planned. The absolute cost is higher than the table's plain-render equivalents suggest,
      about 0.76 s per Langevin step, which puts the two-pair grid at roughly 7 hours on one 3090
      rather than the 670 render-equivalents' 4.5 hours; within the 2× bar.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible. Empty until the grid runs.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only `k` varies across the curves. Same pair, same seed, same
      50 DDIM steps, same guidance, same step size, same starting latent.
- [ ] ⚠️ **Was the measuring tool sound?** The two leak checks from
      [step 25](02-the-corrector-and-the-step-size-it-runs-at.md), the displacement
      column, and the flattening in `k`. Any one of them failing voids the reading.
- [ ] ⚠️ **Is the quantity what the caption says it is?** The residual norm at the settled point is
      a proxy for the distributional gap rather than the gap itself. The corrector does not change
      the function `eps_J - eps_PoE`; it changes where that function is evaluated. And at high noise
      what remains is the non-commutation gap plus the model gap, so only the last steps attribute.
      Both sentences belong in the caption and neither may be dropped for space.
- [ ] ⚠️ **Did the run respect the environment?** All 600 rows present, output under `/datasets`,
      norms upcast to fp32 from fp16 before they were taken, launched under `nohup` outside Slurm
      and harvested by `pgrep` rather than `squeue`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Three rows are owed whatever the result, because they are limits of the design rather than
properties of the answer.

| What the paper says | What it owes alongside it |
|---|---|
| the size of the model's share | that it is measured at the last five denoising steps only, because that is the one place the two errors separate |
| anything about the early window | that steps 0 to 10 are where the [compose rate](../../../context/world/compose-rate.md) is decided and where this measurement cannot attribute, so this plan bounds the timing result rather than explaining it |
| the corrected curve at any `k` | that `k=5` and `k=20` are two different trajectories rather than one point wiggled twice, and that `eps_J` is evaluated where the joint model would never go |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the early window's correction is the sampler's or the model's | nothing in this scope. The two are inseparable at high noise by construction. A different design would be needed, and none is currently known | the strongest version of the timing paragraph. The weaker version, which this scope supports, is that a corrector does or does not reproduce the timing behaviour |
| whether one seed is enough for this curve | a second seed, at the cost of the whole grid again. Deferred until the two pairs are seen to agree or disagree | nothing yet. This file records the single seed as a choice |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Confirm both leak checks passed and a `c` is picked at
[step 25](02-the-corrector-and-the-step-size-it-runs-at.md), then make the first short
run before committing 670 plain-render equivalents to the grid.
