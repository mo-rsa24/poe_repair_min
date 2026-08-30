# 🧪 What is left once the chain settles

Asks how much of the correction survives once a corrector has been run to equilibrium at every noise
level, because what survives at the end of the run is the model's error and what goes away is the
sampler's.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md — <which of the three branches fired>
```

## Recommended skill

▶ `/analyze-run the corrector residual curve, k sweep against denoising step` ✅ reads a finished
   grid against a threshold written before it, which is this plan's whole shape.
   alt: `/defend-results` once the branch has fired, since this is the measurement a reviewer will
   attack first.

## Position in the plan tree

**Step 26 of 30.** Waits on steps 24 and 25. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 24 | [hypothesis-01: the-free-bound-on-the-models-share](hypothesis-01-the-free-bound-on-the-models-share.md) ⚠️ | the lower bound this plan's answer is read against |
| 25 | [instrument-01: the-corrector-and-the-step-size-it-runs-at](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) ⚠️ | the composer and the `c` this plan runs at |
| **26 (current)** | **hypothesis-02: what-is-left-once-the-chain-settles** ⚠️ | **the run the rest of the scope waits on: the correction's size per step against corrector count `k`, judged against a three-way threshold written in source before the answer existed** |
| 27 | [hypothesis-03: does-the-corrector-compose-in-the-same-window](hypothesis-03-does-the-corrector-compose-in-the-same-window.md) ⚠️ | waits on this, softly |
| 21 | [writing-06: mechanism-and-limitations](../../07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md) ⚠️ | cannot be written honestly until this plan returns a size |

Design only. Verdicts and run state live in
[the paired review file](../review/hypothesis-02-what-is-left-once-the-chain-settles.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Where the two errors can be told apart, and where they cannot](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [What it costs](#what-it-costs)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Not run: what this plan leaves alone](#not-run-what-this-plan-leaves-alone)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

Run the Langevin corrector at every noise level so the latent [settles into the distribution](/home-mscluster/mmolefe/goal-setting/learning/sampler-correctors-for-composition/plans/11-stationary-and-detailed-balance.md) the
product-of-experts score actually describes, then measure what is left of the correction at that
settled point. The part that goes away when the chain settles belongs to the sampler. The part
still there at the end of the run belongs to the model.

> A Langevin corrector is a small repeated random walk that nudges the latent along the score and
> adds a little noise each time. Run for long enough at a fixed noise level it forgets where it
> started and settles wherever the score says the probability actually is.

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

**The correction, `r_t`.** The per-step gap `eps_J - eps_PoE` between what the joined prompt
predicts and what adding the two separate prompts predicts, both evaluated at the same latent.
Every result in this project so far is about this quantity.

**The sampler's share.** Summing two diffused scores gives the score of the product of the two
diffused marginals, and that sequence of distributions is not the forward diffusion of anything.
Reverse diffusion is only valid on a sequence that is, so running it on the summed score does not
sample the product it was asked for. Markov-chain correction does not need a valid forward process,
only a target at each noise level, so with enough corrector steps it samples the product exactly.
This is the share a corrector can take away, and it goes to zero as noise goes to zero.

**The model's share.** `p(cat)·p(dog)` is not `p(cat and dog)`. Multiplying two distributions asks
for one thing that is both animals, which is the [chimera](../../../context/world/chimera.md). No
corrector touches this, and it does not vanish at zero noise.

**The corrector count, `k`.** How many unadjusted Langevin steps run at each of the 50 noise levels
before the reverse step is taken. `k=0` is plain product-of-experts.

**The settled point, `x_t^(k)`.** The latent after those `k` Langevin steps at noise level `t`.
Both networks are evaluated there, so `r_t^(k) = eps_J(x_t^(k)) - eps_PoE(x_t^(k))`.

**The read zone.** The last five denoising steps, where the sampler's share has vanished by
construction and anything left is the model's.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis.** A Langevin corrector run to equilibrium at each noise level removes part of
`r_t` and leaves a part that does not go away. What it leaves at the low-noise end of the run is
the model's error, `p(cat)·p(dog) ≠ p(cat and dog)`.

**If true.** The paper reports two sizes, and the adapter is justified by the share a corrector
cannot reach.

**If false, meaning the curve does not move.** The correction is all the model's, the framing
strengthens, and the sampler comparison becomes a two-sentence related-work note.

**If the curve collapses.** The timing result is reread as a sampler artifact, the sampler
comparison stops being optional and becomes a baseline the paper must beat.

**Rationale.** The project has a causal result and a timing result. Injecting `r_t` raises the
[compose rate](../../../context/world/compose-rate.md) with dose while a norm-matched random control
stays at chance level. Injecting it only into steps 0 to 10 composes 0.656 of 32 pair-and-seed runs,
while steps 20 to 30 onward compose 0.000. Both are in
[the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md).
The threat is that the same picture is what a sampler artifact looks like: the sampler's share is
worst at high noise, and high noise is the early steps, which is exactly the window that decides
the outcome. Nothing measured so far tells the two apart, because `r_t` is both of them added
together.

**Dataset details.** Two pairs at one seed. `a_cat__x__a_dog` seed 9, which blends under plain
product-of-experts, and `a_butterfly__x__a_flower_meadow` seed 9, which composes. Six corrector
counts, 50 denoising steps: 600 rows. No images, no VAE decode, no detector.

**Associated materials.** [The review questions](../review/hypothesis-02-what-is-left-once-the-chain-settles.md),
[the whole corrector design](../source/the-whole-corrector-design.md), and
[the scope's direction](../MASTER_PLAN.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Two corrector counts are two trajectories, not one point wiggled twice.**

`k=5` and `k=20` diverge at the first noise level and never meet again. Any axis label saying "the
correction at step `t`" is wrong here. It is the correction along the `k`-corrected path, and every
caption says so.

**The joint model is evaluated where it would never go.**

`eps_J(x_t^(k))` asks the joint-prompt branch to predict noise at a latent produced by a chain
targeting the product of two marginals. That is a fair thing to ask and an off-distribution
evaluation at the same time. It also gives a third reason the curve could rise with `k`, on top of
a step size that is too large: the joint branch could simply be degrading as the chain walks away
from anywhere it has seen. The pair that composes by default is the control that separates those,
and [the section below](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) says how.

**A ratio with a moving denominator is not a measurement.**

The cached size measure is `‖r_t‖/‖eps_PoE‖`, and the corrector moves both terms. Record the
numerator and the denominator as separate columns and plot the ratio on top of them, so a fall in
the ratio caused by a rising denominator is visible rather than hidden.

**Flat is ambiguous until the chain is shown to have moved.**

A curve flat from `k=1` reads as "the corrector does nothing, it is all the model's share", and it
reads identically when the step size is so small the chain never left where it started. Every point
carries the median relative displacement `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖` beside it. A flat curve
whose displacement sits below `MIN_CHAIN_DISPLACEMENT` means the measurement failed, and it is not
a null.

**The trend is the result, and only once it stops falling.**

The value at any single `k` is a statement about the compute budget, not about the problem. A curve
still falling between the two largest `k` has not equilibrated and licenses no reading at all,
which is why the grid runs to `k=200` rather than stopping at 100. This is enforced by
`MAX_K_INSTABILITY` in source, not by a caption asking the reader to be careful.

**The cache cannot be used.**

The corrector moves `x_t`. Every cached trajectory under `interaction_term/` and `cache_analyses/`
is for a path the corrector is no longer on.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A Langevin corrector run to equilibrium at each noise level removes part of `r_t` and leaves a
part that does not go away, and what it leaves at the low-noise end of the run is the model's
error.**

**Independent variable.** The corrector count `k ∈ {0, 1, 5, 20, 100, 200}` at the step size fixed
by [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md), and nothing else. Same
pair, same seed, same 50 DDIM steps, same guidance 7.5, same starting latent.

**Dependent variables.** Per denoising step `t` and per `k`: the numerator
`‖eps_J(x_t^(k)) - eps_PoE(x_t^(k))‖₂`, the denominator `‖eps_PoE(x_t^(k))‖₂`, their ratio, and the
relative chain displacement. All in fp32, upcast from the fp16 the models run in.

**Confounds, and what each is answered by.**

| What else could produce a falling curve | What answers it |
|---|---|
| the denominator grew rather than the numerator shrinking | both are recorded and plotted separately (task 2.3) |
| the chain diverged and the latents are garbage | the latent-norm bound from the step-size search, and the pair that composes by default |
| the joint branch degrades off-distribution, so any rise comes from the measurement itself rather than from the pair | the pair that composes by default, where the corrector's target is close to the joint and the curve should stay low at all `k` |
| the corrector never moved | the displacement column, with its minimum written in source |
| the chain has not equilibrated at the largest `k` | the `k=100` against `k=200` comparison, with an instability bound in source |

**Falsify condition, three-way, at the pre-registered thresholds.** The thresholds live in source as
module-level constants in `scripts/corrector_residual_curve.py`, following the `MIN_MEDIAN_RATIO`
pattern this repo already uses, so they cannot be moved after the answer is visible.

- **Support, the split is real.** Over the last five denoising steps, the ratio at the `k` where the
  curve has settled has fallen by at least `MIN_DROP_FOR_SPLIT = 0.20` of its `k=0` value. At least
  `MIN_REMAINDER_FOR_SPLIT = 0.20` of it is still there. Something went away and something stayed.
  The size of what stayed is the paper's number for the model's share.
- **Null, it is all the model's.** At every step the change between `k=0` and the settled `k` is
  under `MAX_DRIFT_FOR_NULL = 0.05`, while the median displacement is at or above
  `MIN_CHAIN_DISPLACEMENT = 0.05`. The corrector ran, moved the latent, and changed nothing. The
  null goes into section 7, and the comparison half of this scope becomes a baselines table rather
  than a diagnosis.
- **Inconclusive.** The ratio still moves by more than `MAX_K_INSTABILITY = 0.05` between `k=100`
  and `k=200`, or the displacement is below its minimum, or the pair that composes by default shows
  the same behaviour as the failing pair. Then the step size is wrong or the measurement is reading
  itself. Return to [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) and widen
  the search. Never loosen a threshold.

**Why this matters right now.** This is the number the paper's framing rests on and it has never
been measured here. Step 21 of the running order cannot be written honestly without it.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The paper's argument is that a small, shared, learnable correction fixes
composition, and that a rank-8 adapter can carry it. If most of `r_t` is the sampler's error, a
training-free corrector gets most of the same benefit and the adapter is answering a question the
sampler had already solved. That is the strongest reviewer objection available against this work,
and [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md)
already cite the paper that raises it.

**The approach.** Let a Markov chain do at each noise level what reverse diffusion cannot, then
measure what the correction still is at the settled point. Read the answer where the two errors
separate, and say plainly where they do not.

**Key insights.**

1. Two papers bracket the answer before it is measured. Du et al. conclude in the abstract of
   [arXiv 2302.11552](https://arxiv.org/abs/2302.11552) that "the sampler (not the model) is
   responsible for this failure". Soiffer et al.,
   [arXiv 2606.23920](https://arxiv.org/abs/2606.23920), show that no inference-time technique
   alone produces the target distribution once the composed condition is out of distribution, and
   that existing correctors reduce the gap without closing it. Read together they predict a partial
   result, so this plan is designed to return a size rather than a verdict.
2. The outcome changes the manuscript either way, which is what makes it worth the GPU time.

| What the result says | What changes |
|---|---|
| mostly the model's share | the framing strengthens. The correction is a model-level object, the adapter is the right tool for it, and the sampler comparison becomes a two-sentence related-work note |
| an even split | the paper reports both sizes, and the adapter is justified by the share a corrector cannot reach. Section 7 carries the corrector as the honest alternative for the other share |
| mostly the sampler's share | the timing result is reread as a sampler artifact, the sampler comparison stops being optional and becomes a baseline the paper must beat, and section 7 carries it as a limitation rather than a footnote |

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) ➡️

```
  ‖r_t^(k)‖ / ‖eps_PoE‖        cat x dog, seed 9
  |
  |  k=0    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   the curve already measured
  |  k=1     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |  k=20      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |  k=200       ~~~~~~~~~~~~~~~~~~~~~~~          <- if these stop separating,
  |                                                  the chain has equilibrated
  |  ..........................................   <- what is left here at the
  |                                                  RIGHT-HAND END is the model's
  +----------------------------------------------
     step 0                              step 49
     high noise                          low noise
     both errors mixed                   the sampler's is gone
```

The read is not the whole curve. It is the right-hand end, where the sampler's share has vanished
by construction and anything left is the model. The left-hand end is where the compose rate is
decided and where the two errors cannot be separated, which is the honest limit this plan returns
alongside its number.

## Where the two errors can be told apart, and where they cannot

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

This section is why the measurement is read at the end of the run rather than across it. Getting it
wrong would put a number in the paper that means something other than what the caption says. It is
repeated in [the review file](../review/hypothesis-02-what-is-left-once-the-chain-settles.md) and
is never compressed into a caveat line.

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
not commute. So what the curve bottoms out at under high noise is the non-commutation gap plus the diffused model gap, and
no amount of `k` separates them. The early part of the curve is a size, not an attribution.

**Which lands on the timing result.** The compose-decisive window is steps 0 to 10, the high-noise
end, which is the region where this plan cannot attribute. That limit is stated in the caption and
in the review file rather than argued away. What the plan can say about the early window is whether
a corrector applied only there changes the compose rate, which is
[step 27](hypothesis-03-does-the-corrector-compose-in-the-same-window.md) and is a behavioural
answer rather than an attribution.

**Why the residual norm is a proxy and not the gap itself.** The corrector does not change the
function `eps_J - eps_PoE`; it changes where that function is evaluated. A falling curve therefore
says the settled latents sit where the two networks agree better, which is evidence about the
distributional gap rather than a measurement of it. The caption owes that sentence.

**What the composing pair controls.** On `a_butterfly__x__a_flower_meadow`, which composes under
plain product-of-experts, `q_t` is already close to the joint and `eps_J` is evaluated somewhere it
has seen. Its curve should be low at `k=0` and should not rise with `k`. If it rises the same way
the failing pair does, the rise is this measurement walking the joint branch off-distribution and no
reading of either curve is licensed. `an_elephant__x__a_penguin` is not used for this, because
whether it composes by default is
[an open question in another review file](../../04-does-the-fix-reach-unseen-pairs/review/instrument-01-the-clean-pair-pool.md),
and a control whose own behaviour is unsettled controls nothing.

## Description: what to build

⬅️ [Previous](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`scripts/corrector_residual_curve.py`.** Carries the five thresholds as module-level constants,
   runs the `k` grid, writes `residual_curves.json`, and prints which of the three branches fired.
   It is also the script [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md)
   added `--check-identity` and `--step-size-search` to, so the thresholds and the search live in
   one file.
2. **The grid.** `k ∈ {0, 1, 5, 20, 100, 200}` over all 50 steps, on two pairs at seed 9. Per
   `(pair, seed, k, t)`: numerator, denominator, ratio, relative displacement, latent norm.
3. **One figure.** Ratio against denoising step, one curve per `k`, one panel per pair, with the
   two norms on a second row.

Output lands at
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/residual_curves.json`, never
under `/home-mscluster`.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 3 of [the scope's direction](../MASTER_PLAN.md): measure the correction's size per
denoising step against corrector count `k`, on one failing pair and one composing pair, and apply a
three-way threshold written in source before the answer was visible.

**Goals**

1. `residual_curves.json` with its 600 rows: 2 pairs × 6 `k` values × 50 steps.
2. One of the three branches printed, with the numbers it was judged against.
3. The figure, with the read zone marked and the un-attributable zone shaded.
4. The limit from
   [the section above](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) carried into
   the caption and the review file, in full.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#what-it-costs) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Everything writes under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`.
  `/home-mscluster` hit 100% once and silently killed checkpointing, so the job script carries a
  disk guard on `/datasets`, the filesystem it actually writes to, per
  [environment/storage.md](../../../environment/storage.md).
- **biggpu allows one job per user**, so the `k` grid runs under `nohup` outside Slurm and `squeue`
  is blind to it. Harvest by `pgrep -af 'corrector|run_corrector'` on the session node, per
  [environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md).
- **The cached trajectories cannot be used.** The corrector moves the latent onto a different path,
  so every cached pair-and-seed trajectory under `interaction_term/` is for a path this plan is not
  on.
- The models run in fp16. Every norm upcasts to fp32 before it is taken, since the differences
  measured here are small enough that fp16 accumulation shows up in the third digit.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so the figure's PDF comes from matplotlib and never from a `pdflatex` pass,
  per [environment/paper.md](../../../environment/paper.md).

## What it costs

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

The arithmetic, so the first short run can check it rather than a guess standing in for it. Guided
product-of-experts is 3 UNet evaluations per call (prompt A, prompt B, unconditional) and the joint
branch is 2 (joint, unconditional). One noise level at corrector count `k` costs `3k + 5`
evaluations, and there are 50 levels.

| `k` | UNet evaluations, one pair-seed | As multiples of a plain 50-step render (150 evaluations) |
|---|---|---|
| 0 | 250 | 1.7 |
| 1 | 400 | 2.7 |
| 5 | 1,000 | 6.7 |
| 20 | 3,250 | 22 |
| 100 | 15,250 | 102 |
| 200 | 30,250 | 202 |
| whole grid | 50,400 | 336 |

Two pairs at one seed is about 670 plain-render equivalents. No VAE decode and no detector
anywhere, so wall time tracks UNet evaluations directly.

## Tasks

⬅️ [Previous](#what-it-costs) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Confirm both leak checks from
      [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) passed and a `c` is
      picked.
  - **Done when:** the picked `c` is quoted in this plan's review file, with the instruction-3.3
    verdict beside it. Without it the grid is 670 plain-render equivalents spent at an undefended
    step size.

▶ **Next: [task 1.1](#1--write-the-gate-script-with-its-bars-in-source)**.

### 1. 🔧 Write the gate script, with its bars in source

◀ **Needs: [instruction 3.3 of step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md#3--read-the-step-size-search-by-eye-before-the-grid-launches)**,
your signed-off `c`.

- [ ] **1.1** Write `scripts/corrector_residual_curve.py` with the five thresholds as module-level
      constants.

    ```python
    MIN_DROP_FOR_SPLIT      = 0.20   # fraction of the k=0 ratio that must go away
    MIN_REMAINDER_FOR_SPLIT = 0.20   # fraction that must still be there
    MAX_DRIFT_FOR_NULL      = 0.05   # below this at every step, with the chain provably moved, is a null
    MIN_CHAIN_DISPLACEMENT  = 0.05   # median ‖x_t^(k)-x_t^(0)‖/‖x_t^(0)‖ below this is a failed instrument
    MAX_K_INSTABILITY       = 0.05   # k=100 against k=200 differing by more than this is not equilibrated
    ```

  - They sit in source so moving one after seeing the answer shows up in a diff.
  - **`MAX_K_INSTABILITY` is where the compute-budget discipline lives.** The value at any single
    `k` is a statement about the compute budget, not about the problem. Only the trend across `k`
    is a result, and only once it stops falling. That rule is enforced by this constant refusing to
    license a reading, not by a sentence asking the reader to remember it.
  - **Done when:** the five constants exist at module level and `--verdict` refuses to print a
    branch when the instability bound is exceeded.
- [ ] **1.2** The first short run, in-session: one pair, one seed, `k ∈ {0, 1}` only.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_residual_curve.py --smoke --k 0,1 --pair a_cat__x__a_dog --seed 9
    ```

  - Check the measured wall time per noise level against [What it costs](#what-it-costs). A cost
    estimate out by more than 2× means the branch count is wrong and the grid is re-planned before
    it launches.
  - **Done when:** the short run prints a per-level time and it is within 2× of the table.

▶ **Next: [task 2.1](#2--run-the-grid-and-plot-it)**, the launch.

### 2. 🚀 Run the grid, and plot it

◀ **Needs: [task 1.2](#1--write-the-gate-script-with-its-bars-in-source)**, so the cost is known
before 670 plain-render equivalents are committed.

- [ ] **2.1** The grid, under `nohup` outside Slurm.

    ```bash
    OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector
    nohup bash scripts/mechanism_study/run_corrector_curve.sh > "$OUT/curve.log" 2>&1 &
    pgrep -af 'corrector|run_corrector'          # squeue is blind to this
    ```

  - `k ∈ {0, 1, 5, 20, 100, 200}` over all 50 steps, on `a_cat__x__a_dog` seed 9 and on
    `a_butterfly__x__a_flower_meadow` seed 9. Write per `(pair, seed, k, t)`: numerator,
    denominator, ratio, relative displacement, latent norm. All fp32, upcast from fp16.
  - Output goes to: `$OUT/residual_curves.json`
  - **Done when:** `residual_curves.json` holds 600 rows (2 pairs × 6 `k` × 50 steps), counted, not
    assumed.
- [ ] **2.2** Apply the three-way threshold in code, print which branch fired, and write the verdict into
      the review file with the numbers it was judged against.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_residual_curve.py --verdict
    ```

  - **Done when:** the review file's pre-registered threshold is ticked with the branch name and the
    numbers, not just the branch name.
- [ ] **2.3** Plot it.
  - One panel per pair. Ratio `‖r_t^(k)‖/‖eps_PoE‖` against denoising step 0 to 49, one curve per
    `k`, with the numerator and denominator on a second row so a moving denominator is visible.
  - Mark the last five steps as the read zone. Shade steps 0 to 10 as the zone where the two errors
    cannot be separated, labelled on the shading rather than in a legend.
  - Figure to
    `paper/iclr/figures/when-the-correction-arrives/mcmc/how-much-of-the-correction-a-corrector-removes.png`,
    with its `.json` sidecar recording which pairs, seeds and settings were drawn.
  - Add the entry to that folder's `README.md` naming the algorithm and what produced it.
  - **Done when:** the PNG, the sidecar and the README entry all exist.
  - 💡 `/analyze-figure` on the result before it goes near a caption: this figure has to survive a
    reviewer reading it as evidence for the opposite conclusion.

▶ **Next: [instruction 4.1](#4--judge-the-curve-by-eye-against-the-printed-verdict)**, the eye
read the code cannot do.

### 3. 📝 Fold the size back into what already cites it

◀ **Needs: [instruction 4.3](#4--judge-the-curve-by-eye-against-the-printed-verdict)**, so the
number being folded is one you have looked at.

- [ ] **3.1** Fold the size into
      [the idea map's claim 2](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md)
      and delete that map's route row, per its own rule that a route row is deleted once the result
      is folded back in.
  - **Done when:** claim 2 carries the number and the routes table no longer has the
    `/frame-hypothesis` row.
- [ ] **3.2** Update
      [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md)
      with the measured size, since that plan currently cites Soiffer et al. for a claim this plan
      turns into a number.
  - **Done when:** that plan quotes the number and the branch, in place of the citation standing
    alone.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 4. 📊 Judge the curve by eye against the printed verdict

◀ **Needs: [tasks 2.2 and 2.3](#2--run-the-grid-and-plot-it)**, the printed branch and the figure.

- [ ] **4.1** Open
      `paper/iclr/figures/when-the-correction-arrives/mcmc/how-much-of-the-correction-a-corrector-removes.png`.
  - Check the `k=100` and `k=200` curves lie on top of each other.
  - ✅ If they do, the chain has equilibrated and the branch the code printed can be believed.
  - ❌ If they do not, the chain has not equilibrated whatever the threshold printed, and the honest
    answer is inconclusive. Record that, and go back to
    [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) with a wider search.
- [ ] **4.2** Check the second row of panels: did the numerator fall, or did the denominator rise?
  - Say which in the review file, in one sentence. The ratio alone cannot answer it, which is why
    both are plotted.
- [ ] **4.3** Look at the composing pair's panel.
  - Expected result: low at `k=0`, and it should not rise with `k`.
  - ❌ If its curve behaves like the failing pair's, the measurement is reading itself and the whole
    grid is void. Record that judgement even when it agrees with the printed verdict, since it is
    the one check the code cannot make.

▶ **Next: [task 3.1](#3--fold-the-size-back-into-what-already-cites-it)** if the verdict stands,
otherwise stop and write the inconclusive one.

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** steps 27 and 28 both wait on this branch, and the framing of
> every caption downstream changes with it. Nothing proceeds until the three-way threshold has fired
> and the branch is written down.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector

# The grid, outside Slurm because biggpu allows one job per user.
nohup bash scripts/mechanism_study/run_corrector_curve.sh > "$OUT/curve.log" 2>&1 &
pgrep -af 'corrector|run_corrector'          # squeue is blind to this

# Harvest: the file, then the verdict.
$PY -c "import json;d=json.load(open('$OUT/residual_curves.json'));print(len(d['rows']),'rows')"
# expect 2 pairs x 6 k-values x 50 steps = 600 rows
$PY scripts/corrector_residual_curve.py --verdict     # prints which of the three branches fired
```

**Pass criteria**

- `residual_curves.json` has its 600 rows.
- The verdict prints one of the three branches with the numbers it was judged against.
- Instruction 4 has recorded the eye read beside it, including the composing-pair judgement.

**Fail criteria (STOP)**

- Fewer than 600 rows: a level was skipped or a `k` value crashed and the loop swallowed it. Rerun
  the missing pair-and-`k` combinations; the script is resumable by design.
- The composing pair behaves like the failing pair: the measurement is reading itself, the grid is
  void, and no reading of either curve is licensed.

**Partial pass guidance**

- A null is a pass, not a failure. It selects which of three paragraphs section 7 carries, and all
  three are written in [Why this plan exists](#why-this-plan-exists). Under a null,
  [step 28](baseline-01-superdiff-at-this-repos-fifty-steps.md) and
  [step 29](baseline-02-three-rules-on-one-dose-axis.md) still run, because SuperDiff is a
  published rule a reviewer will ask about either way; they just become a baselines table rather
  than a diagnosis.

**When you get results, answer**
[the review file](../review/hypothesis-02-what-is-left-once-the-chain-settles.md).

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#not-run-what-this-plan-leaves-alone) ➡️

Every figure in this scope is held to
[the standard set in the scope's direction](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `mcmc/how-much-of-the-correction-a-corrector-removes.png` | — | y is the correction's size relative to the product-of-experts prediction, `‖r_t^(k)‖/‖eps_PoE‖`, unitless on a 0-to-1 scale; x is denoising step 0 to 49; one curve per `k ∈ {0,1,5,20,100,200}`; one panel per pair; a second row carrying the two norms separately | `scripts/corrector_residual_curve.py` | ⏳ | **Main text, if the check passes.** It is the dynamics of the correction under a corrector, which is the whole question. Sidecar `.json` beside it records the pairs, seeds and settings drawn. From `corrector/residual_curves.json` |

**Three sentences this figure's caption owes**, all argued for above and none droppable for space.
The axis is the correction along the `k`-corrected path rather than along one path. `eps_J` is
evaluated off-distribution, and that is a deliberate choice. The residual norm is a proxy for the
distributional gap rather than the gap itself.

### Organization workflow

1. Run the grid, which writes `residual_curves.json` under `/datasets`.
2. Draw the figure into `paper/iclr/figures/when-the-correction-arrives/mcmc/` with its `.json`
   sidecar.
3. Add the entry to that folder's `README.md` naming the algorithm and what produced it, per this
   repo's artifact rule.

## Not run: what this plan leaves alone

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Metropolis adjustment.** The vendored `AnnealedMALASampler` needs a scalar energy, and SDXL is a
score model with no energy head. That is why the convergence read here is the curve flattening in
`k` plus a displacement column rather than an acceptance rate, which would have been the cheaper
diagnostic.

**More than one seed on this curve.** This is a diagnostic, not a rate, and seeds cost the
whole grid again. If the two pairs disagree, that is the moment to add seeds, and the review file
records that the single seed was a deliberate choice rather than an oversight.

**Attribution in the early window.** By construction, not by omission. See
[Where the two errors can be told apart](#where-the-two-errors-can-be-told-apart-and-where-they-cannot).

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#not-run-what-this-plan-leaves-alone) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| this plan's verdict | the review file, then [the two literature checks before print](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md), which currently cite a paper for a claim this makes into a number |
| this plan's verdict | [step 27](hypothesis-03-does-the-corrector-compose-in-the-same-window.md), [step 28](baseline-01-superdiff-at-this-repos-fifty-steps.md) and [step 30](idea-01-feynman-kac-correctors-gated.md), all waiting on it |
| a figure lands in `mcmc/` | that folder's `README.md` gains an entry naming the algorithm and what produced it, per this repo's artifact rule |
| the size is measured | [the idea map's claim 2](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md), and its route row is deleted |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-02-what-is-left-once-the-chain-settles.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| `poe_repair/composers/poe_langevin.py` | the composer [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) builds, run here at six values of `k` |
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | the three-branch guided product-of-experts the corrector's drift uses, and the denominator this plan records |
| [scripts/correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) | where `‖r_t‖/‖eps_PoE‖` is computed today, at line 101, so the corrected version measures the same quantity the uncorrected curves did |
| [composition/reduce_reuse_recycle/anneal_samplers.py](../../../composition/reduce_reuse_recycle/anneal_samplers.py) | `AnnealedMALASampler`, read only to record why Metropolis adjustment is not available here |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 27, does the corrector compose in the same window](hypothesis-03-does-the-corrector-compose-in-the-same-window.md).
It waits on this plan only softly, since a flat curve here does not imply a flat compose rate. The
corrector can relocate the trajectory without shrinking `‖r_t‖`.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. Seeded with the three
failure modes this plan is most exposed to, so they are recognised rather than rediscovered.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 the residual ratio is flat across every `k`

**When it happens:** the chain never moved, because the step size is too small.
**What you see:** a curve that reads as a clean null.
**Why:** a corrector that does nothing and a corrector that was never on produce the same curve.
**How to fix:** read the displacement column against `MIN_CHAIN_DISPLACEMENT`, then widen the
step-size search upward at
[step 25 task 2.1](instrument-01-the-corrector-and-the-step-size-it-runs-at.md#2--fix-the-step-size-before-anything-is-read).

#### 🔴 the ratio rises monotonically with `k`

**When it happens:** the step size is too large and the chain is diverging, or the joint branch is
degrading off-distribution.
**What you see:** a curve going the wrong way, which is not one of the three branches.
**Why:** two different causes with the same signature.
**How to fix:** check the latent norm against 1.5× the uncorrected latent first; that is the
diverging case and the fix is a smaller `c`. Then check the composing-pair panel; if it rises too,
the measurement is reading itself and the grid is void.

#### 🟡 the grid finished but `residual_curves.json` has fewer than 600 rows

**When it happens:** a level was skipped, or a `k` value crashed and the loop swallowed it.
**What you see:** a plausible-looking file with missing pair-and-`k` combinations.
**Why:** the loop catches per-combination failures so one crash does not kill the whole grid.
**How to fix:** count rows per `(pair, k)` and rerun the missing combinations; the script is
resumable by design.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
