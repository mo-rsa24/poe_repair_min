# Is the gap the sampler's or the model's?

> **This scope's plans sit in the one running order, not in a sequence of their own.** They are
> steps 24 to 30 of the `## Running order` table in the
> [repo root MASTER_PLAN.md](../../MASTER_PLAN.md), interleaved with every other scope's plans.
> The `## Plans` list below is this folder's contents and its dependencies, never a list to work
> down top to bottom.

**Nothing here has been run.** Every plan in this scope is at zero, which is the reason it is its
own scope rather than one step inside a scope that is half finished.

## Table of contents

- [The overall claim](#the-overall-claim)
- [What is this plan](#what-is-this-plan)
- [Why this plan exists](#why-this-plan-exists)
- [What this scope actually does (visual)](#what-this-scope-actually-does-visual)
- [High-level overview](#high-level-overview)
- [Where the two errors can be told apart, and where they cannot](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
- [Purpose and goals](#purpose-and-goals)
- [Mission](#mission)
- [Objectives](#objectives)
- [Goals](#goals)
- [Expected Outcome](#expected-outcome)
- [Definition of Done](#definition-of-done)
- [The figure bar every plan here is held to](#the-figure-bar-every-plan-here-is-held-to)
- [Sub-Scopes](#sub-scopes)
- [Plans](#plans)
- [What must survive into the plan files](#what-must-survive-into-the-plan-files)
- [Environment Context](#environment-context)
- [Process Diagram](#process-diagram)
- [Glossary](#glossary)

## The overall claim

📋 [TOC](#table-of-contents) | [Next](#what-is-this-plan) ➡️

**A [Langevin corrector](/home-mscluster/mmolefe/goal-setting/learning/sampler-correctors-for-composition/plans/21-langevin-dynamics.md) run to equilibrium at each noise level removes part of the correction `r_t`
and leaves a part that does not go away, and what it leaves at the low-noise end of the run is the
model's error rather than the sampler's.** This scope returns that share as a number with its
limit attached, and puts three composition rules on one dose axis so the paper has a baseline row
rather than an assertion.

## What is this plan

⬅️ [Previous](#the-overall-claim) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

The correction the paper is built on, `r_t = eps_J - eps_PoE`, is two errors added together and
nothing in this repo has ever separated them. One of them is the sampler's: summing two diffused
scores gives the score of the product of two diffused marginals, and reverse diffusion is not
valid on that sequence. It goes to zero as the noise goes to zero and a Markov-chain corrector can
take it away. The other is the model's: `p(cat)·p(dog)` is not `p(cat and dog)`, no corrector
touches it, and it does not vanish at zero noise.

This scope builds the corrector, calibrates it, and measures what survives it. It then compares
plain product-of-experts, SuperDiff and the corrector along one generalised dose axis, so the
three rules are read against each other rather than as three unrelated pictures.

## Why this plan exists

⬅️ [Previous](#what-is-this-plan) | 📋 [TOC](#table-of-contents) | [Next](#what-this-scope-actually-does-visual) ➡️

**The paper's timing result and a sampler artifact look identical.**

[The timing plan](../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md)
found that injecting the correction into steps 0 to 10 composes 0.656 of 32 cells while steps 20
to 30 onward compose 0.000. The sampler's share of the error is worst at high noise, and high
noise is the early steps. The picture that supports the paper's claim is also the picture a
sampler artifact would draw.

**It is the strongest objection a reviewer has available.**

If most of the correction is the sampler's, a training-free corrector gets most of the same
benefit and the rank-8 adapter is answering a question the sampler had already solved. Du et al.
conclude in the abstract of [arXiv 2302.11552](https://arxiv.org/abs/2302.11552) that "the sampler
(not the model) is responsible for this failure". Soiffer et al.,
[arXiv 2606.23920](https://arxiv.org/abs/2606.23920), show that no inference-time technique alone
produces the target distribution once the composed condition is out of distribution, and that
existing correctors reduce the gap without closing it. Read together they predict a partial
result, which is why this scope is designed to return a size rather than a verdict.

**The outcome changes the manuscript whichever way it lands.**

| What the result says | What changes |
|---|---|
| mostly the model's share | the framing strengthens. The correction is a model-level object, the adapter is the right instrument, and the sampler comparison becomes a two-sentence related-work note |
| an even split | the paper reports both sizes, and the adapter is justified by the share a corrector cannot reach. Section 7 carries the corrector as the honest alternative for the other share |
| mostly the sampler's share | the timing result is reread as a sampler artifact, the sampler comparison stops being optional and becomes a baseline the paper must beat, and section 7 carries it as a limitation rather than a footnote |

## What this scope actually does (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#high-level-overview) ➡️

The measurement the whole scope turns on, drawn as it will look if the split is real:

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

The read is the right-hand end, where the sampler's share has vanished by construction and
anything left is the model's. The left-hand end is where the compose rate is decided and where the
two cannot be separated, which is the limit this scope returns beside its number.

## High-level overview

⬅️ [Previous](#what-this-scope-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Next](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) ➡️

Seven plans in two halves, separated by one gate.

**The diagnostic half, steps 24 to 27.** A free bound read off cached files, then a Langevin
corrector composer beside the existing ones, then the step size fixed before any curve is read,
then the curve that sizes one error against the other, then the window sweep rerun with the
corrector in place of the injected correction.

**The comparison half, steps 28 to 30.** SuperDiff wired at this repo's 50 steps, two grids
putting three composition rules on one dose axis, and a full read of the Feynman-Kac paper gated
on whether the first two justify it.

The two halves ask different questions. The first sizes an error, the second compares methods.
They are one scope because the second is only worth running under some outcomes of the first, and
the gate between them is written into the plans rather than left to judgement.

The corrector follows
[poe_internal.py](../../poe_repair/composers/poe_internal.py)'s per-step intervention pattern and
the vendored `AnnealedULASampler` at
[anneal_samplers.py](../../composition/reduce_reuse_recycle/anneal_samplers.py). At noise level
`t` the drift is the product-of-experts score `s_t = -eps_PoE / sqrt(1 - ᾱ_t)`. The update is
`x ← x + δ_t·s_t + sqrt(2·δ_t)·z` with `z` standard normal, repeated `k` times, and then the
ordinary DDIM step is taken. The step size is `δ_t = c·β_t`, following the vendored sampler's
`la_step_sizes = diffusion.betas * 0.035`, with `c` fixed by step 25.

## Where the two errors can be told apart, and where they cannot

⬅️ [Previous](#high-level-overview) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goals) ➡️

This section is why the measurement is read at the end of the run rather than across it. Getting
it wrong would put a number in the paper that means something other than what the caption says. It
belongs to step 26, the gate, and to that plan's review file, and it is stated rather than argued
away.

**What the corrector converges to.** Langevin driven by the product-of-experts score at noise
level `t` has stationary distribution `q_t = p_t(cat)·p_t(dog)`, the product of the two diffused
marginals. That is the distribution the summed score is the exact score of. At `t = 0` the noise
is gone and `q_0 = p(cat)·p(dog)`, which is the product of experts the method asked for in the
first place. A converged corrector therefore delivers the product exactly, and that is the whole
of Du et al.'s claim.

**What is left at the end of the run is the model's share, cleanly.** At the last steps the latent
is drawn from `p(cat)·p(dog)`. The residual `eps_J - eps_PoE` is then the joint model disagreeing
with the product on the product's own support. No sampler error is left to contaminate it, because
that error is defined to vanish there. This number is the paper's estimate of the model's share.

**What is left at the start of the run is still both.** At `t > 0` the corrector has settled the
latent into `q_t`, the product of the diffused marginals. What the reverse process needs is the
diffusion of the product, and those two differ precisely because noising and multiplying do not
commute. So the floor at high noise is the non-commutation gap plus the diffused model gap, and no
amount of `k` separates them. The early part of the curve is a size, not an attribution.

**Which lands on the timing result.** The compose-decisive window is steps 0 to 10, the high-noise
end, which is the region where this scope cannot attribute. That limit goes in the caption and in
the review file. What the scope can say about the early window is whether a corrector applied only
there changes the compose rate, which is step 27 and is a behavioural answer rather than an
attribution.

**Why the residual norm is a proxy and not the gap itself.** The corrector does not change the
function `eps_J - eps_PoE`; it changes where that function is evaluated. A falling curve says the
settled latents sit where the two networks agree better, which is evidence about the distributional
gap rather than a measurement of it. The caption owes that sentence.

**What the composing pair controls.** On `a_butterfly__x__a_flower_meadow`, which composes under
plain product-of-experts, `q_t` is already close to the joint and `eps_J` is evaluated somewhere it
has seen. Its curve should be low at `k=0` and should not rise with `k`. If it rises the way the
failing pair does, the rise is the probe walking the joint branch off-distribution and no reading
of either curve is licensed. `an_elephant__x__a_penguin` is not used for this, because whether it
composes by default is
[an open question in another review file](../04-does-the-fix-reach-unseen-pairs/review/instrument-01-the-clean-pair-pool.md).

## Purpose and goals

⬅️ [Previous](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) | 📋 [TOC](#table-of-contents) | [Next](#mission) ➡️

**Purpose**

It serves the paper's central framing, which is that a small, shared, learnable correction fixes
composition and a rank-8 adapter can carry it. That framing rests on the correction being a
model-level object. Nothing has tested it.
[gate-01](../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md)
already cites Soiffer et al. for a claim this scope turns into a number, and
[writing-06](../07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md)
cannot be written honestly until that number exists.

**Goals** are numbered under [Goals](#goals) below.

## Mission

⬅️ [Previous](#purpose-and-goals) | 📋 [TOC](#table-of-contents) | [Next](#objectives) ➡️

Size the sampler's share of the correction against the model's share, on the one part of the run
where the two can be told apart, and say plainly where they cannot. Then put plain
product-of-experts, SuperDiff and a Langevin corrector on one dose axis, so the paper's own rule
is read against two published ones instead of standing alone.

## Objectives

⬅️ [Previous](#mission) | 📋 [TOC](#table-of-contents) | [Next](#goals) ➡️

1. Bound the model's share for free, off cached files, before any corrector exists, so everything
   downstream is capped before it is paid for.
2. Build a Langevin corrector composer that is proven inert when switched off, and fix its one
   free parameter, the step size, before any curve is read.
3. Measure the correction's size per denoising step against corrector count `k`, on one failing
   pair and one composing pair, and apply a three-way bar that was written in source before the
   answer was visible.
4. Rerun the nine-position window sweep with the corrector in place of the injected correction, so
   the two mechanisms can be compared at the same moments of the run.
5. Wire SuperDiff at this repo's settings and check it still works there, then compare three rules
   along one generalised dose axis.
6. Decide whether Feynman-Kac correctors are built or cited, with the reason recorded either way.

## Goals

⬅️ [Previous](#objectives) | 📋 [TOC](#table-of-contents) | [Next](#expected-outcome) ➡️

1. **The free bound.** The correction's size at the last denoising step, on the uncorrected path,
   read from cached files, stated with its unit, its trajectory, and whether the unnormalised
   numerator was recoverable. A size bounded away from zero there is a floor under the model's
   share before a single corrector step runs.
2. **The corrector is inert when off.** `k=0` reproduces plain product-of-experts byte-identical,
   and so does `k=200` with the corrector window placed past the last step.
3. **The step size is fixed and the pick is defended.** The largest `c` that neither stalls
   (median relative displacement below `MIN_CHAIN_DISPLACEMENT`) nor diverges (latent norm past
   1.5× the uncorrected latent, or a ratio rising monotonically with `k`), with the whole search
   table recorded and a human verdict on whether the range was adequate.
4. **The gate fires one of three branches.** Support if, over the last five denoising steps, the
   ratio at the plateaued `k` has fallen by at least `MIN_DROP_FOR_SPLIT = 0.20` of its `k=0`
   value with at least `MIN_REMAINDER_FOR_SPLIT = 0.20` still there. Null if the change is under
   `MAX_DRIFT_FOR_NULL = 0.05` at every step while the median displacement is at or above
   `MIN_CHAIN_DISPLACEMENT = 0.05`. Inconclusive if `k=100` and `k=200` still differ by more than
   `MAX_K_INSTABILITY = 0.05`, or the displacement is below its floor, or the composing pair
   behaves like the failing pair. Every bar sits in source, so moving one after the answer is
   visible shows up in a diff.
5. **The timing comparison is recorded either way.** Whether the corrector's compose rate peaks in
   the same window the injected correction does. Same window is a strong result, two different
   mechanisms acting at the same moment. A different window is stronger and gets its own
   paragraph.
6. **SuperDiff is comparable or is said not to be.** It renders at SDXL base, DDIM, 50 steps,
   guidance 7.5, and the 200-step against 50-step check records whether the 50-step render still
   composes. If it does not, every comparison against it is between a working rule and a crippled
   one, and that sentence goes in the caption.
7. **The three rules travel one axis.** Two grids on the generalised dose axis, with the delivered
   absolute amount per row made visible or matched, and the cells no rule can reach at `λ=1`
   labelled rather than left to look like failures.

## Expected Outcome

⬅️ [Previous](#goals) | 📋 [TOC](#table-of-contents) | [Next](#definition-of-done) ➡️

A number the paper can print: the share of the correction still present at the low-noise end of the
run once the Markov chain has equilibrated, measured on one failing pair and one composing pair,
with the whole `k` grid shown so the trend is visible rather than asserted, and with the limit
stated that the compose-decisive window is the one place this measurement cannot attribute. Beside
it, two dose grids comparing three composition rules, and a decision on Feynman-Kac. A null is a
result here and closes the scope honestly rather than failing it.

## Definition of Done

⬅️ [Previous](#expected-outcome) | 📋 [TOC](#table-of-contents) | [Next](#the-figure-bar-every-plan-here-is-held-to) ➡️

1. The free bound is written into its review file as a number with its unit, the trajectory it was
   cached along, and one line on what it does to the claim.
2. Both leak checks pass, and the step-size search table is filled with a picked `c` and a
   recorded human verdict on whether the tested range was adequate.
3. `residual_curves.json` holds its 600 rows (2 pairs × 6 `k` values × 50 steps), the three-way bar
   has printed one branch with the numbers it was judged against, and the eye read of the `k=100`
   against `k=200` overlay is recorded beside it.
4. The gate plan and its review file both carry the sentence that steps 0 to 10 are where the
   compose rate is decided and where this measurement cannot attribute. It is never dropped for
   space and never argued away.
5. The corrector's window figure exists in
   `paper/iclr/figures/when-the-correction-arrives/mcmc/` beside the injected-correction version in
   `poe/`, and the same-window question is answered either way.
6. SuperDiff renders at 50 steps, the 200-against-50 parity check is recorded, and `eps_M` per step
   is exposed so `r_t^SD = eps_J - eps_M` can be formed.
7. Both dose grids exist in
   `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each with its `.json` sidecar
   and a `README.md` entry naming which rule produced it, and the `λ=1` column is classified per
   row.
8. Feynman-Kac is decided, built or cited, with the reason recorded, including when the answer is
   "not run and why".
9. Every plan's review file has every pre-registered question answered, including the ones whose
   answer is "not run and why".
10. Every figure this scope produced passes
    [the figure bar](#the-figure-bar-every-plan-here-is-held-to) and carries a `.json` sidecar
    recording which cells, seeds and settings were drawn.
11. The measured size is folded into
    [the idea map's claim 2](../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md),
    its route row deleted, and
    [gate-01](../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md)
    updated, since it currently cites a paper for a claim this scope turns into a number.
12. The scope has a recall gallery: run `/recap-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md`
    once every plan above is ✅, and record the Artifact URL it publishes.

## The figure bar every plan here is held to

⬅️ [Previous](#definition-of-done) | 📋 [TOC](#table-of-contents) | [Next](#sub-scopes) ➡️

This scope exists as a scope partly so its figures are held to the same bar as the rest of the
paper's. Every Figure Catalog row in every plan below inherits this list, and a figure that fails
one of these is redrawn rather than captioned around.

**The picture explains and the words confirm.** If it needs a paragraph to make sense, the picture
is wrong and more words will not save it.

**One panel, one job.** Two things to notice means two panels.

**Axes named in plain words on the axis,** with the exact quantity in the caption.

**A label on the thing,** never a legend the reader has to look up.

**Every number carries its unit and its meaning.** Not "0.387 against 0.023", but what the quantity
is, on what scale, for which arm.

**Caveats and provenance live in the caption.** Moved off the face of the figure, never dropped. A
caveat hidden is a false claim by omission.

**A `.json` sidecar sits beside every figure** recording which cells, seeds and settings were
drawn, so a caption can be checked later without rerunning anything.

**A measurement earns a main-text slot only if it advances understanding of the dynamics of failure
and correction.** Anything else is prose.

## Sub-Scopes

⬅️ [Previous](#the-figure-bar-every-plan-here-is-held-to) | 📋 [TOC](#table-of-contents) | [Next](#plans) ➡️

None. If the Feynman-Kac read at step 30 comes back saying it should be built, that build is a
sub-scope and not a task, because it is a second corrector family with its own instrument.

## Plans

⬅️ [Previous](#sub-scopes) | 📋 [TOC](#table-of-contents) | [Next](#what-must-survive-into-the-plan-files) ➡️

The Step column is the position in the [root running order](../../MASTER_PLAN.md), which is the one
order. The Waits on column is the real dependency, which is what to read when deciding what can
start today.

| Step | Plan | Run kind | What it does | Waits on | Status |
|---|---|---|---|---|---|
| 24 | [the free bound on the model's share](plans/hypothesis-01-the-free-bound-on-the-models-share.md) | tests the claim | reads the correction's size as the run approaches zero noise off cached files, and turns it into a floor under the model's share. No GPU, no queue | nothing | ⚠️ not started |
| 25 | [the corrector, and the step size it runs at](plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md) | builds an instrument | builds `poe_repair/composers/poe_langevin.py`, proves it inert at `k=0` and with the window past the last step, and fixes the step-size multiplier `c` by search | nothing | ⚠️ not started |
| 26 | [what is left once the chain settles](plans/hypothesis-02-what-is-left-once-the-chain-settles.md) | tests the claim | the gate: ‖r_t^(k)‖ and ‖eps_PoE‖ per step against `k ∈ {0,1,5,20,100,200}`, two pairs, one seed, no images and no detector. Applies the three-way bar | 24, 25 | ⚠️ not started |
| 27 | [does the corrector compose in the same window](plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md) | tests the claim | the nine-position window sweep rerun with the corrector in place of the injected correction, matched to the existing figure | 26 | ⚠️ not started |
| 28 | [SuperDiff at this repo's fifty steps](plans/baseline-01-superdiff-at-this-repos-fifty-steps.md) | establishes a baseline | wires SuperDiff into `poe_repair/composers/superdiff.py`, matches it to 50 steps at guidance 7.5, and checks with its own 200-against-50 render whether matching broke it | 26 | ⚠️ not started |
| 29 | [three rules on one dose axis](plans/baseline-02-three-rules-on-one-dose-axis.md) | establishes a baseline | the generalised injection `eps_M + λ·r_t^M`, and the two dose grids that compare four rows along it | 25, 28 | ⚠️ not started |
| 30 | [Feynman-Kac correctors, gated](plans/idea-01-feynman-kac-correctors-gated.md) | explores | a full read of arXiv 2503.02819 and a built-or-cited decision, gated on whether the corrector arm moved anything | 26, 29 | ⚠️ not started |

Each plan carries its own `## Environment Facts This Plan Depends On` field, its own Figure Catalog
held to [the figure bar](#the-figure-bar-every-plan-here-is-held-to), and a paired file in `review/`
whose questions were written before any run. They were authored from
[the whole corrector design](source/the-whole-corrector-design.md), read alongside
[the questions pre-registered against it](source/the-questions-pre-registered-against-it.md); both
are deleted once nothing points at them.

**No procedure files.** `procedures/` does not exist here on purpose: every task is one do-able
unit, and the two manual arcs (reading the step-size search table at step 25, judging the gate curve
at step 26) are short enough to live in those plans' Instructions lanes rather than in files of
their own.

**What hand-authoring this scope skipped.** No `diagram-prompts.md` was seeded here, because the
scope this work came out of has none either, so no plan above carries an `## Illustrations` section
and every `## What happens (visual)` is an ASCII sketch. `/init-master-plan`'s map step is what
would fill it. This line is a finding until it is either resolved or deliberately closed.

## What must survive into the plan files

⬅️ [Previous](#plans) | 📋 [TOC](#table-of-contents) | [Next](#environment-context) ➡️

These are the parts of the design that took longest to get right and are easiest to lose in a
reshuffle. Each one names the plan that owns it, so `/populate-plans` cannot drop it quietly.

| What must survive | Which plan owns it |
|---|---|
| [Where the two errors can be told apart, and where they cannot](#where-the-two-errors-can-be-told-apart-and-where-they-cannot), in full, including that the compose-decisive window is exactly where attribution is impossible | step 26, in the plan and again in its review file |
| The value at any single `k` is a statement about compute budget, not about the problem. Only the trend across `k` is a result, and only once it plateaus, which is why the grid runs to `k=200` rather than stopping at 100 | step 26, as a bar and as a caption sentence |
| The `a_butterfly__x__a_flower_meadow` control, and why `an_elephant__x__a_penguin` is not used for it | steps 26 and 27 |
| Two corrector counts are two trajectories, not one point wiggled twice, so no axis may be labelled "the correction at step t" | steps 26 and 27, in every caption |
| A ratio with a moving denominator is not a measurement: numerator and denominator are separate recorded columns, plotted beneath the ratio | step 26 |
| A flat curve with displacement below its floor is a failed instrument, not a null | steps 25 and 26 |
| The rows deliver different absolute amounts at the same `λ`, because `‖r_t^M‖` differs per row. [The timing plan](../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md) met this and answered it with a `--mode matched` arm, so either put the delivered total on each row label or run matched | step 29 |
| The corrector rows will not reproduce the joint render at `λ=1`, because the chain has already left the joint trajectory. Those cells are labelled not-an-identity, which makes the `λ=1` column a free classifier of what each rule is | step 29 |
| The corrector's window figure files to `paper/iclr/figures/when-the-correction-arrives/mcmc/`, the four existing PNGs are already moved into `poe/`, and the dose grids file to `paper/iclr/figures/how-much-is-added/across-composition-rules/`, because the timing folder's name is a question about timing | steps 27 and 29 |

## Environment Context

⬅️ [Previous](#what-must-survive-into-the-plan-files) | 📋 [TOC](#table-of-contents) | [Next](#process-diagram) ➡️

See [environment/00-INDEX.md](../../environment/00-INDEX.md) for this project's
environment and architecture facts. Read before drafting or checking any plan in this scope.

The facts this scope's work rides on, which every plan below names in its own
`## Environment Facts This Plan Depends On` field:

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Everything writes under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`.
  `/home-mscluster` hit 100% once and silently killed checkpointing, so every job script carries a
  disk guard on `/datasets`, the filesystem it actually writes to, per
  [environment/storage.md](../../environment/storage.md).
- biggpu allows one job per user, so the `k` grid runs under `nohup` outside Slurm and `squeue` is
  blind to it. Harvest by `pgrep -af 'corrector|sweep'` on the session node.
- The models run in fp16. Every norm upcasts to fp32 before it is taken, since the differences
  measured here are small enough that fp16 accumulation shows up in the third digit.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so figure PDFs come from matplotlib and never from a `pdflatex` pass.

## Process Diagram

⬅️ [Previous](#environment-context) | 📋 [TOC](#table-of-contents) | [Next](#glossary) ➡️

How the seven plans gate each other. The free bound and the instrument start today and in parallel;
everything else waits on the gate.

```
  24 free bound  ────────┐
  (no GPU, caps          │
   everything below)     ▼
                      26 THE GATE  ──┬──▶ 27 corrector window figure
  25 corrector build ────┘           │
     + step size                     ├──▶ 28 SuperDiff at 50 steps
     (proven inert                   │         │
      before it is                   │         ▼
      believed)                      │    29 three rules, one dose axis
                                     │         │
                                     └─────────┴──▶ 30 Feynman-Kac: built or cited
                                                     (closes unrun if the gate was null)
```

The gate is the only hard stop. A null there does not close the scope: it turns the comparison half
from a diagnosis into a baselines table, and every caption downstream changes with it.

No rendered process image exists for this scope. Generating one needs a `diagram-prompts.md`, which
this scope does not have, per the note under [Plans](#plans).

## Glossary

⬅️ [Previous](#process-diagram) | 📋 [TOC](#table-of-contents)

Terms used only in this scope, one plain line each. The shared vocabulary (PoE, chimera, Mono, the
correction `r_t`, λ, seed against pair, cell) is in the [root MASTER_PLAN.md](../../MASTER_PLAN.md)
and is not repeated here.

- **The correction, `r_t`:** the per-step gap `eps_J - eps_PoE` between what the joined prompt
  predicts and what adding the two separate prompts predicts, both evaluated at the same latent.
- **The sampler's share:** summing two diffused scores gives the score of the product of the two
  diffused marginals, and that sequence of distributions is not the forward diffusion of anything.
  Reverse diffusion is only valid on a sequence that is. Markov-chain correction needs no valid
  forward process, only a target at each noise level, so enough corrector steps sample the product
  exactly. This is the share a corrector can take away, and it goes to zero as noise goes to zero.
- **The model's share:** `p(cat)·p(dog)` is not `p(cat and dog)`. Multiplying two distributions asks
  for one thing that is both animals, which is the chimera. No corrector touches this, and it does
  not vanish at zero noise.
- **The corrector count, `k`:** how many unadjusted Langevin steps run at each of the 50 noise
  levels before the reverse step is taken. `k=0` is plain product-of-experts.
- **The settled point, `x_t^(k)`:** the latent after those `k` Langevin steps at noise level `t`.
  Both networks are evaluated there, so `r_t^(k) = eps_J(x_t^(k)) - eps_PoE(x_t^(k))`.
- **The read zone:** the last five denoising steps, where the sampler's share has vanished by
  construction and anything left is the model's.
- **The generalised dose axis:** for any composition rule `M` with a per-step prediction `eps_M`,
  define `r_t^M = eps_J - eps_M` and inject `eps_M + λ·r_t^M`. At `λ=0` the rule runs alone, at
  `λ=1` the prediction is `eps_J` exactly. Every rule then travels the same axis, so a `λ` column
  is a matched comparison rather than four unrelated pictures.
- **`eps_PoE`, `eps_J`:** the model's raw per-step predictions, from plain product-of-experts and
  from the joined prompt. The correction is the difference between them, `r_t = eps_J - eps_PoE`,
  and `‖eps_PoE‖` is the denominator this scope reports the correction's size against.
- **Compose-rate:** the fraction of pictures showing two separate animals rather than one blended
  one, decided by the validated instance-count detector.
- **Norm-matched:** a comparison scaled to the same length as the real correction at every step, so
  a control that fails cannot have failed for being too weak. Only direction differs.
- **The step-size multiplier, `c`:** the one free parameter of the corrector, where
  `δ_t = c·β_t`. Fixed by search at step 25 before any curve is read, because at the wrong value
  the chain either never moves or diverges, and both imitate a scientific answer.
