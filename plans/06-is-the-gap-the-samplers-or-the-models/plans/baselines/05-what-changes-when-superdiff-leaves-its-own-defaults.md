# 🔌 What changes when SuperDiff leaves its own defaults

Does SuperDiff still compose when it runs at this repo's usual 50 steps instead of its own default
of 200, and does clamping its blending weight `kappa` change that answer?

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md — <does it still compose at 50 steps>
```

## Recommended skill

▶ `/replicate https://github.com/necludov/super-diffusion` ✅ wiring a published method into this
   repo's interface, at this repo's settings, with a parity check against the authors' defaults, is
   what that skill is for.
   alt: `/learn-codebase` on the vendored source first if the pipeline's step-count handling is not
   obvious from a read.

## Position in the plan tree

**Step 28 of 30.** Waits on step 26. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](../hypothesis/03-what-is-left-once-the-chain-settles.md) ⚠️ | the answer that decides how this half is framed. A null there turns it into a baselines table rather than a diagnosis, and the framing of every caption changes with it |
| **28 (current)** | **baseline-01: what-changes-when-superdiff-leaves-its-own-defaults** ⚠️ | **wires a published composition rule into this repo, checks whether cutting its step count from 200 to 50 breaks it, and whether clamping its blending weight changes the answer** |
| 29 | [baseline-02: three-rules-on-one-amount-axis](06-three-rules-on-one-amount-axis.md) ⚠️ | needs the per-step prediction this plan exposes, and puts SuperDiff's row on this project's own shared `λ` axis |

Design only. Verdicts and run state live in
[the paired review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The check before moving on](#the-check-before-moving-on)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Wire SuperDiff into this repo as a composer, at SDXL base, guidance 7.5, and check two things at
once: whether cutting its step count from its own default of 200 down to this repo's usual 50
still composes, and whether clamping its per-step blending weight (`kappa`) changes that answer.
This pipeline has no DDIM mode to match against; it is a hand-written stochastic integrator with
no scheduler object at all, confirmed by reading its 531-line source. "DDIM" does not apply here
and does not appear anywhere else in this plan.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/composers/superdiff.py`, a third composer beside plain
product-of-experts and the Langevin corrector, wrapping `SuperDiffSDXLPipeline` from
[superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0) (`pipeline.py`, read in
full for this plan), source and citation at
[necludov/super-diffusion](https://github.com/necludov/super-diffusion).

> The Langevin corrector is an extra step slipped in between the sampler's own steps. It nudges
> the latent along the model's score and adds a little noise, so the state settles onto what the
> model says is likely instead of only following the sampler's path.

**What it does.** SuperDiff is a composition rule derived from the continuity equation rather than
from naive score addition. Skreta et al.,
[arXiv 2412.17762](https://arxiv.org/abs/2412.17762), ICLR 2025 Spotlight. It is a published
alternative to the rule this paper is about, and a reviewer will ask about it whichever way
[step 26](../hypothesis/03-what-is-left-once-the-chain-settles.md) came back.

**Key components.** The composer wrapper around `SuperDiffSDXLPipeline`, a clamp on its blending
weight `kappa` (the pipeline ships with none), and a hook exposing the per-step prediction `eps_M`
so that `r_t^SD = eps_J - eps_M` can be formed. That last one is what
[step 29](06-three-rules-on-one-amount-axis.md) needs and is easy to leave out.

**Testing approach.** A small grid, on this scope's own standard pair of pairs,
`a_cat__x__a_dog` (the hard case) and `a_butterfly__x__a_flower_meadow` (the easy case), both at
seed 9, matching how `residual_curves.json` already uses them elsewhere in this scope. Crossed on
step count (200, 50) and on the `kappa` clamp (on, off): 8 renders. If a step count or the clamp
changes whether either pair composes, every later comparison against SuperDiff needs that sentence
in its caption.

**Associated materials.** [The review questions](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md),
[the whole corrector design](../../source/the-whole-corrector-design.md), and the register row for
2412.17762 in [the reading register](../../../standing/literature/reading-register.md), at abstract
level since 2026-08-12.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This pipeline has no DDIM mode.** Reading `pipeline.py` in full: `_forward` is a hand-written
Euler-Maruyama integrator with no scheduler object, no `eta` parameter, and no branch for
deterministic sampling anywhere in the file. It injects fresh Gaussian noise
(`torch.empty_like(latents).normal_(generator=self.generator)`) at every step except the last
three, unconditionally. "Run it at DDIM" was never an option this pipeline offers, at any step
count, so this is not a design choice to fix, it is a word that does not apply here.

**The default is 200 steps and this repo runs 50.** `pipeline.py` sets `num_inference_steps=200`
as its own default, confirmed in both the source and the model card
([superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0)). Everything else in
this project runs at 50. That is still the axis worth measuring directly, unrelated to the DDIM
question above.

**`kappa` is unclamped in the shipped pipeline, and that is a real risk at 50 steps.** `kappa`'s
denominator is `((noise_pred_text_o - noise_pred_text_b)**2).sum(...)`, and nothing in the source
stops it from spiking when that term is small. At 50 steps, `dsigma` is larger per step than at
200, so a spike has fewer steps left to be absorbed. Whether clamping `kappa` to roughly
[−0.5, 1.5], holding it at 0.5 for the first ~10% of steps, actually changes the outcome is
exactly what the clamped/unclamped half of this plan's grid measures, rather than assumes.

**A baseline freezes on landing.**

Per this project's run conventions, a baseline may not change any claim. What it can do is give the
comparison something to be measured against, and that only works if the comparison is a fair one.

**The per-step prediction is the deliverable that is easiest to skip.**

Rendering a picture with SuperDiff is satisfying and insufficient. Step 29 needs `eps_M` at every
step, and a wrapper that only returns the final image cannot supply it.

**The cache cannot be used.** SuperDiff follows its own trajectory.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**SuperDiff runs in this repo at SDXL base and guidance 7.5, on `a_cat__x__a_dog` and
`a_butterfly__x__a_flower_meadow` (seed 9), crossed on step count (200, 50) and on a `kappa` clamp
(on, off), exposes its per-step prediction, and each of the 8 cells is recorded as composing or
not.**

**Independent variables.** Step count (200 against 50) and the `kappa` clamp (on against off),
crossed on 2 fixed pairs. Nothing else differs between cells.

**Dependent variable.** Whether each render composes, by the detector and by eye.

**Falsify condition.** The threshold sits on the 8-cell table, read for three things: whether step
count moves the composes verdict, whether the `kappa` clamp moves it, and whether the hard pair and
the easy pair disagree. How good SuperDiff's pictures are beyond composing is a separate question
this plan does not judge.

- **Pass.** Both pairs compose at 200 and at 50 steps, with the clamp on. SuperDiff is comparable
  and step 29 proceeds with it as an ordinary row.
- **Fail.** A pair that composes at 200 steps stops composing at 50, even with the clamp on. This
  does not stop the plan: it puts a sentence in every caption that compares against SuperDiff at
  this repo's step count. What is not acceptable is discovering this inside the grid across
  correction amounts, where it would read as SuperDiff being weak.
- **Ambiguous but informative.** The clamp changes the verdict at 50 steps (composes clamped,
  fails unclamped, or the reverse). That is itself worth a caption sentence: the failure mode was
  the estimator's unclamped `kappa`, not the step count on its own.
- **Inconclusive.** Neither pair composes at either step count. Then the pairs are wrong for this
  check, not the method. Try one more pair before recording anything.

**Why this matters right now.** It is a reviewer's first question about a paper proposing a
composition fix, and it is asked whether or not step 26 came back with a split.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The paper compares its own composition rule against nothing published. That is a
gap a reviewer closes for you, unfavourably.

**The approach.** Wire one published rule at this repo's own settings, and measure what two things
cost before any comparison is drawn: cutting its step count, and running it with the safety clamp
its own shipped code does not have.

**Key insights.**

1. This pipeline has no DDIM option, so matching this repo's sampler was never available as a
   choice. Step count is the one axis that is.
2. `kappa` ships unclamped, and a 50-step run is exactly the setting where an unclamped spike has
   the least room to recover. Testing clamped against unclamped separates a real step-count effect
   from an estimator artefact, rather than conflating them.
3. Exposing `eps_M` per step is what turns a picture into a row on the amount axis. Without it,
   SuperDiff can be shown but not compared.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  2 pairs x 2 step counts x 2 kappa settings = 8 renders

                    kappa clamped           kappa unclamped
              200 steps   50 steps    200 steps   50 steps
a_cat x        compose?    compose?    compose?    compose?
a_dog

butterfly x    compose?    compose?    compose?    compose?
flower_meadow

  read across each row: does cutting steps break it?
  read across clamped vs unclamped at 50 steps: was it the steps, or the estimator?
```

This plan also builds seven supplementary sheets of its own (group 4), all at 200 steps: for
each `kappa` setting (the pipeline's own, or forced to 0, 0.25, 0.5, 0.75, 1) a grid with seeds
down the rows and `λ` across the columns, `λ` being the fraction of `r_t^SD` added back. Step 29's
grids use the same `λ` axis to compare SuperDiff against the other three rules; these sheets
instead hold the rule fixed and show what its own blending weight does to that axis.

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`poe_repair/composers/superdiff.py`.** `SuperDiffSDXLPipeline` from
   [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0) wrapped in this
   repo's composer interface, so it is callable the same way
   [poe.py](../../../poe_repair/composers/poe.py) is.
2. **The `kappa` clamp.** Roughly [−0.5, 1.5], holding `kappa` at 0.5 for the first ~10% of steps,
   toggleable so the grid can render both settings. The shipped pipeline has no clamp anywhere.
3. **The 8-cell grid.** `a_cat__x__a_dog` and `a_butterfly__x__a_flower_meadow`, seed 9, at 200
   and 50 steps, clamped and unclamped.
4. **The per-step prediction hook.** `eps_M` exposed at each step so `r_t^SD = eps_J - eps_M` can
   be formed, which is what the generalised amount axis at
   [step 29](06-three-rules-on-one-amount-axis.md) needs.

Renders write under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 5 of [the scope's direction](../../MASTER_PLAN.md): wire SuperDiff at this repo's
settings and check it still works there, before three rules are compared along one amount axis.

**Goals**

1. `poe_repair/composers/superdiff.py` exists and renders at SDXL base, guidance 7.5, with a
   toggleable `kappa` clamp.
2. The 8-cell grid (2 pairs × 2 step counts × 2 clamp settings) is recorded, with a composes
   verdict per cell.
3. `eps_M` is available per step, verified by forming `r_t^SD` on one render and printing its
   per-step norm.
4. The seven κ × λ sheets exist at 200 steps (cat×dog at six `kappa` settings, butterfly×meadow
   at `kappa=0.5`; each seeds 9 to 12 by `λ ∈ {0, 0.25, 0.5, 0.75, 1}`), filed under
   `across-composition-rules/` with sidecars and README entries.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Renders and weights write under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`, with a disk
  guard on `/datasets`, per [environment/storage.md](../../../../environment/storage.md). The
  SuperDiff checkpoint (7.11GB on the Hub) is a model download and must not land on
  `/home-mscluster`.
- biggpu allows one job per user. Render 1 of the 8 is timed in-session before the rest are
  queued, since no wall-clock cost for this specific pipeline has been measured anywhere in this
  repo yet. If timing shows the full 8 will not fit in-session, the remaining renders move to the
  `nohup`-outside-Slurm path the rest of this scope already uses; read
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md)
  either way.
- **The cached trajectories cannot be used.** SuperDiff follows its own path.
- The models run in fp16, and anything normed upcasts to fp32 first.
- SDXL base 1.0 via `SuperDiffSDXLPipeline`, guidance 7.5, height/width 1024 (the pipeline's own
  recommended default), latents 4×128×128 at 1024². No scheduler or `eta` option exists on this
  pipeline; step count is set via its `num_inference_steps` argument only.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/, report/,
      and any learning journey that names this project, in case a term this plan mentions is
      already defined or explained somewhere else in the repo.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.
- [ ] **0.3** Confirm which way
      [step 26](../hypothesis/03-what-is-left-once-the-chain-settles.md) came out, and record it in
      this plan's review file.
  - A null there turns this half of the scope into a baselines table rather than a diagnosis, which
    changes the framing of every caption this plan and step 29 produce. The runs are the same
    either way; the sentences around them are not.

    > A null means the correction's size at the end of the run came out the same with the chain
    > running as without it.

  - **Done when:** the branch is quoted in the review file.

▶ **Next: [task 1.1](#1--wire-the-pipeline)**.

### 1. 🔌 Wire the pipeline

◀ **Needs: [task 0.3](#0--check-this-plan-before-working-from-it)**, so the captions are
framed correctly from the start.

- [x] **1.1** Wire `SuperDiffSDXLPipeline` into `poe_repair/composers/superdiff.py`.
  - Source: [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0)
    (`pipeline.py`), citation and background at
    [necludov/super-diffusion](https://github.com/necludov/super-diffusion).
  - Follow this repo's composer interface, the one
    [poe.py](../../../poe_repair/composers/poe.py) implements.
  - Time the first render (any one of the 8 cells) and record wall time in the review file, since
    this repo has no prior measurement for this pipeline.
  - **Done when:** one render completes, lands under
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`, and its wall
    time is recorded.
  - Reimplemented directly rather than via `trust_remote_code`, since exposing `eps_M` and
    clamping `kappa` both need the sampling loop's internals. Result and wall time in
    [the review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).
- [x] **1.2** Expose `eps_M` per step so `r_t^SD = eps_J - eps_M` can be formed.
  - **Done when:** `r_t^SD` is formed on one render and its per-step norm printed for all steps of
    that render, proving the hook returns a prediction and not a placeholder.
  - Result in [the review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).
- [x] **1.3** Implement the `kappa` clamp: roughly [−0.5, 1.5], holding `kappa` at 0.5 for the
      first ~10% of steps, as a toggle the grid render can switch on or off.
  - The shipped `pipeline.py` computes `kappa` with no clamp anywhere in the file (verified by
    reading the source in full for this plan). `dsigma` is larger per step at 50 steps than at
    200, so an early `kappa` spike has less room to recover from at the setting this project
    actually needs.
  - **Done when:** the clamp is active and verified on one step where `kappa` would otherwise
    exceed the range, with the pre-clamp and post-clamp values both printed.
  - Result in [the review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

▶ **Next: [task 2.1](#2--the-eight-cell-grid)**.

### 2. 🚀 The eight-cell grid

◀ **Needs: [task 1.3](#1--wire-the-pipeline)**, so there is something to render with and a clamp
to toggle.

- [x] **2.1** Render all 8 cells: `a_cat__x__a_dog` and `a_butterfly__x__a_flower_meadow` (seed 9,
      this scope's standard pair of pairs), at 200 and 50 steps, with the `kappa` clamp on and off.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/parity/`
  - **Done when:** all 8 renders exist, and the scored verdict for each is written beside it.
- [x] **2.2** Record the 8-cell table in the review file: pair, step count, clamp setting, detector
      verdict, eye verdict placeholder (filled by instruction 3.3).
  - Read for whether step count moves the verdict, whether the clamp moves it, and whether the two
    pairs disagree, per [the claim](#the-claim)'s falsify condition.
  - **Done when:** the review file's pre-registered question is ticked with all 8 detector
    verdicts entered.
  - Full table and verdict in
    [the review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

▶ **Next: [task 4.1](#4--the-kappa-by-lambda-sheets)**, since the detector's word on a published
method is not enough to put in a caption, and instruction 3.1 needs both this group and group 4
rendered first.

### 4. 📊 The kappa-by-lambda sheets

◀ **Needs: [task 1.3](#1--wire-the-pipeline)**, the `kappa` clamp toggle and the `eps_M` hook.
Independent of the eight-cell grid in group 2, reuses the same wiring.

- [x] **4.1** Render the κ × λ set at 200 steps. One figure per (pair, `kappa` setting); inside a
      figure, rows are seeds 9 to 12 and columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}`, the fraction
      of `r_t^SD = eps_J - eps_M` added back: inject `eps_M + λ·r_t^SD`. So `λ=0` is SuperDiff
      alone at that `kappa`, and `λ=1` is the joint prompt's prediction exactly.
  - `kappa` settings: `balanced` (the pipeline's own `kappa`, unclamped, its published
    behaviour) and forced to 0, 0.25, 0.5, 0.75, 1. cat×dog gets all six; butterfly×meadow gets
    `kappa=0.5` only. Seven figures, 140 renders.
  - This is step 29's generalised amount axis applied inside one rule, with that rule's own
    blending weight held fixed per figure. It reads differently from
    [step 29](06-three-rules-on-one-amount-axis.md)'s grids, which vary the rule and hold nothing
    else fixed.
  - Composer knobs: `superdiff.run(..., kappa_clamp=False, kappa_override=<kappa or None>, lam=<λ>)`.
    Output under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/lambda_sweep/`.
  - **Done when:** 140 renders exist, each with `r_t^SD`'s per-step norm in its sidecar.
- [x] **4.2** Draw the seven sheets: `cat_dog_grid_200_steps_kappa_{000,025,050,075,100,balanced}.png`
      and `butterfly_meadow_grid_200_steps_kappa_050.png`, each 4 seeds by 5 `λ`, labels drawn on
      the sheet, a missing tile drawn as a red box rather than skipped.
  - Save to `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each with a `.json`
    sidecar naming every tile's source render, plus a `README.md` entry per sheet.
  - **Done when:** all seven sheets, their sidecars, and the README entries exist with no red
    boxes.

▶ **Next: [instruction 3.1](#3--look-at-the-eight-renders-yourself)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.
- [ ] **Promote the register row.** Move
      [the reading register](../../../standing/literature/reading-register.md)'s row for 2412.17762
      from abstract level to full read, since wiring the method required reading it.
  - Done when: the row records the promotion, dated, as a promotion of the 2026-08-12 row rather
    than a first read.

▶ **Next: [the check before moving on](#the-check-before-moving-on).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Look at the eight renders yourself

◀ **Needs: [task 2.1](#2--the-eight-cell-grid)**, all 8 renders.

- [x] **3.1** Open all 8 renders, grouped by pair, so the 200-vs-50 and clamped-vs-unclamped
      comparisons sit side by side.
  - Expected result: 8 pictures, differing only in step count and clamp setting within each pair.
  - ✅ If a cell shows two separate things, mark it composes.
  - ❌ If a cell blends into one thing, mark it does not. It is a caption sentence from here on,
    not a footnote.
- [x] **3.2** If neither pair composes anywhere in the grid, try one more pair before writing
      anything down.
  - A published method failing on every cell of its own grid is more likely the pairs than the
    method, and recording it as a method failure would be unfair and wrong.
- [x] **3.3** Write your eye verdict for each of the 8 cells into the review file beside the
      detector's.
  - Where they disagree, the eye is the one cited, following the practice
    [the timing verdict](../../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
    set for this project.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 29](06-three-rules-on-one-amount-axis.md).

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** step 29 puts SuperDiff on a shared axis with this project's own
> rule. If it does not compose here, or the clamp changes whether it does, that has to be known
> before step 29 draws any conclusion from a row that used it silently unclamped or at the wrong
> step count.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SD=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff

ls -l "$SD/parity/"          # all 8 renders: 2 pairs x 2 step counts x 2 clamp settings
# the per-step prediction hook: one norm per step, not one number total
$PY -c "import json;d=json.load(open('$SD/parity/eps_m_norms.json'));print(len(d['steps']),'steps')"
```

**Pass criteria**

- `superdiff.py` renders at SDXL base, guidance 7.5, with a toggleable `kappa` clamp.
- All 8 grid cells exist and are scored.
- `eps_M` is available at every step of at least one render, verified by forming `r_t^SD`.
- Instruction 3.3 has recorded eye verdicts for all 8 cells beside the detector's.

**Fail criteria (STOP)**

- `eps_M` cannot be exposed per step. Step 29 cannot form the generalised amount axis for this row,
  and the plan is re-scoped to a picture-only comparison rather than proceeding as if the axis
  existed.

**Partial pass guidance**

- Any cell not composing is a partial pass, and it is recorded rather than worked around.
  SuperDiff still runs at step 29; every caption comparing against it carries whatever sentence
  the 8-cell table earned.

**When you get results, answer**
[the review file](../../review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| the eight-cell grid | — | 2 pairs × 2 step counts × 2 `kappa`-clamp settings, composes or not per cell | task 2.1 | ✅ | **Not a paper figure.** It measures whether SuperDiff was set up fairly here, not the phenomenon, so it stays in the review file. If any cell fails to compose, that render is worth keeping as the evidence behind the caption sentence, filed under `paper/iclr/figures/when-the-correction-arrives/superdiff/` with a `README.md` entry |
| `across-composition-rules/cat_dog_grid_200_steps_kappa_{000,025,050,075,100,balanced}.png` (six sheets) | — | one sheet per `kappa` setting at 200 steps; rows are seeds 9 to 12, columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}`, the fraction of `r_t^SD` added back. Each tile is the final render | task 4.2 | ✅ | **Supplementary.** Reads left to right as SuperDiff-alone becoming the joint prompt, with the rule's own blending weight held fixed per sheet. `balanced` is the pipeline's own unclamped `kappa` |
| `across-composition-rules/butterfly_meadow_grid_200_steps_kappa_050.png` | — | same layout, the easy pair at `kappa=0.5` only | task 4.2 | ✅ | **Supplementary.** One sheet, so the easy pair is seen on the same axis without paying for six |

Step 29's own two grids ([06-three-rules-on-one-amount-axis.md](06-three-rules-on-one-amount-axis.md))
already put SuperDiff's row on this project's shared `λ` axis, across seeds 9 to 12, using the
`eps_M` hook this plan exposes. No further figure is owed there from this plan.

### Organization workflow

1. Render all 8 grid pictures and all 140 κ × λ renders under `/datasets`.
2. Keep the 8-cell grid in the review file unless a cell failed to compose, in which case file that
   render under `paper/iclr/figures/when-the-correction-arrives/superdiff/` with its sidecar and
   README entry, because a caption sentence needs its evidence reachable.
3. Draw the seven κ × λ sheets into
   `paper/iclr/figures/how-much-is-added/across-composition-rules/`, each with its sidecar and
   README entry.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the 8-cell verdict | every caption at [step 29](06-three-rules-on-one-amount-axis.md) that compares against SuperDiff |
| the full read happens | [the reading register](../../../standing/literature/reading-register.md)'s row for 2412.17762 moves from abstract level to full read, as a promotion of the 2026-08-12 row |
| a figure lands in `superdiff/` | that folder's `README.md` gains an entry naming the algorithm and what produced it |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | this repo's composer interface, which `superdiff.py` implements so it is callable the same way |
| [necludov/super-diffusion](https://github.com/necludov/super-diffusion) | the authors' implementation and citation; no SDXL code lives here, only SD1.4 examples |
| [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0), `pipeline.py` | the actual SDXL pipeline this plan wires, read in full: confirms `num_inference_steps=200` default, `guidance_scale=7.5` default, no scheduler/`eta` option, and no clamp on `kappa` anywhere |
| [the reading register](../../../standing/literature/reading-register.md) | the row for 2412.17762, at abstract level since 2026-08-12, promoted by this plan's close out |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 29, three rules on one amount axis](06-three-rules-on-one-amount-axis.md). It needs the
per-step prediction this plan exposes, and it is where SuperDiff's row joins this project's own
shared `λ` axis and gets compared against the other three rules.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>4 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the wrapper returns an image but not a per-step prediction

**When it happens:** wrapping a published pipeline by its top-level call rather than its denoising
loop.
**What you see:** SuperDiff renders fine, and step 29 cannot form `r_t^SD` for its row.
**Why:** the generalised amount axis needs `eps_M` at every step, which a top-level wrapper discards.
**How to fix:** task 1.2 verifies the hook by printing per-step norms before anything else is built
on it.

#### 🟡 the checkpoint downloads to `/home-mscluster`

**When it happens:** a Hugging Face pipeline pulled without setting the cache directory.
**What you see:** a full home filesystem, which has already once silently killed checkpointing here
(`poe-disk-001`).
**Why:** the default cache is under `$HOME`.
**How to fix:** point the cache at `/datasets` before the first pull. See
[environment/storage.md](../../../../environment/storage.md).

#### 🟡 this pipeline has no DDIM mode

**When it happens:** assuming the composer wrapper can be told to sample deterministically, the way
the rest of this project's composers do.
**What you see:** no `eta`, no scheduler parameter, nothing to set. `pipeline.py`'s `_forward`
injects fresh Gaussian noise every step except the last three, unconditionally, with no code path
around it.
**Why:** it is a hand-written Euler-Maruyama integrator, not a wrapper around a diffusers scheduler.
**How to fix:** don't try. Wire it as shipped, and note in every caption that its sampling is not
controllable the way this project's other composers are.

#### 🟡 `kappa` can spike, and there is nothing built in to catch it

**When it happens:** any run, more likely at 50 steps where `dsigma` is larger per step than at
200.
**What you see:** asymmetric domination of one concept over the other in the rendered image.
**Why:** `kappa`'s denominator can go small, and the shipped `pipeline.py` has no clamp anywhere
near the `kappa` computation, confirmed by reading the source in full.
**How to fix:** task 1.3 adds the clamp; the clamped/unclamped grid in task 2.1 measures whether it
actually changes the outcome.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
