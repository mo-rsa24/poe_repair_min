# 🎚️ How much of the correction is the sampler's, and how much is the model's?

## Recommended prompt (after run completes)

After you finish this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against the global and project
catalogs, and adds new entries. New errors propagate to all affected plan files.

---

## What this document is

**The whole design, in one file, and the source the scope's plan files are authored from.** It is
not itself a plan and carries no live checkboxes: its eight task groups are steps 24 to 30 of the
[root running order](../../../MASTER_PLAN.md), one plan each, listed in
[this scope's MASTER_PLAN.md](../MASTER_PLAN.md). Read it whole before authoring or executing any
of them, then delete it once every task group has landed in a plan file.

| Step | Plan | What it does |
|------|------|-------------|
| 6 | [hypothesis-03: when-in-the-run-it-matters](../../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md) ◑ | Found the cliff at steps 0 to 10, which is also where a sampler artifact would live |
| 15 | [gate-01: two-literature-checks-before-print](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) ⚠️ | Already cites Soiffer et al. as the reason to train a fix rather than trust a sampler. This scope is the measurement behind that sentence |
| **24 to 30** | **this scope's seven plans** ⚠️ | **Size the sampler's share against the model's share, and compare three composition rules on one dose axis** |
| 21 | [writing-06: mechanism-and-limitations](../../07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md) ⚠️ | Cannot be written honestly until step 26 returns a size |

Design only. The questions it is judged against are in
[the pre-registered review questions](the-questions-pre-registered-against-it.md), which split into
one review file per plan.

---

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
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Not run: what this plan leaves alone](#not-run-what-this-plan-leaves-alone)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## What this asks, in one line

Run a Langevin corrector at every noise level so the latent settles into the distribution the
product-of-experts score actually describes, then measure what is left of the correction at that
settled point. The part that goes away when the chain settles belongs to the sampler. The part
still there at the end of the run belongs to the model.

---

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

**The correction, `r_t`.** The per-step gap `eps_J - eps_PoE` between what the joined prompt
predicts and what adding the two separate prompts predicts, both evaluated at the same latent.
Every result in this scope so far is about this quantity.

**Error A, the sampler's share.** Summing two diffused scores gives the score of the product of
the two diffused marginals, and that sequence of distributions is not the forward diffusion of
anything. Reverse diffusion is only valid on a sequence that is, so running it on the summed
score does not sample the product it was asked for. Markov-chain correction does not need a valid
forward process, only a target at each noise level, so with enough corrector steps it samples the
product exactly. Error A is the share a corrector can take away, and it goes to zero as noise
goes to zero.

**Error B, the model's share.** `p(cat)·p(dog)` is not `p(cat and dog)`. Multiplying two
distributions asks for one thing that is both animals, which is the chimera. No corrector touches
this, and it does not vanish at zero noise.

**The corrector count, `k`.** How many unadjusted Langevin steps run at each of the 50 noise
levels before the reverse step is taken. `k=0` is plain product-of-experts.

**The settled point, `x_t^(k)`.** The latent after those `k` Langevin steps at noise level `t`.
Both networks are evaluated there, so `r_t^(k) = eps_J(x_t^(k)) - eps_PoE(x_t^(k))`.

**The generalised dose axis.** For any composition rule `M` with a per-step prediction `eps_M`,
define `r_t^M = eps_J - eps_M` and inject `eps_M + λ·r_t^M`. At `λ=0` the rule runs alone, at
`λ=1` the prediction is `eps_J` exactly. Every rule then travels the same axis, so a `λ` column is
a matched comparison across rules rather than four unrelated pictures.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

The scope has a causal result and a timing result. Injecting `r_t` raises the compose rate with
dose while a norm-matched random control stays at the floor, and injecting it only into steps 0
to 10 composes 0.656 of 32 cells while steps 20 to 30 onward compose 0.000. Both are in
[the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md), which also records
that the correction is about 2.7 times larger late than early, so it works where it is smallest,
and that tripling the late dose changes almost nothing.

The threat is that the same picture is what a sampler artifact looks like. Error A is worst at
high noise, and high noise is the early steps, which is exactly the window that decides the
outcome. Nothing measured so far tells the two apart, because `r_t` is both of them added
together and no run in this repo has ever separated them.

Two papers bracket the answer before it is measured. Du et al. conclude in the abstract of
[arXiv 2302.11552](https://arxiv.org/abs/2302.11552) that "the sampler (not the model) is
responsible for this failure", which is the strong version of error A. Soiffer et al.,
[arXiv 2606.23920](https://arxiv.org/abs/2606.23920), show that no inference-time technique alone
produces the target distribution once the composed condition is out of distribution, and that
existing correctors reduce the gap without closing it. Read together they predict a partial
result, so this plan is designed to return a size rather than a verdict.

Du et al.'s sampler code is already vendored here, at
[composition/reduce_reuse_recycle/anneal_samplers.py](../../../composition/reduce_reuse_recycle/anneal_samplers.py).
`AnnealedULASampler` is 25 lines and is the reference the new composer follows.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The cache cannot be used.**

The corrector moves `x_t`. Every cached trajectory in `outputs/interaction_term/` and
`cache_analyses/` is for a path the corrector is no longer on, so nothing here reads them except
the one free bound in task 1, which reads the uncorrected path on purpose.

**Two corrector counts are two trajectories, not one point wiggled twice.**

`k=5` and `k=20` diverge at the first noise level and never meet again. Any axis label saying
"the correction at step t" is wrong here. It is the correction along the `k`-corrected path, and
the caption says so.

**The joint model is evaluated where it would never go.**

`eps_J(x_t^(k))` asks the joint-prompt branch to predict noise at a latent produced by a chain
targeting the product of two marginals. That is a legitimate probe and an off-distribution
evaluation at the same time. It also gives a third reason the curve could rise with `k`, on top
of a step size that is too large: the joint branch could simply be degrading as the chain walks
away from anywhere it has seen. The pair that composes by default is the control that separates
those, and [the section below](#where-the-two-errors-can-be-told-apart-and-where-they-cannot)
says how.

**A ratio with a moving denominator is not a measurement.**

The cached size measure is `‖r_t‖/‖eps_PoE‖`, and the corrector moves both terms. Record the
numerator and the denominator as separate columns and plot the ratio on top of them, so a fall in
the ratio caused by a rising denominator is visible rather than hidden.

**Flat is ambiguous until the chain is shown to have moved.**

A curve flat from `k=1` reads as "the corrector does nothing, it is all error B", and it reads
identically when the step size is so small the chain never left where it started. Every point
carries the median relative displacement `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖` beside it, and a flat
curve with a displacement under the floor is a failed instrument, not a null.

**The trend is the result, and only once it plateaus.**

The value at any single `k` is a statement about the compute budget. A curve still falling between
the two largest `k` has not equilibrated and licenses no reading at all, which is why the grid
runs to `k=200` rather than stopping at 100.

**`k=1` against `k=2` shows nothing.**

Langevin needs many steps, and the step size matters as much as the count. The step-size search is
task 3 and it happens before any curve is read, not after one looks wrong.

---

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Claim.** A Langevin corrector run to equilibrium at each noise level removes part of `r_t` and
leaves a part that does not go away. What it leaves at the low-noise end of the run is the model's
error, `p(cat)·p(dog) ≠ p(cat and dog)`.

**Independent variable.** The corrector count `k ∈ {0, 1, 5, 20, 100, 200}` at a step size fixed
by task 3, and nothing else. Same pair, same seed, same 50 DDIM steps, same guidance 7.5.

**Dependent variables.** Per denoising step `t` and per `k`: the numerator
`‖eps_J(x_t^(k)) - eps_PoE(x_t^(k))‖₂`, the denominator `‖eps_PoE(x_t^(k))‖₂`, their ratio, and
the relative chain displacement. All in fp32, upcast from the fp16 the models run in.

**Confounds, and what each is answered by.**

| What else could produce a falling curve | What answers it |
|---|---|
| the denominator grew rather than the numerator shrinking | both are recorded and plotted separately (task 4) |
| the chain diverged and the latents are garbage | latent norm bound in the step-size search, and the composing-pair arm (task 3, task 4) |
| the joint branch degrades off-distribution, so any rise is the probe not the pair | the composing-pair arm, where the corrector's target is close to the joint and the curve should stay low at all `k` |
| the corrector never moved | the displacement column, with a floor in source |
| the chain has not equilibrated at the largest `k` | the `k=100` against `k=200` comparison, with an instability bound in source |

**Falsify condition, three-way, at the pre-registered bars.** The bars live in source as constants
in `scripts/corrector_residual_curve.py`, following the `MIN_MEDIAN_RATIO` pattern, so they cannot
be moved after the answer is visible.

- **Support, the split is real.** Over the last five denoising steps, the ratio at the plateaued
  `k` has fallen by at least `MIN_DROP_FOR_SPLIT = 0.20` of its `k=0` value. At least
  `MIN_REMAINDER_FOR_SPLIT = 0.20` of it is still there. Something went away and something stayed.
  The size of what stayed is the paper's number for error B.
- **Null, it is all the model.** At every step the change between `k=0` and the plateaued `k` is
  under `MAX_DRIFT_FOR_NULL = 0.05`, while the median displacement is at or above
  `MIN_CHAIN_DISPLACEMENT = 0.05`. The corrector ran, moved the latent, and changed nothing. Stop
  at task 4, write the null into section 7, and the method-comparison half of this plan becomes a
  baselines table rather than a diagnosis.
- **Inconclusive.** The ratio still moves by more than `MAX_K_INSTABILITY = 0.05` between `k=100`
  and `k=200`, or the displacement is below its floor, or the composing-pair arm shows the same
  behaviour as the failing pair. Then the step size is wrong or the probe is measuring itself.
  Return to task 3 and widen the search. Never loosen a bar.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

The paper's argument is that a small, shared, learnable correction fixes composition, and that a
rank-8 adapter can carry it. If most of `r_t` is error A, a training-free corrector gets most of
the same benefit and the adapter is answering a question the sampler had already solved. That is
the strongest reviewer objection available against this work, and
[gate-01](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) already cites the paper that raises it.

The outcome changes the manuscript either way, which is what makes it worth the GPU time.

| What the result says | What changes |
|---|---|
| mostly error B | the framing strengthens. The correction is a model-level object, the adapter is the right instrument, and the sampler comparison becomes a two-sentence related-work note |
| an even split | the paper reports both sizes, and the adapter is justified by the share a corrector cannot reach. Section 7 carries the corrector as the honest alternative for the other share |
| mostly error A | the timing result is reread as a sampler artifact, the sampler comparison stops being optional and becomes a baseline the paper must beat, and section 7 carries it as a limitation rather than a footnote |

---

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
  |                                                  RIGHT-HAND END is error B
  +----------------------------------------------
     step 0                              step 49
     high noise                          low noise
     A and B mixed                       A is gone
```

The read is not the whole curve. It is the right-hand end, where error A has vanished by
construction and anything left is the model. The left-hand end is where the compose rate is
decided and where the two errors cannot be separated, which is the honest limit this plan returns
alongside its number.

---

## Where the two errors can be told apart, and where they cannot

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

This section is why the measurement is read at the end of the run rather than across it. Getting
it wrong would put a number in the paper that means something other than what the caption says.

**What the corrector converges to.** Langevin driven by the product-of-experts score at noise
level `t` has stationary distribution `q_t = p_t(cat)·p_t(dog)`, the product of the two diffused
marginals. That is the distribution the summed score is the exact score of. At `t = 0` the noise
is gone and `q_0 = p(cat)·p(dog)`, which is the product of experts the method asked for in the
first place. So a converged corrector delivers the product exactly, and that is the whole of
Du et al.'s claim.

**What is left at the end of the run is error B, cleanly.** At the last steps the latent is drawn
from `p(cat)·p(dog)`. The residual `eps_J - eps_PoE` is then the joint model disagreeing with the
product on the product's own support. There is no sampler error left to contaminate it, because
error A is defined to vanish there. This number is the paper's estimate of the model's share.

**What is left at the start of the run is still both.** At `t > 0` the corrector has settled the
latent into `q_t`, the product of the diffused marginals. The thing the reverse process would need
is the diffusion of the product, and those two differ precisely because noising and multiplying do
not commute. So the floor at high noise is the non-commutation gap plus the diffused model gap,
and no amount of `k` separates them. The early part of the curve is a size, not an attribution.

**Which lands on the timing result.** The compose-decisive window is steps 0 to 10, the high-noise
end, which is the region where this plan cannot attribute. That limit is stated in the caption and
in the review file rather than argued away. What the plan can say about the early window is
whether a corrector applied only there changes the compose rate, which is task 5 and is a
behavioural answer rather than an attribution.

**Why the residual norm is a proxy and not the gap itself.** The corrector does not change the
function `eps_J - eps_PoE`; it changes where that function is evaluated. A falling curve therefore
says the settled latents sit where the two networks agree better, which is evidence about the
distributional gap rather than a measurement of it. The caption owes that sentence.

**What the composing pair controls.** On `a_butterfly__x__a_flower_meadow`, which composes under
plain product-of-experts, `q_t` is already close to the joint and `eps_J` is evaluated somewhere
it has seen. Its curve should be low at `k=0` and should not rise with `k`. If it rises the same
way the failing pair does, the rise is the probe walking the joint branch off-distribution and no
reading of either curve is licensed. `an_elephant__x__a_penguin` is not used for this, because
whether it composes by default is
[an open question in another review file](../../04-does-the-fix-reach-unseen-pairs/review/instrument-01-the-clean-pair-pool.md).

---

## Description: what to build

⬅️ [Previous](#where-the-two-errors-can-be-told-apart-and-where-they-cannot) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

Three things, in two halves separated by a gate.

**The diagnostic half, tasks 1 to 5.** A Langevin corrector composer beside the existing ones, a
step-size search, and a curve that sizes error A against error B. Then, only if the curve falls, a
recreation of the sliding-window sweep with the corrector in place of the injected `r_t`.

**The comparison half, tasks 6 to 8.** SuperDiff wired at this repo's 50 steps, and two grids
putting product-of-experts, SuperDiff and the corrector on one generalised dose axis. Then a full
read of the Feynman-Kac paper, gated on whether the first two justify it.

The two halves ask different questions. The first sizes an error, the second compares methods. They
share a plan file because the second is only worth running under some outcomes of the first, and
the gate between them is written into the tasks rather than left to judgement.

The corrector follows
[poe_internal.py](../../../poe_repair/composers/poe_internal.py)'s per-step intervention
pattern and the vendored `AnnealedULASampler`. At noise level `t` the drift is the
product-of-experts score `s_t = -eps_PoE / sqrt(1 - ᾱ_t)`. The update is
`x ← x + δ_t·s_t + sqrt(2·δ_t)·z` with `z` standard normal, repeated `k` times. Then the ordinary
DDIM step is taken. The step size is `δ_t = c·β_t`, following the vendored sampler's
`la_step_sizes = diffusion.betas * 0.035`, with `c` fixed by task 3. It takes a step window so the
corrector can be on inside a range and off elsewhere, matching the `r_t` window sweep's interface.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

It sizes the sampler's share of the correction against the model's share, which is the one number
the paper's framing rests on and has never been measured here. It also gives the manuscript a
baseline row for two published composition rules on the same axis as its own.

**Goal**

A number with a stated limit: the share of `r_t` still present at the low-noise end of the run once
the Markov chain has equilibrated. Measured on one failing pair and one composing pair, with the
whole `k` grid shown so the trend is visible rather than asserted. Plus, if the gate passes, two grids
comparing three composition rules along one dose axis.

---

## Environment Facts This Plan Depends On

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Everything writes under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`.
  `/home-mscluster` hit 100% once and silently killed checkpointing, so the job script carries a
  disk guard on `/datasets`, the filesystem it actually writes to, per
  [environment/storage.md](../../../environment/storage.md).
- biggpu allows one job per user, so the `k` grid runs under `nohup` outside Slurm and `squeue` is
  blind to it. Harvest by `pgrep -af 'corrector|sweep'` on the session node.
- The models run in fp16. Every norm upcasts to fp32 before it is taken, since the differences
  measured here are small enough that fp16 accumulation shows up in the third digit.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so figure PDFs come from matplotlib and never from a `pdflatex` pass.

---

## What it costs

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

The arithmetic, so a smoke run can check it rather than a guess standing in for it. Guided
product-of-experts is 3 UNet evaluations per call (prompt A, prompt B, unconditional) and the
joint branch is 2 (joint, unconditional). One noise level at corrector count `k` costs `3k + 5`
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

Two pairs at one seed is about 670 plain-render equivalents, and the step-size search adds roughly
110 more. No VAE decode and no detector anywhere in task 4, so wall time tracks UNet evaluations
directly. Task 1 costs nothing at all: it reads files already on disk.

---

## Tasks

⬅️ [Previous](#what-it-costs) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Re-read [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md)
      and confirm the three numbers this plan quotes are still what it says. They are 0.656 at
      steps 0 to 10, 0.000 from steps 20 to 30 onward, and the correction 2.7 times larger late
      than early.
      If any has moved, the threat this plan answers has changed shape and the claim needs
      rewriting before anything runs.
- [ ] **0.2** Confirm nothing else in the tree already measures a corrected residual.
      `grep -rn "langevin\|corrector" plans/ scripts/ poe_repair/`. The vendored Du et al. code
      under `composition/reduce_reuse_recycle/` is expected and is a reference, not a collision.

▶ **Next: task 1.1**, the read that costs no GPU.

### 1. 🧮 The free bound on the model's share

Run group: **hypothesis**. Costs nothing, reads files already on disk, and can change the design
of everything after it. Error A vanishes at zero noise and error B does not, so a correction size
still bounded away from zero at the last denoising step is already a floor under error B, without
a single new sample.

- [ ] **1.1** Read the per-step correction size as the run approaches zero noise, from
      `paper/iclr/figures/correction-size-over-the-denoising-run.json` (50 steps, 3 seeds, two
      pairs) and `cache_analyses/step_collapse.json` beside `snr_collapse.json` across pairs, whose
      `peak_at_edge` field speaks to high-noise concentration.
- [ ] **1.2** State three things in the review file or the read is not worth having. First, the
      cached measure is the ratio `‖r_t‖/‖eps_PoE‖` and this question needs the unnormalised
      numerator, so recover it or say it is unrecoverable. Second, which trajectory the residuals
      were cached along, since at λ=0 that is the plain product-of-experts path and the answer is
      about that path only. Third, whether both arms were evaluated at the same latent at each
      step, because if they were not, the late-step values carry accumulated path difference as
      well as rule difference and cannot be read at all.
- [ ] **1.3** Write the floor into the review file as a number with its unit and its meaning, and
      say what it does to the claim. A large floor means error B is already big and the `k` grid is
      a sizing exercise. A floor near zero, honestly measured, is the first evidence for error A
      and raises the stakes on everything below.

▶ **Next: task 2.1**. Task 1's answer does not block the build, only the reading of task 4.

### 2. 🔧 The corrector, and proof it changes nothing when switched off

Run group: **hypothesis**, instrument half.

◀ **Needs: task 0.2**, so the build is not duplicating something already here.

- [ ] **2.1** Write `poe_repair/composers/poe_langevin.py`, beside
      [poe.py](../../../poe_repair/composers/poe.py), following
      [poe_internal.py](../../../poe_repair/composers/poe_internal.py)'s per-step
      intervention pattern and the vendored `AnnealedULASampler`. Arguments: `k`, the step-size
      multiplier `c`, and a `corrector_window` tuple so the corrector can be on inside a range and
      off elsewhere. Method name format follows `poe_internal.py`'s: `poe_langevin_k<NNN>_c<NNN>`,
      plus `_w<start>-<end>` when a window is set.
- [ ] **2.2** The leak check, before any measurement. `k=0` must reproduce plain
      product-of-experts byte-identical, through the same `--check-identity` pattern
      [interaction_term_window.py](../../../scripts/interaction_term_window.py) uses at its
      line 71. Compare against a `k=0` run of the new composer rather than against `run_cfg_poe`,
      for the same reason the window plan gives: only the corrector logic may differ, not the
      batch shape.
- [ ] **2.3** A second leak check the window sweep did not need: with a corrector window placed
      past the last step, `k=200` must also be byte-identical to plain product-of-experts. That
      catches a corrector that runs outside its window, which the first check cannot see.

▶ **Next: task 3.1**, which fixes the one free parameter before any curve is read.

### 3. 📏 The step size, fixed before anything is read

Run group: **hypothesis**, instrument half. This is a task, not a footnote: at the wrong step size
the chain either never moves or diverges, and both of those imitate a scientific answer.

◀ **Needs: task 2.2**, so a moving chain can be distinguished from a broken composer.

- [ ] **3.1** Sweep the multiplier `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` on
      `a_cat__x__a_dog` seed 9, where `δ_t = c·β_t`. The vendored sampler's 0.035 at `k=20` is the
      centre of the range, not an assumption.
- [ ] **3.2** Record per `c`: the median relative displacement
      `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖` over the 50 levels, the maximum latent norm as a multiple of
      the uncorrected latent's norm, and whether the residual ratio rises monotonically with `k`.
- [ ] **3.3** Pick the largest `c` that neither stalls (displacement below
      `MIN_CHAIN_DISPLACEMENT`) nor diverges (latent norm past 1.5× the uncorrected latent, or a
      monotone rise in the ratio with `k`). Write the picked value and the whole search table into
      the review file. A search where every `c` fails is a finding and stops the plan here.

▶ **Next: task 4.1**, the gate.

### 4. 📊 The gate: what is left once the chain settles

Run group: **hypothesis**. This is the plan's load-bearing measurement and its bars are already
written in [The claim](#the-claim). No images, no detector, no scorer.

◀ **Needs: task 3.3**, the fixed step size.

- [ ] **4.1** Write `scripts/corrector_residual_curve.py` with the five bars as module-level
      constants: `MIN_DROP_FOR_SPLIT = 0.20`, `MIN_REMAINDER_FOR_SPLIT = 0.20`,
      `MAX_DRIFT_FOR_NULL = 0.05`, `MIN_CHAIN_DISPLACEMENT = 0.05`, `MAX_K_INSTABILITY = 0.05`.
      They sit in source so moving one after seeing the answer shows up in a diff.
- [ ] **4.2** Smoke in-session: one pair, one seed, `k ∈ {0, 1}` only, and check the measured
      wall time per noise level against [What it costs](#what-it-costs). A cost estimate that is
      out by more than 2× means the branch count is wrong and the grid is re-planned before it
      launches.
- [ ] **4.3** The grid, under `nohup` outside Slurm: `k ∈ {0, 1, 5, 20, 100, 200}` over all 50
      steps, on `a_cat__x__a_dog` seed 9 and on `a_butterfly__x__a_flower_meadow` seed 9. Write per
      `(pair, seed, k, t)`: numerator, denominator, ratio, relative displacement, latent norm. All
      fp32, upcast from fp16. Output to
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/residual_curves.json`.
- [ ] **4.4** Plot it. One panel per pair, ratio against denoising step 0 to 49, one curve per `k`,
      with the numerator and denominator on a second row so a moving denominator is visible. Mark
      the last five steps as the read zone and shade steps 0 to 10 as the zone where the two errors
      cannot be separated. Figure to
      `paper/iclr/figures/when-the-correction-arrives/mcmc/how-much-of-the-correction-a-corrector-removes.png`
      with its `.json` sidecar.
- [ ] **4.5** Apply the three-way bar from [The claim](#the-claim) in code, print which branch
      fired, and write the verdict into the review file with the numbers it was judged against.

▶ **Next: the engagement gate**, then either task 5 or a stop.

### 5. 🖼️ Only if the curve falls: the same window sweep with the corrector

Run group: **hypothesis**. The matched comparison against the `r_t` window sweep: same pair, same
seeds, same nine positions, corrector in place of injected `r_t`.

◀ **Needs: task 4.5** returning support or an even split, and the engagement gate below.

- [ ] **5.1** Recreate the sliding-window grid with the corrector at the `k` on the flat part of
      task 4's curve. `a_cat__x__a_dog` seeds 9 to 12 as rows, the same nine ten-step window
      positions as columns, green border on detector-composed cells, plus a tenth column with the
      corrector on for all 50 steps. Figure to
      `paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`.
- [ ] **5.2** Put it beside
      `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png`
      and record in the review file whether the corrector's compose rate peaks in the same window
      the injected correction does. Same window is a strong result: two different mechanisms
      acting at the same moment. A different window is a stronger one and needs its own paragraph.

**The gate on this task is soft, deliberately.** A flat curve in task 4 does not strictly imply no
change in compose rate, because the corrector could relocate the trajectory without shrinking
`‖r_t‖`. If task 4 comes back flat and this task still composes, that combination is the finding
and it goes in the review file as such. Run task 5 anyway in that case, and say in the review file
that it was run against a flat gate.

▶ **Next: task 6.1**, the comparison half.

### 6. 🔌 SuperDiff at this repo's settings

Run group: **baseline**. A published composition rule, run so its numbers are comparable to this
repo's, and nothing more.

◀ **Needs: task 4.5**, because a null there turns this half into a baselines table rather than a
diagnosis, and the framing of every caption changes with it.

- [ ] **6.1** Wire the pipeline from
      [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0), source at
      [necludov/super-diffusion](https://github.com/necludov/super-diffusion), into
      `poe_repair/composers/superdiff.py`.
- [ ] **6.2** Match it to SDXL base, DDIM, 50 steps, guidance 7.5. Its default is 200 steps, so
      this is a task with its own check: render the same pair and seed at 200 and at 50 steps and
      record whether the 50-step render still composes. If matching the step count breaks the
      method, the comparison is between a working rule and a crippled one, and that has to be said
      rather than discovered in the grid.
- [ ] **6.3** Expose `eps_M` per step so `r_t^SD = eps_J - eps_M` can be formed, which is what the
      generalised dose axis needs.

▶ **Next: task 7.1**.

### 7. 🎛️ Three rules on one dose axis

Run group: **baseline**.

◀ **Needs: tasks 6.3 and 2.1**, so every row can produce a per-step prediction.

- [ ] **7.1** Implement the generalised injection: for rule `M`, inject `eps_M + λ·r_t^M` where
      `r_t^M = eps_J - eps_M`. Four rows: product-of-experts with `r_t`, SuperDiff with `r_t^SD`,
      corrector at `k=1` with `r_t^(1)`, corrector at `k=5` with `r_t^(5)`.
- [ ] **7.2** Grid one: columns are seeds 9 to 12, `λ` fixed at 0.75, the value
      [F5](../../../paper/iclr/figures/F5-one-dial-three-instruments.png) already reads at.
      Render the same grid at `λ=0` in the same pass, which gives the methods-alone figure for
      free.
- [ ] **7.3** Grid two: columns are `λ ∈ {0, 0.25, 0.5, 0.75, 1}`, seed fixed at 9.
- [ ] **7.4** Handle the two things this grid is known to hit rather than discovering them. The
      rows deliver different absolute amounts at the same `λ`, because `‖r_t^M‖` differs per row.
      [The timing plan](../../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md) met this and answered it with
      a `--mode matched` arm, so either put the delivered total on each row label or run matched.
      And the corrector rows will not reproduce the joint render at `λ=1`, because the chain has
      already moved off the joint trajectory; label those cells not-an-identity. A row that fails
      to converge at `λ=1` is doing something beyond combining scores, which makes the `λ=1` column
      a free classifier of what each rule is.
- [ ] **7.5** Figures to `paper/iclr/figures/how-much-is-added/across-composition-rules/`, not to
      the timing folder. These are dose figures and the timing folder's name is a question about
      timing. Add a `README.md` entry per figure saying which rule produced it.

▶ **Next: task 8.1**, gated.

### 8. 📖 Feynman-Kac correctors, gated

Run group: **idea**. Blocked on a full read because no usable implementation was found.

◀ **Needs: tasks 4.5 and 7.4**. Run this only if the corrector arm moved something. If the gate
returned a null, the Feynman-Kac read is a related-work paragraph and this task closes unrun.

- [ ] **8.1** Promote [arXiv 2503.02819](https://arxiv.org/abs/2503.02819) from its abstract-level
      row in [the reading register](../../standing/literature/reading-register.md) to a full
      read, and record what would have to be implemented and at what cost.
- [ ] **8.2** Decide in the review file whether it is built or cited, and say which, with the
      reason.

### Close out. 🔄 Record what this plan taught

- [ ] **C.1** Answer every pre-registered question in
      [the review file](the-questions-pre-registered-against-it.md), including the
      ones whose answer is "not run and why".
- [ ] **C.2** Fold the size into
      [the idea map's claim 2](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md)
      and delete that map's `/frame-hypothesis` route row, per its own rule that a route row is
      deleted once the result is folded back in.
- [ ] **C.3** Update
      [gate-01](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) with the measured size, since it
      currently cites Soiffer et al. for a claim this plan turns into a number.

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 9. 🖱️ Read the step-size search by eye before the grid launches

◀ **Needs: task 3.2**, the search table.

- [ ] **9.1** Open the search table Claude writes into the review file. For each `c`, look at the
      three columns: median displacement, maximum latent norm as a multiple of the uncorrected
      latent, and whether the ratio rose with `k`.
- [ ] **9.2** Confirm the picked `c` sits in the middle of the usable range rather than at its
      edge. A pick at the smallest or largest tested `c` means the range was wrong, and the fix is
      to extend the sweep, not to accept the edge.
- [ ] **9.3** Write your verdict in one line into the review file's step-size section: the picked
      `c`, and whether the range was adequate. This is the decision the grid's whole cost rides on.

▶ **Next: task 4.3** launches once you have signed off here.

### 10. 📊 Judge the gate curve by eye against the printed verdict

◀ **Needs: task 4.4**, the figure, and task 4.5, the printed branch.

- [ ] **10.1** Open
      `paper/iclr/figures/when-the-correction-arrives/mcmc/how-much-of-the-correction-a-corrector-removes.png`.
      Check the `k=100` and `k=200` curves lie on top of each other. If they do not, the chain has
      not equilibrated whatever the bar printed, and the honest answer is inconclusive.
- [ ] **10.2** Check the second row: did the numerator fall, or did the denominator rise? Say which
      in the review file, in one sentence.
- [ ] **10.3** Look at the composing pair's panel. If its curve behaves like the failing pair's,
      the probe is measuring itself and the whole grid is void. Record that judgement even when it
      agrees with the printed verdict, since it is the one check the code cannot make.

▶ **Next: instruction 11.1** if the gate passed, otherwise stop and write the null.

### 11. 🖱️ Judge the two dose grids by eye

◀ **Needs: tasks 7.2 and 7.3**.

- [ ] **11.1** For grid two, walk the `λ=1` column across all four rows. The two
      product-of-experts-family rows should land on the joint render. The corrector rows should
      not, and that is expected. Record for each row whether it converged, since that column is a
      free classifier of what each rule is doing.
- [ ] **11.2** For grid one, count composed cells per row by eye and compare against the detector's
      count. [The timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md) records
      that the detector and the eye disagree on cat and dog often enough that the eye read is the
      one cited, so do both and cite the eye where they differ.
- [ ] **11.3** Write both counts into the review file, side by side, with the disagreements named
      cell by cell.

▶ **Next: task C.1**, closing the review file.

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

Between task 4 and task 5, nothing proceeds until the three-way bar has fired and the branch is
written down.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector

# The leak checks. Both must pass before any measurement is believed.
$PY scripts/corrector_residual_curve.py --check-identity --pair a_cat__x__a_dog --seed 9
$PY scripts/corrector_residual_curve.py --check-identity --window off --k 200 \
    --pair a_cat__x__a_dog --seed 9

# The step-size search, in-session, before the grid.
$PY scripts/corrector_residual_curve.py --step-size-search --k 20 --pair a_cat__x__a_dog --seed 9

# The grid, outside Slurm because biggpu allows one job per user.
nohup bash scripts/mechanism_study/run_corrector_curve.sh > "$OUT/curve.log" 2>&1 &
pgrep -af 'corrector|run_corrector'          # squeue is blind to this

# Harvest: the file, then the verdict.
$PY -c "import json;d=json.load(open('$OUT/residual_curves.json'));print(len(d['rows']),'rows')"
# expect 2 pairs x 6 k-values x 50 steps = 600 rows
$PY scripts/corrector_residual_curve.py --verdict     # prints which of the three branches fired
```

**Done when** `residual_curves.json` has its 600 rows, the verdict has printed one of the three
branches with the numbers it was judged against, and instruction 10 has recorded the eye read
beside it.

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#not-run-what-this-plan-leaves-alone) ➡️

The rule this plan is held to: a measurement earns a main-text slot only if it advances
understanding of the dynamics of failure and correction. Everything else is prose.

| Figure | What is on it | Main text? |
|---|---|---|
| `mcmc/how-much-of-the-correction-a-corrector-removes.png` | y is `‖r_t^(k)‖/‖eps_PoE‖`, x is denoising step 0 to 49, one curve per `k ∈ {0,1,5,20,100,200}`, one panel per pair, second row carrying the two norms separately. From `corrector/residual_curves.json` | **yes**, if the gate passes. It is the dynamics of the correction under a corrector, which is the whole question |
| `mcmc/samples-as-a-ten-step-corrector-window-slides.png` | rows are seeds 9 to 12, columns are the nine window positions plus a corrector-on-all-50 column, cells are the final picture, green border where the detector scored composed | **yes**, if it differs from the injected-`r_t` version. Identical behaviour is one sentence of prose beside the existing figure |
| `how-much-is-added/across-composition-rules/rules-at-one-dose.png` | rows are the four rules, columns are seeds 9 to 12 at `λ=0.75`, plus the same grid at `λ=0` | no. Supplementary: it varies the seed, which the paper already establishes elsewhere |
| `how-much-is-added/across-composition-rules/rules-as-the-dose-rises.png` | rows are the four rules, columns are `λ ∈ {0,0.25,0.5,0.75,1}` at seed 9, `λ=1` cells labelled not-an-identity where the rule cannot reach it | **yes**, if the gate passes. The `λ=1` column separates rules that combine scores from rules that do something else |
| the step-size search | the five `c` values against displacement and latent norm | no. It belongs in the review file, since it measures the instrument rather than the phenomenon |

Every caption owes three sentences this plan has already argued for. The axis is the correction
along the `k`-corrected path rather than along one path. `eps_J` is evaluated off-distribution, and
that is a deliberate probe. The residual norm is a proxy for the distributional gap rather than the
gap itself.

---

## Not run: what this plan leaves alone

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Metropolis adjustment.** The vendored `AnnealedMALASampler` needs a scalar energy, and SDXL is a
score model with no energy head. That is why the convergence read here is a plateau in `k` plus a
displacement column rather than an acceptance rate, which would have been the cheaper diagnostic.

**Other samplers on the same rule.** DDIM against DDPM against Euler, testing whether the window
sits at fixed noise levels rather than fixed step numbers, is
[generalization-01](../../03-does-the-correction-cause-composition/plans/generalization-01-other-models-and-samplers.md). That plan varies the
integrator under one composition rule; this one varies the composition rule under one integrator.
Neither answers the other.

**More than one seed on the gate curve.** Task 4 is a diagnostic, not a rate, and seeds cost the
whole grid again. If the two pairs disagree, that is the moment to add seeds, and the review file
records that the single seed was a deliberate choice rather than an oversight.

**The corrector as a paper baseline.** If the gate returns a null, the corrector is not a method
this paper competes with and no compose-rate arm is owed. Tasks 6 and 7 still run, because
SuperDiff is a published rule a reviewer will ask about either way.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#not-run-what-this-plan-leaves-alone) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the gate's verdict | the review file, then [gate-01](../../03-does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md), which currently cites a paper for a claim this makes into a number |
| a figure lands in `mcmc/` or `superdiff/` | that folder's `README.md` gains an entry naming the algorithm and what produced it, per this repo's artifact rule |
| the full reads happen | [the reading register](../../standing/literature/reading-register.md) rows for 2302.11552, 2412.17762 and 2503.02819 move from abstract-level to full read. They are promotions of rows dated 2026-08-12, not first reads |
| the size is measured | [the idea map's claim 2](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md), and its route row is deleted |
| the plan's status | the scope [MASTER_PLAN.md](../../03-does-the-correction-cause-composition/MASTER_PLAN.md) table and the root running order, both by `sync-plan-tree` rather than by hand |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [composition/reduce_reuse_recycle/anneal_samplers.py](../../../composition/reduce_reuse_recycle/anneal_samplers.py) | `AnnealedULASampler` at line 218 is the 25-line reference the new composer follows, and `inf_sample.py` line 230 is where its step size `betas * 0.035` at `k=20` comes from |
| [poe_repair/composers/poe_internal.py](../../../poe_repair/composers/poe_internal.py) | the per-step intervention pattern, the method-name format, and the `correction_window` argument shape |
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | the three-branch guided product-of-experts the corrector's drift uses |
| [scripts/interaction_term_window.py](../../../scripts/interaction_term_window.py) | `check_identity` at line 71, the leak-check pattern task 2.2 reuses |
| [scripts/correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) | where `‖r_t‖/‖eps_PoE‖` is computed today, at line 101, which task 1 has to unpick |
| [scripts/interaction_term_dose_matched.py](../../../scripts/interaction_term_dose_matched.py) | the `--mode matched` arm task 7.4 reuses so rows deliver comparable totals |

---

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Task 1, the free bound. It costs no GPU, it reads files already on disk, and its answer changes how
much the rest of the plan is worth. If the correction is already large at the last denoising step,
error B has a floor before a single corrector step runs.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

Filled by `/ingest-error-pattern --from-run-log` after the plan runs. Seeded with the three failure
modes this plan is most exposed to, so they are recognised rather than rediscovered.

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| the residual ratio is flat across every `k` | the chain never moved: step size too small | the displacement column against `MIN_CHAIN_DISPLACEMENT` | widen the step-size search upward, task 3.1 |
| the ratio rises monotonically with `k` | step size too large and the chain is diverging, or the joint branch degrading off-distribution | the latent norm against 1.5× the uncorrected latent, then the composing-pair panel | the first is a smaller `c`; the second voids the probe and needs a different diagnostic |
| the grid finished but `residual_curves.json` has fewer than 600 rows | a level was skipped, or a `k` value crashed and the loop swallowed it | row count per `(pair, k)` | rerun the missing cells; the script is resumable by design |
