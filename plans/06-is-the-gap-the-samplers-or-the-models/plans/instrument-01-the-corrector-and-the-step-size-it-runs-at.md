# 🔬 The corrector, and the step size it runs at

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md — <one line on which c was picked and why>
```

## Recommended skill

— custom; no skill fits. Writing a composer against this repo's own per-step intervention pattern
is project-specific work, and a skill that generalised it would fight the pattern rather than
follow it.

## Position in the plan tree

**Step 25 of 30.** Waits on nothing. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 24 | [hypothesis-01: the-free-bound-on-the-models-share](hypothesis-01-the-free-bound-on-the-models-share.md) ⚠️ | the free read that caps what this instrument is worth. Does not block this build |
| **25 (current)** | **instrument-01: the-corrector-and-the-step-size-it-runs-at** ⚠️ | **builds the Langevin corrector composer, proves it inert when switched off, and fixes the one free parameter before any curve is read** |
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](hypothesis-02-what-is-left-once-the-chain-settles.md) ⚠️ | the gate, which cannot be believed unless this plan passed |
| 29 | [baseline-02: three-rules-on-one-dose-axis](baseline-02-three-rules-on-one-dose-axis.md) ⚠️ | needs the composer built here to produce its two corrector rows |

Design only. Verdicts and run state live in
[the paired review file](../review/instrument-01-the-corrector-and-the-step-size-it-runs-at.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
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
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

Build a composer that runs `k` [Langevin corrector steps](/home-mscluster/mmolefe/goal-setting/learning/sampler-correctors-for-composition/plans/22-predictor-corrector-sampling.md) at each of the 50 noise levels before
taking the reverse step. Prove it produces byte-identical output to plain product-of-experts when
it is switched off. Then fix its step size by search, before any measurement is read off it.

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

**The corrector count, `k`.** How many unadjusted Langevin steps run at each of the 50 noise
levels before the reverse step is taken. `k=0` is plain product-of-experts.

**The step-size multiplier, `c`.** The corrector's one free parameter, where the Langevin step size
at noise level `t` is `δ_t = c·β_t`. The vendored sampler uses 0.035 at `k=20`, which is the centre
of the search range rather than an assumption inherited.

**The relative displacement.** `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖`, how far the chain actually moved at
noise level `t`. It is the difference between a corrector that does nothing and a corrector that
was never on.

**The corrector window.** A range of steps inside which the corrector acts and outside which it
does not, matching the interface the injected-correction window sweep already uses.

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** A new composer, `poe_repair/composers/poe_langevin.py`, sitting beside the existing
ones. At noise level `t` the drift is the product-of-experts score
`s_t = -eps_PoE / sqrt(1 - ᾱ_t)`. The update is `x ← x + δ_t·s_t + sqrt(2·δ_t)·z` with `z` standard
normal, repeated `k` times. Then the ordinary DDIM step is taken.

**What it does.** It lets the latent settle into the distribution the product-of-experts score
actually describes, before the reverse step moves on. Everything downstream in this scope is a
measurement taken on the path this composer produces.

**Key components.** The vendored `AnnealedULASampler` at
[anneal_samplers.py](../../../composition/reduce_reuse_recycle/anneal_samplers.py) line 218 is the
25-line reference. [poe_internal.py](../../../poe_repair/composers/poe_internal.py) is the per-step
intervention pattern, the method-name format, and the window-argument shape.
[poe.py](../../../poe_repair/composers/poe.py) is the three-branch guided product-of-experts the
drift uses.

**Testing approach.** Two leak checks and one search. The first check proves the composer is inert
at `k=0`. The second proves it is inert with the window placed past the last step even at `k=200`,
which catches a corrector running outside its window that the first check cannot see. The search
fixes `c` before any curve exists to be disappointed by.

**Associated materials.** [The review questions](../review/instrument-01-the-corrector-and-the-step-size-it-runs-at.md),
[the whole corrector design](../source/the-whole-corrector-design.md), and
[the scope's direction](../MASTER_PLAN.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected effort: the composer plus the search, roughly 110 plain-render equivalents of GPU.**

The two leak checks are one cell each. The search is five `c` values at `k=20` on one pair and one
seed, which at `3k + 5 = 65` UNet evaluations per noise level over 50 levels is about 22 plain
renders per value.

**The cache cannot be used from this plan onward.**

The corrector moves `x_t`. Every cached trajectory under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/` and its `cache_analyses/` is for a
path the corrector is no longer on. Only [step 24](hypothesis-01-the-free-bound-on-the-models-share.md)
reads them, and it does so on purpose because it is asking about the uncorrected path.

**At the wrong step size the chain either never moves or diverges, and both imitate a result.**

A curve flat from `k=1` reads as "the corrector does nothing" and reads identically when the step
size is so small the chain never left where it started. This is why the search is a task with its
own verdict rather than a footnote adjusted after a curve looks wrong.

**`k=1` against `k=2` shows nothing.**

Langevin needs many steps, and the step size matters as much as the count. Nothing here is judged
on a two-point comparison.

**Prerequisites.** None. This plan is a build and can start today.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A Langevin corrector composer exists, is proven to change nothing when switched off, and runs at
a step size picked by a recorded search rather than inherited from another codebase.**

**Independent variable.** The step-size multiplier `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` on
`a_cat__x__a_dog` seed 9, and nothing else.

**Dependent variables.** Per `c`: the median relative displacement over the 50 noise levels, the
maximum latent norm as a multiple of the uncorrected latent's norm, and whether the residual ratio
rises monotonically with `k`.

**Falsify condition, pre-registered.** The bars live in source as module-level constants, following
the `MIN_MEDIAN_RATIO` pattern this repo already uses, so they cannot be moved after the answer is
visible.

- **Pass.** Both leak checks are byte-identical, and at least one `c` neither stalls (median
  displacement below `MIN_CHAIN_DISPLACEMENT = 0.05`) nor diverges (maximum latent norm past 1.5×
  the uncorrected latent, or a ratio rising monotonically with `k`). The pick is the largest such
  `c`.
- **Fail, and it stops the scope here.** Either leak check differs by a single byte, which means
  the composer is not inert and every later comparison is against a moving baseline. Or every `c`
  in the range fails. That second one is a finding rather than a bug: it says this corrector cannot
  be run stably on this model at these settings, and it is written up as such.
- **Inconclusive.** The picked `c` sits at the smallest or the largest value tested, which means
  the range was wrong. The fix is to extend the sweep, never to accept the edge.

**Why this matters right now.** Steps 26, 27 and 29 all measure something on the path this composer
produces. A composer that leaks, or one running at a step size nobody defended, makes all three
uninterpretable in a way no later check would catch.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope's whole argument rests on comparing a corrected path against an
uncorrected one. If the composer changes anything when it is supposed to be off, the comparison is
between two unknowns.

**The approach.** Build the composer against the pattern the repo already uses for per-step
intervention. Then prove inertness twice: once with the corrector count at zero, and once with the
count at its maximum and the window placed where it cannot act. Then fix the step size by a search
whose table is recorded whether or not it is flattering.

**Key insights.**

1. Two leak checks catch different bugs. `k=0` catches a composer that computes something extra.
   The window-past-the-end check at `k=200` catches a corrector that ignores its own window, which
   the first check cannot see because at `k=0` there is nothing to ignore.
2. The step size is the one place a null result can be manufactured by accident, so it is fixed
   before any curve exists rather than tuned after one disappoints.
3. The search table is evidence about the instrument, not about the phenomenon, which is why it
   lives in the review file and not in a paper figure.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  one noise level t, corrector count k
  ┌──────────────────────────────────────────────────────────┐
  │  x_t ──▶ [ Langevin ] ──▶ [ Langevin ] ──▶ ... k times    │
  │           x += δ·s + √(2δ)·z                              │
  │           s = -eps_PoE / √(1-ᾱ_t),  δ = c·β_t             │
  │                            │                              │
  │                            ▼  x_t^(k), the settled point   │
  │                     [ ordinary DDIM step ]                 │
  └──────────────────────────────────────────────────────────┘
                              │
                              ▼  next noise level

  k = 0                  ──▶  must be byte-identical to plain PoE
  k = 200, window off    ──▶  must ALSO be byte-identical
  c too small            ──▶  displacement under the floor: chain never moved
  c too large            ──▶  latent norm past 1.5x: chain diverged
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`poe_repair/composers/poe_langevin.py`.** Beside
   [poe.py](../../../poe_repair/composers/poe.py), following
   [poe_internal.py](../../../poe_repair/composers/poe_internal.py)'s per-step intervention
   pattern and the vendored `AnnealedULASampler`. Arguments: `k`, the step-size multiplier `c`, and
   a `corrector_window` tuple so the corrector can be on inside a range and off elsewhere. Method
   name format follows `poe_internal.py`'s: `poe_langevin_k<NNN>_c<NNN>`, plus `_w<start>-<end>`
   when a window is set.
2. **The identity check path.** Reuse `check_identity` at
   [interaction_term_window.py](../../../scripts/interaction_term_window.py) line 71, so the leak
   check compares like against like: a `k=0` run of the new composer, not `run_cfg_poe`, for the
   same reason the window plan gives. Only the corrector logic may differ, not the batch shape.
3. **The step-size search.** A `--step-size-search` mode on
   `scripts/corrector_residual_curve.py`, carrying `MIN_CHAIN_DISPLACEMENT = 0.05` and the 1.5×
   latent-norm bound as module-level constants. It writes the five-row table the review file holds.

Output lands under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`, never under
`/home-mscluster`.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 2 of [the scope's direction](../MASTER_PLAN.md): build a Langevin corrector
composer that is proven inert when switched off, and fix its one free parameter before any curve is
read.

**Goals**

1. `poe_repair/composers/poe_langevin.py` exists and takes `k`, `c` and a corrector window.
2. Both leak checks report byte-identical output.
3. The five-row search table is filled in the review file, with a picked `c` and a recorded human
   verdict on whether the tested range was adequate.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Everything writes under `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`.
  `/home-mscluster` hit 100% once and silently killed checkpointing, so the search script carries a
  disk guard on `/datasets`, the filesystem it actually writes to, per
  [environment/storage.md](../../../environment/storage.md).
- **The cached trajectories cannot be used by this plan or anything after it.** The corrector moves
  the latent onto a different path, so every cached cell under `interaction_term/` describes a run
  this composer is not doing.
- The models run in fp16. Every norm upcasts to fp32 before it is taken, since the differences
  measured here are small enough that fp16 accumulation shows up in the third digit.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- The search runs in-session rather than under Slurm, because it is short. The rule that biggpu
  allows one job per user bites at [step 26](hypothesis-02-what-is-left-once-the-chain-settles.md),
  not here. Read [environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md)
  before launching anything onto a node this session is not on: a relative path in the launch line
  resolves against `$HOME` rather than the repo, and the failure is silent.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Confirm nothing else in the tree already measures a corrected residual.

    ```bash
    grep -rn "langevin\|corrector" plans/ scripts/ poe_repair/
    ```

  - The vendored Du et al. code under `composition/reduce_reuse_recycle/` is expected and is a
    reference, not a collision.
  - **Done when:** every hit outside that folder has been read and named as either unrelated or a
    collision, in one line in the review file.

▶ **Next: [task 1.1](#1--build-the-corrector-composer)**, the build.

### 1. 🔧 Build the corrector composer

◀ **Needs: [task 0.2](#0--preflight-check-this-plan-before-working-from-it)**, so the build is not
duplicating something already here.

- [ ] **1.1** Write `poe_repair/composers/poe_langevin.py`.
  - Follows [poe_internal.py](../../../poe_repair/composers/poe_internal.py)'s per-step
    intervention pattern and the vendored `AnnealedULASampler`.
  - Arguments: `k`, the step-size multiplier `c`, and a `corrector_window` tuple.
  - Method name format: `poe_langevin_k<NNN>_c<NNN>`, plus `_w<start>-<end>` when a window is set.
  - **Done when:** the module imports and a `k=0` render completes, producing a file under
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`.
- [ ] **1.2** The leak check, before any measurement.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_residual_curve.py --check-identity --pair a_cat__x__a_dog --seed 9
    ```

  - `k=0` must reproduce plain product-of-experts byte-identical, through the same
    `--check-identity` pattern
    [interaction_term_window.py](../../../scripts/interaction_term_window.py) uses at line 71.
    Compare against a `k=0` run of the new composer rather than against `run_cfg_poe`, so only the
    corrector logic differs and not the batch shape.
  - **Done when:** the command prints its identity-pass string. A single differing byte is a fail,
    not a rounding note.
- [ ] **1.3** The second leak check, which the window sweep never needed.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_residual_curve.py --check-identity --window off --k 200 \
        --pair a_cat__x__a_dog --seed 9
    ```

  - With a corrector window placed past the last step, `k=200` must also be byte-identical to plain
    product-of-experts. This catches a corrector that runs outside its window, which the first
    check cannot see.
  - **Done when:** the command prints its identity-pass string, and both results are recorded in
    the review file.

▶ **Next: [task 2.1](#2--fix-the-step-size-before-anything-is-read)**, which fixes the one free
parameter.

### 2. 📏 Fix the step size before anything is read

◀ **Needs: [task 1.2](#1--build-the-corrector-composer)**, so a moving chain can be told apart from
a broken composer.

- [ ] **2.1** Sweep the multiplier `c ∈ {0.01, 0.035, 0.1, 0.3, 1.0}` at `k=20` on
      `a_cat__x__a_dog` seed 9, where `δ_t = c·β_t`.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_residual_curve.py --step-size-search --k 20 \
        --pair a_cat__x__a_dog --seed 9
    ```

  - The vendored sampler's 0.035 at `k=20` is the centre of the range, not an assumption.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/step_size_search.json`
  - **Done when:** that file exists with five rows, one per `c`.
- [ ] **2.2** Record per `c`: the median relative displacement `‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖` over
      the 50 levels, the maximum latent norm as a multiple of the uncorrected latent's norm, and
      whether the residual ratio rises monotonically with `k`.
  - **Done when:** the five-row table in the review file is filled, every cell populated.
- [ ] **2.3** Pick the largest `c` that neither stalls (displacement below
      `MIN_CHAIN_DISPLACEMENT = 0.05`) nor diverges (latent norm past 1.5× the uncorrected latent,
      or a monotone rise in the ratio with `k`). Write the picked value and the whole search table
      into the review file.
  - A search where every `c` fails is a finding and stops the scope here, written up rather than
    worked around.
  - **Done when:** the review file names the picked `c` and holds the whole table, including the
    rows that failed.

▶ **Next: [instruction 3.1](#3--read-the-step-size-search-by-eye-before-the-grid-launches)**, your
sign-off on the pick. Step 26's whole cost rides on it.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the engagement gate](#the-engagement-gate).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 🖱️ Read the step-size search by eye before the grid launches

◀ **Needs: [task 2.2](#2--fix-the-step-size-before-anything-is-read)**, the search table.

- [ ] **3.1** Open the search table Claude writes into
      [the review file](../review/instrument-01-the-corrector-and-the-step-size-it-runs-at.md).
  - For each `c`, look at the three columns: median displacement, maximum latent norm as a multiple
    of the uncorrected latent, and whether the ratio rose with `k`.
  - Expected result: displacement rises with `c`, and the latent norm stays near 1.0 until some `c`
    where it runs away.
  - ✅ If the three columns tell that story, the instrument behaves and the pick is meaningful.
  - ❌ If displacement is flat across all five `c` values, the corrector is not being applied at
    all. That is a build bug, not a step-size finding. Go back to task 1.2.
- [ ] **3.2** Confirm the picked `c` sits in the middle of the usable range rather than at its edge.
  - A pick at the smallest or largest tested `c` means the range was wrong, and the fix is to
    extend the sweep, not to accept the edge.
  - ❌ If the pick is at an edge, say so and send task 2.1 back with a wider range.
- [ ] **3.3** Write your verdict in one line into the review file's step-size section: the picked
      `c`, and whether the range was adequate.
  - This is the decision the whole cost of step 26 rides on, and it is the one check the code
    cannot make.

▶ **Next: [step 26's task 1.1](hypothesis-02-what-is-left-once-the-chain-settles.md#1--write-the-gate-script-with-its-bars-in-source)**,
which launches only once you have signed off here.

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** a composer that leaks, or a step size nobody defended, makes
> steps 26, 27 and 29 uninterpretable in a way none of their own checks would catch.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector

# Both leak checks. Both must pass before any measurement is believed.
$PY scripts/corrector_residual_curve.py --check-identity --pair a_cat__x__a_dog --seed 9
$PY scripts/corrector_residual_curve.py --check-identity --window off --k 200 \
    --pair a_cat__x__a_dog --seed 9

# The search, in-session, before the grid.
$PY scripts/corrector_residual_curve.py --step-size-search --k 20 --pair a_cat__x__a_dog --seed 9
$PY -c "import json;d=json.load(open('$OUT/step_size_search.json'));print(len(d['rows']),'rows')"
# expect 5 rows, one per c
```

**Pass criteria**

- Both leak checks print byte-identical.
- `step_size_search.json` has its five rows.
- A `c` is picked, it is not at an edge of the tested range, and instruction 3.3 has recorded the
  human verdict beside it.

**Fail criteria (STOP)**

- Either leak check differs. The composer is not inert and nothing downstream may run.
- Every `c` fails both ways. Write it up as a finding: this corrector cannot be run stably here.

**Partial pass guidance**

- A pick at an edge is a partial pass. Extend the sweep and rerun rather than proceeding; a step
  size at the edge of the tested range is a step size nobody has bounded.

**When you get results, answer**
[the review file](../review/instrument-01-the-corrector-and-the-step-size-it-runs-at.md).

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The bar every figure in this scope is held to is
[in the scope's MASTER_PLAN](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| the step-size search table | — | five `c` values against median displacement, maximum latent norm as a multiple of the uncorrected latent, and whether the ratio rose with `k` | `scripts/corrector_residual_curve.py --step-size-search` | ⏳ | **Not a figure and never a paper figure.** It measures the instrument rather than the phenomenon, so it lives as a table in the review file |

### Organization workflow

The search table stays in the review file. Nothing from this plan is filed under
`paper/iclr/figures/`.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the picked `c` | [the gate plan](hypothesis-02-what-is-left-once-the-chain-settles.md) and [the dose-axis plan](baseline-02-three-rules-on-one-dose-axis.md), both of which run the composer at this value |
| a leak check fails | the whole scope stops; the scope [MASTER_PLAN.md](../MASTER_PLAN.md) records it and no downstream plan starts |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/instrument-01-the-corrector-and-the-step-size-it-runs-at.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [composition/reduce_reuse_recycle/anneal_samplers.py](../../../composition/reduce_reuse_recycle/anneal_samplers.py) | `AnnealedULASampler` at line 218 is the 25-line reference the new composer follows, and `inf_sample.py` line 230 is where its step size `betas * 0.035` at `k=20` comes from |
| [poe_repair/composers/poe_internal.py](../../../poe_repair/composers/poe_internal.py) | the per-step intervention pattern, the method-name format, and the `correction_window` argument shape |
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | the three-branch guided product-of-experts the corrector's drift uses |
| [scripts/interaction_term_window.py](../../../scripts/interaction_term_window.py) | `check_identity` at line 71, the leak-check pattern tasks 1.2 and 1.3 reuse |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 26, what is left once the chain settles](hypothesis-02-what-is-left-once-the-chain-settles.md).
It is the scope's load-bearing measurement, and it runs the composer built here at the `c` picked
here.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>3 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. Seeded with the three
failure modes this plan is most exposed to.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the leak check passes at `k=0` but the corrector ignores its window

**When it happens:** a window argument that is parsed but never consulted in the per-step branch.
**What you see:** the first leak check passes, and every later measurement is contaminated by a
corrector running where it was told not to.
**Why:** at `k=0` there is nothing to gate, so the first check cannot see a broken gate.
**How to fix:** task 1.3, the `k=200` check with the window past the last step, which is the only
thing that catches it.

#### 🟡 the displacement column is flat across every `c`

**When it happens:** the corrector is built but not actually applied to the latent.
**What you see:** a search table where all five rows look identical and every `c` reads as
"stalled".
**Why:** a stalled chain and an unapplied corrector produce the same numbers.
**How to fix:** confirm the update line runs by asserting the latent changed at one noise level
before trusting any row, then re-run task 2.1.

#### 🟡 a relative path in a launch line resolves against `$HOME`

**When it happens:** launching onto a node this session is not on.
**What you see:** a silent failure, catalogued as `poe-launch-001`.
**Why:** the launch line's working directory is not the repo.
**How to fix:** absolute paths in every launch line. See
[environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md).

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
