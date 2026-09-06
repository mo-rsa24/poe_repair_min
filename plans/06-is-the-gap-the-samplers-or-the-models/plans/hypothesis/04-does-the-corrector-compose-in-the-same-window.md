# 🧪 Does the corrector compose in the same window?

This plan asks whether a corrector switched on inside a ten-step window changes what comes out of
the run, and whether the window where it works is the same early window the injected correction
works in.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md — <same window or a different one>
```

## Recommended skill

▶ `/pair-figure` ✅ this figure only means anything beside the injected-correction version, and
   pairing two grids so the comparison is the thing the reader sees is exactly what that skill
   decides.
   alt: `/analyze-figure` on the existing `poe/samples-as-a-ten-step-window-slides.png` first, to
   fix the layout this one has to match.

## Position in the plan tree

**Step 27 of 30.** Waits on step 26. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 6 | [hypothesis-03: when-in-the-run-it-matters](../../../03-does-the-correction-cause-composition/plans/hypothesis/05-when-in-the-run-it-matters.md) ◑ | generated the injected-correction renders at all nine window positions, which this plan is matched against render for render |
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](03-what-is-left-once-the-chain-settles.md) ⚠️ | the measurement this plan waits on, though not strictly |
| **27 (current)** | **hypothesis-03: does-the-corrector-compose-in-the-same-window** ⚠️ | **generates the nine window positions again with the corrector in place of the injected correction, and asks whether the [compose rate](../../../../context/world/compose-rate.md) peaks at the same moment** |
| 29 | [baseline-02: three-rules-on-one-amount-axis](../baselines/06-three-rules-on-one-amount-axis.md) ⚠️ | the other half of the comparison, on how much is added rather than timing |

Design only. Verdicts and run state live in
[the paired review file](../../review/04-does-the-corrector-compose-in-the-same-window.md).

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
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Slide a ten-step corrector window across the run, at the same nine positions the injected-correction
renders used, and ask whether the [compose rate](../../../../context/world/compose-rate.md) peaks
in the same window that the injected correction does.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis.** A corrector switched on only inside a ten-step window changes the compose rate,
and the window where it works is the same early window the injected correction works in.

**If true, same window.** Two different mechanisms acting at the same moment of the run. That is a
strong result and it goes beside the existing figure.

**If true, different window.** Stronger still, and it needs its own paragraph, because two
mechanisms that both fix composition at different moments say something about the run that neither
says alone.

**If false, no window works.** The corrector relocates the trajectory without changing what comes
out, which bounds what a training-free corrector can do at this compute budget.

**Rationale.** [Step 26](03-what-is-left-once-the-chain-settles.md) measures a norm and
cannot attribute anything in the early window, which is where the compose rate is decided. This
plan asks a behavioural question about that same window instead, which is answerable there.

**Dataset details.** `a_cat__x__a_dog` seeds 9 to 12 as rows. Nine ten-step window positions as
columns, matched to the existing figure, plus a tenth column with the corrector on for all 50 steps.
Forty renders, each decoded and scored.

**Associated materials.** [The review questions](../../review/04-does-the-corrector-compose-in-the-same-window.md),
[the whole corrector design](../../source/the-whole-corrector-design.md), and the figure this one is
matched against, `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png`.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This plan waits on step 26, but not strictly.**

A flat curve at [step 26](03-what-is-left-once-the-chain-settles.md) does not strictly
imply no change in compose rate, because the corrector can relocate the trajectory without
shrinking [`‖r_t‖`](../../../../context/world/interaction-term.md#words-this-file-uses). If step 26 comes back flat and this plan still composes, that combination is the
finding and it goes in the review file as such. Run this plan anyway in that case, and record that
it was run against a flat curve at step 26.

**This plan decodes and scores, so it costs more than the number of renders suggests.**

Forty renders at 50 steps each, plus a VAE decode and a detector pass on each one. Unlike step 26,
wall time here does not track UNet evaluations alone.

**The comparison only works if the layout matches.**

Same pair, same seeds, same nine positions, same green-border rule for a render the detector scored
as composed. A figure that is nearly the same is worse than one that is obviously different,
because the reader will compare them anyway.

**The corrector runs at the `k` on the flat part of step 26's curve.**

Not at the largest `k`. The flat part is where the chain has equilibrated, and anything past it is
compute spent for nothing.

**The cache cannot be used.** The corrector moves the latent onto a different path.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A ten-step corrector window slid across the run produces a compose-rate profile that either does
or does not peak where the injected correction's does, and the answer is recorded either way.**

**Independent variable.** The position of the ten-step corrector window, nine positions across the
50 steps, plus an all-50 column. Seeds 9 to 12 as the repeat axis.

**Dependent variable.** Whether the detector scores each render as composed, and the compose rate
per column as seeds composed out of 4.

**Falsify condition.** There is no pass or fail here. This is a comparison recorded either way, and
the review file's question is written so that both answers are reportable. What would make it
uninterpretable is a layout that does not match the existing figure, since the whole read is the
comparison between the two.

**Why this matters right now.** It is the only thing this scope can say about the compose-decisive
early window, because [step 26](03-what-is-left-once-the-chain-settles.md) cannot
attribute there by construction.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope's headline measurement is silent in exactly the window the paper's
timing result is about. A number that cannot speak where the question is asked needs a companion
that can.

**The approach.** Ask a behavioural question instead of an attributional one. The corrector either
changes what comes out of a given window or it does not, and that is answerable at high noise where
attribution is not.

**Key insights.**

1. Matching the existing figure render for render is what turns two figures into one comparison.
   Any deviation in pair, seeds, positions or border rule has to be paid for in caption prose.
2. Same window and different window are both results. Only "we did not look" is a failure.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  corrector ON only inside the bracket, OFF everywhere else

  window position ->   0-10  5-15  10-20  15-25  20-30  25-35  30-40  35-45  40-50   all
  seed  9              [ ]   [ ]   [ ]    [ ]    [ ]    [ ]    [ ]    [ ]    [ ]     [ ]
  seed 10              [ ]   [ ]   [ ]    [ ]    [ ]    [ ]    [ ]    [ ]    [ ]     [ ]
  seed 11              [ ]   [ ]   [ ]    [ ]    [ ]    [ ]    [ ]    [ ]    [ ]     [ ]
  seed 12              [ ]   [ ]   [ ]    [ ]    [ ]    [ ]    [ ]    [ ]    [ ]     [ ]
                         ^
                         green border where the detector scored composed

  the injected-correction version of this grid peaks HERE, at the early columns.
  the question is whether this one does too.
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The run across the ten columns.** The composer from
   [step 25](../tools/02-the-corrector-and-the-step-size-it-runs-at.md) already takes a
   `corrector_window` tuple, so this is a driver over nine positions plus an all-50 condition, at
   the `k` on the flat part of step 26's curve.
2. **The figure.** Rows are seeds 9 to 12, columns are the nine window positions plus the all-50
   column, each square is the final picture, green border where the detector scored composed.
3. **The comparison, written down.** One paragraph in the review file saying whether the peak
   column matches the injected-correction figure's.
4. **The read-out every parallel session shares, and the fidelity read.** Two sheets per pair,
   rows the held-out seeds 9 to 16, from the seed's cached initial noise. The first puts the joint
   prompt, plain product-of-experts and the corrector on all 50 steps side by side. The second
   starts from the rank-32 adapter at λ 1.2 on all 50 steps and adds `k ∈ {0, 5, 20}` corrector
   steps on the last fifteen steps only (35 to 49), with the drift taken from the corrected
   prediction, so the chain settles into the corrected model's distribution rather than plain
   PoE's. The tail sheet is scored twice: compose (instance count) and sharpness (Laplacian
   variance of the greyscale render, the same measure plan 07 of scope 01 used). The control
   pair `a_butterfly__x__a_flower_meadow` runs on the same seeds so a corrector that breaks what
   already works is caught. Driver: `scripts/corrector_window_sweep.py` (`--sheet`, `--tail`,
   `--figures`, `--wandb`); thresholds in that file as `FIDELITY_MIN_SHARPNESS_RISE` and
   `FIDELITY_MAX_COMPOSE_LOSS_SEEDS`.

Figure to
`paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`,
with its `.json` sidecar. The `mcmc/` subfolder is where every corrector figure in this scope
files, because the folder's own `README.md` splits the timing question by which composition rule
supplied the term.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 4 of [the scope's direction](../../MASTER_PLAN.md). It generates the nine window
positions again with the corrector in place of the injected correction, so the two mechanisms can
be compared at the same moments of the run.

**Goals**

1. Forty scored renders, four seeds by ten columns.
2. The figure, matched render for render to the injected-correction version and filed beside it.
3. One recorded answer to whether the peak window matches, with the number of seeds composed per
   column.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Renders and their scored grid write under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/`, with a disk guard on
  `/datasets`, the filesystem actually written to, per
  [environment/storage.md](../../../../environment/storage.md). Only the finished figure and its
  sidecar go into the repo, under `paper/iclr/figures/`.
- biggpu allows one job per user, so this runs under `nohup` outside Slurm and `squeue` is
  blind to it. Harvest by `pgrep` on the session node, per
  [environment/hpc/execution-protocol.md](../../../../environment/hpc/execution-protocol.md).
- **The cached trajectories cannot be used.** The corrector moves the latent onto a different path.
- The tail condition attaches the rank-32 adapter at step 30050,
  `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`.
  The windowed adapter sampler leaves the adapter enabled when it returns, so the tail stage runs
  in its own process and no pure-corrector or reference render shares a process with it.
- The control pair's held-out cache holds seeds 9 to 12 only. Seeds 13 to 16 start from the
  from-seed float16 draw, which is the same construction as the cache's step-0 latent (checked
  equal on every cached cell before the runs).
- The models run in fp16, and anything normed upcasts to fp32 first.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so the figure's PDF comes from matplotlib, per
  [environment/paper.md](../../../../environment/paper.md).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Cross-reference this plan's terms against context/, environment/, runbook/,
      report/, and any learning journey that names this project, in case a term this plan mentions
      is already defined or explained somewhere else in the repo.
  - Paste: `/xref-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md`
  - Done when: the scan comes back with no candidates, or its proposed links have been applied.
- [x] **0.3** Read the existing injected-correction figure and write down the four layout facts this
      figure has to match: the pair, the seeds, the nine window positions, and the exact rule the
      green border encodes.

    ```bash
    ls -l paper/iclr/figures/when-the-correction-arrives/poe/
    sed -n '1,60p' paper/iclr/figures/when-the-correction-arrives/README.md
    ```

  - **Done when:** the four layout facts are quoted in this plan's review file, so a mismatch later
    is visible rather than argued about.

▶ **Next: [task 1.1](#1--generate-the-ten-window-columns)**.

### 1. 🚀 Generate the ten window columns

◀ **Needs: [task 2.2 of step 26](03-what-is-left-once-the-chain-settles.md#2--run-the-grid-and-plot-it)**,
whose curve says which `k` sits on the flat part. Run this plan even if that curve came back flat,
and record that it was run against a flat curve.

- [x] **1.1** Recreate the sliding-window grid with the corrector at the `k` on the flat part of
      step 26's curve.
  - `a_cat__x__a_dog` seeds 9 to 12 as rows, the same nine ten-step window positions as columns,
    plus a tenth column with the corrector on for all 50 steps.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/window_curves_mcmc.json`
  - **Done when:** that file holds 40 scored renders, counted rather than assumed.
- [x] **1.2** Draw the figure.
  - Rows are seeds 9 to 12, columns are the ten positions, each square is the final picture, green
    border where the detector scored composed. Column labels name the window in plain words on the
    axis ("steps 0 to 10") rather than in a legend.
  - Figure to
    `paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`,
    with its `.json` sidecar recording every render, the seeds, `k`, `c` and the border rule.
  - Add the entry to that folder's `README.md` naming the algorithm and what produced it.
  - **Done when:** the PNG, the sidecar and the README entry all exist.
- [x] **1.3** Confirm the timing folder's split by composition rule is intact, so this figure lands
      where the folder's own README says it should.

    ```bash
    ls paper/iclr/figures/when-the-correction-arrives/poe/ \
       paper/iclr/figures/when-the-correction-arrives/mcmc/
    ```

  - The injected-correction figures live in `poe/` and the corrector's in `mcmc/`, because the
    folder asks a question about timing and the composition rule is the variable inside it. The
    split and the README were put in place when this scope was created; this task confirms nothing
    has drifted rather than performing the move again.
  - **Done when:** `poe/` still holds the injected-correction renders and `mcmc/` holds this plan's
    figure and nothing else.

▶ **Next: [instruction 2.1](#2--read-the-two-grids-side-by-side)**, the comparison only a person
can make, and [task 3.1](#3--the-eight-seed-sheets-and-the-corrector-on-the-tail-of-the-adapter-run)
in parallel.

### 3. 🖼️ The eight-seed sheets, and the corrector on the tail of the adapter run

◀ **Needs: [task 2.2 of step 26](03-what-is-left-once-the-chain-settles.md#2--run-the-grid-and-plot-it)**
for the sheet's `k`, and [step 25's picked `c`](../tools/02-the-corrector-and-the-step-size-it-runs-at.md)
for both.

- [x] **3.1** The eight-seed sheet per pair: joint prompt, plain PoE, corrector on all 50 steps.

    ```bash
    OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector
    ssh <node> 'STAGE=sheet GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > '"$OUT"'/logs/sheet.log 2>&1 &'
    ```

  - Both pairs, seeds 9 to 16, at the flat-part `k` and the picked `c`. Plain PoE is the composer
    at `k=0`, byte-identical to the reference sampler; the joint prompt is plain CFG on the joint
    embedding.
  - Output goes to: `$OUT/sheet_scores.json`, renders under `$OUT/sheet/`.
  - **Done when:** the file holds 16 rows (2 pairs × 8 seeds), each with three scored tiles.
- [x] **3.2** The tail condition: the rank-32 λ 1.2 run with the corrector on steps 35 to 49 only.

    ```bash
    ssh <node> 'STAGE=tail GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > '"$OUT"'/logs/tail.log 2>&1 &'
    ```

  - `k ∈ {0, 5, 20}`, both pairs, seeds 9 to 16: 48 renders, each scored for compose and for
    sharpness. `k=0` is the adapter alone, so the corrector's addition is visible in the sheet.
  - Output goes to: `$OUT/tail_fidelity.json` with the printed branch.
  - **Done when:** the file holds 48 rows and the branch line is one of support, null,
    composition breaks, inconclusive or no branch fired, with the numbers it was judged on.
- [ ] **3.3** Draw the four sheets, and log everything to W&B.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_window_sweep.py --figures --wandb
    ```

  - Sheets to `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-<pair>-eight-seed-sheet.png`
    and `corrector-on-adapter-tail-<pair>-eight-seed-sheet.png`, each with a `.json` sidecar
    carrying the compose rate per column and every tile's score, plus a `README.md` entry each.
  - W&B project `prime_lab/poe-repair-animals-compose`: the sheets as images, the sidecars as
    one artifact, and the run id written into the review file's `## Runs` table.
  - **Done when:** the four PNGs, their sidecars and the README entries exist, and the review
    file carries the W&B run id.

▶ **Next: [instruction 4.1](#4--read-the-tail-sheet-by-eye)**, the sharpness read only an eye can
confirm, and [task 5.1](#5--the-clean-tail-the-adapter-early-the-frozen-model-after) once the
tail condition has returned its branch.

### 5. 🧹 The clean tail: the adapter early, the frozen model after

◀ **Needs: [task 3.2](#3--the-eight-seed-sheets-and-the-corrector-on-the-tail-of-the-adapter-run)**,
whose `k = 0` renders are this group's baseline, and task 3.1's plain-PoE renders, which set its
target band.

The softness of the adapter's renders is the goal here, and the tail condition in group 3 showed
the corrector on the corrected score does not remove it (sharpness +8%, inside the band). This
group changes what the tail is: the adapter sets the composition on the early steps only, the
frozen model's own plain-PoE step takes over after a cutoff, and the corrector, when on, settles
the latent into the frozen model's low-noise distribution rather than the adapter's. It is the
"correct early, then clean up" idea of
[scope 01's plan 14](../../../01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md),
run here with this scope's corrector as the clean-up; that plan's λ schedules and re-noise cell
stay with it.

- [ ] **5.1** Render the grid: cutoff `∈ {20, 30}` (the adapter on steps `[0, cutoff)` at λ 1.2)
      by `k ∈ {0, 5, 20}` corrector steps on the frozen score inside steps 35 to 49, both pairs,
      seeds 9 to 16, from the seed's cached noise.

    ```bash
    OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector
    ssh <node> 'STAGE=clean GPU=<idx> EXTRA="--c 3" nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > '"$OUT"'/logs/clean_tail.log 2>&1 &'
    ```

  - The cutoffs bracket where the corrected run commits (median step 15, range up to 35 in
    [where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md)).
  - Output goes to: `$OUT/clean_tail.json`, renders under `$OUT/clean_tail/`.
  - **Done when:** the file holds 96 rows (2 pairs × 8 seeds × 2 cutoffs × 3 `k`) and prints a
    branch.
- [ ] **5.2** Draw the sheet per pair and log it.

    ```bash
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY scripts/corrector_window_sweep.py --figures --wandb
    ```

  - Sheets to `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-clean-tail-<pair>-eight-seed-sheet.png`:
    rows seeds 9 to 16, columns the joint prompt, plain PoE, the adapter alone, then the six
    conditions, each tile with its count and sharpness; sidecar with the band, the baseline and
    the branch.
  - **Done when:** both PNGs, sidecars and README entries exist and the review file carries the
    W&B run id.

▶ **Next: [instruction 4.4](#4--read-the-tail-sheet-by-eye)**, the eye read of the clean-tail
sheet.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [what has to pass before this runs](#what-has-to-pass-before-this-runs).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 2. 👁️ Read the two grids side by side

◀ **Needs: [task 1.2](#1--generate-the-ten-window-columns)**, the new figure.

- [x] **2.1** Open both figures next to each other.
  - `paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`
  - `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png`
  - Expected result: two grids with identical rows and columns, differing only in what was switched
    on inside the window.
  - ❌ If the layouts differ in any of the four facts recorded at task 0.3, the comparison is not
    matched. Fix the figure rather than explaining the difference in the caption.
- [x] **2.2** Count composed renders per column by eye, on both grids, and note which column peaks
      on each.
  - [The timing verdict](../../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
    records that the detector and the eye disagree on cat and dog often enough that the eye read is
    the one cited. Do both, and cite the eye where they differ.
- [x] **2.3** Record in the review file whether the corrector's compose rate peaks in the same
      window the injected correction does, with the per-column counts beside it.
  - Same window is a strong result: two different mechanisms acting at the same moment.
  - A different window is a stronger one and needs its own paragraph.
  - No window composing at all is also recorded, and it bounds what a training-free corrector does
    at this compute budget.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 28](../baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

### 4. 👁️ Read the tail sheet by eye

◀ **Needs: [task 3.3](#3--the-eight-seed-sheets-and-the-corrector-on-the-tail-of-the-adapter-run)**,
the tail sheet.

- [x] **4.1** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png`.
  - Row by row, compare the `k=0` tile (the adapter alone) with the `k=20` tile: is the `k=20`
    one crisper, the same, or softer, and are both animals still there?
  - The Laplacian variance printed on each tile is the number the branch was judged on; a tile
    the number calls sharper that the eye calls noisier is the case the measure cannot see, and
    it is recorded in the review file as such.
- [x] **4.2** Open the butterfly × meadow tail sheet and check nothing that composed at `k=0` has
      lost its butterfly at `k=20`.
- [x] **4.3** Write the eye read in one line beside the printed branch in the review file.
- [ ] **4.4** Open `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-clean-tail-cat-dog-eight-seed-sheet.png`.
  - Row by row, is any clean-tail column both as crisp as the plain-PoE column and still two
    animals? Name the seeds where it is, and the seeds where the hand-off to the frozen model
    lost an animal.
  - Write the read in one line beside the printed branch in the review file.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**.

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the scope's only statement about the window the paper's
> timing result is about, and it is worth nothing unless it is matched render for render to the
> figure it is compared against.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector

pgrep -af 'corrector|window'          # squeue is blind to this
$PY -c "import json;d=json.load(open('$OUT/window_curves_mcmc.json'));print(len(d['cells']),'cells')"
# expect 4 seeds x 10 columns = 40 cells

ls -l paper/iclr/figures/when-the-correction-arrives/mcmc/
```

**Pass criteria**

- 40 scored renders in `window_curves_mcmc.json`.
- 16 rows in `sheet_scores.json` and 48 in `tail_fidelity.json`, with the tail branch printed.
- The figure and its `.json` sidecar exist under `mcmc/`, with a `README.md` entry.
- The layout matches the four facts recorded at task 0.3.
- Instruction 2.3 has recorded the same-window answer, whichever way it went.

**Fail criteria (STOP)**

- The layout does not match the existing figure. The two are then not a comparison, and this one
  must be redrawn.

**Partial pass guidance**

- No column composing is still a result. Record it and say what it bounds.

**When you get results, answer**
[the review file](../../review/04-does-the-corrector-compose-in-the-same-window.md).

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-<pair>-eight-seed-sheet.png`, one per pair | — | rows are the held-out seeds 9 to 16, columns the joint prompt, plain PoE, and the corrector on all 50 steps at the flat-part `k`; frame colour is the scorer's verdict; the compose count per column is in the title and the sidecar | `scripts/corrector_window_sweep.py --figures` | ⏳ | **The read-out every parallel session shares.** Sidecar `.json` carries every tile's score, `k`, `c` and the sampler settings. For the control pair the frame is the unvalidated both-concepts read, said on the sheet |
| `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-on-adapter-tail-<pair>-eight-seed-sheet.png`, one per pair | — | rows seeds 9 to 16, columns the joint prompt, plain PoE, the rank-32 λ 1.2 adapter alone, then the adapter plus `k=5` and `k=20` corrector steps on steps 35 to 49; each tile carries its instance count and its Laplacian variance | `scripts/corrector_window_sweep.py --figures` | ⏳ | **The fidelity read.** Sidecar carries the branch, the thresholds and every number. Main text only if the branch is support |
| `artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-clean-tail-<pair>-eight-seed-sheet.png`, one per pair | — | rows seeds 9 to 16; columns the joint prompt, plain PoE, the adapter alone on all 50 steps, then the adapter on steps 0 to 19 or 0 to 29 followed by the frozen model, with 0, 5 or 20 corrector steps on the frozen score in steps 35 to 49; each tile carries its instance count and Laplacian variance | `scripts/corrector_window_sweep.py --figures` | ⏳ | **The fidelity fix under test.** Sidecar carries the plain-PoE band, the adapter-alone baseline, the thresholds and the branch. Main text only if the branch is support |
| `mcmc/samples-as-a-ten-step-corrector-window-slides.png` | — | rows are seeds 9 to 12, columns are the nine ten-step window positions plus a corrector-on-all-50 column, each square is the final generated picture, green border where the detector scored it as two separate animals | the driver at task 1.1 | ⏳ | **Main text, only if it differs from the injected-correction version.** If the behaviour is identical, one sentence of prose beside the existing figure covers it. Sidecar `.json` records every render, the seeds, `k`, `c` and the border rule |

**Two sentences this figure's caption owes.** The corrector runs at one `k`, which is a compute
budget rather than a property of the problem. And the green border encodes the detector's verdict,
which disagrees with the eye on this pair often enough that the eye count is quoted beside it.

### Organization workflow

1. Generate the ten columns, which writes the scored grid under `/datasets`.
2. Draw the figure into `paper/iclr/figures/when-the-correction-arrives/mcmc/` with its sidecar.
3. Add the entry to that folder's `README.md`.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the same-window answer | the review file, then [writing-06](../../../07-writing-the-paper/plans/writing/06-mechanism-and-limitations.md), whose mechanism paragraph this either supports or complicates |
| a figure lands in `mcmc/` | that folder's `README.md` gains an entry naming the algorithm and what produced it |
| the plan's status | the scope [MASTER_PLAN.md](../../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [scripts/interaction_term_window.py](../../../scripts/interaction_term_window.py) | the injected-correction window run this plan mirrors, with its nine positions, its seeds, and its scored-grid format |
| `poe_repair/composers/poe_langevin.py` | the composer from [step 25](../tools/02-the-corrector-and-the-step-size-it-runs-at.md), whose `corrector_window` argument is what this plan slides |
| `paper/iclr/figures/when-the-correction-arrives/README.md` | the folder's own rule for which subfolder a figure lands in, and the exact green-border rule quoted into captions |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 28, what changes when SuperDiff leaves its own defaults](../baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md).
It starts the comparison half, which asks how much is added rather than when.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>2 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 the detector and the eye disagree on cat and dog

**When it happens:** scoring this pair, at any amount of correction or any window.
**What you see:** green borders on renders that do not look composed, or no border on renders that
do.
**Why:** the instance-count scorer is validated but not perfect on this pair, and
[the timing verdict](../../../03-does-the-correction-cause-composition/review/05-when-in-the-run-it-matters.md)
already records the disagreement rate.
**How to fix:** count by eye as well, cite the eye where they differ, and say so in the caption.

#### 🟡 the two grids are nearly matched rather than matched

**When it happens:** redrawing a layout from memory instead of from the existing figure.
**What you see:** two figures a reader compares anyway, with a difference that is not the variable.
**Why:** the four layout facts were not written down before drawing.
**How to fix:** task 0.3 records them first; redraw rather than explain the difference in prose.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
