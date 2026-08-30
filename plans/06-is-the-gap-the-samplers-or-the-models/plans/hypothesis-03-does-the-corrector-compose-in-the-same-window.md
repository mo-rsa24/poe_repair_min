# 🧪 Does the corrector compose in the same window?

This plan asks whether a corrector switched on inside a ten-step window changes what comes out of
the run, and whether the window where it works is the same early window the injected correction
works in.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md — <same window or a different one>
```

## Recommended skill

▶ `/pair-figure` ✅ this figure only means anything beside the injected-correction version, and
   pairing two grids so the comparison is the thing the reader sees is exactly what that skill
   decides.
   alt: `/analyze-figure` on the existing `poe/samples-as-a-ten-step-window-slides.png` first, to
   fix the layout this one has to match.

## Position in the plan tree

**Step 27 of 30.** Waits on step 26. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 6 | [hypothesis-03: when-in-the-run-it-matters](../../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md) ◑ | generated the injected-correction renders at all nine window positions, which this plan is matched against render for render |
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](hypothesis-02-what-is-left-once-the-chain-settles.md) ⚠️ | the measurement this plan waits on, though not strictly |
| **27 (current)** | **hypothesis-03: does-the-corrector-compose-in-the-same-window** ⚠️ | **generates the nine window positions again with the corrector in place of the injected correction, and asks whether the [compose rate](context/world/compose-rate.md) peaks at the same moment** |
| 29 | [baseline-02: three-rules-on-one-dose-axis](baseline-02-three-rules-on-one-dose-axis.md) ⚠️ | the other half of the comparison, on dose rather than timing |

Design only. Verdicts and run state live in
[the paired review file](../review/hypothesis-03-does-the-corrector-compose-in-the-same-window.md).

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
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Slide a ten-step corrector window across the run, at the same nine positions the injected-correction
renders used, and ask whether the compose rate peaks in the same window that the injected correction
does.

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

**Rationale.** [Step 26](hypothesis-02-what-is-left-once-the-chain-settles.md) measures a norm and
cannot attribute anything in the early window, which is where the compose rate is decided. This
plan asks a behavioural question about that same window instead, which is answerable there.

**Dataset details.** `a_cat__x__a_dog` seeds 9 to 12 as rows. Nine ten-step window positions as
columns, matched to the existing figure, plus a tenth column with the corrector on for all 50 steps.
Forty renders, each decoded and scored.

**Associated materials.** [The review questions](../review/hypothesis-03-does-the-corrector-compose-in-the-same-window.md),
[the whole corrector design](../source/the-whole-corrector-design.md), and the figure this one is
matched against, `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png`.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This plan waits on step 26, but not strictly.**

A flat curve at [step 26](hypothesis-02-what-is-left-once-the-chain-settles.md) does not strictly
imply no change in compose rate, because the corrector can relocate the trajectory without
shrinking `‖r_t‖`. If step 26 comes back flat and this plan still composes, that combination is the
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
early window, because [step 26](hypothesis-02-what-is-left-once-the-chain-settles.md) cannot
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
   [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) already takes a
   `corrector_window` tuple, so this is a driver over nine positions plus an all-50 condition, at
   the `k` on the flat part of step 26's curve.
2. **The figure.** Rows are seeds 9 to 12, columns are the nine window positions plus the all-50
   column, each square is the final picture, green border where the detector scored composed.
3. **The comparison, written down.** One paragraph in the review file saying whether the peak
   column matches the injected-correction figure's.

Figure to
`paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`,
with its `.json` sidecar. The `mcmc/` subfolder is where every corrector figure in this scope
files, because the folder's own `README.md` splits the timing question by which composition rule
supplied the term.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 4 of [the scope's direction](../MASTER_PLAN.md). It generates the nine window
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
  [environment/storage.md](../../../environment/storage.md). Only the finished figure and its
  sidecar go into the repo, under `paper/iclr/figures/`.
- biggpu allows one job per user, so this runs under `nohup` outside Slurm and `squeue` is
  blind to it. Harvest by `pgrep` on the session node, per
  [environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md).
- **The cached trajectories cannot be used.** The corrector moves the latent onto a different path.
- The models run in fp16, and anything normed upcasts to fp32 first.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².
- No system LaTeX here, so the figure's PDF comes from matplotlib, per
  [environment/paper.md](../../../environment/paper.md).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Read the existing injected-correction figure and write down the four layout facts this
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

◀ **Needs: [task 2.2 of step 26](hypothesis-02-what-is-left-once-the-chain-settles.md#2--run-the-grid-and-plot-it)**,
whose curve says which `k` sits on the flat part. Run this plan even if that curve came back flat,
and record that it was run against a flat curve.

- [ ] **1.1** Recreate the sliding-window grid with the corrector at the `k` on the flat part of
      step 26's curve.
  - `a_cat__x__a_dog` seeds 9 to 12 as rows, the same nine ten-step window positions as columns,
    plus a tenth column with the corrector on for all 50 steps.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/window_curves_mcmc.json`
  - **Done when:** that file holds 40 scored renders, counted rather than assumed.
- [ ] **1.2** Draw the figure.
  - Rows are seeds 9 to 12, columns are the ten positions, each square is the final picture, green
    border where the detector scored composed. Column labels name the window in plain words on the
    axis ("steps 0 to 10") rather than in a legend.
  - Figure to
    `paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`,
    with its `.json` sidecar recording every render, the seeds, `k`, `c` and the border rule.
  - Add the entry to that folder's `README.md` naming the algorithm and what produced it.
  - **Done when:** the PNG, the sidecar and the README entry all exist.
- [ ] **1.3** Confirm the timing folder's split by composition rule is intact, so this figure lands
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
can make.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the engagement gate](#the-engagement-gate).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 2. 👁️ Read the two grids side by side

◀ **Needs: [task 1.2](#1--generate-the-ten-window-columns)**, the new figure.

- [ ] **2.1** Open both figures next to each other.
  - `paper/iclr/figures/when-the-correction-arrives/mcmc/samples-as-a-ten-step-corrector-window-slides.png`
  - `paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png`
  - Expected result: two grids with identical rows and columns, differing only in what was switched
    on inside the window.
  - ❌ If the layouts differ in any of the four facts recorded at task 0.2, the comparison is not
    matched. Fix the figure rather than explaining the difference in the caption.
- [ ] **2.2** Count composed renders per column by eye, on both grids, and note which column peaks
      on each.
  - [The timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md)
    records that the detector and the eye disagree on cat and dog often enough that the eye read is
    the one cited. Do both, and cite the eye where they differ.
- [ ] **2.3** Record in the review file whether the corrector's compose rate peaks in the same
      window the injected correction does, with the per-column counts beside it.
  - Same window is a strong result: two different mechanisms acting at the same moment.
  - A different window is a stronger one and needs its own paragraph.
  - No window composing at all is also recorded, and it bounds what a training-free corrector does
    at this compute budget.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 28](baseline-01-superdiff-at-this-repos-fifty-steps.md).

## The engagement gate

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
- The figure and its `.json` sidecar exist under `mcmc/`, with a `README.md` entry.
- The layout matches the four facts recorded at task 0.2.
- Instruction 2.3 has recorded the same-window answer, whichever way it went.

**Fail criteria (STOP)**

- The layout does not match the existing figure. The two are then not a comparison, and this one
  must be redrawn.

**Partial pass guidance**

- No column composing is still a result. Record it and say what it bounds.

**When you get results, answer**
[the review file](../review/hypothesis-03-does-the-corrector-compose-in-the-same-window.md).

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The standard every figure in this scope is held to is
[in the scope's MASTER_PLAN](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
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
| the same-window answer | the review file, then [writing-06](../../07-writing-the-paper/plans/writing-06-mechanism-and-limitations.md), whose mechanism paragraph this either supports or complicates |
| a figure lands in `mcmc/` | that folder's `README.md` gains an entry naming the algorithm and what produced it |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-03-does-the-corrector-compose-in-the-same-window.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [scripts/interaction_term_window.py](../../../scripts/interaction_term_window.py) | the injected-correction window run this plan mirrors, with its nine positions, its seeds, and its scored-grid format |
| `poe_repair/composers/poe_langevin.py` | the composer from [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md), whose `corrector_window` argument is what this plan slides |
| `paper/iclr/figures/when-the-correction-arrives/README.md` | the folder's own rule for which subfolder a figure lands in, and the exact green-border rule quoted into captions |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 28, SuperDiff at this repo's fifty steps](baseline-01-superdiff-at-this-repos-fifty-steps.md).
It starts the comparison half, which asks about dose rather than timing.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>2 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 the detector and the eye disagree on cat and dog

**When it happens:** scoring this pair, at any dose or window.
**What you see:** green borders on renders that do not look composed, or no border on renders that
do.
**Why:** the instance-count scorer is validated but not perfect on this pair, and
[the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md)
already records the disagreement rate.
**How to fix:** count by eye as well, cite the eye where they differ, and say so in the caption.

#### 🟡 the two grids are nearly matched rather than matched

**When it happens:** redrawing a layout from memory instead of from the existing figure.
**What you see:** two figures a reader compares anyway, with a difference that is not the variable.
**Why:** the four layout facts were not written down before drawing.
**How to fix:** task 0.2 records them first; redraw rather than explain the difference in prose.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
