# ⏱️ When in the denoising run is the correction needed?

## Recommended prompt (after run completes)

After you finish this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against the global and project
catalogs, and adds new entries. New errors propagate to all affected plan files.

---

## Position in the plan tree

**Step 6 of 22.** Waits on steps 4 and 5. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 5 (previous) | ~~[hypothesis-04: what-the-cached-runs-already-show](hypothesis-04-what-the-cached-runs-already-show.md)~~ ✅ | Measured the fork step, step 16, from cached trajectories |
| **6 (current)** | **hypothesis-03: when-in-the-run-it-matters** ◑ | **Answered: the cliff is at the start. The timing tab is still owed** |
| 7 (next) | ~~[hypothesis-05: the-same-story-from-three-sides](hypothesis-05-the-same-story-from-three-sides.md)~~ ✅ | Corroborated the story from three independent measures |

Design only. Verdicts and run state live in
[../review/hypothesis-03-when-in-the-run-it-matters.md](../review/hypothesis-03-when-in-the-run-it-matters.md).

---

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [How wide the window is, and why not the obvious rule](#how-wide-the-window-is-and-why-not-the-obvious-rule)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Not run: the conditioning-window half](#not-run-the-conditioning-window-half)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Illustrations](#illustrations)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## What this asks, in one line

Let the correction act only inside a narrow window of the 50 denoising steps, slide that
window from start to finish, and measure the compose rate at each position. A peak says
when the correction is needed; a flat curve says it is needed throughout.

---

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment**

A sliding window over the 50 denoising steps, gating only the injected correction. Nine
placements, eight held-out pairs, four seeds.

**What it found**

The answer is in, and it is sharper than a peak. The compose rate falls off a cliff: the earliest
window composes, one notch later is much worse, and from a fifth of the way in it is zero and
stays there. Every number is in the
[review file](../review/hypothesis-03-when-in-the-run-it-matters.md), never in this plan.

**Two things the result changed**

The peak does not sit where the correction is large. It sits where the correction is at its
smallest, so size is ruled out as the explanation.

The peak does not sit at the fork step either, the same moment estimated from cached
trajectories in step 5. The two disagree, so neither may be printed as confirming the other, and
no caption may draw the fork step as a band behind the timing curve.

**Where it stands**

The bar is answered and so is every pre-registered question, including two follow-ons added after
the timing result to remove its one confound. Eight figures are built. What remains is manual:
driving the correction-timing tab by hand in the inspector.

**Associated materials**

- Review file, the status source: [../review/hypothesis-03-when-in-the-run-it-matters.md](../review/hypothesis-03-when-in-the-run-it-matters.md)
- Figure register, which caps every caption: [paper/iclr/figures.md](../../../../../paper/iclr/figures.md), rows F4a to F4h
- Grid definition both runner and scorer read: `poe_repair/experiments/interaction_term/window_grid.py`

**For the full picture**

[Scope master plan](../MASTER_PLAN.md). Serves Definition-of-Done item 4 and Goal 2.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime**

The timing grid is the largest generation grid in the program: 9 windows × 8 pairs × 4 seeds = 288
cells at about 31 seconds each, roughly 2.5 hours on one A6000. It has already run.

**Prerequisites**

Steps 4 and 5. Step 5 supplies the fork step this experiment's peak is compared against.

**Environment Facts This Plan Depends On**

- W2 harness: `scripts/interaction_term_window.py` over `run_teacher_residual`'s
  `correction_window`, which gates only the injected r_t and leaves conditioning on at every step.
  Never reuse W1's uncond-outside gating for W2, that confounds losing-the-correction with
  losing-conditioning.
- Largest generation grid in the program: 9 windows × 8 pairs × 4 seeds = 288 cells at about 31s
  each, roughly 2.5 hours on one A6000.
- Output goes to /datasets via `POE_REPAIR_OUTPUT_ROOT`, which the sampler reads; the disk guard
  checks the filesystem `$OUT` actually resolves to.
- The scorer refuses to pool runs under 40 steps, because a window at step 20 of a 20-step run is
  a different moment from step 20 of a 50-step run.
- The inspector is served on a cluster node, so reaching it needs an SSH forward. The launcher
  prints the exact `ssh -L` line to paste into a second terminal.

**Project tracking**

W&B project `prime_lab/poe-repair-animals-compose`. W&B owns the numbers; this plan owns the
design and the review file owns the verdict.

**Known issues**

See the [Error Matrix](#error-matrix) at the bottom of this file.

---

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Sliding a fixed-width window across the run, gating only the correction, measures causally when
the correction is needed, with no window assumed in advance.**

**Why this matters right now:** it is the difference between a fix that must be applied throughout
and one that only has to be applied at the start. The second is a far cheaper method and a far
stronger claim, and only this experiment can tell them apart.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The question**

Every earlier result says the correction works. None of them says *when* it has to act. Without
that, the method has to be applied at every step because nobody knows which steps are load-bearing.

**The approach**

Hold everything else fixed and move one thing: the stretch of steps during which the correction is
allowed to act. Because conditioning never switches off, the only difference across the nine
placements is when the correction lands.

**Key insights**

1. **The base must stay fully conditioned.** Gating the prompt as well would confound losing the
   correction with losing conditioning, and the answer would mean nothing. That is a different
   experiment, named below and not run.

2. **An all-off window must reproduce plain PoE exactly.** If gating leaks, every window position
   is contaminated and the curve is measuring the leak. The identity check runs before any grid.

3. **The width had to be fixed before any window ran.** Choosing it after seeing the curve would
   let the width be tuned until the answer looked good. The reasoning is recorded in its own
   section below.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#how-wide-the-window-is-and-why-not-the-obvious-rule) ➡️

Nine placements of a width-10 window across 50 denoising steps, stride 5. The correction is
injected only inside the shaded stretch; the prompt is on everywhere in every row.

```
step   0    10    20    30    40    50
       |----|-----|-----|-----|-----|
w1     ####                              window 0-10
w2       ####                            window 5-15
w3          ####                         window 10-20
w4            ####                       window 15-25
w5               ####                    window 20-30
w6                 ####                  window 25-35
w7                    ####               window 30-40
w8                      ####             window 35-45
w9                         ####          window 40-50
       ^^^^                    ^
       the cliff               the fork step (step 16) sits inside
       lives here              w3 and w4 only, by design
```

Each row is 8 pairs × 4 seeds = 32 scored cells. The fork step lands inside only two of the nine
placements, which is what lets the sweep disagree with it rather than being built to agree.

---

## How wide the window is, and why not the obvious rule

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

The width had to be fixed before any window ran. The obvious rule was to take it
from the correction's size: make the window as wide as the narrowest band of
steps carrying half of the total ‖r_t‖. That rule gives 25 steps, half the
trajectory, because ‖r_t‖ barely changes over the run: each fifth of the run
carries 15-22% of the total and the largest step is 1.8x the smallest
(`scripts/window_width.py`, artifact `cache_analyses/window_width.json`). At
width 25 every placement contains the fork step, so the curve would come out
flat whatever the truth is, and the experiment would test nothing.

So the width is set by what the sweep has to be able to resolve, not by where
the correction is large. Width 10 at stride 5 gives nine placements, each a
fifth of the run, with the fork step inside only two of them (10-20 and 15-25).
If timing matters those two win; if nothing peaks, timing does not matter, and
that is a finding rather than a failure. Both numbers live in
`poe_repair/experiments/interaction_term/window_grid.py`, which the runner, the
scorer, the strip, and the inspector all read, so no two of them can disagree
about which grid was run.

---

## Description: what to build

⬅️ [Previous](#how-wide-the-window-is-and-why-not-the-obvious-rule) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

One sliding-window experiment over the same eight held-out pairs and four seeds
the dose sweep used. The base is full guided PoE at every step, with the prompt
on throughout; a width-10 window slid across the 50 steps at stride 5 gates only
the injected r_t. Because conditioning never switches off, the only thing that
changes across the nine positions is when the correction acts.

Two follow-on runs were added after the timing result, to remove its one confound. The windows
differ in when the correction lands and also in how much of it lands, because the correction grows
through the run. The swap run gives the late window exactly the total that works early; the matched
run rescales every window to deliver the same total. Both are cat × dog, four seeds, through the
same sampler as the sweep.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose**

It measures causally when the interaction term is needed, with no window assumed
in advance. The peak is then compared against the fork step, which is the same
moment estimated a different way, from cached trajectories rather than from new
runs. Serves DoD 4 and Goal 2.

**Goal**

Register slot **F4**, in two halves read together: compose rate against window
centre with the fork step drawn on it, and a strip of one cell across all nine
window positions with the scorer's verdict on each.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 1. 🔧 The harness and the grid

- [x] **1.1** W2 harness: window-gated r_t injection on an always-conditioned PoE
      base, and prove the λ=0-outside-window case equals plain PoE exactly.
      `scripts/interaction_term_window.py --window off --check-identity`.
- [x] **1.2** Fix the window width and the grid in source, from the ‖r_t‖-vs-step
      curve. The section above records what the curve actually supports.
      `scripts/window_width.py`, `window_grid.py`.
- [x] **1.3** Smoke in-session on one pair, one seed, three windows: early, over the
      fork step, and late. `SMOKE=1 bash scripts/mechanism_study/run_window_sweep.sh`.
- [x] **1.4** Full W2 grid: 288 cells, resumable, then score every image.
      `bash scripts/mechanism_study/run_window_sweep.sh`, then
      `python scripts/plot_window_curves.py`.
- [x] **1.5** The curve and the image strips. `scripts/plot_window_curves.py` for the
      curve with the fork step drawn on it, `scripts/window_strip.py` for the
      same cell across all nine windows.

▶ **Next: task 2.1**, the runs that untie timing from dose.

### 2. 📊 Untying timing from dose, and the two ends of the run

◀ **Needs: tasks 1.4 and 1.5**, so the nine-window curve exists to be challenged.

- [x] **2.1** The swap run: give the late window exactly the correction total that works early,
      cat × dog, four seeds. `scripts/interaction_term_dose_matched.py --mode swap`, artifacts
      `dose_matched/swap_manifest.json` and `swap_scores.json`.
- [x] **2.2** The matched run: rescale every one of the nine windows to deliver the same total,
      then re-score. `--mode matched`, artifact `dose_matched/matched_scores.json`.
- [x] **2.3** The front-loaded and back-loaded sweeps: five prefix cutoffs with the correction on
      then off, and five suffix cutoffs with it off then on. `scripts/longer_correction_grid.py`,
      `later_start_grid.py`, `plot_growing_window_curves.py`, scored into
      `growing_window_curves.json`.
- [x] **2.4** Build the eight F4 figures from the scored files.
      `size_vs_timing.py`, `window_position_grid.py`, `window_map_all_pairs.py`, `timing_vs_dose.py`,
      `timing_cliff_matched_dose.py`, `longer_correction_grid.py`, `later_start_grid.py`, `caption_readback.py`.
- [x] **2.5** Build the manifest the inspector's timing tab reads.
      `scripts/build_window_manifest.py`, artifact `window/window_inspector_manifest.json`.

▶ **Next: instruction 3.1**, which drives that manifest by hand.

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 🖱️ Drive the correction-timing tab by hand

◀ **Needs: task 2.5** done, so `window_inspector_manifest.json` exists for the tab to read.

3.1 **Start the inspector and open the tunnel**
   - On the cluster node: `bash scripts/run_lora_inspector.sh`
   - It prints an `ssh -L` line. Paste that line into a second terminal on your laptop and leave
     it running: the app is served on the node, so without the forward the URL resolves to nothing
   - Open the forwarded `localhost` URL in a browser
   - ✅ the inspector loads and a "Correction timing" tab is present
   - ❌ the page does not load: check the forward is still up before restarting the app

3.2 **Slide the window and watch three things move together**
   - Drag the window slider from the earliest position to the latest
   - Watch the picture, the scorer's verdict, and the marker on the curve
   - Expected result: all three move together, and the pictures visibly stop composing early in
     the sweep
   - ✅ the cliff is visible by eye, in the same place the curve puts it
   - ❌ the picture and the curve marker disagree: the manifest and the curve were built from
     different grids; rebuild both from `window_grid.py`

3.3 **Pin one window and compare two**
   - Pin the earliest window, then slide to a late one
   - Record what differs, in one line, in the review file
   - This is the check the numbers cannot make: whether the failure looks like one fused animal
     rather than two separate ones

▶ **Next: instruction 4.1.**

### 4. 📋 Close the register gaps this plan's figures opened

◀ **Needs: task 2.4** done, so all eight figures exist on disk.

4.1 **Give F4f a register row, or retire the figure**
   - `paper/iclr/figures/how-many-seeds-composed-as-the-window-moves.{png,pdf,json}` exists and is tracked, and no row in
     `paper/iclr/figures.md` describes it
   - A figure with no row has no ceiling on what its caption may claim
   - ✅ a row exists naming its claim sentence, its layout, its evidence and its build command
   - ❌ nothing in the paper needs it: retire the figure rather than leaving it unregistered

4.2 **Flip the F4g and F4h rows from reserved to built**
   - Both rows read `reserved` in `paper/iclr/figures.md`; both figures exist on disk and are
     tracked
   - Read each PDF cold first (it is the same judgment as 3.2), then flip the status and write the
     caption caps
   - ✅ the register's status matches what is on disk for every F4 row

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters**: this plan's answer is what the paper's timing claim rests on. A
> leaking harness would make every window position wrong in the same direction, which looks exactly
> like a result.

- **Pass criteria**
  - The all-off window reproduces plain PoE byte-identically
  - A mid-trajectory window visibly changes the output
  - All 288 cells present, no missing or skipped windows
  - The review file's pre-registered bar is answered

- **Fail criteria (STOP)**
  - The all-off window differs from plain PoE, meaning the gating leaks. Fix before any grid.

- **Partial pass guidance**
  - A flat curve is a finding, not a failure: it would say timing does not matter. Record it as
    the answer rather than widening the grid until something peaks.

**When you get results, answer** [the review file](../review/hypothesis-03-when-in-the-run-it-matters.md)
**or move to** [Next step](#next-step).

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#not-run-the-conditioning-window-half) ➡️

These are result plots rather than diagrams of a system, so the subject-versus-process split the
Lane column carries does not apply. It is filled `—` deliberately rather than guessed. Every
figure below exists on disk and is tracked in git.

| Item | Lane | What it shows | Built by | Register status |
|------|------|---------------|----------|-----------------|
| F4a when it arrives | — | Every cat × dog cell in the window grid, 9 windows across, 4 seeds down, time reading left to right | `window_position_grid.py` | **built** |
| F4b size is not timing | — | Correction size per step against compose rate per window, on one step axis | `size_vs_timing.py` | **built** |
| F4c the cliff in language | — | The same nine windows read by caption similarity instead of by counting animals | `caption_readback.py` | **built** |
| F4d timing not dose | — | 2×2 of real samples: early and late windows crossed with each other's delivered total | `timing_vs_dose.py` | **built** |
| F4e cliff survives dose-matching | — | The nine-window rate at full strength and with every window rescaled to one total | `timing_cliff_matched_dose.py` | **built** |
| F4f the window map | — | The window grid itself | `window_map_all_pairs.py` | ⚠️ no register row (instruction 4.1) |
| F4g more start, same ceiling | — | Front-loaded sweep: five prefix cutoffs, correction on then off | `longer_correction_grid.py` | ⚠️ `reserved`, file exists (instruction 4.2) |
| F4h too late to fix | — | Back-loaded sweep: five suffix cutoffs, correction off then on | `later_start_grid.py` | ⚠️ `reserved`, file exists (instruction 4.2) |

Numbers live in each figure's sidecar `.json` and in its register row, never in this plan.

---

## Not run: the conditioning-window half

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

W1 slides the same fixed-width window over the prompt rather than over the
correction, and would say whether the prompt and the correction are needed at
the same moment. It corroborates but F4 can carry its claim on W2 alone, so it
is not run here and the caption says the conditioning-window comparison is
future work. Running it means adding sliding fixed-width schedules to
`poe_repair/experiments/cfg_window_without_lora` and scoring old and new outputs
with the compose scorer.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#not-run-the-conditioning-window-half) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

**Why this matters**: three files can disagree about one figure. This plan says what to build, the
register says what may be claimed, and the review file says what was found. F4f and the two
`reserved` rows are what that disagreement looks like when nobody reconciles it.

**After completion**:

1. **Capture learnings and errors**: run `/ingest-error-pattern --from-run-log`.
2. **Update this plan file**: `/sync-plan-tree` regenerates the Error Matrix and strikes the task
   lines that landed.
3. **Update the register**: flip each row's status and write its caption caps (instructions 4.1
   and 4.2).
4. **Answer the review file**: the verdict lives there, never here.

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Capture patterns | `/ingest-error-pattern --from-run-log` | Manual, after completion | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Automatic, by ingest-error-pattern | This file's Error Matrix regenerated |
| Reconcile the register | Manual edit of `paper/iclr/figures.md` | Manual, instructions 4.1 and 4.2 | Every F4 row's status matches disk |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#illustrations) ➡️

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window

# The leak check: nothing injected must leave the trajectory untouched.
$PY scripts/interaction_term_window.py --pair a_cat__x__a_dog --seed 9 \
    --window off --check-identity        # expect: byte-identical to PoE

# The grid. Resumable, skips any cell whose image exists.
bash scripts/mechanism_study/run_window_sweep.sh
find $OUT/pairs -name '*_w*.png' | wc -l  # expect 288 for the sliding sweep

# Score, then the two halves of F4, then the manifest the inspector reads.
$PY scripts/plot_window_curves.py         # curve + peak band, prints missing windows
$PY scripts/window_strip.py --pair a_cat__x__a_dog --seed 9
$PY scripts/build_window_manifest.py

# Drive it by hand: the Correction timing tab.
bash scripts/run_lora_inspector.sh        # prints the ssh -L line to tunnel with
```

**File:** [scripts/interaction_term_window.py](../../../../../scripts/interaction_term_window.py)
**What it does:** gates only the injected r_t over `run_teacher_residual`'s `correction_window`,
leaving conditioning on at every step. `--check-identity` is the leak check.

**File:** [scripts/interaction_term_dose_matched.py](../../../../../scripts/interaction_term_dose_matched.py)
**What it does:** `--mode swap` crosses the early and late windows against each other's delivered
total; `--mode matched` rescales all nine windows to one total.

**File:** `poe_repair/experiments/interaction_term/window_grid.py`
**What it does:** holds the width and the stride. The runner, the scorer, the strip and the
inspector all read it, so no two of them can disagree about which grid was run.

---

## Illustrations

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Two pieces in the scope's illustrated map,
[plans/closing-the-compositional-gap/diagram-prompts.md](../../../diagram-prompts.md), both
marked `[planned]`. They use the map's pinned cast, so they read as part of that set rather than
as one-off pictures.

**Prompt 2a (Subject): one thing changes, which is when the correction acts**

Nine step strips stacked, each with its window bracket one notch further right, the fork step
drawn as a vertical guide crossing only two of them. What the experiment is made of, with no
sequencing in it.

*(image not yet generated; save under `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/assets/` and link it here)*

**Prompt 2a (Process): the stages that produced the timing answer**

Seven stages from fixing the width in source to driving the timing tab by hand, with the leak
check's failure branch drawn and the register gaps hanging off the figures card.

*(image not yet generated; save under the same `assets/` folder and link it here)*

---

## Next step

⬅️ [Previous](#illustrations) | 📋 [TOC](#table-of-contents)

1. Drive the correction-timing tab by hand, instruction 3.1 in
   [Instructions](#instructions). The manifest it reads is already built.
2. Close the register gaps: instruction 4.1 for F4f, 4.2 for the two `reserved` rows.
3. F4 goes through step 13's `/design-figure` pass, carrying two cautions the review settled.
   The caption may not claim the conditioning-window comparison, which was not run. And it may
   not draw step 16 as a band behind the timing curve, because the peak and the fork step
   disagree.

Then step 7, [hypothesis-05: the same story from three sides](hypothesis-05-the-same-story-from-three-sides.md), already done.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

**Purpose**: known issues and their fixes. Regenerated by `/sync-plan-tree --update-error-matrices`
after `/ingest-error-pattern` appends new patterns.

#### From global catalog

None of the current global entries apply. This plan trains nothing and runs no distributed job:
`py-001`, `dist-001`, `dist-002`, `mem-001`, `train-001` and `train-002` all describe failures of
a training run.

#### From project catalog

- **poe-score-001**, scorer returns all zeros or NaNs despite valid inputs. Every point on the
  timing curve is a scored rate, so a degenerate scoring pass would draw a clean cliff that means
  nothing.
- **poe-score-002**, compose-rate stuck at 0.0 even though the fix is active. This is what the
  tail of the curve looks like when it is real, which is exactly why the failure mode has to be
  ruled out rather than assumed away. The review records that the detector disagrees with the
  pictures on seed 12, and that the eye read is the one cited there.
- **poe-launch-001**, SSH-direct launch dies instantly because relative paths resolve in `$HOME`
  rather than the repo. The 2.5-hour grid was started with `nohup` outside Slurm, which is exactly
  the launch path this entry describes.

---

**Auto-update note:** this section is regenerated by `/sync-plan-tree`. Do not edit it by hand;
changes are overwritten on the next sync.
