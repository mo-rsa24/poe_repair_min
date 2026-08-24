# 📊 The figures that carry the transfer argument

## Recommended prompt (after run completes)

After you finish this plan and want to ingest error patterns into the catalogs, use this prompt:

```
/ingest-error-pattern --from-run-log
```

This extracts error patterns from the run transcript, deduplicates against the global and project
catalogs, and adds new entries. New errors propagate to all affected plan files.

---

## Position in the plan tree

**Step 14 of 22.** Waits on steps 11 and 12. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 13 (previous) | [figure-01: the-seven-paper-figures](../../does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md) ◑ | The figures the causal scope owes the paper (F6 needs a decision) |
| **14 (current)** | **figure-01: the-transfer-figures** ◑ | **F8a and F8b built; the F8 pair waits on the leave-one-pair-out sweep** |
| 15 (next) | [gate-01: two-literature-checks-before-print](../../does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) ⚠️ | The two `/pressure-test` passes before anything is written |

---

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
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## What this asks, in one line

Turn the transfer runs into figures a reviewer reads in order: the fix arrives during training,
it reaches pairs the adapter never trained on, arrival and transfer come apart, and an
animals-only pool beats a size-matched mixed one.

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this plan is about**

Turning the transfer runs into figures a reviewer reads in order. Nothing here trains anything or
scores anything: every figure reads a file some earlier run already wrote.

**What the reader should end up believing**

Four things, one per figure, in reading order. The fix arrives during training. It reaches pairs
the adapter never trained on. Arrival and transfer come apart when they disagree. An animals-only
pool beats a size-matched mixed pool on the same held-out pairs.

**Words this plan uses**

The four owed figures carry short internal names. They are this plan's names only, and they are
not the paper's slot names.

- **A2, delivery-live:** how far the correction travelled toward its target, over training steps.
- **A3, transfer:** compose-rate against the fraction of pairs held out.
- **A4, delivery against transfer:** the same x-axis with compose-rate and direction-cosine as
  twin panels, so the two axes can be seen to diverge.
- **A5, pool contrast:** animals-only against size-matched mixed, per held-out animal pair.

**Oracle** names the correction computed from the joint prompt, the thing the adapter is trained
to imitate. It is the target, not a shippable method: at full dose it reproduces the joint render
by construction.

The paper's register calls the finished pair **F8**, and its two already-built neighbours **F8a**
and **F8b**. A3 and A5 are what F8 is assembled from.

**Where it stands**

Two figures are built and in the register as **built**: F8a (one pooled adapter composes on pairs
it never trained on) and F8b (the adapter against the oracle correction it imitates). Both read
files that already existed, so neither waited on this plan's dependencies. The four owed figures
have not started, because A2 needs step 9's live curves and A3, A4 and A5 need the fifteen-run
sweep from step 11 and the mixed pool from step 12.

**Associated materials**

- Figure register, which caps every caption: [paper/iclr/figures.md](../../../../../paper/iclr/figures.md), rows F8a, F8b, F8
- Sweep review file (feeds A3, A4): [review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../review/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md)
- Contrast review file (feeds A5): [review/baseline-01-the-size-matched-control-pool.md](../review/baseline-01-the-size-matched-control-pool.md)
- This plan's own record: [review/figure-01-the-transfer-figures.md](../review/figure-01-the-transfer-figures.md)

**For the full picture**

[Scope master plan](../MASTER_PLAN.md), Definition-of-Done item 5.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected effort**

No GPU and no queue. Each figure is a plotting script over a JSON file that already exists, so a
figure is minutes of compute and the time goes into the design decision before it. Budget one
`/design-figure` round per figure.

**Prerequisites**

A2 needs the three live curves from step 9 logged. A3 and A4 need the fifteen-run leaderboard from
step 11. A5 needs the mixed-pool contrast from step 12. A figure whose input file does not exist is
halted, not approximated.

**Environment Facts This Plan Depends On**

- The `co3` absolute python path, because the figure scripts run outside any activated shell. See
  [docs/ENVIRONMENT.md](../../../../../docs/ENVIRONMENT.md).
- Figure outputs land in `paper/iclr/figures/` inside the repo, which is on `/home-mscluster`. This
  is the one exception to the `/datasets` rule and it is safe because a figure is kilobytes, not
  gigabytes. The run outputs these figures read stay on `/datasets`.
- No system LaTeX exists here, so figure text is rendered by matplotlib's own mathtext, never by a
  LaTeX pass.

**The register caps the caption, not the result**

A caption may not claim past its register row's sentence. Where a result would support more, the
register row gets widened first, on evidence, and the caption follows.

**Known issues**

See the [Error Matrix](#error-matrix) section at the bottom of this file for the catalog of known
issues and their fixes.

---

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The transfer evidence becomes four reviewer-facing figures, one claim each, each designed through
`/design-figure` before it is built and each capped by its register row.**

**Why this matters right now:** step 15 is the literature check before anything is written, and
step 16 onward is the writing itself. A figure that does not exist by then is a claim the paper
cannot make.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap**

The runs from steps 11 and 12 produce leaderboards and JSON files. Nobody reads a leaderboard and
comes away believing a fix transfers. The evidence only becomes an argument once someone can see
it.

**The approach**

Four simple figures instead of one dense one, in reading order, each answering one question. The
form of each is decided through `/design-figure` before it is built, so the layout and the honest
limits are settled while the answer is still unknown.

**Key insights**

1. **Splitting delivery from transfer is the whole point.** A floor compose-rate means either the
   correction never arrived or it arrived pointing the wrong way, and those are different findings.
   A4 exists so the two can be seen apart rather than argued about.

2. **The pool contrast needs pairing, not averaging.** A5 is paired bars per held-out animal pair,
   because a mean over pairs hides which pairs the mixed pool actually loses on.

3. **A figure built before its data lands is a fabrication.** Each figure halts on a missing input
   file rather than plotting a partial one.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

Which run feeds which figure, and which of them exist today:

```
step 9   three live curves            ──────────────►  A2  delivery-live        (owed)
         (instrument-02)

step 10  pooled adapter run           ──┬───────────►  F8a one adapter          (BUILT)
         compose_rate.json             │
                                       └──┐
step  8  dose sweep                       ├────────►  F8b adapter vs oracle    (BUILT)
         dose_curves.json             ────┘

step 11  fifteen-run sweep            ──┬───────────►  A3  transfer            (owed)
         leaderboard + curve            └───────────►  A4  delivery vs transfer (owed)
                                          │
step 12  size-matched mixed pool      ────┴─────────►  A5  pool contrast        (owed)

                                       A3 + A5  ────►  register slot F8  (reserved)
```

The two built figures sit outside the A2 to A5 chain: both were assembled from files that already
existed, so neither waited on steps 11 and 12.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **A2, delivery-live.** Fraction-of-distance-reached on the y-axis, training step on the x-axis,
   one faint line per held-out pair, with the ~40% plateau drawn as a horizontal reference line.
   Reads the live curves logged by step 9. Saves to `paper/iclr/figures/`.

2. **A3, transfer.** Compose-rate on the y-axis, fraction of pairs held out on the x-axis. The
   do-no-harm baseline is drawn as a band rather than a line. Real held-out thumbnails are pinned
   at a couple of x positions, so the number has a picture beside it. Reads step 11's leaderboard.

3. **A4, delivery against transfer.** Compose-rate and direction-cosine as twin panels sharing one
   x-axis, so a run where the correction points right but under-delivers is visibly different from
   one where it points wrong. Reads the same leaderboard as A3.

4. **A5, pool contrast.** Paired bars, animals-only against size-matched mixed, one pair of bars
   per held-out animal pair, with the same-pair pairing drawn explicitly rather than implied by
   position. Reads step 12's contrast.

Each of the four is designed through `/design-figure` first, and the design note is what the gate
checks for.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose**

Serves Definition-of-Done item 5 of the scope: turn the run outputs into the reviewer-facing
evidence set. F1 belongs to the compose-scorer scope, not here.

**Goals**

1. A2, A3, A4 and A5 each exist as a non-empty figure file with a sidecar JSON holding the numbers
   drawn.
2. Each was designed through `/design-figure` before being built, evidenced by a design note per
   figure.
3. Register slot F8 is assembled from A3 and A5 and its row flips from `reserved` to `built`.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 1. 📊 Figures already built from files that existed

- [x] **1.1** F8a, one pooled adapter composes on pairs it never trained on: two panels on a
  training-step axis, the second unpooled per pair so the aggregate cannot be carried by one easy
  pair.
  - Built by `python scripts/make_f8.py`, reads the pooled run's `compose_rate.json` and
    `pair_pool.json`
  - Outputs `paper/iclr/figures/compose-rate-as-the-lora-trains.{png,pdf,json}`
  - Which commit built it, and whether it has been judged: the
    [review file](../review/figure-01-the-transfer-figures.md)
- [x] **1.2** F8b, the shippable adapter against the oracle correction it imitates: one horizontal
  dot row per pair, four markers per row.
  - Built by `python scripts/make_f8b.py`, reads `dose_curves.json` and the pooled run's
    `compose_rate.json`, no new runs
  - Outputs `paper/iclr/figures/compose-rate-by-pair-for-lora-against-the-joint-prompt-correction.{png,pdf,json}`
  - Which commit built it, and whether it has been judged: the
    [review file](../review/figure-01-the-transfer-figures.md)

▶ **Next: instruction 3.1**, the eye check both figures still owe.

### 2. 📊 The four figures this plan owes

◀ **Needs: steps 11 and 12** to have landed, so the leaderboard and the contrast exist to plot.

- [ ] **2.1** **[needs /design-figure]** A2 delivery-live: fraction-of-distance-reached over
  training, one faint line per held-out pair, the ~40% plateau drawn as a reference line.
- [ ] **2.2** **[needs /design-figure]** A3 (A) transfer: compose-rate vs fraction held out,
  do-no-harm baseline as a band, real held-out thumbnails pinned at a couple of points.
- [ ] **2.3** **[needs /design-figure]** A4 (A) real: compose-rate and direction-cosine as
  twin panels sharing an x-axis, so divergence (delivery vs transfer) is visible.
- [ ] **2.4** **[needs /design-figure]** A5 (B) pool: paired bars, animals vs mixed, per
  held-out animal pair, same-pair pairing explicit.

▶ **Next: instruction 3.1** for each figure as it lands, then the engagement gate.

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them. A figure is judged by eye, and that judgment is not something a script can make.

### 3. 👁️ Judge each figure and cap its caption

◀ **Needs: task 1.1, 1.2, or any of 2.1 to 2.4** done, so there is a PDF to open.

3.1 **Open the PDF and read it cold**
   - Open `paper/iclr/figures/<figure>.pdf` in a viewer, not the PNG in a chat window
   - Read only the figure and its axis labels, without the caption
   - Expected result: you can say what the figure claims in one sentence, without help
   - ✅ the sentence you say matches the row's sentence in
     [paper/iclr/figures.md](../../../../../paper/iclr/figures.md); mark verified
   - ❌ you cannot say it, or you say something the row does not claim; the figure goes back to
     `/design-figure` rather than getting a longer caption

3.2 **Check the caption against its register row**
   - Open `paper/iclr/figures.md`, find the row for this figure
   - Compare the row's claim sentence against what the figure actually shows
   - ✅ the caption claims at or below the row; mark verified
   - ❌ the figure supports more than the row says; widen the register row first, on evidence, then
     write the caption. Never widen the caption alone

3.3 **Flip the register row's status**
   - Edit the figure's row in `paper/iclr/figures.md`, status column, `reserved` → `**built**`
   - Add the caption caps in the same row: what the figure cannot be read to say
   - Expected result: the row names the output path, the build command, and the sidecar JSON

▶ **Next: instruction 4.1**, recording the verdict where the record lives.

### 4. 📋 Record the verdict in the review file

◀ **Needs: instruction 3.3** done, so the register and the figure agree.

4.1 **Answer the plan's review questions**
   - Open [review/figure-01-the-transfer-figures.md](../review/figure-01-the-transfer-figures.md)
   - Answer the pre-registered question for the figure that just landed
   - Record which run and which checkpoint step the figure read, beside the answer
   - ✅ every question for a built figure is answered or explicitly marked unknown with a next
     action named
   - ❌ a built figure with no answered question is the state this convention exists to prevent:
     the work is paid for and the value is uncollected

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters**: steps 15 and 16 write the paper against these figures. A figure
> that does not exist, or exists but was never judged by eye, becomes a claim the paper cannot
> support.

- **Pass criteria**
  - Each of A2 to A5 exists as a non-empty figure file with its sidecar JSON
  - Each was designed through `/design-figure` before building, evidenced by a design note per
    figure
  - A script asserts the four figure files are present and non-empty
  - Each built figure has been read cold (instruction 3.1) and its register row flipped (3.3)

- **Fail criteria (STOP)**
  - A figure's underlying data is missing, for example the leaderboard from step 11 or the contrast
    from step 12 has not landed. Halt that figure. Do not fabricate a plot from absent data.

- **Partial pass guidance**
  - Figures whose inputs exist may land while the rest wait. F8a and F8b are exactly this case:
    both were built from files that already existed, well before steps 11 and 12.
  - Record which figures are held and on what, in the review file, so the gap is stated rather than
    implied.

**When you get results, answer** [the review file](../review/figure-01-the-transfer-figures.md)
**or move to** [Next step](#next-step).

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

These are result plots, not diagrams of a system, so the subject-versus-process split that the
Lane column carries does not apply to them. The column is filled `—` deliberately rather than
guessed.

### Built

| Item | Lane | What it shows | Built by | Output | Register row |
|------|------|---------------|----------|--------|--------------|
| F8a one adapter transfers | — | Compose rate over training steps, trained-on against held-out, with the uncorrected floor; second panel unpooled per pair | `python scripts/make_f8.py` | `paper/iclr/figures/compose-rate-as-the-lora-trains.{png,pdf,json}` | F8a, **built** |
| F8b adapter against the oracle | — | One dot row per pair: no injection, oracle at two doses, adapter at step 60000 | `python scripts/make_f8b.py` | `paper/iclr/figures/compose-rate-by-pair-for-lora-against-the-joint-prompt-correction.{png,pdf,json}` | F8b, **built** |

Numbers for both live in their sidecar `.json` and in the register row, never in this plan.

### Owed

| Item | Lane | What it shows | Reads | Status |
|------|------|---------------|-------|--------|
| A2 delivery-live | — | Fraction-of-distance-reached over training, one line per held-out pair, ~40% plateau as reference | Step 9's live curves | ⏳ not started |
| A3 transfer | — | Compose-rate against fraction held out, do-no-harm band, thumbnails pinned | Step 11 leaderboard | ⏳ blocked on step 11 |
| A4 delivery against transfer | — | Compose-rate and direction-cosine as twin panels on one x-axis | Step 11 leaderboard | ⏳ blocked on step 11 |
| A5 pool contrast | — | Paired bars, animals against size-matched mixed, per held-out pair | Step 12 contrast | ⏳ blocked on step 12 |
| F8 (the register slot) | — | Leaderboard plus degradation curve, assembled from A3 and A5 | A3 and A5 | ⏳ reserved |

#### Organization workflow

1. Design each owed figure through `/design-figure` and keep the design note.
2. Build it once its input file exists.
3. Judge it by eye and flip its register row (instructions 3.1 to 3.3).
4. Answer its review question (instruction 4.1).

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

**Why this matters**: three files can disagree about one figure. The plan says what to build, the
register says what may be claimed, and the review file says what was found. Keeping them in step is
what stops a caption drifting past its evidence.

**After completion**:

1. **Capture learnings and errors**: run `/ingest-error-pattern --from-run-log`.
2. **Update this plan file**: `/sync-plan-tree` reads the catalogs and regenerates the Error Matrix,
   and strikes through the task lines that landed.
3. **Update the register**: flip the row's status and write its caption caps (instruction 3.3).
4. **Answer the review file**: the verdict lives there, never in this plan (instruction 4.1).

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Capture patterns | `/ingest-error-pattern --from-run-log` | Manual, after completion | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Automatic, by ingest-error-pattern | This file's Error Matrix regenerated |
| Flip the register row | Manual edit of `paper/iclr/figures.md` | Manual, instruction 3.3 | The caption's ceiling matches the figure |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** [scripts/make_f8.py](../../../../../scripts/make_f8.py)
**Function:** `main`
**What it does:** reads the pooled run's `compose_rate.json` and `pair_pool.json`, draws the two
panels, writes PNG, PDF and a sidecar JSON holding every number drawn plus its caption caps.

```python
# outline
src  = RUN / "compose_rate.json"      # scored pooled run
pool = RUN / "pair_pool.json"         # which pairs were trained on
# panel (a): trained-on vs held-out compose rate over steps, PoE floor dotted
# panel (b): the held-out curves unpooled, one per pair
fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)   # OUT_DIR = paper/iclr/figures
```

**File:** [scripts/make_f8b.py](../../../../../scripts/make_f8b.py)
**Function:** `main`
**What it does:** reads `dose_curves.json` for the oracle arm and the pooled run's
`compose_rate.json` for the adapter arm, draws one dot row per pair, writes the same three outputs.
No new runs.

**Not yet written:** the four scripts behind A2 to A5. Each is authored after its `/design-figure`
round, not before.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/design-figure` ✅: designs each figure (form, computation, honest limits) before
   it's built. alt: `/plan-figures` to sequence the A2–A5 set as one coherent cascade.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

Step 15, [gate-01: two literature checks before print](../../does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md):
the two `/pressure-test` passes that run before any of this is written up.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

**Purpose**: known issues and their fixes. Regenerated by `/sync-plan-tree --update-error-matrices`
after `/ingest-error-pattern` appends new patterns.

#### From global catalog

None of the current global entries apply. This plan runs no training, no distributed job, and no
GPU work: `py-001`, `dist-001`, `dist-002`, `mem-001`, `train-001` and `train-002` all describe
failures of a training run, and a plotting script over an existing JSON file cannot hit them.

#### From project catalog

- **poe-score-001**, scorer returns all zeros or NaNs despite valid inputs. Relevant because every
  figure here plots compose-rate straight out of a scored run's JSON. A figure drawn over a
  degenerate scoring pass looks like a null result. Check the sidecar's counts before believing a
  flat curve.
- **poe-score-002**, compose-rate stuck at 0.0 even though the fix is active. Same reason: it is
  the failure that most looks like a real finding when plotted.
- **poe-lora-001**, fraction-of-distance-reached plateaus at 20% instead of the expected 40%.
  Directly relevant to A2, whose reference line is drawn at the plateau.
- **poe-lora-002**, direction-cosine diverges below 0.3 despite low training loss. Directly
  relevant to A4, which plots direction-cosine against compose-rate to tell delivery-null from
  no-transfer.

---

**Auto-update note:** this section is regenerated by `/sync-plan-tree`. Do not edit it by hand;
changes are overwritten on the next sync.
