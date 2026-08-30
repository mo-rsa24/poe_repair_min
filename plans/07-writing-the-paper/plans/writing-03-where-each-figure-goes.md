# 🗺️ Which figure goes where, and which ones we do not have yet

This plan asks which section each of the paper's figures belongs in, which of them exist already,
and what run has to finish before each missing one can be drawn.

**Step 19 of 22.** Waits on steps 13 and 14. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 18 | [the results skeleton](writing-05-the-results-skeleton.md) | ⚠️ |
| **19** | **this plan** | **⚠️** |
| 20 | [method and introduction](writing-04-method-and-introduction.md) | ⚠️ |

## What this asks, in one line
Give every figure the register has reserved a place for its section, split main text from appendix against the page limit, and hand the owed rows back to the result scopes as their run order.

## Description
Decide the paper's figure set. Each reserved place gets its section, its owning
scope, and a mark saying whether the figure exists today or is owed by an
unfinished run.

## Purpose
This is the phase-1 deliverable. It does two jobs at once: it fixes the paper's
visual argument, and its "owed" column becomes the run order for the two result
scopes. Without it, the figure work is prioritised by whatever finishes first
rather than by what the paper needs. Serves DoD 5.

## Goal
The register `paper/iclr/figures.md` already holds the eight reserved places with claims and
owners. This plan's product is the LAYOUT on top of it: a section per reserved place, main text
against appendix within the page limit, and the owed rows ordered into the run order the result
scopes follow.

## Environment Facts This Plan Depends On
- This scope produces no figures. Production stays with the owning scopes:
  `does-the-fix-reach-unseen-pairs/plans/figure-01-the-transfer-figures.md` (A2 to A5, which fill F8 in the register) and
  `does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md` (the seven-figure cascade).
- What exists today: the pooled held-out read from
  `does-the-fix-reach-unseen-pairs/plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md`
  (out_out 0.96 at step 60k). Everything in the does-the-correction-cause-composition cascade is
  owed, and only plan 00, which builds the measuring tools, is complete.

  > Held-out pairs are the animal pairs the adapter never trained on. A held-out number is the
  > compose rate measured on those pairs only, so it says whether the fix reaches beyond what it
  > was fitted to.

- Two figures in the does-the-correction-cause-composition cascade are drawings rather than
  measurements (the three-regime diagram and the method schematic). They have ready-to-paste
  image prompts and do not depend on any run.

## Tasks
- [ ] check the register's eight reserved places against the section order, looking for a claim
      the order makes with no figure reserved for it, and a reserved place the order never uses
- [ ] tag each row with its owning scope and plan file
- [ ] mark each row have-it or owed, and for owed, name the run that closes it
- [ ] split main text from appendix, and count pages against the ICLR limit
- [ ] order the owed rows by how much the paper needs them; that order is the
      run order handed back to the two result scopes

## Success/Failure Outcomes
- **the layout**
  - Success: every reserved place names one claim and one owner, and the owed rows are
    ordered. A reader of the table can say what the paper argues without the
    prose.
  - Failure: more main-text figures than the page limit allows, or two rows
    making the same claim. Cut to appendix rather than shrinking figures.
- **the run order**
  - Success: the two result scopes can start work from this table alone.
  - Failure: an owed row whose closing run is not named. That row is a wish
    rather than a plan.

## Recommended skill
▶ `/plan-figures` ✅ to lock the order the story is told in and the 3 to 6 load-bearing
   figures, reading from `plans/04-does-the-fix-reach-unseen-pairs/plans/figure-01-the-transfer-figures.md`
   and `plans/03-does-the-correction-cause-composition/plans/figure-01-the-seven-paper-figures.md`.

## Engagement Instructions
```bash
cat paper/iclr/FIGURES.md      # expect one row per slot: claim, section, owner, have-it/owed
grep -c "owed" paper/iclr/FIGURES.md    # expect a nonzero count, each with a named run
```
