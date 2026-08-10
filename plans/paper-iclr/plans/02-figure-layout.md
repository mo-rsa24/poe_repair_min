# 🗺️ Which figure goes where, and which ones we do not have yet

## What this asks, in one line
Place every figure slot from the register into a section, split main text from appendix against the page limit, and hand the owed rows back to the result scopes as their run order.

## Description
Decide the paper's figure set: every slot, its section, its owning scope, and
whether the figure exists today or is owed by an unfinished run.

## Purpose
This is the phase-1 deliverable. It does two jobs at once: it fixes the paper's
visual argument, and its "owed" column becomes the run order for the two result
scopes. Without it, the figure work is prioritised by whatever finishes first
rather than by what the paper needs. Serves DoD 5.

## Goal
The register `paper/iclr/figures.md` already holds the eight slots with claims and owners.
This plan's product is the LAYOUT on top of it: a section per slot, main text against appendix
within the page limit, and the owed rows ordered into the run order the result scopes follow.

## Environment Facts This Plan Depends On
- This scope produces no figures. Production stays with the owning scopes:
  `animals-compose-transfer/plans/05-figures.md` (A2-A5, feeding register slot F8) and
  `interaction-term/plans/10-figures.md` (the seven-figure cascade).
- What exists today: the pooled held-out read from
  `animals-compose-transfer/plans/03a-phase1-pooled.md` (out_out 0.96 at step
  60k). Everything in the interaction-term cascade is owed; only plan 00, the
  instruments, is complete.
- Two figures in the interaction-term cascade are illustrative, not measured
  (the three-regime diagram and the method schematic). They have ready-to-paste
  image prompts and do not depend on any run.

## Tasks
- [ ] check the register's eight slots against the spine: any claim the spine makes with no slot, any slot the spine never uses
- [ ] tag each row with its owning scope and plan file
- [ ] mark each row have-it or owed, and for owed, name the run that closes it
- [ ] split main text from appendix, and count pages against the ICLR limit
- [ ] order the owed rows by how much the paper needs them; that order is the
      run order handed back to the two result scopes

## Success/Failure Outcomes
- **the layout**
  - Success: every slot names one claim and one owner, and the owed rows are
    ordered. A reader of the table can say what the paper argues without the
    prose.
  - Failure: more main-text figures than the page limit allows, or two rows
    making the same claim. Cut to appendix rather than shrinking figures.
- **the run order**
  - Success: the two result scopes can start work from this table alone.
  - Failure: an owed row whose closing run is not named. That row is a wish,
    not a plan.

## Recommended skill
▶ `/plan-figures` ✅ to lock the narrative spine and the 3-6 load-bearing
   figures, reading from `plans/animals-compose-transfer/plans/05-figures.md`
   and `plans/interaction-term/plans/10-figures.md`.

## Engagement Instructions
```bash
cat paper/iclr/FIGURES.md      # expect one row per slot: claim, section, owner, have-it/owed
grep -c "owed" paper/iclr/FIGURES.md    # expect a nonzero count, each with a named run
```
