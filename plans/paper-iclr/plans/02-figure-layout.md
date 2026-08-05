# 🗺️ Which figure goes where, and which ones we do not have yet

## Description
Decide the paper's figure set: every slot, its section, its owning scope, and
whether the figure exists today or is owed by an unfinished run.

## Purpose
This is the phase-1 deliverable. It does two jobs at once: it fixes the paper's
visual argument, and its "owed" column becomes the run order for the two result
scopes. Without it, the figure work is prioritised by whatever finishes first
rather than by what the paper needs. Serves DoD 5.

## Goal
`paper/iclr/FIGURES.md`: a table with one row per figure slot, each carrying the
claim it makes, its section, its owning scope and plan file, and have-it or owed.

## Environment Facts This Plan Depends On
- This scope produces no figures. Production stays with the owning scopes:
  `animals-compose-transfer/plans/05-figures.md` (F2-F5) and
  `interaction-term/plans/10-figures.md` (the seven-figure cascade).
- What exists today: the pooled held-out read from
  `animals-compose-transfer/plans/03a-phase1-pooled.md` (out_out 0.96 at step
  60k). Everything in the interaction-term cascade is owed; only plan 00, the
  instruments, is complete.
- Two figures in the interaction-term cascade are illustrative, not measured
  (the three-regime diagram and the method schematic). They have ready-to-paste
  image prompts and do not depend on any run.

## Tasks
- [ ] ⚠️ list every figure slot the spine implies, one row per slot, claim named
- [ ] ⚠️ tag each row with its owning scope and plan file
- [ ] ⚠️ mark each row have-it or owed, and for owed, name the run that closes it
- [ ] ⚠️ split main text from appendix, and count pages against the ICLR limit
- [ ] ⚠️ order the owed rows by how much the paper needs them; that order is the
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
