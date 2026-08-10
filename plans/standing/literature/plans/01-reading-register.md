# 📚 The reading register

## Description
One register of every paper read for this project, with what we take from it and
which plan it touches. Re-entered whenever a paper is read, a new idea needs a
source, or the paper's claim changes.

## Purpose
Two failure modes this prevents. Designing an experiment for a question the field
already answered, which costs GPU-days and draws "we already know this". And
trying a method with no source, which produces a number nobody can defend.

## Goal
`docs/reading-register.md`: one row per paper, kept current, readable by the
related-work section without further work.

## Environment Facts This Plan Depends On
- No GPU and no queue. Nothing here is submitted as a job.
- Network access for paper fetches, available on the compute nodes.

## Tasks
- [ ] ⚠️ Create `docs/reading-register.md` with the columns: paper (arXiv id),
      claim, what it proves, what we borrow, plans it touches, date read.
- [ ] ⚠️ Back-fill it from the seven papers already reconciled on the
      interaction-term question, so the register starts current rather than empty.
- [ ] ⚠️ Standing: whenever a run in `PARKING_LOT.md` tries an idea, add or cite
      the register row it came from. A tried idea with no row is the thing this
      plan exists to catch.
- [ ] ⚠️ Standing: run `/pressure-test` on the paper's headline claim whenever
      that claim changes, and record the verdict as a register row.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|

No runs. Nothing in this scope executes.

## Success/Failure Outcomes
- **The register is current**
  - Success: every paper cited in the manuscript has a row, and every idea-trying
    run in `PARKING_LOT.md` names one.
  - Failure: a row exists for a paper nobody read, or a run cites no source. Both
    mean the register has become decoration.

## Recommended skill
▶ `/paper-scout` to find work · `/unpack-paper` to read one properly ·
`/pressure-test` for the novelty verdict.

## Engagement Instructions
```bash
# every idea-trying run in the parking lot should name a register row
grep -c "^|" docs/reading-register.md          # rows in the register
grep -n "tries an idea" PARKING_LOT.md         # runs that owe a source
```
