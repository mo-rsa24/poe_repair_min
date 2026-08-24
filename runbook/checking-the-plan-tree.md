# Checking the plan tree

Navigation: 📋 [Index](00-INDEX.md)

Contents: [1. Run the state check](#1-run-the-state-check)

## 1. Run the state check

`ran 2026-08-24`

The four-check report-only pass named in `CLAUDE.md`'s "What the conventions look like in this
repo": stale in-flight claims, unharvested output, orphaned plan files, and jargon that won't
read cold. About 7 seconds over the whole tree. The session-start hook already runs checks 1
and 2 automatically; run this by hand for checks 3 and 4, or any time after a sweep that moved
plan files.

```bash
python3 scripts/plan_pulse.py
```

✅ Real output, 2026-08-24, 180 plan files:

```
plan pulse: 180 plan files under plans

STALE: task lines claiming a run is in flight with nothing in flight (1)
  plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md:20

UNHARVESTED: output newer than the plan that owns it (0)

ORPHAN: markdown no task names (5)
  plans/retrofit-poe-repair-min.md
  plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/assets/F1-schematic-prompt.md
  plans/closing-the-compositional-gap/plans/can-we-trust-the-compose-rate/claim-under-test.md
  plans/closing-the-compositional-gap/diagrams/process-versions/01-2026-08-18.md
  plans/closing-the-compositional-gap/diagrams/process-versions/02-2026-08-19.md

DEBRIS: narration a plan should not carry (6)
  [six lines, in plans/standing/artifact-reconciliation/inventory/, superseded-decision narration]

RUNSTATE: run state in a design plan, belongs in its review file (0)

UNJUDGED: a finished run with review questions still unanswered (3)
  [three review files, closing-the-compositional-gap scope, listed with their open question lines]

JARGON: design plans that will not read cold (4)
  [four plan files, missing a plain opener or carrying over-long sentences]
```

**Reading the report.** `STALE` and `RUNSTATE` are almost always real: fix the plan line. `ORPHAN`
and `DEBRIS` are usually intentional (a standing reference document, a superseded-and-marked
section) rather than a defect; check each by name before touching it. `UNJUDGED` is a pointer to
review files with open questions, which is `verify-plan`'s and the scope owner's job to close, not
this check's. `JARGON` flags plans for a `/plain-speak` pass.

No ❌ branch: this is a read-only report and has no failure mode beyond the interpreter or the
script itself not running.
