# ⚖️ Promote this, or close it

Design only. The decision lives in [../review/gate-02-promote-or-close.md](../review/gate-02-promote-or-close.md).

## What this asks, in one line
Given what the other three plans found, does this scope earn a numbered step in the paper, a
wording change only, or nothing?

## Why this plan exists
A scope that tries an idea has to be able to end. Without a plan whose whole job is to write
the ending, the likely outcome is that the work goes quiet rather than concluding, and the
labelled set sits unused while the paper still prints an unqualified 94%.

## Description
Read the three verdicts, judge them against the two promotion levels in the master plan, and
write the decision. If the big level fires, this plan also states what happens to the certified
scorer and to the sibling scope's finished runs, which is the part most likely to be skipped.

## Purpose
Serves Objective 4 and Definition-of-Done item 9.

## Goal
A decision in `review/gate-02-promote-or-close.md` naming which promotion level fired, with the
numbers that decided it, and the consequence for `scorer_validated.json` written out.

## Environment Facts This Plan Depends On
- Runs in session. No GPU, no queue, no disk.
- Touches `outputs/compose_scorer/scorer_validated.json` only by describing what should happen
  to it. Any actual re-certification is a separate plan, because
  `does-the-fix-reach-unseen-pairs` reads that file as a precondition before it starts.

## Success/Failure Outcomes
- **The small level fires.** The band and the coverage number go to `writing-the-paper` as
  proposed wording for `writing-06`, plus a cap on F2's caption. Wording only, no paper-table
  row. This is what we expect.
- **The big level fires.** A numbered step in the root `## Running order` and a group-1 plan in
  `does-the-correction-cause-composition`. This plan must then also say whether the winning
  metric re-certifies or replaces `scorer_validated.json`, and what that means for the runs
  `does-the-fix-reach-unseen-pairs` has already finished against the old certificate. Leaving
  that unstated would silently invalidate a sibling scope's results.
- **Neither fires.** The scope closes. The labelled set stays as a reusable instrument and the
  limitations paragraph still goes to `writing-06`. Written down as the finding, not left as an
  absence.

## Tasks
- [ ] Read the three verdicts: `gate-01`'s literature outcome, `instrument-01`'s two rates
  and coverage number, `idea-01`'s agreement table.
- [ ] Judge against the two promotion levels in `MASTER_PLAN.md`, quoting the numbers that
  decided it rather than summarising them.
- [ ] If the big level fired: state re-certify or replace for `scorer_validated.json`, name
  the consequence for `does-the-fix-reach-unseen-pairs`, and open the group-1 plan in
  `does-the-correction-cause-composition`.
- [ ] Hand the wording to `writing-the-paper` either way, because the band belongs in
  `writing-06` under every outcome.
- [ ] Mark the scope's status in `MASTER_PLAN.md`: promoted, wording-only, or closed.

## Engagement Instructions
The decision file names exactly one of the three outcomes and quotes the deciding numbers with
their denominators. If the big level fired, it contains a sentence on `scorer_validated.json`
and a sentence on the sibling scope's finished runs; absent either, the decision is incomplete.
The handoff to `writing-06` is present under every outcome, including closure. The scope's
status line in `MASTER_PLAN.md` matches the decision.

STOP: any of the three upstream verdicts missing → halt. A promotion decision made on two of
three inputs is a guess.

## Recommended skill
▶ `/defend-results` ✅: attack the promotion call before the supervisor does, especially the
   claim that a 10-point gap means the curve is contaminated rather than that the labelled set
   is small. alt: `/triage-plan` if the big level fires and the follow-on work needs routing.
