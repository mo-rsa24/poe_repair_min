# 🎯 Review: does this scope reach the paper, change a sentence, or close?

**Unanswered, and blocked until the other three verdicts exist.** This file judges
[../plans/gate-02-promote-or-close.md](../plans/gate-02-promote-or-close.md), which reads them
and writes the ending.

## Recommended prompt (when the three verdicts are in)

```
/sync-plan-tree plans/closing-the-compositional-gap/plans/can-we-trust-the-compose-rate
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/gate-02-promote-or-close.md) | the three promotion levels and what each one costs |
| **this file** | **the verdict: which level fired, and what follows from it** |
| [what it reads](instrument-01-the-three-state-labelled-set.md) | the two false-compose rates |
| [what it reads](idea-01-what-the-current-benchmarks-score.md) | the best agreement score |
| [what it reads](gate-01-is-this-hole-already-known.md) | whether the hole is already published |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The small promotion**: the band and the coverage number reach `writing-06` as proposed
  wording, plus a cap on F2's caption reading 94% as an upper bound. Wording only. No row in the
  root `MASTER_PLAN.md` paper table. This is what we expect to happen.
- **The big promotion**: a numbered step in the root `## Running order` and a new group-1 plan in
  `does-the-correction-cause-composition`. Earned only by a contaminated verdict at the 10-point
  bar or a candidate clearing the 95%-versus-85% bar.
- **Closure**: neither fires. The labelled set stays as a reusable instrument, the limitations
  paragraph still goes to `writing-06`, and that is written down as the finding.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Not a run: a decision.** Judged by whether it could have gone the other way, and by whether it
names its consequences rather than only its verdict.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

No compute. The inputs are the other three verdicts in this folder.

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the decision | Decision | | reading only, no GPU | the promotion level, and the handoffs it triggers | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Which promotion level fired, and with which numbers?
      Quote the deciding numbers with their denominators rather than summarising them: the two
      false-compose rates from `instrument-01`, the best agreement score from `idea-01`, and
      `gate-01`'s outcome. Exactly one of the three levels is named. This is the only question
      that moves anything.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ If the big promotion fired, what happens to `scorer_validated.json`?
      Re-certified or replaced, stated in one sentence. That file is the precondition
      `does-the-fix-reach-unseen-pairs` checks before it starts, so leaving this unstated would
      change a sibling scope's entry condition without saying so.
- [ ] ⚠️ If the big promotion fired, what happens to the runs already finished against the old
      certificate?
      `does-the-fix-reach-unseen-pairs` has completed runs scored by the current scorer. Say
      whether they stand, need re-scoring, or become bounded claims. An unanswered version of
      this question silently invalidates finished work.
- [ ] ⚠️ Did the wording reach `writing-06` under every outcome, including closure?
      The band on 94% belongs in the paper whether or not this scope produced anything else.
      Closure is not a reason to skip the handoff.
- [ ] ⚠️ Does the scope's status line in `MASTER_PLAN.md` match this decision?
      Promoted, wording-only, or closed. A scope whose master plan does not state its own ending
      reads as still running.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the decision itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible. Nothing yet: the decision has not been made.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** The three levels must all have been reachable. A decision
      where the numbers could only ever have produced the small promotion has not decided
      anything, and saying so is more useful than naming a level.
- [ ] ⚠️ **Was the instrument sound?** The deciding numbers come from three verdicts that must
      each be complete first. A level named from a partial verdict is a guess wearing a
      threshold. Confirm all three files carry answers before reading them.
- [x] ✅ **Did the run respect the environment?** Not applicable. No compute, no output directory,
      no flags. The inputs are three markdown files in this folder.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the band on the compose rate | it goes to `writing-06` under every outcome, closure included. This is the one handoff that does not depend on which level fired |
| F2's caption | the cap reading 94% as an upper bound, if the small promotion or better fired |
| any number produced before a re-certification | whether it stands, needs re-scoring, or becomes a bounded claim. Silence here invalidates finished work in a sibling scope without saying so |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the other three verdicts in this folder | the scope's ending, and its status line in `MASTER_PLAN.md`, which reads as still running until this is written |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Wait for [gate-01](gate-01-is-this-hole-already-known.md),
[instrument-01](instrument-01-the-three-state-labelled-set.md) and
[idea-01](idea-01-what-the-current-benchmarks-score.md) to carry answers, then write the ending.
