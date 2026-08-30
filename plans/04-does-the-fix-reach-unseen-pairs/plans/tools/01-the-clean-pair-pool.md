# 🧬 A clean pool, built only once the scorer is trusted

Build the set of animal pairs every later step in this scope is measured on, and start only
after the compose-scorer has been proven to work.

**Step 3 of 22.** Waits on nothing. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 2 | ~~[fixing the size measure before any result](../../../03-does-the-correction-cause-composition/plans/tools/02-fix-the-size-measure-before-any-result.md)~~ | ✅ |
| **3** | **this plan** | **✅** |
| 4 | [03-more-correction-more-composition](../../../03-does-the-correction-cause-composition/plans/hypothesis/03-more-correction-more-composition.md) | ◑ 6.3GB owed |

Design only. Verdicts live in [the review file for this plan](../../review/01-the-clean-pair-pool.md).

## What this asks, in one line
Build the set of animal pairs this whole claim runs on. Three conditions, each closing a
specific hole:

- **No animal word repeats anywhere in the pool.** Otherwise a transfer win could just be
  the model having seen that animal already.
- **Every training pair is proven to fail by default**, measured over eight seeds rather
  than judged by eye, so there is something for the fix to fix.
- **Nothing starts until the validated scorer exists.** Without a trusted scorer every
  number this claim produces is unreadable.

## Why this plan exists
Leaving one pair out is only a fair transfer test if no animal word repeats across
pairs. If "wolf" appeared in a training pair and in the held-out pair, the model would
have seen the concept, so a compose would prove nothing. This plan builds a pool where
no animal word is shared between pairs, and confirms each pair actually fails by
default, so the test in plan 03 starts from clean ground.

> Held-out: the pair the model is deliberately never trained on, kept back so it can be
> tested on something genuinely new.

## Description
The pool the scope runs on, plus the condition that has to hold before the scope may
start at all. First, check what the sibling scope owes this one: the compose-scorer must
have written
`scorer_validated.json` with a passing verdict. Without it there is no trusted scorer,
so the scope halts. Then curate ~15 blend-prone animal×animal pairs where no animal
word repeats across pairs (~15 pairs ≈ ~30 distinct animals). Finalise `pair_pool.yaml`
by confirming each training pair fails by default, measured as a fail-rate over 8 seeds
by the compose-scorer, not by eye. Keep a few compose-by-default pairs as a do-no-harm
control.

## Purpose
Serves Objective 1 (Pool) and Definition-of-Done item 1. It is also the way in: this is
where an unattended run enforces the dependency on the compose-scorer.

## Goal
`pair_pool.yaml` finalised with no animal word shared between pairs: ~15 blend-prone
animal pairs plus a few controls, each training pair's fails-by-default rate over 8 seeds
recorded, and none of it started until `scorer_validated.json` says the scorer passed.

The pool on disk: `artifacts/results/does-the-fix-reach-unseen-pairs/{pair_pool.yaml, pair_prompts.yaml}`,
19 pairs (15 blend-prone, cat×dog as the known-failure reference, 3 compose-by-default
controls), 38 distinct animals, no word repeated.

## Environment Facts This Plan Depends On
- `co3` python at its absolute path. The fail-rate scoring pass (8 seeds × 19 pairs) fit the
  in-session GPU; no biggpu/bigbatch request was needed for a set this size.
- Output lives at `artifacts/results/does-the-fix-reach-unseen-pairs/{pair_pool.yaml,
  pair_prompts.yaml, fail_rate.{json,md}}`, moved there from the old `outputs/` location by an
  earlier retrofit pass.

## Tasks
- [x] Precondition check: assert `scorer_validated.json` (from artifacts/plans/completed/compose-scorer)
  exists and its pass flag is true.
- [x] Curate the first-draft pair list: 15 blend-prone animal×animal pairs,
  token-disjoint. Candidate list + blend rationale recorded in pair_pool.yaml comments.
- [x] Write `pair_pool.yaml` and confirm it loads through
  `pair_pool.py` with its built-in train/held-out overlap assertion passing (exit 0).
  Also passed a stronger animal-token-disjointness + prompt-coverage check.
- [x] Score each pair fails-by-default over 8 seeds with the compose-scorer (fail-
  RATE, not eyeball). Keep pairs above the fail-rate threshold as training pairs;
  set aside a few compose-by-default pairs as the do-no-harm control.
 

## Engagement Instructions
MUST PASS before this counts as done, checked by a script with nobody watching: (a)
`scorer_validated.json` exists AND its pass flag is true; (b) `pair_pool.yaml` loads and
the pair_pool.py overlap assertion passes (exit 0); (c) each training pair's recorded
fail-rate is above threshold and each control pair's is below.
STOP: if `scorer_validated.json` is missing or its verdict is not pass → HALT, because
the compose-scorer this depends on has not delivered. If fewer than 10 blend-prone pairs
with no shared animal word survive scoring → halt, since the pool is then too small to
leave one pair out. If a pair meant for training turns out to compose by default → drop
it, then check again that at least 10 remain.

## Recommended skill
▶ `/run-experiment` ✅: drives the fail-rate scoring pass over 8 seeds and confirms
   the pool. alt: `/debug-config` for the pair_pool.yaml load + overlap assertion.
