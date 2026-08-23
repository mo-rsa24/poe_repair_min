# 🔬 Review: is the pair pool clean enough to test transfer on?

**Answered, and the pool is in use.** This file judges
[../plans/instrument-01-the-clean-pair-pool.md](../plans/instrument-01-the-clean-pair-pool.md),
the gate the whole transfer claim starts behind. Every later run in this scope draws its pairs
from what is checked here.

One thing checked here is contradicted by a sibling claim's run, and it is not settled. See
[Still open](#still-open) before reading any do-no-harm result out of this scope.

## Recommended prompt (if the pool changes)

```
/verify-plan ../plans/instrument-01-the-clean-pair-pool.md
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/instrument-01-the-clean-pair-pool.md) | how the pool is built, and the checks it has to survive |
| **this file** | **the verdict: the pool is clean, with one control pair in dispute** |
| [what it unblocks](hypothesis-01-does-one-pooled-fix-transfer-at-all.md) | every run in this scope draws its pairs from here |

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

- **The pool**: the set of animal pairs this claim runs on, split into pairs the fix trains on
  and pairs it is tested on.
- **No word repeats**: no animal name appears in more than one pair anywhere in the pool. Without
  that, a transfer win could just be the model having seen that animal during training.
- **Fails by default**: the pair produces one blended animal, not two, when nothing is corrected.
  Measured as a rate over eight starting seeds rather than judged by eye, because there has to be
  something for the fix to fix.
- **Control pairs**: pairs that compose fine with no correction at all. They are here to catch a
  fix that breaks things that already worked.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Not a run: an instrument.** Judged by whether its checks could have failed, not by what they
found. A failure here blocks every plan in the scope rather than closing one.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| fail-rate scoring, 8 seeds per pair, instance-count scorer | Instrument | 2026-07-30 | 8 seeds × 19 pairs | `outputs/animals_compose_transfer/fail_rate.{json,md}` | done |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ Does a trusted scorer exist to measure any of this with?
      Yes. `outputs/compose_scorer/scorer_validated.json` records `pass: true` using the
      instance-counting read. The read it does NOT use is recorded in the same file: the
      embedding-distance reads were built, tested and rejected because they could not separate
      the hard pair. Without this file nothing in this scope may start.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Does any animal word repeat across pairs?
      No. 19 pairs, 38 distinct animals, and `pair_pool.py`'s overlap assertion passes. The pool
      is 15 blend-prone pairs, plus cat×dog as a known-failure reference, plus 3 pairs that
      compose fine as controls.
- [x] ✅ Does every training pair actually fail by default?
      Yes, and measured rather than eyeballed. Nine pairs fail on all 8 seeds. Donkey×pony fails
      on 0.75 of them and crocodile×alligator on 0.62, the two easiest in the pool. The held-out
      blend pairs are deliberately unscored here: scoring them would be running the transfer test
      early.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible.

- [ ] 🟡 Does `an_elephant__x__a_penguin` actually compose by default? Raised by the dose sweep
      scoring it 0 of 4 at strength 0. Unresolvable from the data in hand, which is why it sits
      in [Still open](#still-open) rather than being answered here.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Not applicable in the usual sense: nothing is compared here.
      The check that stands in for it is that the held-out pairs were deliberately left unscored,
      so building the instrument did not leak the transfer test.
- [ ] 🟡 **Was the instrument sound?** The scorer is validated by `scorer_validated.json` and the
      fail rates are measured over eight seeds rather than eyeballed. What is not sound is the
      control-pair assumption: one of the three control pairs scores as a failure elsewhere.
- [x] ✅ **Did the run respect the environment?** Output landed at
      `outputs/animals_compose_transfer/fail_rate.{json,md}`, and the overlap assertion in
      `pair_pool.py` ran and passed rather than being skipped on an empty pool.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the fix does no harm to pairs that already compose | that this scope may have no working do-no-harm control, per the contradiction below. Owed by `writing-06-mechanism-and-limitations` |
| every training pair fails by default | that two pairs do not fail on every seed: donkey×pony at 0.75 and crocodile×alligator at 0.62 |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

`an_elephant__x__a_penguin` is listed here as one of three compose-by-default control pairs, and
that is what the do-no-harm check rests on. The dose review reports it scoring 0 of 4 at strength
0, with all four images single fused creatures:
[hypothesis-02 of the other claim](../../does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md).

Both cannot be true. Either the pool's assumption about that pair is wrong, or the two runs
measured it under conditions different enough to explain the gap. It is not resolved here, and it
is not resolved by preferring the more convenient answer.

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether elephant×penguin composes by default | re-scoring that pair under both runs' conditions and finding what differs between them | the do-no-harm claim, the limitations sentence in `writing-06-mechanism-and-limitations`, and reading the fifteen-run sweep as do-no-harm-clean |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Re-score `an_elephant__x__a_penguin` under both runs' conditions, and record the answer on both
sides of the contradiction rather than only here.
