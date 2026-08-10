# Review: is the pair pool clean enough to test transfer on?

**Answered, and the pool is in use.** This file judges
[../plans/instrument-01-the-clean-pair-pool.md](../plans/instrument-01-the-clean-pair-pool.md),
the gate the whole transfer claim starts behind. Every later run in this scope draws its pairs
from what is checked here.

## Words this file uses
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
**Not a run: an instrument.** Judged by whether its checks could have failed, not by what they
found.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| fail-rate scoring, 8 seeds per pair, instance-count scorer | Instrument | 2026-07-30 | `outputs/animals_compose_transfer/fail_rate.{json,md}` | done |

## The questions

- [x] ✅ Does a trusted scorer exist to measure any of this with?
      Yes. `outputs/compose_scorer/scorer_validated.json` records `pass: true` using the
      instance-counting read. The read it does NOT use is recorded in the same file: the
      embedding-distance reads were built, tested and rejected because they could not separate
      the hard pair. Without this file nothing in this scope may start.
- [x] ✅ Does any animal word repeat across pairs?
      No. 19 pairs, 38 distinct animals, and `pair_pool.py`'s overlap assertion passes. The pool
      is 15 blend-prone pairs, plus cat×dog as a known-failure reference, plus 3 pairs that
      compose fine as controls.
- [x] ✅ Does every training pair actually fail by default?
      Yes, and measured rather than eyeballed. Nine pairs fail on all 8 seeds. Donkey×pony fails
      on 0.75 of them and crocodile×alligator on 0.62, the two easiest in the pool. The held-out
      blend pairs are deliberately unscored here: scoring them would be running the transfer test
      early.

## A contradiction with the dose review, unresolved

`an_elephant__x__a_penguin` is listed here as one of three compose-by-default control pairs, and
that is what the do-no-harm check rests on. The dose review reports it scoring 0 of 4 at strength
0, with all four images single fused creatures:
[hypothesis-02 of the other claim](../../does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md).

Both cannot be true. Either the pool's assumption about that pair is wrong, or the two runs
measured it under conditions different enough to explain the gap. It is not resolved here, and it
is not resolved by preferring the more convenient answer. What follows from it either way: this
scope may have **no working do-no-harm control**, which is a limitations sentence owed by
`writing-06-mechanism-and-limitations`, and the fifteen-run sweep should not be read as
do-no-harm-clean until it is settled.
