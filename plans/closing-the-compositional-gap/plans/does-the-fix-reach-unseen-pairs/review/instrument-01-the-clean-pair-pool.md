# 🔬 Review: the pool and the precondition

Verdicts for [../plans/instrument-01-the-clean-pair-pool.md](../plans/instrument-01-the-clean-pair-pool.md).

## Run kind
**Tests the claim** (the entry gate: no trusted scorer, no scope).

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| fail-rate scoring, 8 seeds per pair, instance-count scorer | Tests the claim | 2026-07-30 | `outputs/animals_compose_transfer/fail_rate.{json,md}` | done |

## The questions

- [x] ✅ Does the trusted scorer exist? Yes: `outputs/compose_scorer/scorer_validated.json`,
      pass=true, instance-count read (the embedding reads were rejected; the contract records why).
- [x] ✅ Is the pool token-disjoint? Yes: 19 pairs, 38 distinct animals, no animal word repeats,
      `pair_pool.py`'s overlap assertion passes. 15 blend-prone rotation pairs, cat×dog as the
      known-failure reference, 3 compose-by-default controls.
- [x] ✅ Does every training pair actually fail by default? Yes, measured, not eyeballed: 9 pairs
      at fail-rate 1.00 over 8 seeds, donkey×pony 0.75, crocodile×alligator 0.62. Controls kept
      aside. Held-out blend pairs are unscored by design: they are the transfer test.
