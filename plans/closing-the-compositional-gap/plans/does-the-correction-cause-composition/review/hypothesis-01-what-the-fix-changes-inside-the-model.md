# Review: what does the fix change inside the model?

**Answered, and it replicated, with one caveat the paper owes.** This file judges
[../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md](../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md).
Its answers fill register slot **F7**, and the caveat below is what caps that figure's caption.

The question in plain terms: every word in the prompt decides both *where* in the image to look
and *what* to write there. Our account of why the fix works says it changes the second, not the
first. This is whether that survives 64 cells the fix never trained on.

## Words this file uses
- **A cell**: one animal pair at one starting seed. 8 pairs by 8 seeds gives the 64.
- **Where it looks / what it paints**: the two maps captured per word, with the adapter off and
  again on, from the identical starting state.
- **Pattern, not brightness**: the two maps are not on a common scale, and the adapter dims one of
  them by about 25% overall. So the comparison strips out uniform brightness and compares only
  the change a rescaling cannot explain. The design plan explains why the obvious comparison
  gives the opposite answer.
- **The ratio**: how much more the painted content's pattern moved than the attention's. Above 1
  supports the account; below 1 contradicts it.
- **The shuffled-map floor**: the same measurement on deliberately scrambled maps, which is what
  "no real effect" looks like.

## Run kind
**Tests the claim.** So a failure of the pre-registered bar closes the plan and opens one
follow-on. It did not fail.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| 64-cell probe sweep on mscluster109, 8 held-out pairs × seeds 9 to 16, adapter off against on at matched steps | Tests the claim | 2026-08-05 | `/datasets/.../interaction_term/reprobe/`, 384 token-step rows, `verdict.json` | done, 0 failed |
| one-cell smoke, an_eagle__x__a_hawk seed 9 (plus frog/toad, seal/walrus, cat/dog) | Tests the claim | before the sweep | `docs/evidence/F7-mechanism-reprobe/smoke_eagle_hawk.png` | done |

Run directly on the node rather than as a queued job, because biggpu allows one job per user and
an interactive session held the slot.

## The pre-registered bar

The bar lives in the source of `scripts/mechanism_study/reprobe_table.py`, written before the
sweep ran: **median ratio at least 1.2, AND at least 75% of rows above 1.** Both conditions, so
one strong pair cannot carry a weak average.

- [x] ✅ Does the adapter change what a word paints more than where it looks, across held-out
      pairs and seeds?
      **Yes.** Median content-to-weight pattern ratio **1.52**. 373 of 384 rows above 1, which is
      97%. Six times the headroom over a shuffled-map noise floor. All 8 pairs clear the bar
      individually:

      | Pair | Ratio | Pair | Ratio |
      |---|---|---|---|
      | a_frog__x__a_toad | 2.15 | an_eagle__x__a_hawk | 1.78 |
      | a_cat__x__a_dog | 1.73 | a_cow__x__a_buffalo | 1.50 |
      | an_elephant__x__a_penguin | 1.45 | a_goose__x__a_swan | 1.41 |
      | a_seal__x__a_walrus | 1.37 | a_leopard__x__a_jaguar | 1.16 |

      The early worry was a_leopard__x__a_jaguar reading 1.15 on seed 9 alone; across all eight
      of its seeds it medians 1.16, the weakest in the pool and still above 1.
      ✓ verified (64 cells, 384 rows, median 1.52 against the 1.2 bar, `verdict.json`)

## The caveat the write-up owes

- [x] 🟡 Is the effect specific to pairs that need fixing?
      **No, and this must be stated in the paper rather than hidden.** Split by role: the six
      unseen transfer pairs median 1.45, the reference pair 1.73, and the control pair
      (an_elephant__x__a_penguin, which composes fine with no adapter at all) reads 1.45,
      mid-range.

      So the effect is present on a pair that needs no fixing. That weakens the claim from "this
      is how the fix works" toward "this is what the adapter does to any pair it touches".
      Distinguishing the two would need a pair the adapter demonstrably does not help, which we
      do not have. **F7's caption must be written to the narrower sentence.**

## Was the instrument sound

- [x] ✅ Does the probe read the right word for every pair?
      Yes, after a fix. The probe originally used a hardcoded token position that is correct for
      "a cat" and "a dog" and wrong for any pair whose animal name splits into pieces. Three pool
      pairs split: walrus (`wal`+`rus`), chimpanzee (`chim`+`pan`+`zee`), and porpoise. One of
      them, a_seal__x__a_walrus, is in this sweep, so it would have measured a word fragment and
      raised no error. The map is now derived per pair from the tokenizer and checked across all
      19 pool pairs: 0 mismatches.
- [x] ✅ Do the maps look like anything?
      Yes. The smoke cells render a bird head and a frog head in profile, not noise.

## Still open

- [ ] ⚠️ The figure's statistical entity: per-seed points, or one point per pair. A design
      decision for `/pair-figure`, recorded here because F7's caption depends on it.
