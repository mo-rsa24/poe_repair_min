# 🧪 Review: what does the fix change inside the model?

**Answered, and it replicated, with one caveat the paper owes.** This file judges
[../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md](../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md).
Its answers fill register slot **F7**, and the caveat below is what caps that figure's caption.

The question in plain terms: every word in the prompt decides both *where* in the image to look
and *what* to write there. Our account of why the fix works says it changes the second, not the
first. This is whether that survives 64 cells the fix never trained on.

## Recommended prompt (to write the figure)

```
/design-figure F7 the mechanism reprobe, one point per pair with seeds as a spread
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md) | the two maps, why the obvious comparison gives the opposite answer, the bar in source |
| **this file** | **the verdict: the ratio is 1.52, it replicated, and one caveat caps the caption** |
| [the register](../../../../../paper/iclr/figures.md) | F7's row, whose claim the caption may not exceed |

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

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** So a failure of the pre-registered bar closes the plan and opens one
follow-on. It did not fail.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| 64-cell probe sweep on mscluster109, 8 held-out pairs × seeds 9 to 16, adapter off against on at matched steps | Tests the claim | 2026-08-05 | 64 cells, 384 token-step rows | `/datasets/.../interaction_term/reprobe/`, 384 token-step rows, `verdict.json` | done, 0 failed |
| one-cell smoke, an_eagle__x__a_hawk seed 9 (plus frog/toad, seal/walrus, cat/dog) | Tests the claim | before the sweep | 4 cells | `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/smoke_eagle_hawk.png` | done |

Run directly on the node rather than as a queued job, because biggpu allows one job per user and
an interactive session held the slot.

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

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

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] 🟡 Is the effect specific to pairs that need fixing?
      **No, and this must be stated in the paper rather than hidden.** Split by role: the six
      unseen transfer pairs median 1.45, the reference pair 1.73, and the control pair
      (an_elephant__x__a_penguin, which composes fine with no adapter at all) reads 1.45,
      mid-range.

      So the effect is present on a pair that needs no fixing. That weakens the claim from "this
      is how the fix works" toward "this is what the adapter does to any pair it touches".
      Distinguishing the two would need a pair the adapter demonstrably does not help, which we
      do not have. **F7's caption must be written to the narrower sentence.**

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible.

- [x] ✅ What is the figure's statistical entity? One point per pair, with its seeds as a light
      spread behind it. The data nests three deep, 384 rows inside 64 cells inside 8 pairs, and
      the rows within a cell are the same image read at different steps and tokens. They are not
      independent, so plotting rows as points would inflate the sample 48-fold and turn a claim
      about eight pairs into an apparent claim about hundreds of observations.

      Seeds stay visible rather than averaged away, because the spread is what tells a reader
      whether the effect is a property of the pair or of one lucky run. This matches what F3 and
      F4b already do, so a point means the same thing everywhere in the paper.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** One axis differs: the adapter off against on, from the
      identical starting state at matched steps. Nothing else changes between the two readings of
      a cell, which is what makes the pattern difference attributable to the adapter.
- [x] ✅ **Was the instrument sound?** Two checks, both passed.
      *Does the probe read the right word for every pair?* Yes, after a fix. The probe originally
      used a hardcoded token position that is correct for "a cat" and "a dog" and wrong for any
      pair whose animal name splits into pieces. Three pool pairs split: walrus (`wal`+`rus`),
      chimpanzee (`chim`+`pan`+`zee`), and porpoise. One of them, a_seal__x__a_walrus, is in this
      sweep, so it would have measured a word fragment and raised no error. The map is now derived
      per pair from the tokenizer and checked across all 19 pool pairs: 0 mismatches.
      *Do the maps look like anything?* Yes. The smoke cells render a bird head and a frog head in
      profile, not noise.
- [x] ✅ **Did the run respect the environment?** Output landed under `/datasets`, all 64 cells
      completed with 0 failed, and the run was placed directly on mscluster109 rather than queued,
      because biggpu allows one job per user and an interactive session held the slot. Harvest it
      by `pgrep`, not `squeue`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| F7's caption | the narrower sentence: the adapter changes what a word paints more than where it looks **on any pair it touches**, not specifically on pairs that need fixing. The control pair reads 1.45, mid-range. Reason and numbers under [Written before the run](#written-before-the-run-answered-after) |
| F7's sample size | n=8 pairs, reporting the per-pair median. It may not quote 384 or 64 as a sample size |
| the pre-registered bar | median 1.52 and 97% of rows above one were computed over rows, and stay as the pre-registered number, cited as such rather than as the figure's statistic |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the effect is how the fix works, or what the adapter does to any pair | a pair the adapter demonstrably does not help, which the pool does not currently contain | nothing is blocked; F7 is written to the narrower sentence instead. Widening it later needs that pair |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Build F7 to the narrower caption, one point per pair with seeds as a spread, n=8.
