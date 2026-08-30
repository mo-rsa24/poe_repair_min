# 🧪 Review: what does the fix change inside the model?

**Answered, and it replicated, with one caveat the paper owes.** This file judges
[the design plan](../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md).
Its answers fill **F7** in the figure register, and the caveat below is what caps that figure's
caption.

The question in plain terms. Every word in the prompt decides both *where* in the image to look
and *what* to write there. Our account of why the fix works says the adapter changes what a word
writes and leaves where it looks alone. The experiment asks whether that holds across 64
pair-and-seed runs the fix never trained on.

## Recommended prompt (to write the figure)

```
/design-figure F7 the mechanism re-measurement, one point per pair with seeds as a spread
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-01-what-the-fix-changes-inside-the-model.md) | the two maps, why the obvious comparison gives the opposite answer, the threshold in source |
| **this file** | **the verdict: the ratio is 1.52, it replicated, and one caveat caps the caption** |
| [the register](../../../paper/iclr/figures.md) | F7's row, whose claim the caption may not exceed |

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

- **A run**: one animal pair at one starting seed. 8 pairs by 8 seeds gives the 64.
- **Where it looks / what it paints**: the two maps captured per word, with the adapter off and
  again on, from the identical starting state.
- **Pattern, not brightness**: the two maps are not on a common scale, and the adapter dims one of
  them by about 25% overall. So the comparison strips out uniform brightness and compares only
  the change a rescaling cannot explain. The design plan explains why the obvious comparison
  gives the opposite answer.
- **The ratio**: how much more the painted content's pattern moved than the attention's. Above 1
  supports the account; below 1 contradicts it.
- **What no real effect looks like**: the same measurement run on deliberately scrambled maps,
  which is the reading to beat.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** So missing the pre-registered threshold closes the plan and opens one
follow-on. It did not miss.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the 64-run measurement on mscluster109, 8 held-out pairs × seeds 9 to 16, adapter off against on at matched steps | Tests the claim | 2026-08-05 | 64 runs, 384 token-step rows | `/datasets/.../interaction_term/reprobe/`, 384 token-step rows, `verdict.json` | done, 0 failed |
| a first short run on an_eagle__x__a_hawk seed 9 (plus frog/toad, seal/walrus, cat/dog) | Tests the claim | before the full 64 | 4 runs | `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/smoke_eagle_hawk.png` | done |

> Held-out means the pair was never used to train the adapter, so what it reads here cannot come
> from the adapter having already seen it.

Run directly on the node rather than as a queued job, because biggpu allows one job per user and
an interactive session was already using the one allowed.

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

The threshold lives in the source of `scripts/mechanism_study/reprobe_table.py` and was written
before any of these runs happened. It is **a median ratio of at least 1.2, AND at least 75% of
rows above 1.** Both conditions have to hold, so one strong pair cannot carry a weak average.

- [x] ✅ Does the adapter change what a word paints more than where it looks, across held-out
      pairs and seeds?
      **Yes.** Median content-to-weight pattern ratio **1.52**. 373 of 384 rows above 1, which is
      97%. That sits six times clear of what deliberately scrambled maps read. All 8 pairs pass
      on their own:

      | Pair | Ratio | Pair | Ratio |
      |---|---|---|---|
      | a_frog__x__a_toad | 2.15 | an_eagle__x__a_hawk | 1.78 |
      | a_cat__x__a_dog | 1.73 | a_cow__x__a_buffalo | 1.50 |
      | an_elephant__x__a_penguin | 1.45 | a_goose__x__a_swan | 1.41 |
      | a_seal__x__a_walrus | 1.37 | a_leopard__x__a_jaguar | 1.16 |

      The early worry was a_leopard__x__a_jaguar reading 1.15 on seed 9 alone; across all eight
      of its seeds it medians 1.16, the weakest in the pool and still above 1.
      ✓ verified (64 runs, 384 rows, median 1.52 against the 1.2 threshold, `verdict.json`)

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

Questions the result itself raised. **Nothing here may ever become a pre-registered threshold**,
because it was written with the answer already visible.

- [x] ✅ What is the figure's statistical entity? One point per pair, with its seeds as a light
      spread behind it. The data nests three deep, 384 rows inside 64 runs inside 8 pairs, and
      the rows within one run are the same image read at different steps and tokens. They are not
      independent, so plotting rows as points would inflate the sample 48-fold and turn a claim
      about eight pairs into an apparent claim about hundreds of observations.

      Seeds stay visible rather than averaged away, because the spread is what tells a reader
      whether the effect is a property of the pair or of one lucky run. This matches what F3 and
      F4b already do, so a point means the same thing everywhere in the paper.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** One axis differs: the adapter off against on, from the
      identical starting state at matched steps. Nothing else changes between the two readings of
      one run, which is what makes the pattern difference attributable to the adapter.
- [x] ✅ **Was the instrument sound?** Two checks, both passed.
      *Does the measurement read the right word for every pair?* Yes, after a fix. The measuring
      code originally used a hardcoded token position that is correct for "a cat" and "a dog" and
      wrong for any pair whose animal name splits into pieces. Three pool pairs split: walrus
      (`wal`+`rus`), chimpanzee (`chim`+`pan`+`zee`), and porpoise. One of them,
      a_seal__x__a_walrus, is in these 64 runs, so it would have measured a word fragment and
      raised no error. The map is now derived per pair from the tokenizer and checked across all
      19 pool pairs: 0 mismatches.
      *Do the maps look like anything?* Yes. The four short first runs render a bird head and a
      frog head in profile, and nothing resembling noise.
- [x] ✅ **Did the run respect the environment?** Output landed under `/datasets`, all 64 runs
      completed with 0 failed, and the work was placed directly on mscluster109 rather than
      queued, because biggpu allows one job per user and an interactive session was already using
      the one allowed. Harvest it with `pgrep`, since `squeue` cannot see it.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| F7's caption | the narrower sentence: the adapter changes what a word paints more than where it looks **on any pair it touches**, not specifically on pairs that need fixing. The control pair reads 1.45, mid-range. Reason and numbers under [Written before the run](#written-before-the-run-answered-after) |
| F7's sample size | n=8 pairs, reporting the per-pair median. It may not quote 384 or 64 as a sample size |
| the pre-registered threshold | median 1.52 and 97% of rows above one were computed over rows, and stay as the pre-registered number, cited as such rather than as the figure's statistic |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether the effect is how the fix works, or what the adapter does to any pair | a pair the adapter demonstrably does not help, which the pool does not currently contain | nothing is blocked; F7 is written to the narrower sentence instead. Widening it later needs that pair |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Build F7 to the narrower caption, one point per pair with seeds as a spread, n=8.
