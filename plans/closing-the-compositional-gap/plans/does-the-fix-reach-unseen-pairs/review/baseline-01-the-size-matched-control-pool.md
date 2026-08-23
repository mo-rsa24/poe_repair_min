# 🧪 Review: would any pool of the same size have worked?

**Nothing has run yet.** This file holds the question, written before the run. It judges
[../plans/baseline-01-the-size-matched-control-pool.md](../plans/baseline-01-the-size-matched-control-pool.md),
and its answer is what lets the transfer claim survive the obvious objection.

The objection: if training on fifteen animal pairs helps unseen animal pairs, maybe training on
*any* fifteen pairs would have helped just as much, and the win is really about how much data
there was.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed in a way worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baseline-01-the-size-matched-control-pool.md) | how the mixed pool is built, and what is held identical between the two |
| **this file** | **the verdict: whether the animals pool beat a pool of the same size** |
| [the claim this defends](hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | the transfer rate this control makes attributable |

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

- **Size-matched**: the control pool holds exactly as many pairs as the animals pool, so the
  amount of training signal is identical and only the content differs.
- **The mixed pool**: the same count of pairs, built from scenes, styles and objects instead of
  animals.
- **The identical held-out set**: both pools are tested on the same unseen animal pairs. Testing
  them on different sets would make the comparison meaningless.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Produces a competitor.** It freezes the moment it lands: no tuning it up after our own number
is known, and no tuning it down either.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| one mixed-pool run, same pair count, same held-out set | Produces a competitor | | | the contrast per held-out pair | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Does the animals pool beat the size-matched mixed pool on the identical held-out animal
      pairs? A win attributes the transfer to what is in the pool rather than to how much of it
      there is. A tie or a loss bounds the claim, and that boundary goes in the paper as a
      sentence rather than being left out.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Nothing beyond the bar. This run exists to answer one question.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible. Nothing yet: the run has not started.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** The pools must differ in content and in nothing else.
      Print the realised pair count of both pools and confirm they match, rather than trusting
      the config to have built what it claims.
- [ ] ⚠️ **Was the instrument sound?** Both pools must be scored by the same scorer over their
      own output directories only, on the identical held-out set.
- [ ] ⚠️ **Did the run respect the environment?** Output under `/datasets`, and the pool-selection
      flag confirmed to have selected a non-empty set of mixed pairs rather than silently falling
      back to the animals pool.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the transfer is about what is in the pool, not how much of it there is | this comparison, with the mixed pool's number beside the animals pool's. If it ties or loses, the bounding sentence goes in the paper rather than being left out |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the mixed-pool run | the attribution sentence in the transfer claim, and any reviewer asking whether data volume explains the win |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Build the mixed pool per [the design](../plans/baseline-01-the-size-matched-control-pool.md), then
launch the single run.
