# 🌍 The animal pair

An animal pair is the unit every experiment in the current scope is built from: two animal
prompts (for example "a cat" and "a dog"), tested together across several seeds and render
methods. Whether a pair "fails" (blends into a [chimera](chimera.md)) or "composes" is judged per
seed by the scorer in [compose-rate.md](compose-rate.md).

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What an animal pair is](#what-an-animal-pair-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-an-animal-pair-is) ➡️

- **Pair slug**: the on-disk and in-code name for a pair, always the form `a_cat__x__a_dog`
  (article, animal, double underscore, `x`, double underscore, article, animal). Kept in this
  exact form everywhere on disk even though prose says "a cat and a dog"; changing the slug form
  was costed and rejected (`plans/retrofit-poe-repair-min.md`: "costs 3,049 files and 456
  directories and buys nothing a card does not already give").
- **Seed**: a re-roll of the starting noise only; the pair of concepts stays the same. Contrast
  with changing the pair itself, which is the harder generalisation test.
- **Group**: an older, six-way taxonomy (co-occurrence, factorisation, object-plus-scene,
  dual-object, entanglement, concept collision) that classified pairs by *why* composition is
  hard, not just by species. See [What people get wrong](#what-people-get-wrong) for its current
  status.
- **Cell**: one (pair, seed) point, the smallest unit a run or a figure is built from.

## What an animal pair is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**Two animal names, joined into one prompt pair, chosen so that PoE composition reliably fails on
them.** ✅

`EXPERIMENTS.md`'s selection warning states this plainly: pairs entered the current pool "by
failing" under plain PoE (8 seeds each, instance-count scorer). 15 of 17 pairs in
`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` fail on all 8 seeds; the two least-reliable
failures are `a_donkey__x__a_pony` (6/8) and `a_crocodile__x__an_alligator` (5/8). ✅

**The pool is deliberately biased toward failure, which the project's own documents flag as a
scoping limit, not an oversight.** ✅

`EXPERIMENTS.md`: "No analysis inside this pool can say what predicts PoE failure, because the
pool has no successes to contrast against. Any such analysis is conditioned on the outcome." A
follow-on experiment (EXP-02, status pending as of this build) is designed to build a second pool
spanning both successes and failures. ✅

**One pair is a deliberate exception: it composes.** ✅

`plans/retrofit-poe-repair-min.md`'s "Words this uses": `a_butterfly__x__a_flower_meadow` "is the
deliberate exception: it is the control that composes and the paper opens on it." 🔍 (This file has
not independently verified its fail-rate in `fail_rate.md`, which does not list it; flagged in
[Still open](../00-INDEX.md#still-open).)

## What it looks like

Navigation: ⬅️ [What an animal pair is](#what-an-animal-pair-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

The PoE and Mono renders in [poe-composition.md](poe-composition.md#what-it-looks-like) are the
`a_cat__x__a_dog` pair, seed 42: the project's own reference pair, present in almost every walk of
this repository.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**The pair is the axis every transfer claim is measured across.** ✅

Whether a LoRA trained on one pair (or one group of pairs) composes on a pair it never trained on
is Objective 3 and Objective 4 of `MASTER_PLAN.md`'s five-rung ladder, and is the harder of the two
generalisation axes (pair vs seed) per `MASTER_PLAN.md`'s Glossary: "a new pair is the harder
test."

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `pair_slug` | which two animals this cell is about | `a_cat__x__a_dog` (example) | [Dictionary § pair_slug](../data/02-dictionary.md#pair_slug) |
| `seed` | which noise re-roll produced this cell | `42` (example) | [Dictionary § seed](../data/02-dictionary.md#seed) |
| `fail_rate` | how often plain PoE fails this pair, over 8 seeds | `1.00` (example, 8 of 8) | [Dictionary § fail_rate](../data/02-dictionary.md#fail_rate-compose_rate) |
| `group` | the six-way taxonomy label some pairs still carry | `group6_coherent_collision` (example) | [Dictionary § group](../data/02-dictionary.md#group-group_label) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The pair count is not settled across the repo's own documents.** ⚠️

`EXPERIMENTS.md` (its axes table) says "17 in the current pool, plus a new spread set built in
EXP-02" (EXP-02 status: pending). `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` lists exactly 17
pairs (1 dissimilar control, 1 reference, 15 train). `plans/retrofit-poe-repair-min.md`'s "Words
this uses" instead names "the 20 animal pairs" as the current scope boundary, plus 7 non-animal
pairs from an earlier era that stay on disk but are out of the live claim. This file cannot tell
which count is current without checking which pairs, beyond the 17, exist under the pool's
directory today; listed under [Still open](../00-INDEX.md#still-open). 🔍

**A pair from the older six-group taxonomy is not automatically part of the current pool.** ✍️

`plans/shelved/phases/09-lora-taxonomy-single-seed.md` (a shelved plan) describes representative
pairs for Groups 1-5 (dolphin×ocean-wave, dog×oil-painting-style, mailbox×snowfield,
typewriter×cactus, and a deferred, unresolved Group 5) that are not animal-vs-animal pairs and sit
outside the current animal-only scope, per the retrofit plan's animal-pair boundary. Group 6
(`a_cat__x__a_dog`) is the one taxonomy group that overlaps the live pool.

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The pool is biased toward failure by construction | Read in `EXPERIMENTS.md`, "The selection warning" | 2026-08-24 |
| The 17-row fail-rate table | Read in `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` | 2026-08-24 |
| The butterfly control pair and the animal-pair boundary | Read in `plans/retrofit-poe-repair-min.md`, "Words this uses" | 2026-08-24 |
| The six-group taxonomy and its representative pairs | Read in `plans/shelved/phases/09-lora-taxonomy-single-seed.md` | 2026-08-24 |
| Pair vs seed, which is the harder test | Read in `MASTER_PLAN.md`, Glossary | 2026-08-24 |
