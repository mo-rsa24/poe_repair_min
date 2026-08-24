# 🌍 What would count as this project working

## Table of contents

- [The test](#the-test)
- [What a good result looks like](#what-a-good-result-looks-like)
- [What a bad result looks like](#what-a-bad-result-looks-like)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)

## The test

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-a-good-result-looks-like) ➡️

**Does a LoRA trained to predict the interaction term make PoE compose like Mono, and does that
fix carry to pairs and seeds it never trained on?** ✅

Quoted directly from `MASTER_PLAN.md`'s Mission. The five objectives it lists are a ladder, each
widening the held-out set: overfit one pair and seed, survive new seeds of the same pair, transfer
to a sibling pair, generalise within a group of similar pairs, and finally span the whole studied
set of pairs held out on both pair and seed at once. The project succeeds at whichever rung it
reaches, per its own Expected Outcome: "at minimum a per-pair/per-group catalogue backed by
evidence, at most a single taxonomy-spanning LoRA." ✅

## What a good result looks like

Navigation: ⬅️ [The test](#the-test) | 📋 [TOC](#table-of-contents) | [Next](#what-a-bad-result-looks-like) ➡️

**More correction, more composition, with the controls flat.** ✅

`MASTER_PLAN.md` step 4 (`hypothesis-02-more-correction-more-composition`) names this the
headline claim and marks it done. The falsification shape, from `EXPERIMENTS.md`'s design
convention: a dose-response curve where compose rate rises as the interaction term's strength (λ)
rises, while a control that injects a random vector of the same size stays flat. ✍️

**A held-out check landing above a bar fixed before the run.** ✅

Objective 2's bar, read in `MASTER_PLAN.md`'s Goals: composing on at least 3 of 4 held-out seeds
for the representative pair, per group. Objective 3's bar: composing on at least 2 of 4 held-out
seeds for a held-out sibling pair. Bars are fixed in the plan file before the run, per this
project's convention that a threshold not visible in a diff can be adjusted after seeing the
answer.

## What a bad result looks like

Navigation: ⬅️ [What a good result looks like](#what-a-good-result-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

**A compose rate that does not clear the scorer's own uncertainty.** ✅

`evidence/f2-lambda1-audit/README.md`: a headline of 94% at full correction strength has a true
rate bounded around 87% to 94%, once the 32 cells behind it are opened by eye. A result that
depends on reading the difference between, say, 90% and 94% as a real effect is not trustworthy on
this scorer without a tighter check.

**Failing cells that fit none of the project's own failure modes.** ✍️

`EXPERIMENTS.md` EXP-05 pre-registers two remaining explanations for adapter failure (aiming
wrong, delivering too little) and states its own null: fewer than 40% of failing cells landing in
either mode would mean the two-mode decomposition is wrong, and a large group fitting neither
would mean the corrector emits a fine correction and the run still fails for an unexplained reason.

## What this cannot tell you

Navigation: ⬅️ [What a bad result looks like](#what-a-bad-result-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Whether the two right concepts are present, only whether two instances of some concept are.** ✅

The validated scorer counts distinct "animal" instances; it cannot tell a cat-and-dog image from a
two-dog image by construction. See
[world/compose-rate.md § What people get wrong](../world/compose-rate.md#what-people-get-wrong).
`evidence/f2-lambda1-audit/02-two-of-one/` is the one confirmed case of this in the current
32-cell audit.

**Anything about pairs outside the animal-pair pool.** ✍️

Per `plans/retrofit-poe-repair-min.md`'s "Words this uses": the animal pairs are the current
scope; the seven non-animal pairs from an earlier cross-taxonomy era and the six-group taxonomy
they came from (see [world/animal-pair.md § What people get wrong](../world/animal-pair.md#what-people-get-wrong))
are not part of the live claim.

**Whether the fix generalises past SDXL, or past this one detector-based scorer.** 🔍

Inferred from the project's own background-experiment table in `MASTER_PLAN.md`: a second model
and sampler, and a check of whether any published metric agrees with human judgement better than
this one, are both listed as not-yet-started rows, not settled findings.

## Where this came from

Navigation: ⬅️ [What this cannot tell you](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The mission statement and the five-rung ladder | Read in `MASTER_PLAN.md`, Mission and Objectives | 2026-08-24 |
| The dose-response and held-out-seed bars | Read in `MASTER_PLAN.md`, Goals, and `EXPERIMENTS.md` | 2026-08-24 |
| The scorer's true-rate bound at λ=1 | Read in `evidence/f2-lambda1-audit/README.md` | 2026-08-24 |
| The two-mode failure classification and its null | Read in `EXPERIMENTS.md`, EXP-05 | 2026-08-24 |
| The animal-pair scope boundary | Read in `plans/retrofit-poe-repair-min.md`, "Words this uses" | 2026-08-24 |
