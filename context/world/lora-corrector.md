# 🌍 The LoRA corrector

A LoRA (low-rank adapter) here is a small set of extra trained weights bolted onto SDXL's
cross-attention layers, trained to predict the interaction term (see
[interaction-term.md](interaction-term.md)) at every denoising step and add it back into the PoE
run, without ever seeing the literal joint prompt at inference. It is the project's one shippable
fix, as distinct from the interaction term itself, which is only ever computed from the joint
prompt during training and is not something a deployed system has access to.

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What the LoRA corrector is](#what-the-lora-corrector-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-lora-corrector-is) ➡️

- **Rank-8**: the size of the low-rank adapter; a smaller number of trained parameters than
  retraining the attention weights outright.
- **Cross-attention (`attn2`)**: the layer inside SDXL's UNet where the text prompt's embedding
  enters the image-generation computation; this is where the LoRA is attached.
- **Mono-free**: the property that matters most about this fix: at inference, the LoRA never
  receives or encodes the joint prompt. Only its own trained weights and the PoE run's own state
  are used.
- Struck word: **adapter**. Current prose says **lora** in full, per
  `plans/retrofit-poe-repair-min.md`'s struck-words list; "adapter" survives only inside code
  identifiers and older documents quoted verbatim.

## What the LoRA corrector is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**A rank-8 LoRA on SDXL's cross-attention layers, trained on the cached interaction term, added at
inference without seeing the joint prompt.** ✅

Quoted from `MASTER_PLAN.md`'s Mission: "we train a rank-8 cross-attention LoRA on the cached
guided residual r_t = ε̃_J − ε̃_PoE so that, at inference and without ever encoding the joint
prompt (Mono-free), the corrected PoE prediction moves toward the Mono ceiling." ✅

**It is trained per configuration, not once for the whole project.** ✍️

Different training runs pool over different sets of pairs and seeds, matching the project's
five-rung ladder (see [purpose/03-what-working-means.md](../purpose/03-what-working-means.md#the-test)):
one pair and one seed, one pair pooled over several seeds, one group of pairs pooled together, or
the whole studied set at once. Which pooling unit is deployable is the project's open research
question, not a settled fact this file can assert.

## What it looks like

Navigation: ⬅️ [What the LoRA corrector is](#what-the-lora-corrector-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

A LoRA is a set of trained numeric weights, not a directly picturable object. What it produces
(a corrected render, ideally closer to the Mono image than plain PoE) is picturable, and an
example of that comparison is owed:

> 📷 **Picture wanted**: one cell's PoE render, LoRA-corrected render, and Mono render side by
> side, so a reader can see what the correction visibly moved. Not yet selected; a strong
> candidate is any cell behind `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`'s
> `passing_spaces: instance_count` result. Save as
> `images/world/05-lora-corrected-vs-poe-vs-mono.png`.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**It is the one artefact the project can ship.** ✅

The interaction term itself needs the joint prompt to compute, which defeats the purpose of PoE
composition. `MASTER_PLAN.md`'s Expected Outcome names the LoRA (or a catalogue of LoRAs, one per
group) as "A deployable, Mono-free PoE corrector whose reach is characterised" — the project's
actual deliverable.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `arm` | one of the values this column takes is the LoRA-corrected render | `lora_residual_inject` (example, a method name in `poe_repair/methods/_sampling.py`) | [Dictionary § arm](../data/02-dictionary.md#arm) |
| `pair_slug` | which pair a given LoRA checkpoint was trained on or evaluated against | `a_cat__x__a_dog` (example) | [Dictionary § pair_slug](../data/02-dictionary.md#pair_slug) |
| `group` | the pooling unit a LoRA checkpoint was trained over, when pooled by group | `group6_coherent_collision` (example) | [Dictionary § group](../data/02-dictionary.md#group-group_label) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A LoRA that composes on its training pair is not evidence it generalises.** ✍️

`context/research-guidelines.md` records this as a live worry, not a resolved one: whether a
single-pair-trained LoRA's apparent transfer to a sibling pair is a real hit or "a memorised
correction that happens to fit" is exactly why the project's own rung 3 (`does-the-fix-reach-unseen-pairs`)
was downgraded from a publication gate to an optional smoke test, and rung 4 (group-wise pooling on
concept-disjoint pairs) was named the reviewer-credible version instead
(`plans/retrofit-poe-repair-min.md`'s reference to `report/experiments-log.md` EXP-03).

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The LoRA's rank, attachment point and Mono-free property | Read in `MASTER_PLAN.md`, Mission | 2026-08-24 |
| The deployable-artefact framing | Read in `MASTER_PLAN.md`, Expected Outcome | 2026-08-24 |
| The single-pair-transfer confound and its downgrade | Read in `context/research-guidelines.md` and `report/experiments-log.md` | 2026-08-24 |
| `arm` value `lora_residual_inject` | Read in `README.md`'s repo layout, `poe_repair/methods/_sampling.py` listing | 2026-08-24 |
