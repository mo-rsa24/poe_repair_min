# 🌍 Chimera

A chimera is what PoE composition produces when it fails: one creature with parts of both
requested concepts fused together, instead of two separate concepts in one scene. It is the
project's name for the failure that [interaction-term.md](interaction-term.md) tries to explain
and [lora-corrector.md](lora-corrector.md) tries to fix.

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What a chimera is](#what-a-chimera-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-a-chimera-is) ➡️

- **Blend**: this project's word for a chimera result when discussing the scorer's binary label
  (`compose` vs `blend`), see [compose-rate.md](compose-rate.md).
- **Single concept / noise**: the two other named failure modes alongside chimera, per
  `MASTER_PLAN.md`'s Mission line: "PoE usually fails (chimera / single concept / noise)".

## What a chimera is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**One blended creature carrying features of both requested concepts, instead of two distinct
concepts in the frame.** ✅

`MASTER_PLAN.md`'s Glossary, verbatim: "Chimera: that failure. One animal with parts of both,
instead of two animals." It is defined specifically against the animal-pair pool this project
currently studies (see [animal-pair.md](animal-pair.md)); the same underlying failure on
non-animal pairs is described more generally in `MASTER_PLAN.md`'s Mission as "single concept /
noise" alongside it. ✅

## What it looks like

Navigation: ⬅️ [What a chimera is](#what-a-chimera-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

The plain PoE render in [poe-composition.md](poe-composition.md#what-it-looks-like) (copied from
`data/pilot/seed_42/a_cat__x__a_dog/poe.png`) is this project's own chimera example: one fused
creature rather than a cat beside a dog.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**A chimera is the visible symptom; the interaction term is the measured cause the project
targets.** ✅

The paper's abstract states the causal claim directly: PoE "can fail catastrophically, producing a
single blended chimera instead of a scene containing both concepts, and we trace this failure to a
specific, correctable gap" (`paper/iclr/iclr2027_conference.tex`). Whether a given render is a
chimera or a real composition is exactly what the scorer in
[compose-rate.md](compose-rate.md) is built to call.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `n_instances` | how many distinct animal instances the scorer's detector found; 1 usually means a chimera or a dropped concept | `1` (example, from `catdog_poe_blend` in `scorer_validated.json`) | [Dictionary § n_instances](../data/02-dictionary.md#n_instances) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A chimera is not the only failure the instance-count scorer catches, and "2 instances" is not
proof of a real chimera-free success.** ✍️

`evidence/f2-lambda1-audit/02-two-of-one/` holds one confirmed case of two instances of the *same*
animal (two dogs, no cat) scored as a compose success, because the detector cannot tell which two
animals it found, only how many. See
[compose-rate.md § What people get wrong](compose-rate.md#what-people-get-wrong).

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The definition of a chimera | Read in `MASTER_PLAN.md`, Glossary | 2026-08-24 |
| The chimera example image | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/poe.png` | 2026-08-24 |
| The causal claim linking chimera to the interaction term | Read in `paper/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| The "two of the same animal" scorer miss | Read in `evidence/f2-lambda1-audit/README.md` | 2026-08-24 |
