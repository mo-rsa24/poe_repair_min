# 🌍 The interaction term (r_t)

The interaction term is the paper's central object: the per-step gap between what SDXL would
predict if it saw the literal joint prompt (Mono) and what it actually predicts under PoE
composition. The project's claim is that this gap is what PoE composition drops, and that adding
it back in fixes the chimera failure. [lora-corrector.md](lora-corrector.md) is the trained
stand-in for this term that never sees the joint prompt.

> 🖼️ **Diagram wanted**: the process-lane piece "how the interaction term is computed and injected
> back in", per [diagram-prompts.md](../diagram-prompts.md). Not yet rendered.

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What the interaction term is](#what-the-interaction-term-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-the-interaction-term-is) ➡️

- **r_t (also Δ_t, "the residual")**: the same quantity as "the interaction term", written as
  a symbol in code and plan files. `r_t = ε̃_Mono − ε̃_PoE` (some earlier documents write the
  Mono-minus-PoE order as `ε̃_J − ε̃_PoE`; both name the same gap, see
  [What people get wrong](#what-people-get-wrong)).
  Struck word: **oracle**. Some plan files still call this quantity "the oracle" when describing it
  as the target a trained corrector imitates. Current prose calls it the joint-prompt correction or
  the interaction term instead, per `plans/retrofit-poe-repair-min.md`'s struck-words list.
- **λ (lambda)**: the dial for how much of the interaction term to add back at inference, from 0
  (plain PoE) to 1 (the full correction).
- **Window**: the span of denoising steps over which the correction is injected. It does not have
  to cover the whole run.
- **Commitment step**: the step after which a run's outcome stops changing, measured separately
  per pair (see [How it shows up in the data](#how-it-shows-up-in-the-data)).

## What the interaction term is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**The step-by-step difference between the Mono prediction and the PoE prediction, made explicit as
the signal linear score composition omits.** ✅

Quoted from the paper's abstract (`paper/iclr/iclr2027_conference.tex`): "the interaction term:
the per-step residual between the model's joint-prompt prediction and its product-of-experts
prediction, which makes explicit the signal omitted by linear score composition." ✅

**It is measured directly in the pipeline's cached output, not estimated indirectly.** ✅

A pilot cell's `summary.json` carries `d_t_poe_vs_mono`, a per-step magnitude array (50 values,
step 0 near 0.0 rising to about 0.246 by the final step for `a_cat__x__a_dog` seed 42), and
`d_T_poe_vs_mono`, the same quantity's final value. Read directly,
`data/pilot/seed_42/a_cat__x__a_dog/summary.json`. ✅

**It grows through the run rather than being a fixed offset**, and how far into the run it needs
to be added (the window) is a separate, measured question from how big it is (the strength, λ).
`report/experiments-log.md`'s EXP-04 found the best window centred at steps 0-10 for every one of 8 tested
pairs, regardless of how late that pair's own outcome settles (its "commitment step", which ranges
from step 18 to step 36 across pairs). ✅

## What it looks like

Navigation: ⬅️ [What the interaction term is](#what-the-interaction-term-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

There is no single picture of the interaction term itself: it is a per-step vector in the model's
noise-prediction space, not a rendered image. The two images in
[poe-composition.md](poe-composition.md#what-it-looks-like) are its endpoints (the Mono target and
the PoE render it is measured against); a rendered picture of the term's own trajectory is the
diagram slot above.

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**It is the whole causal claim.** ✅ If adding this term back in (at increasing λ) reliably raises
the compose rate while a same-sized random vector does not, the project has evidence the term is
what PoE composition is missing, not just a correlate. `MASTER_PLAN.md` step 4 records this
dose-response result as done.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `d_T_poe_vs_mono` | the interaction term's final-step size for one cell | `0.246` (example, `a_cat__x__a_dog` seed 42) | [Dictionary § d_T / d_t](../data/02-dictionary.md#d_t-d_t) |
| `d_t_poe_vs_mono` | the same quantity at every one of the 50 steps | array of 50 floats, `0.0` to `0.246` (example) | [Dictionary § d_T / d_t](../data/02-dictionary.md#d_t-d_t) |
| `lambda` | how much of the term is injected at inference | `1.0` (example, full strength) | [Dictionary § lambda](../data/02-dictionary.md#lambda) |
| `step` | which of the 50 denoising steps a value belongs to | `10` (example) | [Dictionary § step](../data/02-dictionary.md#step) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The window that works is not the window where the run visibly commits.** ⚠️

`report/experiments-log.md` EXP-01 and EXP-04 together found something the project itself calls
counter-intuitive: every studied pair's outcome settles (commitment step) well after the window
where the correction actually works (steps 0-10 for all 8 tested pairs, while commitment ranges
from step 18 to 36). The correction stops mattering 8 to 26 steps before the picture "looks
decided" by the running-estimate measure. Either the decision happens earlier than that measure
can see, or the measure tracks the wrong event; `report/experiments-log.md` records this as unresolved and
not the registered finding.

**The sign convention is written both ways across the repo's own documents.** 🔍

`MASTER_PLAN.md`'s glossary writes `r_t = ε̃_J − ε̃_PoE`; the memory note titled
"correction-shared-early-part" (not read as part of this build) is recorded by the maintainer as
using `r_t = ε̃_Mono − ε̃_PoE`. Both name Mono-minus-PoE; `ε̃_J` is read here as shorthand for the
joint-prompt (Mono) prediction, not a third quantity, but this file has not confirmed that by
reading the code that computes it. Listed under [Still open](../00-INDEX.md#still-open).

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The interaction term's definition | Read in `paper/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| Its measured per-step and final values for one cell | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/summary.json` | 2026-08-24 |
| The window-vs-commitment-step finding | Read in `report/experiments-log.md`, EXP-01 and EXP-04 | 2026-08-24 |
| The struck word "oracle" and its replacement | Read in `plans/retrofit-poe-repair-min.md`, "Words this uses" | 2026-08-24 |
