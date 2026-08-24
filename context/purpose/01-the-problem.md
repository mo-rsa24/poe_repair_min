# 🌍 The problem this project solves

Text-to-image diffusion models like Stable Diffusion XL (SDXL) are usually asked for one prompt
at a time. One cheap way to make a pretrained model draw two named concepts at once, without
retraining it, is to run two "expert" predictions side by side, one per concept, and combine them
at every denoising step. This is Product-of-Experts (PoE) composition, and it frequently fails: it
paints a blend of the two concepts (a chimera) or drops one of them, rather than a scene showing
both. This project asks why, and tests one specific fix. What a chimera and PoE composition
actually are, with an example image, is in [world/poe-composition.md](../world/poe-composition.md)
and [world/chimera.md](../world/chimera.md); this file is only the why.

## Table of contents

- [Who has the problem](#who-has-the-problem)
- [What it costs them today](#what-it-costs-them-today)
- [What this project does about it](#what-this-project-does-about-it)
- [Where this came from](#where-this-came-from)

## Who has the problem

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-it-costs-them-today) ➡️

**Anyone who wants a pretrained diffusion model to render two named concepts in one image without
retraining the whole model or hand-writing a joint prompt for every possible pair of concepts.** ✍️

The alternative to PoE composition is to type the joint prompt directly ("a cat and a dog") and
let the model's own training handle it. That works (this project calls it Mono, see
[world/poe-composition.md § What PoE composition is](../world/poe-composition.md#what-poe-composition-is)), but it
defeats the purpose of a compositional method: at scale, the whole point is to combine pretrained
single-concept experts instead of retraining or hand-prompting for every combination.

## What it costs them today

Navigation: ⬅️ [Who has the problem](#who-has-the-problem) | 📋 [TOC](#table-of-contents) | [Next](#what-this-project-does-about-it) ➡️

**PoE composition on SDXL blends or drops concepts far more often than it renders both.** ✅

Measured directly: `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`, 8 seeds per pair under the
validated instance-count scorer (see
[world/compose-rate.md](../world/compose-rate.md#what-a-compose-rate-is)). 15 of 17 pairs in that
table fail on 8 of 8 seeds (fail-rate 1.00), including the reference pair `a_cat__x__a_dog`. Two
pairs fail less often: `a_donkey__x__a_pony` at 6 of 8 and `a_crocodile__x__an_alligator` at 5 of
8. A dissimilar control pair, `an_elephant__x__a_penguin`, also fails 8 of 8, so "the two concepts
are visually different" does not predict success on its own; `EXPERIMENTS.md`'s EXP-03 registers
this as a pre-registered null rather than the hypothesis. 🔍

**When PoE does render both concepts, the render still is not trustworthy on its own.** ✍️

`artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` opened all 32 of the strongest-correction cells behind a
94% headline compose rate by eye: 1 of 32 was a real scorer error (two of the same animal, called
a success), and 5 more could not be called confidently either way, most because the pool
deliberately contains look-alike pairs (leopard vs jaguar, cow vs buffalo). The card's own
conclusion: the true rate sits somewhere around 87% to 94%, and a printed rate should say so
rather than stand alone. This is why
[world/compose-rate.md § What people get wrong](../world/compose-rate.md#what-people-get-wrong)
exists.

## What this project does about it

Navigation: ⬅️ [What it costs them today](#what-it-costs-them-today) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**It traces the PoE failure to one specific, measurable gap and tests whether closing that gap
fixes composition.** ✅

The gap is the interaction term: the per-step difference between the prediction the model would
make if it saw the literal joint prompt (Mono) and the prediction PoE actually makes by combining
two separate experts. The paper's own framing (`paper/iclr/iclr2027_conference.tex`, abstract):
"[PoE] can fail catastrophically, producing a single blended chimera instead of a scene containing
both concepts, and we trace this failure to a specific, correctable gap. This gap is the
interaction term... the signal omitted by linear score composition." What the interaction term is,
concretely, is in [world/interaction-term.md](../world/interaction-term.md). ✅

**The fix under test is a small trained corrector, not the joint prompt itself.** ✅

A rank-8 low-rank adapter (LoRA) on SDXL's cross-attention layers is trained to predict the
interaction term at every denoising step and add its prediction back into the PoE run, without the
LoRA ever seeing the joint prompt at inference (Mono-free). What a LoRA does here is in
[world/lora-corrector.md](../world/lora-corrector.md). The open research question is not whether
this closes the gap on one memorised pair (it does, per `MASTER_PLAN.md` Objective 1), but how far
the fix reaches: one trained pair, a pooled set of seeds, a group of similar pairs, or a single
corrector spanning the whole studied set of pairs. That reach is what
[purpose/03-what-working-means.md](03-what-working-means.md) defines as the test.

## Where this came from

Navigation: ⬅️ [What this project does about it](#what-this-project-does-about-it) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| PoE composition fails on 15 of 17 pool pairs at fail-rate 1.00 | Read in `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` | 2026-08-24 |
| The headline compose rate needs a stated bound, not a bare number | Read in `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` | 2026-08-24 |
| The interaction term is the gap the fix targets | Read in `paper/iclr/iclr2027_conference.tex`, abstract | 2026-08-24 |
| The reach question (one pair vs group vs whole taxonomy) is the open question | Read in `MASTER_PLAN.md`, Mission and Objectives | 2026-08-24 |
