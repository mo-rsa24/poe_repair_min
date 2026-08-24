# 🌍 What we produce

For a chosen pair of concepts, a seed, and a render method, the pipeline produces a generated
image and a small set of numbers describing how that image was made and whether it succeeded.
This file says what those outputs are, in the words a reader of the paper or the plan tree would
use, and links to the dictionary for the exact fields.

## Table of contents

- [The output, concretely](#the-output-concretely)
- [Who reads it](#who-reads-it)
- [What they do differently](#what-they-do-differently)
- [Where this came from](#where-this-came-from)

## The output, concretely

Navigation: 📋 [TOC](#table-of-contents) | [Next](#who-reads-it) ➡️

**One generated image per (pair, seed, render method, step schedule), plus the per-step latent
trajectory it was decoded from.** ✅

A cell such as `data/pilot/seed_42/a_cat__x__a_dog/` holds `poe.png` (plain PoE), `monolithic.png`
(Mono, the joint-prompt cheat), `solo_a.png` and `solo_b.png` (each concept alone, the two
"experts"), `trajectory_manifold.png` (a 2-D projection of the denoising path), and
`summary.json` (the per-step and final size of the interaction term, `d_t_poe_vs_mono` and
`d_T_poe_vs_mono`; see [data/02-dictionary.md § d_T](../data/02-dictionary.md#d_t-d_t)). Read
this session directly. ✅

**One compose/blend label and a compose rate, per pair, from the scorer.** ✅

`artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` and `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`
carry the validated verdict: an image counts as "compose" when a GroundingDINO detector finds at
least two distinct "animal" instances after non-max suppression. See
[world/compose-rate.md](../world/compose-rate.md) for what that scorer can and cannot tell a
reader. ✅

**One trained LoRA checkpoint per training configuration**, a set of weights on SDXL's
cross-attention layers, used at inference to reconstruct the interaction term without seeing the
joint prompt. See [world/lora-corrector.md](../world/lora-corrector.md). ✍️

## Who reads it

Navigation: ⬅️ [The output, concretely](#the-output-concretely) | 📋 [TOC](#table-of-contents) | [Next](#what-they-do-differently) ➡️

**The paper's evidence register.** ✅

The images, compose rates, and trajectory figures feed a set of named figures tracked in
`MASTER_PLAN.md`'s paper table (steps 13 and 14) and written up into the ICLR 2027 submission at
`paper/iclr/`. A figure's own reasoning lives beside the plan that owns it, never copied into this
folder (see [Cross-linking](../00-INDEX.md) in the format this folder follows).

**The plan tree's verdicts.** ✅

Every run is judged against a bar written before the run, per this project's `CLAUDE.md` and
`~/.claude/EXPERIMENT_CONVENTIONS.md`. W&B (project `prime_lab/poe-repair-animals-compose`) holds
the curves; the plan tree holds only the verdict, the run id, and the bar it was judged against.

## What they do differently

Navigation: ⬅️ [Who reads it](#who-reads-it) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A verdict on transfer reach changes what the paper claims to deliver.** ✍️

Per `MASTER_PLAN.md`'s Expected Outcome: if a single LoRA composes across the whole studied set of
pairs and seeds, the paper claims one deployable corrector. If it does not, the paper claims a
catalogue: one trained corrector per group of similar pairs, each backed by its own held-out-seed
and held-out-pair evidence. Both are documented outcomes; neither is a failure of the project, per
`MASTER_PLAN.md`'s Definition of Done.

## Where this came from

Navigation: ⬅️ [What they do differently](#what-they-do-differently) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| A pilot cell's five images and its summary.json fields | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/` | 2026-08-24 |
| The compose/blend rule and its scorer | Read in `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` | 2026-08-24 |
| W&B owns the numbers, the plan tree owns the verdict | Read in project `CLAUDE.md` | 2026-08-24 |
| The two possible shapes of the deliverable (single LoRA vs catalogue) | Read in `MASTER_PLAN.md`, Expected Outcome | 2026-08-24 |
