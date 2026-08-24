# 🌍 Where the data comes from

Every number and image in this project traces back to one SDXL sampling run, cached at every
denoising step, then scored by a fixed rule. This file walks that journey once, stage by stage,
so a reader can place any file they find on disk.

> 🖼️ **Diagram wanted**: the process-lane piece "prompt pair to cached trajectory to scorer to
> figure", per [diagram-prompts.md](../diagram-prompts.md). Not yet rendered.

## Table of contents

- [The journey](#the-journey)
- [The two filesystems](#the-two-filesystems)
- [Where this came from](#where-this-came-from)

## The journey

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-two-filesystems) ➡️

**① A pair and a seed choose the run.** ✅

Two animal prompts (see [world/animal-pair.md](../world/animal-pair.md)) and a seed fix which
model call is made. `poe_repair/pairs.py` names the base pair list; the fail-rate table that
selects the current pool now lives at
`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` (the directory this file's
sources were read from, `outputs/animals_compose_transfer/`, no longer exists on disk, see
[Still open](../00-INDEX.md#still-open)). `data/pilot/seed_<n>/<slug>/` is the per-cell output
layout this project has used since the pilot stage. ✅

**② SDXL runs the 50-step denoising schedule under one or more render methods.** ✅

For a given cell, several render methods can be run: plain PoE, Mono (the literal joint prompt),
PoE plus the interaction term injected over a window at a chosen strength (λ), or PoE plus the
trained LoRA's own prediction. `poe_repair/methods/_sampling.py` (per `README.md`'s repo layout)
holds the functions for each: `run_cfg`, `run_cfg_poe`, `run_teacher_residual`,
`run_lora_residual_inject`. Model: `stabilityai/stable-diffusion-xl-base-1.0`, verified directly
from `data/pilot/seed_42/a_cat__x__a_dog/summary.json`'s `model_id` field. ✅

**③ Per-step state is cached, not only the final image.** ✅

Because every step's latent state is needed to measure the interaction term and to decode
intermediate estimates, the run caches enough to recompute the model's running estimate of the
finished image at any step (the Tweedie-formula estimate `EXPERIMENTS.md`'s EXP-01 uses). This is
what lets several later measurements (commitment step, window sweeps) be re-derived from the cache
alone, with no further GPU sampling, per `EXPERIMENTS.md`'s repeated "cache only" compute notes. 🔍

**④ A cell's images and summary are written to disk.** ✅

A pilot cell (`data/pilot/seed_42/a_cat__x__a_dog/`) holds five PNGs (PoE, Mono, each concept
alone, a trajectory-manifold plot) and a `summary.json` carrying the interaction term's per-step
and final magnitude. Full-pool runs write the equivalent to `outputs/<family>/` on
`/home-mscluster` or its mirror on `/datasets` (see [The two filesystems](#the-two-filesystems)).

**⑤ The compose-rate scorer reads the final image and produces a label.** ✅

`GroundingDINO` (`IDEA-Research/grounding-dino-tiny`) is run against the rendered image, counts
distinct "animal" instances, and the rule in
[world/compose-rate.md](../world/compose-rate.md#what-a-compose-rate-is) turns that count into a
`compose` or `blend` label. Per-pair rates are aggregated across seeds
(`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`).

**⑥ Labelled cells feed figures and review-file verdicts.** ✍️

A figure (tracked in `MASTER_PLAN.md`'s paper table) reads a set of cells' labels and images
directly; a plan's review file records the verdict against a bar fixed before the run
(`~/.claude/EXPERIMENT_CONVENTIONS.md`). Neither figures nor verdicts are restated in this folder;
see [Cross-linking](../00-INDEX.md).

## The two filesystems

Navigation: ⬅️ [The journey](#the-journey) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Run bytes live on two filesystems that are not always in sync, and a script has to be told which
one it is writing to.** ✅

`/home-mscluster` is the git working tree; `/datasets` is a larger, non-git mount used for large
artefacts, per this project's own `CLAUDE.md`: "Large artifacts go to `/datasets` only, because
`/home-mscluster` hit 100% once and silently killed checkpointing." `RETROFIT.md`'s census recorded
several run families split across both, with some folders (e.g. `interaction_term/dose`, 6.3G)
existing as real data on one filesystem and a zero-byte placeholder on the other, as of
2026-08-23. 🔍 This context file does not restate the environment's paths, sizes, or disk-guard
rule; the file that owns them is
[environment/storage.md § Access paths](../../environment/storage.md#access-paths).

## Where this came from

Navigation: ⬅️ [The two filesystems](#the-two-filesystems) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The render-method functions and their names | Read in `README.md`, repo layout section | 2026-08-24 |
| A cell's on-disk shape and its model id | Read directly, `data/pilot/seed_42/a_cat__x__a_dog/summary.json` | 2026-08-24 |
| The scorer's compose/blend rule | Read in `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` | 2026-08-24 |
| The two-filesystem split and its sync gaps | Read in `RETROFIT.md`, section 2 ("Moving") | 2026-08-24 |
| The "cache only" measurement principle | Read in `EXPERIMENTS.md`, EXP-01 and EXP-04 compute notes | 2026-08-24 |
