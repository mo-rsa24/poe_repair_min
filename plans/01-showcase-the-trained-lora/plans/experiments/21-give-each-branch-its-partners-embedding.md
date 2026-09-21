# 🧪 Give each branch its partner's embedding: does the adapter fit the early correction better when it is told the other animal?

**This plan asks one question: if each single-prompt branch of the adapter also receives the pooled text embedding of the other prompt, does the held-out fit to the true correction in the early steps rise from the branch-blind 0.925 toward the training pairs' 0.985, and does the 30k render get nearer the joint-prompt image?**

**Step 63 in the root running order. Waits on [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) and [20-an-ema-of-the-adapter-weights](20-an-ema-of-the-adapter-weights.md) for the regularisation setting it trains on, so that this run differs from the best regularised run on the partner axis only.**

## Recommended prompt (after this plan completes)

```
/analyze-run phase1_r32_partner_40k
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 57 (feeds this) | [15-experiment-d-weight-decay](15-experiment-d-weight-decay.md) | whether weight decay 0.1 is the base setting |
| 62 (feeds this) | [20-an-ema-of-the-adapter-weights](20-an-ema-of-the-adapter-weights.md) | whether the EMA is part of the base setting |
| **63 (current)** | **21: the partner embedding** | the architecture-side fix: the branch is told the other animal |
| next | [05-assemble-the-showcase-figures](../figures/05-assemble-the-showcase-figures.md) | the wall, from whichever checkpoint the three fidelity plans leave best |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Words this plan uses](#words-this-plan-uses)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **A branch**: one of the three UNet forwards the adapter's correction is assembled from: "a cat" alone, "a dog" alone, and the empty prompt. Each branch today sees only its own prompt; the adapter has to read the other animal off the latent (the pressure-test route, part 1).
- **The pooled embedding**: SDXL's second text-encoder vector for a prompt (`text_embeds`, 1280 numbers), fed to the UNet beside the token sequence. The token sequence carries the words; the pooled vector carries a summary of the prompt.
- **The partner embedding**: the "a cat" branch receives `pool_cat + pool_dog` as its pooled vector and the "a dog" branch `pool_dog + pool_cat`; the empty branch is unchanged. The token sequences stay their own, so the branches remain distinct. Applied only when the adapter is enabled: the frozen product-of-experts forward and every reference render are untouched.
- **The on-cache fit**: the cosine between the adapter's correction and the true correction on the cached training states, no sampling, per pair tier and per step bucket, from `scripts/showcase/fit_cosine_on_cache.py` ([the recipe](../../../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#2-measure-how-well-a-checkpoint-fits-the-cached-correction)). The branch-blind rank-8 adapter at 30k reads 0.985 on training pairs and 0.925 on cat × dog in the early bucket, 0.961 against 0.748 late.
- **Drift** and **compose count**: as in plan 20, from the 8-seed grid at λ 1.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** The route's part 5 puts branch blindness as a soft ceiling: the adapter reaches 0.97 on training pairs, so the latent carries enough of the partner to read, yet the early held-out fit sits at 0.925 against 0.985. The partner embedding is the smallest change that gives the branch the other animal directly, and the on-cache fit is a read that needs no render.

**The hypothesis.** With the partner embedding, the early held-out fit rises to at least 0.955, and the 30k render's drift is more negative than the same-setting branch-blind run's by the 0.03 the grids tell apart.

**If true.** The adapter's design changes for the paper, and the transfer story gains a mechanism sentence: the correction is a function of the pair the branch is told about.

**If false.** Branch blindness is not the early gap's cause; the late gap (0.961 against 0.748), which more pairs address, is the whole story, and the pair-scaling plan in scope 04 moves up.

**Dataset.** The same 11 pairs and 88 cells; held-out cat × dog seeds 9 to 16 for the grid; the cache's held-out tiers for the fit.

**Associated materials.**
- Review questions: [the review file](../../review/21-give-each-branch-its-partners-embedding.md)
- The argument: [the pressure-test route, parts 2, 5 and 7](../../../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
- The fit numbers this is judged against: [run 1 in the idea map](../../../../artifacts/ideas/improving-the-pooled-lora-run/run-01-fit-cosine-r8-030000.md)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Cost.** One rank-32 training to 40k (13 to 16 hours on an RTX 3090), the on-cache fit at 30k (about 20 minutes, no sampling), two 8-seed grids (about 15 minutes each). One Slurm job plus two short ones.

**Buys.** The review file's one question, and the mechanism sentence if supported.

**Why it waits.** Trained on weight decay 0 it would differ from the 30,050 baseline on one axis but from the regularised runs on two. Training on the setting plans 15 and 20 support keeps every comparison single-axis.

**The inference side.** The injection sampler must add the partner vector on the adapter-enabled forward and nowhere else, and the fit script must encode the pair the same way; both read one helper so the two cannot drift apart.

**W&B project:** `prime_lab/poe-repair-animals-compose`, run id `phase1_r32_partner_40k`.

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- Training as a Slurm job on `bigbatch` with the faulted nodes excluded, or over SSH on a free `biggpu` device ([nodes](../../../../environment/hpc/nodes.md), [execution protocol](../../../../environment/hpc/execution-protocol.md)).
- Checkpoints and grids to `/datasets` only ([storage](../../../../environment/storage.md)).
- The on-cache fit reads fp16 tensors and upcasts before accumulating ([overview](../../../../environment/overview.md)).

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Telling each branch the other animal closes part of the early held-out fit gap and renders nearer the joint-prompt image at the showcase step.**

**Why this matters right now.** It is the one lever on the correction's direction rather than its size or its integration; the other fidelity plans leave the direction where it is.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The route names branch blindness as a soft ceiling and no run has lifted it.

**The solution.** The smallest architectural change that lifts it, read first on the cache where it costs nothing to sample, then on the grid.

**Key insights.**
1. The pooled vector is the right slot: adding to the token sequence would change every cross-attention key and value, while the pooled vector enters once through the time embedding and leaves the per-token attention intact.
2. The frozen forward never sees the partner, so plain PoE and Mono are unchanged and the comparison stays single-axis.
3. The on-cache fit answers the fit question before any render, and the grid answers the render question after.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
branch        tokens (unchanged)     pooled vector, adapter on        pooled vector, adapter off
"a cat"       [a cat ...]            pool_cat + pool_dog              pool_cat
"a dog"       [a dog ...]            pool_dog + pool_cat              pool_dog
""            [...]                  pool_∅                           pool_∅

loss        = || PoE(adapter branches) − eps_J(cached) ||²        as every pooled run
inference   = PoE(frozen, adapter off) + λ · (PoE(adapter on, partner) − PoE(frozen, adapter off))
read 1      on-cache fit cosine, early bucket, cat × dog: 0.925 today
read 2      8-seed grid at 30k, λ 1: drift against the same-setting branch-blind run
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **One helper** `partner_pooled(pool_a, pool_b, pool_e)` returning the three pooled vectors for the adapter-on forward, used by the trainer's step (`_train_one_step`, the `text_embeds` of `cond_3K`), by the injection sampler on its adapter-enabled forward, and by the fit script.
2. **The trainer flag** `--partner-embedding` on the pooled trainer, recorded in `config.json`, default off.
3. **The sampler flag** on `scripts/showcase/lambda_boundary_probe.py`, read from the checkpoint's config so a partner checkpoint cannot be rendered without it.
4. **The fit script flag**, the same way.
5. **The launcher**: `scripts/showcase/regularised_r32.sbatch` with `RUN_ID=phase1_r32_partner_40k` and the flag, on the regularisation setting plans 15 and 20 support.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves goal 4 and the transfer question in scope 04. Checkable outcomes:

1. A training run whose `config.json` reads `partner_embedding: true`, 8 checkpoints on `/datasets`.
2. The fit table at 30k with the early held-out cosine beside the branch-blind number.
3. Two grids at 30k (partner against the same-setting branch-blind run) and the review file's question answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/01-showcase-the-trained-lora/plans/experiments/21-give-each-branch-its-partners-embedding.md`. Done when the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--build-the-partner-path)**.

### 1. 🔧 Build the partner path

◀ **Needs:** plans 15 and 20 read at 30k, so the base setting is known.

- [ ] **1.1 Write the helper and the three flags** (Description items 1 to 4), with a unit check that the adapter-off forward's pooled vectors are byte-identical to today's.
- [ ] **1.2 Smoke the trainer** with `--dry-run --partner-embedding` on the session node's CPU-free path or a free device; done when `config.json` carries the flag and one epoch runs.

▶ **Next: [task 2.1](#2--train-fit-and-render)**.

### 2. 🧪 Train, fit and render

- [ ] **2.1 Launch** `RUN_ID=phase1_r32_partner_40k PARTNER=1 sbatch scripts/showcase/regularised_r32.sbatch` (the launcher gains a `PARTNER` switch in task 1.1) with `WEIGHT_DECAY` and `EMA_DECAY` set to the supported base setting. Record job id, node and W&B id in the review file.
- [ ] **2.2 The on-cache fit at 30k**: [the recipe](../../../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#2-measure-how-well-a-checkpoint-fits-the-cached-correction) with the partner flag; the early-bucket cat × dog cosine into the review file beside 0.925.
- [ ] **2.3 The 30k grid**: the probe recipe, `--windows full --lambdas 1.0`, out-root `figure_r32_partner_030000`; its drift against the same-setting branch-blind 30k probe folder.

▶ **Next: [instruction 3.1](#3--judge)**.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Judge

◀ **Needs: [task 2.3](#2--train-fit-and-render)** done.

3.1 **Open the two 30k grids side by side** (partner, branch-blind). One word per seed into the review file before the numbers.

3.2 **Judge against the bars** in the review file and write the verdict.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

- The unit check in task 1.1: with the adapter off, the pooled vectors are today's, so plain PoE and Mono are unchanged.
- Plans 15 and 17 have their 30k verdicts, so the base setting is not a guess.

**Fail criteria:** the on-cache fit on training pairs falls below 0.95 in the early bucket (the partner vector broke the fit rather than helping it); then the grid is not rendered and the plan closes null on the fit alone.

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

Every figure lands in `artifacts/results/does-telling-each-branch-the-other-animal-help/`, with its card entry in that folder's `README.md`.

| Figure | What is plotted | What it argues | What it may not claim | File |
|---|---|---|---|---|
| fit by tier and bucket | x step bucket (early, commit, late), y cosine between the adapter's and the true correction on cached states; one line per tier (training pairs, seen words unseen pairing, no seen word), partner solid, branch-blind dashed | whether the partner closes the early gap | anything about renders | `fit-cosine-partner-vs-branch-blind.png` |
| the two 30k grids | 8 seeds at λ 1, partner above, branch-blind below, the count under each tile | whether the render moved | anything about later steps | `grid-30k-partner.png`, `grid-30k-branch-blind.png` |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

### 4. 🧹 Close out

- [ ] **4.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (after any red run).
- [ ] **4.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

- `poe_repair/experiments/one_pair_one_seed/trainer.py`: `_train_one_step`, where `cond_3K["text_embeds"]` is built
- `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`: the flag and `config.json`
- `scripts/showcase/lambda_boundary_probe.py`: the adapter-enabled forward in `lambda_window_grid.run_lora_residual_inject_windowed_poe`
- `scripts/showcase/fit_cosine_on_cache.py`: the on-cache fit

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

`/run-experiment` once plans 15 and 20 have their 30k verdicts; `/analyze-run` when the training lands.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

Support: the adapter's design changes and scope 04's transfer plans train with the flag on. Null: the on-policy training arm in the pressure-test route (part 7, item 2) is the last direction-side lever, bounded at 0.04 of cosine by run 2 in the idea map.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

| Symptom | Cause | Fix |
|---|---|---|
| plain PoE through the probe differs from the cached `poe.png` by more than 6 grey levels | the partner vector leaked into the adapter-off forward | the unit check in task 1.1 is the guard; fix the helper's call site |
| the fit on training pairs collapses | the added vector doubles the pooled norm and the time embedding saturates | average the two pooled vectors instead of summing, record the choice, rerun the smoke |
| `has no 'partner_embedding' key` in the probe | a checkpoint from a run without the flag | the probe reads the flag from the checkpoint's config; branch-blind checkpoints render as today |
