# 🧪 Training on the renders that composed

**Variation `06`. The target stops being the model's joint-prompt prediction and becomes the noise added to one of its own renders that a person judged to show both concepts. It escapes a teacher known to be wrong, and it pays for that with a corpus of 43 cells and a noisier target.**

**Step 75 in the root running order. Waits on 74, because it moves the target and the empty branch against `00` at the same time, so it can only be read once `01` has settled the empty branch. Last in the scope by design.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md — the second data path built, the scaling curve read against its bar, and the verdict recorded
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 74 (previous) | [04: freezing the empty branch](04-freezing-the-empty-branch.md) | the empty branch settled, so this plan moves one axis and not two |
| **75 (current)** | **05: training on the renders that composed** | a second data path, gated by a scaling curve |
| next | none | `02` is authored by plan 04's close-out; nothing follows this one |

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
- [Success/Failure Outcomes](#successfailure-outcomes)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Words this plan uses

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

- **The teacher**: the model's own prediction given the joint prompt, which every objective from `00` to `04` trains against. It is known to be wrong on some cells: one scope already drops training cells whose joint-prompt target shows the wrong animals.
- **A render that composed**: an image the model itself drew that a person judged, by eye, to show both concepts as separate things. The scope pool is `cells_v57.json`, 43 cells over 29 pairs. `cells_v56.json`, 72 over 30, is the wider set this plan may reach for only if the curve says 43 is not enough, and reaching for it changes the pool, so it is a second experiment and not a longer version of this one.
- **The second data path**: what this plan builds. Today training reads five cached prediction tensors per step. This path instead takes a render, encodes it to a latent, adds noise drawn now, and trains the composition to predict the noise that was added.
- **The scaling curve**: compose rate against how many cells were trained on, at 10, 20, 30 and 43, each set nested inside the next so the only thing that changes is the count.
- **A high-variance target**: the noise added to one image on one draw. The teacher it replaces is a prediction, which is already an average over what the model believes. Swapping one for the other trades a biased target for a noisy one.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this plan asks**

Is a corpus of 43 hand-picked renders enough to train against, and does escaping the teacher buy more than the noisier target costs?

**Why the corpus is that size and cannot grow**

The rule that promoted this variation was that a filter exists which can tell a cat beside a dog from two dogs. That rule is met for these two cell sets, because a person selected them by eye. It is not met for growing past them: the automatic filter reports both species present on single-species images at 0.323 for turtle × tortoise against a 0.10 bar, and the pool is look-alike species throughout. So the corpus is fixed until someone sits down and picks more by hand.

**Why it is last**

Every other variation in this scope changes how the three branches are combined and holds the target fixed. This one changes the target. Run against `00` it moves two axes at once, which is why it waits for `01` to settle the first.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime.** Building the data path is the bulk of the work. Four nested training runs at 50,000 steps each is about 80 hours if run one after another, or one overnight on four cards.

**Prerequisites.** Plan 04's verdict, so the empty branch is settled and this plan moves one axis. Plan 01's paired cadence, so every render has its weights.

**The corpus.** `artifacts/_shared/cross_pair_pool_configs/cells_v57.json`, the scope pool. It does not contain cat × dog, which is the held-out test pair and never trains. Eight of its 43 cells need [task 1.5](../tools/01-the-three-instrument-fixes.md) first: `a_typewriter__x__a_cactus` seeds 1 to 8 sit in both cache directories and are otherwise taken from the held-out copy.

**The guidance question is already settled elsewhere.** `--null-anchor` exists in `train_pooled.py` and defaults to off. This plan does not touch it.

**Project tracking.** W&B, project `prime_lab/poe-repair-animals-compose`.

**Known issues.** See the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [Storage](../../../../environment/storage.md): the encoded renders are a new artifact on `/datasets`, alongside the existing training cache. They are small next to it, because one render encodes to one latent rather than fifty steps of five tensors.
- [Overview](../../../../environment/overview.md): the VAE encode runs in fp16 and the result is upcast to fp32 before the branches are combined, the same rule the existing path follows.
- [Throughput](../../../../environment/hpc/throughput.md): rank 32 at about 2,750 steps an hour on a 49 GB card. The new path adds one VAE encode per cell, done once and cached, not per step.
- [Nodes](../../../../environment/hpc/nodes.md): four nested runs at once need four cards, so `bigbatch`, pinned from a live idle probe. `biggpu` allows one job per user.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Training against pictures the model actually got right, instead of against its own answer to the joint prompt, reaches a higher compose rate on held-out pairs, and 43 cells is enough to get there.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Every objective so far trains the composition to match the model's own joint-prompt prediction. That prediction is unreliable: a scope already drops training cells because their joint target shows the wrong animals, and two of the four held-out cells in this project have a reference that is visibly wrong. The ceiling is the teacher.

**The solution.** Use pictures instead. The model does sometimes draw both concepts, and when it does, a person can see it. Take those renders, encode them, add known noise, and train the composition to predict the noise. The target is then right by construction exactly where the joint prompt was wrong.

**The risk the variations note does not name.** The noise added to one image on one draw is a high-variance target. The prediction it replaces is already an average. With 43 cells, that variance may cost more than escaping the teacher buys, and nothing in the note's argument rules it out. The scaling curve is what exposes it: if variance dominates, the curve does not rise with cell count.

**What `05` being dead changes here.** Nothing about the design. It removes the alternative. Real photographs were the other way to get a target that is not the model, and retrieval reached every pair while returning watermarked stock and merchandise. Renders the model itself got right are now the only non-model target available.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  the path that exists today                 the path this plan builds
  ──────────────────────────                 ─────────────────────────
  cache: x_t, eps_a, eps_b,                  a render a person judged
         eps_uncond, eps_J                   to show both concepts
    │  five tensors per step                   │
    ▼                                          ▼  VAE encode, once
  target = guided eps_J                      a clean latent x
    ▲                                          │  + noise z drawn now
    │ the model's own answer,                  ▼
    │ wrong on some cells                    x_t = x + sigma_t · z
                                               │
                                               ▼
                                             target = z, the noise added

  THE GATE, before anything is believed:
    train on 10, then 20, then 30, then 43 cells, each nested in the next
    compose rate on held-out pairs against cell count
      rising  → the corpus is enough, read the verdict
      flat    → 43 cells is not enough, or the noisy target eats the gain
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **A second data path** beside `poe_repair/training_cache.py`. It reads a cell from `cells_v57.json`, finds the render, encodes it through the frozen VAE once and caches the latent, then per training step draws noise, forms the noised latent at that step's sigma, and returns the drawn noise as the target. The three branches are computed as they are today.
2. **Four nested cell subsets** at 10, 20, 30 and 43, each a strict subset of the next, drawn from `cells_v57.json` so the only axis that moves is the count.
3. **The scaling curve**: compose rate on the held-out pairs against cell count, with the bar in source.
4. **The verdict**, read against `01` on the same held-out seeds.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objectives 4 and 5 of [the scope](../../MASTER_PLAN.md#objectives): test the variations in the note's order, one axis at a time, and end with one objective named.

1. A second data path exists and a smoke run proves it trains.
2. The scaling curve exists over four nested cell counts.
3. The sizing question is answered either way, before any comparison against `01` is read.
4. If the curve rises, a verdict against `01` on the same seeds.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [ ] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md
  ```
- [ ] 0.2 **Run the following prompt:**
  ```
  /frame-hypothesis training on the renders that composed, in plans/09-designing-the-correction-loss
  ```
  - The scaling curve's bar goes into the review file before the data path is built
  - Produces: the bar, written and dated ahead of the runs

▶ **Next: [tasks 1.1 to 1.3](#1--build-the-second-data-path)**, the data path.

### 1. 🔧 Build the second data path

- [ ] 1.1 **Resolve each cell to its render**
  - Read `artifacts/_shared/cross_pair_pool_configs/cells_v57.json`, 43 cells over 29 pairs, and find the image each (pair, seed) names
  - Produces: a manifest listing every cell with the path to its render, and a printed count
  - Done when the count reads 43, not when the script exits 0. A cell that resolves to nothing is a silently smaller corpus
- [ ] 1.2 **Encode the renders once and cache the latents**
  - Frozen VAE, fp16 encode, result stored fp32, on `/datasets` beside the existing training cache
  - Produces: one latent per cell, and a printed total size
- [ ] 1.3 **Write the sampler side of the path**
  - Per training step: draw noise, form the noised latent at that step's sigma, return the drawn noise as the target. The three branches are computed exactly as they are today
  - Behind a flag, so every earlier variation still runs unchanged from the same code, and the flag is recorded in `config.json`
  - Produces: one new data-path switch
- [ ] 1.4 **Smoke it**
  - Two epochs on 10 cells
  - Produces: a log line naming the data path and the cell count, and a falling loss
  - Done when the log names both, not when the job exits 0

▶ **Next: [tasks 2.1 and 2.2](#2--the-scaling-curve-the-gate)**, the gate.

### 2. 📈 The scaling curve, the gate

◀ **Needs: [task 1.4](#1--build-the-second-data-path)** passing, so there is a path that trains.

- [ ] 2.1 **Build four nested cell subsets**
  - 10, 20, 30 and 43 cells, each a strict subset of the next, from `cells_v57.json`
  - Assert the nesting in code. Four independently sampled subsets would confound count with which cells were drawn
  - Produces: four cell files and a printed check that each is contained in the next
- [ ] 2.2 **Train all four to 50,000 steps**
  - Rank 16, checkpoints and render sets every 2,500 steps, four cards on `bigbatch`
  - Produces: four W&B runs, 20 checkpoints each
- [ ] 2.3 **Draw the curve**
  - Compose rate on the held-out pairs against cell count, at matched steps, with the bar drawn
  - Produces: `compose-rate-vs-cell-count.png` and `06-scaling.json`

▶ **Next: [instruction 4.1](#4--read-the-curve-and-decide)**, the read that decides whether task 3 happens at all.

### 3. 📐 Read it against 01

◀ **Needs: [instruction 4.1](#4--read-the-curve-and-decide)** answered yes, so the corpus is known to be enough.

- [ ] 3.1 **Score the held-out seeds at matched steps against `01`**
  - Produces: `06-per-checkpoint.json`, the same shape as the earlier plans'
- [ ] 3.2 **Read the strips blind**
  - Same procedure as plan 03's, shuffle key saved separately
- [ ] 3.3 **Write `report/designing-the-correction-loss/06-training-on-the-renders-that-composed.md`**

### Close out. 🔄 Record what this plan taught

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md
  ```

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 4. 👓 Read the curve, and decide

◀ **Needs: [task 2.3](#2--the-scaling-curve-the-gate)** done, so the curve exists.

- [ ] 4.1 **Open the curve and read it against the bar**
  - Path: `artifacts/results/designing-the-correction-loss/06-training-on-the-renders-that-composed/compose-rate-vs-cell-count.png`
  - ✅ Rising across the four counts and clearing the bar at 72: the corpus is enough. Go to task 3.1
  - ❌ Flat: 43 cells is not enough, or the noisy target is eating the gain. Stop. Do not run task 3, and do not read a comparison against `01` off four runs that never learned. Record it as a null and say which of the two causes the evidence supports
  - ❌ Rising but still under the bar at 72: the corpus would work and there is not enough of it. That is a different finding, and what it asks for is a person picking more cells by eye
- [ ] 4.2 **Check the four runs are really nested**
  - Open the four cell files and confirm each is contained in the next
  - ❌ Not nested: the curve is confounded and means nothing. Back to task 2.1

### 5. 👁️ Read the strips blind

◀ **Needs: [task 3.2](#3--read-it-against-01)** done.

- [ ] 5.1 **Label the strips without opening the shuffle key**
  - Three labels per panel: clean, unclear, not two
  - ✅ All labelled before unblinding: the read is valid
  - ❌ Key opened first: void, reshuffle and repeat on another day

▶ **Next: [task 3.3](#3--read-it-against-01)**, the write-up.

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#successfailure-outcomes) ➡️

> The corpus cannot grow without a person picking cells by hand, so the sizing question is not a detail. If 43 cells is not enough, this variation is over regardless of what the target is worth.

**The question written before the runs.** Does compose rate on held-out pairs rise with cell count across 10, 20, 30 and 43?

**The bar, in source, not in prose.** `MIN_SCALING_SLOPE` and `MIN_COMPOSE_AT_FULL` in the scaling script, so neither can be adjusted after seeing the curve without showing up in a diff.

**Flat is a null and it is a finding.** It says either the corpus is too small or the noisy target costs more than the biased one did. The evidence separates them: if the loss falls normally but compose rate does not move, the corpus is the limit; if the loss itself is noisier than the earlier runs at matched steps, the target is.

**Partial pass.** Rising but short of the bar at 72 closes the plan with a request rather than a verdict: the corpus works and needs more hand-picked cells.

The review questions are in [the review file](../../review/05-training-on-the-renders-that-composed.md).

---

## Success/Failure Outcomes

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

Task 1.1 and task 2.1 both fail quietly, which is why they are named here.

**Task 1.1, resolving cells to renders.** A cell whose render is missing is skipped and the run trains on fewer cells than its name claims. The failure state is a corpus of 68 called 72, and the whole scaling curve is then drawn against wrong x values. The check is the printed count.

**Task 2.1, the nested subsets.** Four independently drawn subsets would look identical in every log and confound cell count with which cells were drawn. The failure state is a curve that answers a question nobody asked. The check is the containment assertion in code, read again by eye at instruction 4.2.

**Task 1.2, the encode.** A VAE encode at the wrong precision produces latents that are subtly off and a loss that still falls. The failure state is a run that trains successfully on a corrupted target. The check is one decode-and-look on the first cell.

---

## Figure Catalog

⬅️ [Previous](#successfailure-outcomes) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Pending, from [this scope's illustrated map](../../diagram-prompts.md)**

| Figure | Lane | What it shows | Status |
|---|---|---|---|
| [corrloss-01-what-was-written-to-disk-once](../../diagrams/corrloss-01-what-was-written-to-disk-once.png) | subject | the cache this plan adds a second path beside | 🖼️ rendered · opened at task 1.3, so the new path's shape is drawn against the old one |
| [corrloss-04-the-four-places-they-differ](../../diagrams/corrloss-04-the-four-places-they-differ.png) | subject | one machine, four settings; this plan moves the target setting | 🖼️ rendered · opened at task 0.2 |

**Generated during plan execution**

| File | Lane | What it holds | Task | Status |
|---|---|---|---|---|
| `06-scaling.json` | — | compose rate per cell count, with the bar | task 2.3 | ⏳ |
| `compose-rate-vs-cell-count.png` | subject | compose rate against cell count, four points, bar drawn | task 2.3 | ⏳ |
| `06-per-checkpoint.json` | — | compose count and drift against `01` at matched steps | task 3.1 | ⏳ |
| `blind/seed-NN.png` | subject | one strip per seed, ids instead of names | task 3.2 | ⏳ |

**Organization workflow.** Filed under `artifacts/results/designing-the-correction-loss/06-training-on-the-renders-that-composed/` with its card entry.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Fix the bar before the runs | `/frame-hypothesis training on the renders that composed, in plans/09-designing-the-correction-loss` | **task 0.2**, before the data path is built | the scaling bar dated ahead of the runs |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @plans/09-designing-the-correction-loss/plans/hypothesis/05-training-on-the-renders-that-composed.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `poe_repair/training_cache.py`
**Relevant section:** `CellPath` at line 47 and `resolve_cells` at line 188. The new path mirrors this shape rather than replacing it, so both can be selected by a flag.

```python
# the second path, beside the cached one
class RenderCell:
    """One cell whose target is a render the model got right, not its own prediction."""
    pair: str
    seed: int
    render: Path        # the image a person judged to show both concepts
    latent: Path        # its VAE encode, written once

# per training step, instead of reading five cached tensors:
#   x  = load(latent)                      # fp32
#   z  = torch.randn_like(x)               # drawn now, this is the target
#   xt = x + sigma_t * z
#   ... three branches at xt, composed as today, against z
```

**File:** `artifacts/_shared/cross_pair_pool_configs/cells_v57.json`
**Relevant section:** 43 cells over 29 pairs, keyed by pair slug with a list of seeds. `cells_v57.json` holds 43 over 29. Neither contains `a_cat__x__a_dog`, which is the held-out test pair and never trains.

**File:** `poe_repair/experiments/cross_pair_lora_pooling/train_pooled.py`
**Relevant section:** the argument parser, where the data-path switch joins `--null-anchor` at line 136, and the config dump at line 365, where plan 01 made objective flags recordable.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/frame-hypothesis training on the renders that composed, in plans/09-designing-the-correction-loss` ✅ — fixes the scaling bar in the review file before the data path exists, which is the ordering `verify-plan` checks against `git log`.
   alt: `/design-figure` on task 2.3, since four points and a bar can be drawn to look conclusive or honest.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

Nothing follows this plan. `02`, anchoring the empty branch to two animals, is authored by plan 04's close-out and takes its own place in the running order.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>No failures catalogued yet</summary>

**Purpose**: known issues and their fixes, regenerated by `/ingest-error-pattern` and `/sync-plan-tree`.

#### From global catalog

#### From project catalog

---

**Auto-update note:** regenerated by `/sync-plan-tree`. Do not edit by hand.

</details>
