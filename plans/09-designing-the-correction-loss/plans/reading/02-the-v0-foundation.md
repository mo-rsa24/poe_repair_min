# 📚 The V0 foundation

**Five training runs and 136 checkpoints sit on disk with no page that reads them. This plan writes that page, and its shape is the shape every later variation copies.**

**Step 72 in the root running order. Waits on 71 for the paired-checkpoint cadence, though its own reading works on what already exists. Gates nothing, but every later report page is judged against the one it produces.**

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md — six pillar files written, the chart script in scripts/, the 00 page filed with its rank-ablation figure and computed band
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 71 (previous) | [01: the three instrument fixes](../tools/01-the-three-instrument-fixes.md) | the number, the cadence and the config |
| **72 (current)** | **02: the V0 foundation** | the page that says what today's objective produces |
| 73 (next) | [03: the V0a read](../hypothesis/03-the-v0a-read.md) | four experiments already on disk, judged |

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

- **V0, filed as `00`**: the objective running today. All three branches adapted, the target built from the cache, the guidance weight on both sides. A run is `00` when its `config.json` carries none of `compose`, `kappa`, `orth_weight`, `null_anchor`, `energy_penalty`, `loss_space`, `branch_prompt_style` or `ema_decay`.
- **The seed-noise band**: the spread of held-out compose rate across the eight held-out cat × dog seeds at one reference checkpoint. A second checkpoint landing inside that spread is a null. Pre-registered in `review/09-experiment-b-rank-16-32.md` of scope 01, never computed.
- **Matched steps**: comparing two ranks at the same optimizer step. Comparing rank 8 at 30k against rank 32 at 30k is one axis; comparing rank 8 at 30k against rank 16 at 50k is two, and says nothing about rank.
- **Drift**: the DINOv2 distance from the corrected render to the joint-prompt render minus the distance to the plain product render. More negative means closer to the target.
- **A comparison strip**: the labelled three-panel picture each training run writes at a checkpoint epoch, showing the joint-prompt target, the plain product, and the adapter's render.

---

## Quick context: where you are

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**What this plan produces**

One page that answers, in one read, what the objective running today produces at rank 8, 16 and 32, with the pictures on it and the dense detail one click away. The page states what those runs trained on: 88 cells over 11 look-alike animal pairs, which is not the scope pool. They are the history this scope starts from, and no later variation is compared against them.

**Why it is not a write-up of nothing**

The numbers exist. Two report files already carry them with eight rendered figures, and both are dense enough that nobody re-reads them. This plan does not replace those files: it links them. What it adds is the page that makes them findable and the two things they never did, which are a computed seed-noise band and a rank comparison at matched steps.

**The one thing it cannot inherit**

`make_closeout_figures.py` drew the existing eight figures and was kept in a session scratchpad. It is not in the repo, so nothing in that folder can currently be redrawn by anyone.

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime.** No training. The scoring pass over eight seeds at a handful of checkpoints is minutes per checkpoint on a free card. The writing is the bulk of the work.

**Prerequisites.** None blocking. Task 2.3's band needs renders that already exist.

**The standing constraint on every page this plan writes.** Critical information on the face, thumbnails rather than described figures, links rather than restated content, rendered equations where the equation is clearest, short sentences with blank lines between them. A page that would run long is the wrong page.

**Project tracking.** W&B, project `prime_lab/poe-repair-animals-compose`.

**Known issues.** See the [Error Matrix](#error-matrix).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- [Storage](../../../../environment/storage.md): the five runs live under `/datasets/mmolefe/poe_repair_min/outputs/showcase/` except `phase1_r8_100k`, which is in the repo under `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/`. Nothing is moved by this plan.
- [Throughput](../../../../environment/hpc/throughput.md): the file this plan adds the memory-per-rank row to. It has rank-32 timing and no memory column.
- [Overview](../../../../environment/overview.md): cached tensors are float16 and any analysis stacking many of them upcasts to float32 first. The scoring pass in task 2.1 must do so.
- [Nodes](../../../../environment/hpc/nodes.md): the scoring pass needs one healthy card; pin a node rather than trusting `sinfo`.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One page, four panels of thumbnails, one rank table and a link to the detail, and a script that can redraw every figure on it. A reader arriving in six months learns what the current objective produces, at every rank, without opening anything else.**

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Ask what the current objective produces and the answer is scattered across two dense report files, eight figures with no README linking them to an objective, 136 checkpoints on two filesystems, per-checkpoint json in thirteen separate directories, and prose in one markdown file that holds the only copy of the LoRA weight norms. Nothing names an objective anywhere.

**The solution.** One page, filed under the variation's number and name, that carries the verdict in its first line and links everything else.

**Three things the existing write-up left undone.**

1. The seed-noise band was pre-registered as the threshold in two review files and never computed, so neither review file's headline question has been answered in its own terms.
2. The rank comparison was explicitly disclaimed. The existing report says "Rank 8 at 30k, rank 16 at 50k and rank 32 at 30k differ on two axes, so nothing here ranks the ranks", which is exactly what the pre-registered bar asks for.
3. The figures cannot be redrawn.

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  thirteen results.json          ┐
  on /datasets, one per probe    │
                                 ├──▶  00-per-checkpoint.json  ──┐
  LoRA norms, prose only,        │     (one file, every rank,    │
  recomputed from checkpoints    ┘      every checkpoint)        │
                                                                 ▼
  renders already on disk  ──────────────▶  the band  ──▶  make_correction_loss_figures.py
  (8 held-out seeds)                        (computed,          │
                                             never before)      │ eleven figures
                                                                ▼
  __cmp strips on /datasets  ───────────▶  five-across strip ──▶  00's page
  (10 checkpoints per run)                                        one screen
                                                                  ├─ links: two dense reports
                                                                  ├─ links: the runbook recipe
                                                                  └─ disclosure: five runs, aborts
```

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`context/world/the-correction-loss.md`**: one shared file serving every variation. The three branches, the guidance weight, the cached target, rank as a dial, what a checkpoint is. Every variation's report page links it rather than restating it.
2. **An extension to `context/world/lora-corrector.md`**: rank is a dial with three measured values, not the fixed adjective "rank-8".
3. **`runbook/running-things-on-the-cluster/training-a-correction-adapter.md`**: the cold-start recipe, with a table of one row per variation naming the flag that selects it.
4. **A row in `environment/hpc/throughput.md`**: memory per rank, which decides 3090 against A6000.
5. **`00-per-checkpoint.json`**: compose counts, drift and LoRA norms for every rank at every probed checkpoint, in one file.
6. **`scripts/make_correction_loss_figures.py`**: eleven figures from that file, parameterised by run so every later variation reuses it.
7. **`report/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt.md`** and its artifact card.

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves objectives 2 and 3 of [the scope](../../MASTER_PLAN.md#objectives): one number and one name per variation in every pillar, and reading what already ran rather than re-running it.

1. The six pillar files exist and each carries only what belongs on its face.
2. `00`'s page carries the three ranks, the comparison strip, and the rank-ablation figure with its computed band.
3. Every figure on that page can be redrawn from one command.
4. The pre-registered rank question is answered in its own terms, either way.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

For Claude to execute.

### 0. 🧭 Check this plan before working from it

- [ ] 0.1 **Run the following prompt:**
  ```
  /verify-plan @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md
  ```
- [ ] 0.2 **Run the following prompt:**
  ```
  /xref-pillar @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md
  ```

▶ **Next: [tasks 1.1 to 1.4](#1--write-the-four-pillar-entries)**, the four pillar entries.

### 1. 📖 Write the four pillar entries

- [ ] 1.1 **Write `context/world/the-correction-loss.md`**
  - Defines: the three branches `a`, `b`, `u`; the guidance weight `w = 7.5` and why it sits inside the loss; the cached target; rank as a dial; what a checkpoint is and what the save cadence means
  - None of these terms appears anywhere in `context/` today
  - Produces: one file every variation's page links to
  - 💡 read `~/.claude/OUTPUT_GUIDE.md`'s context row first: it names the part of the context format most often got wrong
- [ ] 1.2 **Extend `context/world/lora-corrector.md`**
  - Today it says "rank-8" as a fixed property. Ranks 16 and 32 exist, are measured, and rank is a dial
  - Produces: an edit, not a new file
- [ ] 1.3 **Write `runbook/running-things-on-the-cluster/training-a-correction-adapter.md`**
  - The cold-start recipe: which python, which pool, which cells file, the launch line, what lands where, how long it takes. The runbook can resume and probe a training and has never been able to start one
  - Include the variation table: one row per variation naming its flag, and the reason `bigbatch` rather than `biggpu` (four experiments need four cards; `biggpu` allows one job per user)
  - Produces: one recipe file, with its question added to `runbook/00-INDEX.md`
- [ ] 1.4 **Add the memory-per-rank row to `environment/hpc/throughput.md`**
  - Gathered, not measured: rank 32 holds about 23.5 GB without gradient checkpointing and went out of memory on a 3090; rank 16 with checkpointing peaks at 11.2 GB of a 24 GB card
  - Produces: one row, marked as gathered from the review files that record it

▶ **Next: [tasks 2.1 to 2.5](#2--collect-the-numbers-and-draw-the-figures)**, which need no pillar file to exist first and can run in parallel.

### 2. 📊 Collect the numbers and draw the figures

- [ ] 2.1 **Collect the per-checkpoint numbers into one file**
  - Read the thirteen `results.json` files under `/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_*` and `lambda_boundary_probe*`
  - Recompute the LoRA Frobenius norms from the checkpoints, because they exist only as prose in one markdown file
  - Produces: `artifacts/results/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt/00-per-checkpoint.json`, one row per (rank, checkpoint, λ) with compose count, drift and norm
- [ ] 2.2 **Write `scripts/make_correction_loss_figures.py`**
  - Regenerates all eleven figures from that file: the eight that exist plus the three owed
  - Parameterised by run directory and output directory, so every later variation reuses it rather than getting its own script
  - The eight existing figures must come out matching what the two report files already embed. Where one cannot be reproduced, say so in the script's docstring rather than drawing something close
  - Produces: one script, eleven files
- [ ] 2.3 **Compute the seed-noise band**
  - The spread of held-out compose rate across the eight held-out cat × dog seeds at one reference checkpoint, `phase1_r32_100k` at 30,050
  - The definition is fixed in `plans/01-showcase-the-trained-lora/review/09-experiment-b-rank-16-32.md` and must be used as written there, not re-derived
  - Produces: the band, into `00-per-checkpoint.json`
  - 💡 `/design-figure` before task 2.4: the band can be drawn as an error bar, a shaded region or a reference line, and the three say different things about how strong the claim is
- [ ] 2.4 **Draw the rank-ablation figure**
  - Held-out compose rate on y, rank on x, at matched optimizer steps, band drawn
  - This is the figure `plans/01-showcase-the-trained-lora/plans/experiments/09-experiment-b-rank-16-32.md` catalogued and nobody made
  - Produces: `compose-rate-by-rank-matched-steps.png`
- [ ] 2.5 **Assemble the five-across comparison strip**
  - `[joint target][plain PoE][LoRA 10k][LoRA 30k][LoRA 100k]`, one row, from the `__cmp` renders already written per checkpoint under `samples/per_epoch/`
  - One row for a pair trained on, one for a pair never seen
  - The saved strips repeat the target and plain columns at every checkpoint, so two thirds of their space carries nothing new. The assembled row drops the repeats
  - Produces: `in-sample-strip.png` and `held-out-strip.png`

▶ **Next: [task 3.1](#3--write-the-page)**, which needs every figure above.

### 3. ✍️ Write the page

◀ **Needs: [tasks 1.1 to 2.5](#1--write-the-four-pillar-entries)** done, so every figure and every link target exists.

- [ ] 3.1 **Write `report/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt.md`**
  - Verdict in the first line. The equation rendered. Four panels of thumbnails: in-sample, held-out, the loss, the same seed over time. One rank table. Links to the two dense reports rather than their contents. The five runs and the identification rule behind a disclosure
  - The rank table carries the warning that its three entries sit at three different steps, until task 2.4's figure replaces it with the matched-step comparison
  - Produces: one page, one screen
- [ ] 3.2 **Write the artifact card**
  - `artifacts/results/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt/README.md`, one entry per item with how it was made
  - Produces: the card, per `~/.claude/ARTIFACT_TREE_FORMAT.md`

◀ **Needs: [task 3.1](#3--write-the-page)** done, so there is a page to read.

### Close out. 🔄 Record what this plan taught

- [ ] C.1 **Run the following prompt**, after any red run:
  ```
  /ingest-error-pattern --from-run-log
  ```
- [ ] C.2 **Run the following prompt:**
  ```
  /sync-plan-tree @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md
  ```

---

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

For you to follow manually.

### 4. 👓 Read the page as a stranger

◀ **Needs: [task 3.1](#3--write-the-page)** done.

- [ ] 4.1 **Open the page and read it once, top to bottom, without clicking anything**
  - ✅ You know what the objective produces and at which rank by the end of the first screen: it works
  - ❌ You had to open a link to understand the verdict: the verdict is in the wrong place. Back to 3.1
- [ ] 4.2 **Check each figure for its axes**
  - Every figure names what is on y, what is on x, and what one point is, either on the figure or in one line beneath it
  - ❌ Any figure that does not: back to 2.2, because the fix belongs in the script rather than in the caption
- [ ] 4.3 **Look for restated content**
  - Anything on the face that a linked file already says in full is a cut
  - ❌ Found: back to 3.1

### 5. 📈 Compare the eleven regenerated figures against the eight that exist

◀ **Needs: [task 2.2](#2--collect-the-numbers-and-draw-the-figures)** done.

- [ ] 5.1 **Open each regenerated figure beside the one embedded in the existing report**
  - Path: `artifacts/results/does-training-longer-help-the-pooled-lora/` for the originals
  - ✅ Same curve, same values: the script is right and the figures are now reproducible
  - ❌ Different: do not silently adopt the new one. Record which differs and by how much in the review file, because one of the two is wrong and the existing one is already cited

▶ **Next: nothing. This plan closes here.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> This page is the template every later variation copies. If it is wrong, eight more pages are wrong the same way.

- The six pillar files exist and none restates what it links to
- Every figure on the page can be redrawn by one command
- The seed-noise band is computed from the definition already written, not a new one
- The rank-ablation figure exists and the pre-registered question is answered either way

**Partial pass.** If the eight existing figures cannot be reproduced exactly, the plan still closes: the script draws what it can, the discrepancy is recorded, and the page keeps the originals. An unreproducible figure that is labelled as such is honest; a silently replaced one is not.

The review questions are in [the review file](../../review/02-the-v0-foundation.md).

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

**Pending, from [this scope's illustrated map](../../diagram-prompts.md)**

| Figure | Lane | What it shows | Status |
|---|---|---|---|
| corrloss-01-what-was-written-to-disk-once | subject | the cache's five tensors per entry | opened at task 2.1, to check the json's fields against what the cache holds |
| corrloss-05-checkpoint-and-picture-together | subject | a checkpoint and a render at the same step | opened at task 2.5, to check the strip's layout against it |
| corrloss-capstone-cache-to-one-page-per-objective | subject | the whole system, ending at one page per objective | opened at task 3.1, since this plan writes the first of those pages |

**Generated during plan execution**

| File | Lane | What it holds | Task | Status |
|---|---|---|---|---|
| `00-per-checkpoint.json` | — | compose count, drift and LoRA norm per rank and checkpoint | task 2.1 | ⏳ |
| `compose-rate-by-rank-matched-steps.png` | subject | compose rate against rank at matched steps, band drawn | task 2.4 | ⏳ |
| `in-sample-strip.png` | subject | target, plain PoE, and the adapter at three checkpoints, a pair trained on | task 2.5 | ⏳ |
| `held-out-strip.png` | subject | the same for a pair never seen | task 2.5 | ⏳ |
| `training-loss-three-ranks.png` | subject | fit loss and its step buckets, three ranks | task 2.2 | ⏳ |

**Organization workflow.** Everything is filed under `artifacts/results/designing-the-correction-loss/00-matching-the-composition-to-the-joint-prompt/` with its card entry. The eight existing figures stay where they are, in `artifacts/results/does-training-longer-help-the-pooled-lora/`, and the page links both homes. Moving them would break two report files, a runbook anchor and an artifact card.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md` | **task 0.1**, before any work | conformance and thin instructions reported |
| Cross-reference the plan | `/xref-pillar @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md` | **task 0.2**, before any work | terms already documented elsewhere linked |
| Capture patterns | `/ingest-error-pattern --from-run-log` | **the close out**, after any red run | errors added to the catalogs |
| Bring the tree current | `/sync-plan-tree @plans/09-designing-the-correction-loss/plans/reading/02-the-v0-foundation.md` | **the close out** | statuses, running order and Error Matrix match reality |

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**New file:** `scripts/make_correction_loss_figures.py`

```python
"""Every figure for one correction-loss variation, from one collected json.

Replaces make_closeout_figures.py, which drew the eight existing figures and was
kept in a session scratchpad rather than the repo, so nothing in
artifacts/results/does-training-longer-help-the-pooled-lora/ can currently be
redrawn by anyone.

    co3 python scripts/make_correction_loss_figures.py \
        --numbers artifacts/results/designing-the-correction-loss/<NN-name>/<NN>-per-checkpoint.json \
        --out     artifacts/results/designing-the-correction-loss/<NN-name>/
"""
```

**Read-only source:** the thirteen probe directories under `/datasets/mmolefe/poe_repair_min/outputs/showcase/`. Each `results.json` carries `rows[]` with `seed`, `window`, `lambda`, `image_path`, `max_delta_norm`, `checkpoint_step`, `compose`, `n_instances`, `cat_and_dog` and `drift`.

**Read-only source:** `samples/per_epoch/<epoch>/{in_in,out_out}__<pair>__seed<NN>__cmp.png` under each run, the three-panel strips task 2.5 reassembles. The plain `.png` beside each one is the same render at 1024² without labels baked in.

**The band's definition:** `plans/01-showcase-the-trained-lora/review/09-experiment-b-rank-16-32.md`, the `## Bars` section. Used as written; not re-derived.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ `/report-pulse` ✅ — owns the claim, figure and statistic shape `00`'s page has to carry, and audits `report/` for the discovery routes the new folder needs.
   alt: `/design-figure` on task 2.4 before it is drawn, since how the band is drawn changes how strong the claim looks.

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents)

[03: the V0a read](../hypothesis/03-the-v0a-read.md) judges the four experiments already on disk against the page this plan produced.

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
