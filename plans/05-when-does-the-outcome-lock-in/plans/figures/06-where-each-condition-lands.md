# 🖼️ Where each condition lands, and when it gets there

**What this plan asks:** in the scorer's image-embedding space, where do "a cat" alone, "a dog"
alone, the joint prompt, product-of-experts and product-of-experts plus the trained correction
each land, and at which denoising step does each run commit to where it ends up?

Step 52 in the root running order; waits on nothing for its first three rungs (cache and
existing renders only); its fourth rung is one GPU render job. Next is
[the grid and the figures](05-the-grid-and-the-figures.md), which reads the commit steps this
plan produces.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md — <one line on what actually happened>
```

## Recommended skill

`/design-figure` for the axis-picture layout in rung 2; `/pair-figure` for the endpoint-plus-track
pair in rung 4. Everything else is custom scripts.

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 48 (previous) | [the grid and the figures](05-the-grid-and-the-figures.md) | the speciation table and the scope's paper figures |
| **52 (current)** | **Where each condition lands** | **five conditions as clouds in DINOv2 space, axis pictures decoded through a representation autoencoder, per-step tracks with a commit step each, first on cat×dog then on the unseen pairs** |
| 51 (beside) | [twisted SMC on a learned twist](../../../06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md) | the sampler-side solution whose endpoints join this figure once its full run lands |

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## Quick context: where you are

Navigation: ⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The experiment.** One pair at a time, render the five conditions on the same pinned initial
noise across the eight held-out seeds, embed every final image with the compose scorer's
DINOv2 ViT-S/14 encoder, and draw one point per seed per condition in a single PCA plane fit over
all of them. Then add time: decode each step's Tweedie estimate along a run, embed it the same
way, and draw the run as a track through that plane.

> **DINOv2** is the image encoder the compose scorer already uses; in its space "a cat picture"
> and "a dog picture" sit in separable clusters, which is what makes "where did it land" a
> readable question. The noisy-latent PCA plane the repo already has keeps a quarter of the
> variance and behaves like pixel PCA, so it is kept for "when do paths separate" only.

> **A representation autoencoder (RAE)** is a decoder trained to reconstruct pixels from frozen
> DINOv2 features (Zheng et al. 2025, arXiv 2510.11690, weights under `nyu-visionx` on the Hub).
> Walking along a principal component and decoding each point gives a strip of pictures that
> says what the axis means, in place of a legend.

**The hypothesis.** Product-of-experts endpoints sit in a third region between the single-animal
clouds or inside one of them, and never inside the joint-prompt cloud. Endpoints with the
correction at λ 1.2 move toward the joint-prompt cloud along the same direction seed after seed
(per-seed cosine between the λ 1.0 and λ 1.2 arrows above 0.9). On the tracks, the corrected run
forks from the product-of-experts run inside the first ten steps, which is the window the
injection experiments already named.

**Context details.** The correction is the rank-32 adapter at step 30050 from
`/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`,
applied over the full 50-step window at λ 0, 1.0 and 1.2. Those renders for cat×dog seeds 9 to
16 already exist under `outputs/showcase/figure_r32_030050/renders/full/`. The single-prompt
references render with the same sampler settings (50 DDIM steps, guidance 7.5, 1024 square, the
training cache's held-out step-0 latents).

**This plan's job.** Give the scope one picture in which the correct case, the failure and the
fix are the same kind of object (a point, then a track), on axes whose meaning is shown.

**Associated materials.** Verdict: [the review file](../../review/06-where-each-condition-lands.md).
Design reasoning: [the decision ledger](../../decisions-taken-here.md). The repo's noisy-latent
plane, kept as the secondary view:
`artifacts/drips/showcase-the-trained-adapter/manifold/manifold_data.json`.

**For the full picture.** The scope's [master plan](../../MASTER_PLAN.md).

## Considerations

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime.** Rung 1: the 24 reference renders take about six seconds each on a Blackwell
device, the plot runs on CPU in under a minute. Rung 2: the decoder download plus a few hundred
decodes, minutes. Rung 3: 50 VAE decodes per run and the same number of DINOv2 forwards, cache
only, minutes per pair. Rung 4: eight pairs, eight seeds, five arms at 50 steps with trajectories
saved, about 320 renders, under an hour on one Blackwell device.

**Prerequisites.** The rank-32 step-30050 renders present; the training cache's held-out split
present; `co3_bw` on mscluster110 to 112 (the `co3` build has no kernels for those cards, which is
how this plan's first render attempt died silently on 2026-09-05).

**Project tracking.** Outputs to `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/`;
filed figures to `artifacts/results/where-does-each-condition-land/`. No W&B run for rungs 1 to 3;
rung 4 logs to `prime_lab/poe-repair-animals-compose`.

**Known issues.** See [Error Matrix](#error-matrix).

<details>
<summary>Environment Facts This Plan Depends On</summary>

- `co3_bw` on the Blackwell nodes and `co3` elsewhere; see [nodes](../../../../environment/hpc/nodes.md).
- The [shared-device launch path](../../../../runbook/running-things-on-the-cluster/launching-and-harvesting-a-run.md#2-launch-on-a-shared-device)₁ over SSH with `nohup`, its 1 GB device guard, and harvesting by
  `pgrep` on the node; see [the execution protocol](../../../../environment/hpc/execution-protocol.md).
- Large artifacts to `/datasets` only; the session node's NFS view of `/datasets` can lag the
  launch node by minutes, so count outputs and assemble figures on the launch node.
- fp16 upcast rule for the Tweedie estimate and for every distance computed in rung 3.

</details>

## The claim

Navigation: ⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**One plane, five conditions, eight seeds: the failure is a place, the fix is a direction, and
both are readable without a legend because the axes are pictures.**

**Why this matters right now:** the paper's mechanism section argues from norms and cosines that
nobody can see. This is the figure that lets a reader see the same claim.

## Why this plan exists

Navigation: ⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** The repo has compose rates (a number per condition) and a noisy-latent plane (paths,
no semantics). Nothing shows where each condition's images sit relative to each other.

**The approach.** Borrow the design of Helbling and Chau's Diffusion Explorer (arXiv 2507.01178):
precomputed trajectories, one time slider, arrows drawn on the current sample. Their models are
2D toys, so nothing of theirs is ported; the idioms are rebuilt over our cache.

**Key insights:**
1. The overlap question ("is there a cat-and-dog distribution that PoE misses") is answered by
   distances in the full 384-dimensional space, written to the sidecar. The plane is for looking,
   the sidecar is for checking.
2. Eight seeds show where things land. They do not support a separation claim; the review file
   carries that as an explicit limit.

## What happens (visual)

Navigation: ⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
 PC2 (decoded strip along this axis drawn at the left edge)
  |         .  .            joint prompt "a cat and a dog"  (8 seeds, hull)
  |       .  o  .
  |   x x                      x  PoE, no correction
  |  x  x  --->  +  +          +  PoE + 1.2 x correction (arrow per seed from its x)
  |
  |  [cat cloud]        [dog cloud]
  +--------------------------------------------  PC1 (decoded strip along the bottom)

 rung 3 adds a track per run: step 0 (grey blur, uncommitted) ... step 49 (the point above)
```

## Description: what to build

Navigation: ⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The reference renders** `scripts/showcase/where_each_condition_lands_render.py`: "a cat",
   "a dog", "a cat and a dog" for seeds 9 to 16 on the cache's pinned latents. Launched through
   `scripts/showcase/where_each_condition_lands_shared_device.sh`.
2. **The endpoint figure** `scripts/showcase/where_each_condition_lands_plot.py`: six conditions
   (the three references, PoE at λ 0, corrected at λ 1.0 and λ 1.2), one PCA over all 48 points,
   hulls for the three reference clouds, an arrow per seed from PoE to corrected, the plane's
   variance share printed on the figure, and a sidecar carrying every point's cosine distance to
   each cloud centroid, the per-seed arrow cosines, and the inter-cloud gaps.
3. **The axis pictures** `scripts/showcase/where_each_condition_lands_axes.py`: load the DINOv2-B
   RAE decoder, first reconstruct one known chimera render and one joint render and save both
   beside their originals, then decode seven points along PC1 and PC2 of the endpoint PCA and
   paste them as strips on the figure's axes.
4. **The tracks** `scripts/showcase/where_each_condition_lands_tracks.py`: for one seed, decode the
   Tweedie estimate at every step of the joint run and the PoE run (both from the cache), embed,
   project into the same plane, draw as tracks with the step number every ten steps, and write
   per track the commit step: the first step after which the nearest cloud centroid never changes.
5. **The unseen-pair render** `scripts/showcase/where_each_condition_lands_render.py --pairs`:
   the same five arms for the eight held-out pairs, trajectories saved as `latent_trajectory.pt`
   beside each image, so rungs 2 to 4 run over unseen pairs including the corrected track.

## Purpose and goal

Navigation: ⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

**Purpose.** Objective 2 of [the master plan](../../MASTER_PLAN.md) asks at which step the final
image is decided; this plan reads that step off a semantic track and gives the whole scope its
one explanatory picture.

**Goals:**
1. The endpoint figure and sidecar for cat×dog, filed with its card.
2. Axis strips whose decoded frames are labelled as decoder reconstructions, with the chimera
   reconstruction check saved and judged.
3. A commit step per track, and the fork step between the PoE and corrected tracks, per seed, on
   at least one unseen pair.

## Tasks

Navigation: ⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Paste: `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md`
  - Done when: the report comes back clean, or its proposals have been applied.

▶ **Next: [task 1.1](#1--the-endpoint-figure-on-catdog)**.

### 1. 🖼️ The endpoint figure on cat×dog

◀ **Needs: [task 0.1](#0--check-this-plan-before-working-from-it)**.

- [x] **1.1** Render the three references for seeds 9 to 16 on mscluster110 device 0 through the
  [shared-device launcher](../../../../runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md#1-render-the-references-and-the-per-step-frames-on-a-shared-device)₂ (2026-09-05, PID 397295, six seconds per image).
  - **Done when:** 24 PNGs under `where_each_condition_lands/{solo_a,solo_b,joint}/` and
    `render_manifest.json` beside them, counted on the launch node.
- [x] **1.2** [Run the plot](../../../../runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md#2-make-the-endpoint-figures-and-their-sidecar) on the launch node and copy the figure and sidecar into
  `artifacts/results/where-does-each-condition-land/` with a card entry (2026-09-05; the
  cloud-axes view, the PCA view, the contact sheet, the sidecar and the card are filed).

    ```bash
    ssh mscluster110 'cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min && \
      /home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python scripts/showcase/where_each_condition_lands_plot.py'
    ```

  - **Done when:** `cat-x-dog-in-dino-space.png` and `.json` exist, the sidecar holds 48 rows,
    and the variance share of the plane is printed on the figure.

▶ **Next: [instruction 5.1](#5--read-the-endpoint-figure)**, then [task 2.1](#2--the-axis-pictures).

### 2. 🎨 The axis pictures

◀ **Needs: [task 1.2](#1--the-endpoint-figure-on-catdog)**, so the PCA basis exists.

- [x] **2.1** Write `where_each_condition_lands_axes.py`: the DINOv2-B RAE decoder and its
  ImageNet stats from `nyu-visionx/RAE-collections` sit under
  `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/rae/`; the
  decoder code is vendored at `scripts/showcase/rae_vendor/` (MIT) and the encoder is torch hub
  `dinov2_vitb14_reg`, because the cluster's transformers build predates the class RAE's wrapper
  uses. Two chimera renders (PoE seeds 9 and 10), one joint render and one corrected render are
  reconstructed in `axes/reconstruction_check.png` (2026-09-05, mscluster108 device 1).
  - **Done:** the chimeras survive; the review file records the errors.
- [x] **2.2** Seven points along each cloud axis (mean ± 2 standard deviations of the 48
  projections), built in the decoder's token space because it reads patch tokens rather than the
  plane's class token, decoded and pasted on the cloud-axes figure as
  `axes/cat-x-dog-in-dino-space-cloud-axes-with-axis-pictures.png`, every frame captioned as a
  reconstruction; token-space both-ness agrees with the plane's at Spearman 0.87.
  - **Done:** filed with its card entry; x decodes cat to dog, y decodes one animal to two.

▶ **Next: [task 3.1](#3--the-tracks-from-the-cache)**.

### 3. 🛤️ The tracks from the cache

◀ **Needs: [task 1.2](#1--the-endpoint-figure-on-catdog)**.

- [x] **3.1** The tracks, built as three scripts rather than one: `where_each_condition_lands_trajectories.py`
  re-renders all six conditions for seeds 9 to 16 and decodes the Tweedie estimate at 13 saved
  steps; `where_each_condition_lands_frames_embed.py` [embeds the frames and projects them](../../../../runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md#3-embed-the-frames-and-export-the-pages-data) onto
  the cloud axes into the animated [scene](../../../../artifacts/scenes/where-each-condition-lands/README.md),
  [built from Diffusion Explorer's UI](../../../../runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md#4-build-the-page-from-diffusion-explorers-ui-and-open-it);
  `where_each_condition_lands_dynamics.py` reads the commit step per run (`COMMIT_TOL = 0.10`
  in source: both-ness stays within 0.10 of its final value) and the fork step
  (`FORK_MIN_DIST = 0.05`) into `commit-and-fork-steps.json`, and draws
  `both-ness-over-denoising-steps.png` and `frames-seed-15-strip.png` (2026-09-05).
- [ ] **3.2** Add the four arrows at the current step (A, B, their PoE sum, joint) on the PoE
  track, projected into the plane, with their true norms and pairwise cosines printed beside
  them, for the steps 0, 5, 10, 20 and 40 as a strip.
  - **Done when:** `tracks/cat-x-dog-seed-9-arrows.png` exists.

▶ **Next: [task 4.1](#4--the-unseen-pairs-with-the-corrected-track)**.

### 4. 🚀 The unseen pairs, with the corrected track

◀ **Needs: [tasks 1.2 and 3.1](#3--the-tracks-from-the-cache)**, so the pipeline is proven on one pair.

- [x] **4.1** Extend the render script with `--pairs` over the eight held-out animal pairs and
  five arms (A, B, joint, PoE, PoE plus rank-32 step-30050 correction at λ 1.2, full window),
  seeds 9 to 16, 50 DDIM steps, `latent_trajectory.pt` saved beside every image. Built as
  `scripts/showcase/where_each_condition_lands_pairs_render.py` with its launcher
  `where_each_condition_lands_pairs_shared_device.sh`; in flight on mscluster106 device 1 since
  2026-09-05 15:54 (about 30 s per reference run and 64 s per adapter run on the RTX 8000). A
  first launch on mscluster111 ran on the CPU because that card reads "GPU requires reset" and
  torch sees no device; the launcher now aborts when `torch.cuda` is unavailable.
  - **Done when:** 320 `image_1024.png` and 320 `latent_trajectory.pt` counted on the launch
    node under `where_each_condition_lands/pairs/<pair>/<arm>/seed_<n>/`, and each pair's
    `frames_manifest.json` lists them.
- [x] **4.2** Run the endpoint plane, the contact sheet, the both-ness curves and the commit and
  fork steps over every held-out pair, judged against the bar above, with the instance count
  scored on the same images as a post-hoc column: `scripts/showcase/where_each_condition_lands_pairs_analyze.py`
  on mscluster108 device 1, 2026-09-06, filed under
  `artifacts/results/where-does-each-condition-land/pairs/` with `verdict.json`. Verdict null on
  the both-ness bar (0 of 7 counted pairs pass both halves, 5 overlap); instance count reads the
  correction composing on 8 of 8 pairs. The per-pair track figures and animated pages are not
  drawn; the frames and trajectories for them are on disk.
  - **Done:** one endpoint figure, one contact sheet and one both-ness curve per pair are filed,
    and `verdict.json` holds one row per pair with the commit and fork steps per seed.

▶ **Next: [instruction 5.2](#5--read-the-endpoint-figure)**.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted.

- [ ] **Capture the failures this plan hit.**
  - Paste: `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md`
  - Done when: each failure has a catalog entry, or there were none.
- [ ] **Bring the tree current.**
  - Paste: `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md — <one line>`
  - Done when: statuses match reality.

▶ **Next: what has to pass before this runs.**

## Instructions

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

**For you to follow manually.** Do these yourself.

### 5. 👁️ Read the endpoint figure

◀ **Needs: [task 1.2](#1--the-endpoint-figure-on-catdog)** done.

5.1 **Open** `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.png`.
   - Expected result: three hulls (cat, dog, joint), eight black PoE points, eight arrows to the
     λ 1.2 points, the variance share printed in the corner.
   - ✅ If the PoE points sit outside the joint hull and the arrows point toward it, answer the
     review file's first question ✅ and read the sidecar's `per_seed` cosines to confirm.
   - ❌ If PoE points sit inside the joint hull, or arrows scatter, answer ❌ and copy the seeds
     into "Asked after the result".

5.2 **After task 4.2**, open one unseen pair's track figure and write down, per seed, the commit
   step of the joint track, the commit step of the PoE track, and the fork step. Enter them in
   the review file's table.

▶ **Next: what has to pass before this runs**, then [the grid and the figures](05-the-grid-and-the-figures.md).

## What has to pass before this runs

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **The figure is only as honest as its sidecar**: a 2D plane can put two far-apart points side
> by side, so every claim made from the picture is checked against the full-space distances.

- **Pass criteria:**
  - PoE endpoints' nearest centroid is never the joint cloud on cat×dog.
  - Per-seed cosine between the λ 1.0 and λ 1.2 arrows above 0.9 for at least 6 of 8 seeds.
- **Fail criteria (STOP and reassess, not abandon):**
  - The chimera does not survive RAE reconstruction: the axis strips stay, and no decoded frame
    may stand in for a PoE output anywhere in the figure.
  - Cloud gaps (cosine distance between centroids) under 0.05: the space does not separate the
    conditions and the figure cannot be read.
- **Partial pass guidance:**
  - Some seeds' PoE points nearest the joint cloud: those are the seeds where PoE composes
    already; list them, keep them in the figure, and check them against the scorer's verdict.

**The bar for the unseen pairs (task 4), written before that render runs.** On cat × dog the
nearest-centroid bar above failed and the both-ness read was taken after the result, so it is
pre-registered here for the next eight pairs and nowhere else. Both-ness is the projection of a
render's DINOv2 embedding onto the unit vector from the midpoint of that pair's two
single-animal centroids to its joint-prompt centroid, computed per pair on that pair's own eight
reference renders. Commit step is the first saved step after which a run's both-ness stays
within `COMMIT_TOL = 0.10` of its final value.

- **Support** if, on at least 6 of the 8 held-out pairs, the eight PoE both-ness values and
  the eight λ 1.2 both-ness values do not overlap (`max(PoE) < min(corrected)`), and the
  corrected runs' median commit step is at or before the PoE runs' median commit step.
- **Null** if the two bands overlap on 4 or more pairs.
- **Inconclusive** between, or if the reference clouds of a pair fail to separate (centroid
  cosine gap between the two single animals under 0.05), in which case that pair is excluded
  from the count and named.
- **The expected exception:** elephant × penguin composes by default, so its PoE band is
  expected to sit inside the joint band already; it counts toward neither support nor null and
  is reported on its own line.
- **What would surprise:** a pair where the corrected band sits below the PoE band, meaning
  the correction pushed the pair away from its joint prompt.

**When you get results, answer** [the review file](../../review/06-where-each-condition-lands.md).

## Figure Catalog

Navigation: ⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

#### Pending: to be generated from prompts

| Item | Lane | Prompt file | What it shows | Save to |
|------|------|-------------|---------------|---------|
| (none) | | | | |

#### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|------|------|-------------|--------------|--------|---------|
| Endpoints on cloud axes, cat×dog | — | one point per seed per condition; x is the unit vector from the cat centroid to the dog centroid, y the unit vector from their midpoint to the joint centroid orthogonal to x, both in DINOv2 cosine units; hulls for the three references; an arrow per seed from PoE to λ 1.0 to λ 1.2 | `where_each_condition_lands_plot.py` | ✅ | `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-cloud-axes.png`; the read is in the card |
| Endpoints in the PCA plane, cat×dog | — | the same 48 points in PC1 and PC2 of their embeddings (22% of variance), thumbnails on the four extremes | `where_each_condition_lands_plot.py` | ✅ | `.../cat-x-dog-in-dino-space.png` + `.json` |
| Contact sheet, cat×dog | — | eight seeds by six conditions, the renders behind every point | `where_each_condition_lands_plot.py` | ✅ | `.../cat-x-dog-contact-sheet.png` |
| Dose strip, seeds 10 and 11 | — | joint prompt, PoE, plus λ 1.0, plus λ 1.2 for the two seeds whose dose arrows agree least | PIL over the existing PNGs | ✅ | `.../cat-x-dog-seeds-10-11-dose-strip.png`; every figure above is read in `.../figure-explainer.md` and the numbers are filed in [the finding on where each condition lands](../../../../report/when-does-the-outcome-lock-in/where-does-each-condition-land.md) |
| Reconstruction check | — | chimera and joint renders beside their RAE reconstructions | `where_each_condition_lands_axes.py` | ⏳ | `.../axes/reconstruction_check.png` |
| Endpoint plane with axis pictures | — | the same plane with seven decoded frames along each axis, captioned as decoder reconstructions | `where_each_condition_lands_axes.py` | ⏳ | `.../cat-x-dog-in-dino-space-with-axis-pictures.png` |
| The animated page | — | every run's running estimate as a moving point in the cloud-axes plane, slider over the 50 steps, trails and thumbnails per seed | `where_each_condition_lands_frames_embed.py` + the scene | ✅ | `artifacts/scenes/where-each-condition-lands/`; screenshots `.../scene-step-0.png`, `scene-step-20.png`, `scene-seed-15-step-50.png` |
| Both-ness over steps | — | y both-ness of the running estimate, x denoising step, thin line per seed, thick mean per condition, for joint, PoE and λ 1.2; commit and fork steps in the JSON beside it | `where_each_condition_lands_dynamics.py` | ✅ | `.../both-ness-over-denoising-steps.png` + `commit-and-fork-steps.json` |
| Seed 15 frame strip | — | six conditions by seven saved steps, the decoded running estimate | `where_each_condition_lands_dynamics.py` | ✅ | `.../frames-seed-15-strip.png` |
| Four arrows at five steps | — | A, B, PoE sum and joint predictions projected at steps 0, 5, 10, 20, 40, true norms and cosines printed | `where_each_condition_lands_tracks.py` | ⏳ | `.../tracks/cat-x-dog-seed-9-arrows.png` |
| Per unseen pair: endpoints and tracks | — | the two figures above per held-out pair, the corrected track added, fork step marked | tasks 4.1 to 4.2 | ⏳ | `.../<pair>/` plus `commit_and_fork_steps.json` |

#### Organization workflow

1. Run on the launch node; 2. Read; 3. Copy into `artifacts/results/` with its card; 4. Link here.

## Orchestration: keeping catalogs and plan files in sync

Navigation: ⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md` | **task 0.1** | conformance reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md` | **the close out** | errors catalogued |
| Bring the tree current | `/sync-plan-tree @plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md` | **the close out** | statuses match reality |
| Organize outputs | copy from the launch node + update Figure Catalog | after each rung | deliverables linked |

## Code references

Navigation: ⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**File:** [poe_repair/training_cache.py](../../../../poe_repair/training_cache.py)
**Relevant section:** the per-step record (`x_t`, `eps_a_raw`, `eps_b_raw`, `eps_j_raw`,
`eps_uncond`) and the closed forms for the guided predictions, which rung 3 reads.

**File:** [scripts/showcase/lambda_window_grid.py](../../../../scripts/showcase/lambda_window_grid.py)
**Relevant section:** `run_lora_residual_inject_windowed_poe`, the sampler that applies the
adapter over a window and already tracks the trajectory; rung 4 reuses it with the rank-32
checkpoint and λ 1.2.

**File:** [scripts/trajectory_divergence.py](../../../../scripts/trajectory_divergence.py)
**Relevant section:** `dino_embed_paths`, the DINOv2 ViT-S/14 recipe every embedding here copies.

```python
# rung 3, per step of the PoE run
eps_poe = gs * (eps_a_raw + eps_b_raw - 2 * eps_uncond) + eps_uncond      # fp32
x0_hat  = (x_t - sqrt(1 - abar_t) * eps_poe) / sqrt(abar_t)
img     = vae.decode(x0_hat)                                              # 1024 square
z       = dino(img)                                                        # 384, L2-normed
xy      = (z - mu) @ vt[:2].T                                              # the endpoint plane
```

## Next step

Navigation: ⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

[The grid and the figures](05-the-grid-and-the-figures.md) reads the commit steps written here
beside its own speciation numbers.

## Error Matrix

Navigation: ⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>One failure so far</summary>

#### From global catalog

(Patterns applicable across all projects)

#### From project catalog

- **Wrong python build for the card.** The first reference render on mscluster110 was launched
  with `co3`, whose torch has no kernels for the Blackwell card; the process exited without a
  traceback and left an empty output folder. The launcher now picks `co3_bw` on mscluster110 to
  112, the same switch `scripts/twisted_smc/train_twist.sbatch` uses.

**Auto-update note:** regenerated by `/sync-plan-tree`; do not edit manually.

</details>
