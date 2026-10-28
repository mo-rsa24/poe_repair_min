# 🎯 Structural composition: can a toy reproduce the cat-dog hybrid, and does caption fine-tuning remove it?
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** red circles and blue squares where PoE of "circle" and "square" draws one purple
> overlay while the shapes' own pictures never show one: first from formulas alone, then from one small
> network trained like SDXL, then after [caption fine-tuning](what-poe-loses-when-it-composes.md#92-caption-fine-tuning-done-so-it-can-fail) on pairings it never saw. Every step ends on
> pictures. · **Time:** ~10 h over four sittings, all on the laptop · **Stop point:** the verdict on
> caption fine-tuning

Built to: route-file-template v1

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [Schedule](#schedule)
- [Instructions](#instructions) (you walk these)
- [Tasks](#tasks) (the AI writes these)
- [What already exists on the cluster](#what-already-exists-on-the-cluster)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

**Goal:** reproduce SDXL's cat-dog hybrid on shapes, where every answer can be checked, then test the
first remedy. Why a toy, and what "the model of a word" means here, is in
[§8.1 and §8.2 of the write-up](what-poe-loses-when-it-composes.md#8-structural-experiments-on-the-toy).

**Two phases.**

- **Phase 1 sets up the failure** with formulas only, in six checkpoints you can look at.
- **Phase 2 replaces the formulas** with a network trained like SDXL, checks it fails the same way, then tries fixes.

**Where it runs.** The laptop, in notebooks. Phase 1 needs only numpy and matplotlib. Phase 2 trains a
small network on the RTX 4070, in the `sdxl-local` environment (torch 2.5.1 with CUDA, built
2026-10-22).

**Already in.** A rough first run of phase 1 exists:
[`toy_sampler_hybrids.py`](../../artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py)
and its figure.

<a href="../../artifacts/notes/what-poe-loses-when-it-composes/figures/fig-toy-du-layout.png"><img src="figures/fig-toy-du-layout.jpg" width="420" alt="Du-style rows on red circles and blue squares: product of shapes trained alone gives overlays; product with the pair captioned under each name gives the shapes apart exactly but overlays from the samplers; the sum gives circles or squares"></a>

## What you will achieve

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Phase | What you see at the end | Step |
|---|---|---|
| 1. the failure, from formulas | one pixel worked by hand; the overlay from the exact product, none from the sum, the shapes apart once the pair is included, and what samplers draw (as SDXL runs PoE, and Du's) against the exact answer; then the shape range and guidance | 1 |
| 1b. averaging in SDXL's latent | whether averaging a cat's and a dog's latent gives a ghost or a fused animal, beside PoE | 1b |
| 2a. one trained network | a small network trained on the same captioned pictures draws both shapes for "circle and square", and the overlay under PoE, matching phase 1 | 2, 3 |
| 2b. the [locality test](what-poe-loses-when-it-composes.md#87-the-locality-test) and the switches | a heat map of where the word "circle" reaches into the square's pixels, per switch | 3 |
| 2c. caption fine-tuning | PoE on pairings never shown in fine-tuning: overlays, or the shapes apart | 4, 5 |
| verdict | one page: each hypothesis, its pictures, its reading | 6 |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

▶ **Recommended:** `you: 1 ‖ you: 1b → AI: 2 → you: 3 → AI: 4 → you: 5 → AI: 6`, with `AI: 7` alongside at any time

Each AI step writes a walk file like
[the six-checkpoint walk](today-the-toy-failure-in-six-checkpoints.md): the setup fixed up front, then one
notebook cell per step, each with what you should see. Each of your steps walks it with
`/drip-execute-plan`, typing every cell yourself.

## Schedule

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

| Sitting | Run | Checkpoint |
|---|---|---|
| 1, ~3 h | step 1 | the six checkpoint figures |
| 2, ~3 h | steps 2 and 3, first half | the trained network's pictures beside phase 1's |
| 3, ~2 h | step 3, second half, and step 4 | the locality heat maps per switch |
| 4, ~2 h | steps 5 and 6 | held-out pairings under PoE; the verdict page |

## Instructions

*For you to walk, one cell per message.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

### 1. 🌊 Phase 1: the failure in six checkpoints

You ~3 h · the laptop

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md
```

**The six, and two more:**

1. the training pictures;
2. one word's model, sampled;
3. one pixel worked by hand, then **the exact product: overlays**;
4. the exact sum: no overlays;
5. the caption dial;
6. the [samplers against the exact answer](today-the-toy-failure-in-six-checkpoints.md#6--the-samplers-against-the-exact-answer), **including PoE as SDXL runs it** (the guided sum, DDIM, guidance 1 and 7.5) beside Du's reverse diffusion and MALA;
7. the shape range: the product moves onto an in-between shape both words accept;
8. guidance: surer words pin the overlay, or hand it to the surer word.

Every sample grid has its noise-free row beside it. Stop at any checkpoint whose picture surprises you.

### 1b. 🌊 Averaging in SDXL's latent

You ~1.5 h · the laptop GPU

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/today-averaging-in-the-sdxl-latent.md
```

Asks whether the fused animal comes from SDXL's latent or from its denoiser.

### 3. 🌊 Phase 2a and 2b: one trained network, its locality, its switches

You ~4 h over two sittings · the laptop GPU

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/toy-one-network-many-captions.md
```

Written by step 2. Training runs launch from a cell; the walk moves to the next cell while it trains.

### 5. 🌊 Phase 2c: caption fine-tuning on pairings it never saw

You ~1.5 h · the laptop GPU

```text
/drip-execute-plan plans/10-what-poe-loses-when-it-composes/toy-captions-that-say-contains.md
```

Written by step 4.

## Tasks

*For the AI to write. Each writes one walk file and checks every picture it promises can be drawn.*

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#what-already-exists-on-the-cluster) ➡️

### 2. 🔧 Write the walk for one trained network

You ~10 min · AI ~45 min · Opus, high

```text
Background
Work in /home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. Phase 1 of the toy
(plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md) showed, with
formulas only, that the product of "circle" and "square" over pixels is a purple overlay. Phase 2
replaces the formulas with one small caption-conditioned network trained on the same pictures, the way
SDXL is one network trained on captioned photos.

Objectives
1. Write plans/10-what-poe-loses-when-it-composes/toy-one-network-many-captions.md in the shape of the
   six-checkpoint walk: the setup fixed up front, then one notebook cell per step, each with what I
   should see as a picture and a check.
2. Steps: the dataset with captions as phase 1; a small U-Net denoiser on 8 x 16 pictures conditioned
   on a caption through a learned word embedding, with condition dropout so one network gives
   conditional and unconditional predictions; training on the sdxl-local GPU kernel; samples for each
   caption and for PoE of "circle" and "square", in Du's layout with the dot map, beside phase 1's exact
   pictures; then the locality test as a heat map; then the switches one at a time.
3. Write before each picture what it should show, copied from the write-up's sections 8.6 to 8.8.

What to read
- plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md, sections 8.2 to 8.8.
- The six-checkpoint walk and the notebook it produced:
  artifacts/notes/what-poe-loses-when-it-composes/the-toy-failure.ipynb.

What is already known
- The sdxl-local environment has torch 2.5.1 with CUDA on an RTX 4070 laptop GPU with 8 GB.
- The cluster's run 2: small trained 2D experts match the exact rule closely and do not extrapolate.
- Bradley 2025b: a patch-local model trained on 1 object generates up to 5; a standard U-Net up to 1.
- What to reuse from Du and where to diverge is the table in the write-up's section 8.6: copy the
  schedule, training loop, samplers, composition and energy parameterisation from the cluster's
  rebuild; replace the 2D ResNet with a small convolutional U-Net on 8 x 16 pictures, and the separate
  unconditional models with one caption-conditioned model with condition dropout.
- PoE is run both as SDXL runs it (the guided sum with the unconditional prediction, DDIM, guidance 1
  and 7.5) and with Du's samplers (the six-checkpoint walk's checkpoint 6).

Rules
- Pictures first: every step ends on a figure, and a number appears only as a count of pictures.
- The base network must reproduce phase 1's checkpoint 3 (overlays under PoE) and the joint caption
  must draw both shapes before any switch is tried; say so in the walk as a stop point.
- Each switch changes one thing, as the write-up's section 8.8 lists.

What to hand back
- The walk file, with breadcrumbs and a contents list.
- A one-line timing estimate per training run on the 4070, from a 100-step trial you run yourself.

I judge it when I can read the walk top to bottom and know what picture each cell will draw.
```

### 4. 🔧 Write the walk for caption fine-tuning

You ~10 min · AI ~45 min · Opus, high

<a href="../../artifacts/notes/what-poe-loses-when-it-composes/figures/fig-caption-dial.png"><img src="figures/fig-caption-dial.jpg" width="300" alt="The caption dial: the exact product's share of two separate shapes rising from nothing as each word's pictures start to include the pair"></a>

```text
Background
This continues after phase 2a. If "circle" comes to mean "a picture containing a circle", PoE of
"circle" and "square" can find the two shapes apart. The test is whether that reaches pairings never
shown in fine-tuning, or only teaches the pairs shown.

Objectives
1. Write plans/10-what-poe-loses-when-it-composes/toy-captions-that-say-contains.md, same shape.
2. Steps: more shapes and colours (triangle, star, cross), so each shape can be shown beside many
   others; fine-tune the phase 2a network with each shape captioned only by its own name, beside varied
   others at varied places, holding out circle with square completely; PoE on the held-out pairing and on
   trained pairings, in Du's layout with the dot map; the locality heat map before and after; and "circle"
   alone, to see whether it now draws extra shapes.

What to read
- The write-up's section 9.2, and the phase 2a walk and notebook.
- The first rough run: with the pair captioned under each name, the exact answer held the two shapes
  apart, yet reverse diffusion drew overlays almost every time; so show the sampler's pictures beside
  the network's.

Rules
- Write the pass mark before any picture: the held-out pairing shows the shapes apart on at least half
  its samples, and the heat map in the other shape's region fades. Anything less is a null.

What to hand back
- The walk file.

I judge it when the walk names, before running, the picture that would mean the remedy failed.
```

### 6. 🔧 The verdict page

You ~10 min · AI ~20 min · Opus, high

```text
Background
This closes the toy route. Phases 1, 2a, 2b and 2c are walked, each notebook saved with its figures.

Objectives
1. Write artifacts/notes/what-poe-loses-when-it-composes/toy-verdicts.md: H-S1, H-S2 and the
   caption fine-tuning remedy, each with its pass mark, the figure that shows it, and one plain sentence.
2. Add one row per phase to the write-up's section 8.

Rules
- Copy what the notebooks drew. Do not redraw or recount.

I judge it when each verdict points at its picture.
```

### 7. 🔧 SDXL's caption dial: how often a picture of a cat with a dog is captioned with one name

You ~10 min · AI ~1 h writing, then cluster jobs · Opus, high · runs any time; its numbers feed steps 4 and 6

<a href="../../artifacts/notes/what-poe-loses-when-it-composes/figures/fig-caption-dial-pairings.png"><img src="figures/fig-caption-dial-pairings.jpg" width="300" alt="The toy's caption dial: of 16 pairings, only the pair with itself costs nothing, so the product draws the two shapes apart once both words hold the pair"></a>

```text
Background
Work in /home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. Checkpoint 5 of the
six-checkpoint walk (plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md)
showed on formulas that PoE of "circle" and "square" draws the two shapes apart only when each word's
own pictures include the pair, at share s: at sigma 0.1 any s above 0 flips every sample, and with the
pair under one name only the apart picture gets share s and no more. The write-up's section 9.2 ends
by asking for SDXL's own s. This step measures it on SDXL's public stand-in data.

Objectives
1. Estimate s_cat: of pictures whose caption names a cat and no dog, the share that also show a dog.
   Likewise s_dog with the roles swapped. Report each with a 95% interval and the sample size.
2. As a reference, the share of captions that name both animals among all captions naming either.
3. Write the result to artifacts/results/how-often-a-cat-with-a-dog-is-captioned-with-one-name/
   (caption-counts.md, the numbers as JSON, a contact sheet of the pictures called "both animals"),
   with a README card entry, and one plain row in report/ via the repo's report convention.

What to read
- scripts/caption_wetness.py and scripts/caption_wetness.sbatch: the same three stages (sample
  whole-word caption matches with DuckDB, fetch images from their URLs, score with CLIP ViT-bigG-14)
  already run for camels and dogs. Reuse them; add a mode or a sibling script rather than a new pipeline.
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md: the dataset, its path
  (/datasets/mmolefe/relaion2b-en/parquet/, 128 files, 2.1 billion rows), the DuckDB binary, and how
  a count carries its own file coverage.
- The write-up's section 9.2 and Figure 14.
- environment/overview.md and environment/00-INDEX.md before any job.

What is already known
- The caption search builds its working table on node-local disk ($TMPDIR); on /datasets the same
  scan took 47 minutes (commit 1f97ae9). Slurm sees 12 usable CPUs per bigbatch node (commit e6ceda7).
- The dog pattern in caption_wetness.py is \b(dog|dogs|puppy)\b. Choose and state the cat pattern
  before sampling.
- Every cluster action goes through scripts/cluster.sh from the laptop checkout; commit first.

Rules
- Write the bar before scoring: run CLIP's "is a dog in this picture" question on 50 pictures checked
  by eye, and use the CLIP calls only if they agree with the eye on at least 40 of 50. Otherwise report
  the eye-checked count alone and say so.
- Count from the data. A share with no sample size, or an animal call nobody looked at, is not a result.
- Large outputs on /datasets only; the disk guard checks the filesystem it writes to.

What to hand back
- s_cat and s_dog with intervals and sample sizes, the reference share, the folder path, and the job ids.
- One sentence placing s_cat and s_dog on the toy's dial (checkpoint 5) and on Figure 10's curves.

I judge it when s_cat and s_dog each come with an interval, a sample size, and a contact sheet I can
check by eye.
```

## What already exists on the cluster

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

The cluster project, `reduce-reuse-recycle`, already holds pieces this route can borrow once it is back.

| Piece | Where | What it gives this route |
|---|---|---|
| Du's 2D product and sum, rebuilt | `rebuild/compose.py`, `rebuild/phase2_compose.py` (ring × bar, left + right, matching Du's pictures) | a check that sampler code reproduces a published figure |
| the parts world: 16-pixel Gaussian-mixture experts | `rebuild/parts_world.py`, step file `04-the-parts-world.md` | the same exact-rule method; the hybrid reproduced exactly (claim 9) |
| ideal experts and samplers | `rebuild/ideal.py`, `rebuild/samplers.py`, `anneal_samplers.py` | ULA, MALA, UHA with their tests |
| "contains" against "exactly" labels | idea map `poe-overlap-dial`, claim 1.2 | the caption dial's first evidence |
| chimera mass by position | idea map, run 1 | centred shapes fuse more than shapes at varied places |
| the text-conditioned testbed | layer 11, idea map `sdxl-shaped-hybrid-testbed` | the cluster home for phase 2 |

## Later

Navigation: ⬅️ [Previous](#what-already-exists-on-the-cluster) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

**Bring it to the cluster.** Copy the walks and notebooks into `reduce-reuse-recycle`, as step files in
`artifacts/drips/simple-distributions-rebuild/` next to the parts world, and register the bricks in
`REBUILD_WALK.md`.

**A latent space, and its organisation.** A small autoencoder with a diffusion model on top, three ways:
a pixel-like latent (expect a ghost overlay), a disentangled latent with shape, colour and position as
separate directions (expect one in-between object), and one slot per object (expect two objects). The
write-up's section 8.6 has the table; the cluster's layer 10 found a plain product is not unchanged by an
invertible warp.

**Architecture.** Cross-attention against joint attention, one text encoder against several.

**The other remedies** ([write-up §9.1](what-poe-loses-when-it-composes.md#91-the-remedies-by-where-they-act)):
masked composition with given boxes, masks derived from the model, the learned correction, and merging
inside the network ([§9.5](what-poe-loses-when-it-composes.md#95-composing-in-feature-space)).

## Why this chain

Navigation: ⬅️ [Previous](#later) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

<details><summary>Rationale and the time estimates</summary>

**Formulas before networks.** If the exact product overlays, the core claim is settled before anything is trained. A trained network's overlay can only be blamed on the network once the formula's answer is known.

**Samplers before networks.** The rough run showed reverse diffusion drawing overlays where the exact answer was the shapes apart. That is measured on formulas first, so it is never mistaken for a network failure.

**One switch at a time.** Each switch stands for one route by which "circle" reaches the square's pixels in SDXL. Several at once would leave the cause unreadable.

**Times.** Estimates: about 30 minutes per checkpoint or cell group, longer where a network trains.

</details>

## Next step

Navigation: ⬅️ [Previous](#why-this-chain) | 📋 [TOC](#table-of-contents)

**Start with:** step 1, the six-checkpoint walk, on the laptop.
