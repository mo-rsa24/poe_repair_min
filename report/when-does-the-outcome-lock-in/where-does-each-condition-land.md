# Where does each condition land, and does the correction move it toward the joint prompt?   ❌ null on both pre-registered bars · ✅ support on the post-hoc reads · verified 2026-09-06

**The claim**

On the eight held-out pairs, the both-ness bar written before the render came back null: the
PoE and corrected bands overlap on 5 of the 7 counted pairs. The same finished images, scored
by the validated instance count, compose on every pair once corrected: PoE 0.00 on six of
seven counted pairs, corrected 0.75 to 1.00 on all eight. The null is a verdict on the both-ness
axis as a cross-pair measuring tool, and the instance-count read was not pre-registered for this
render.

On cat × dog, plain product-of-experts renders land between the "a cat" cloud and the "a dog"
cloud, and below the "a cat and a dog" cloud, in the image space the compose scorer uses.

Adding the trained correction moves every seed up into the joint-prompt band.

Over the 50 denoising steps, the joint-prompt run settles by step 10 and the corrected run by
step 15. The plain run wanders until step 22 and often changes animal on the way.

The bar written before the run asked whether each render's nearest cloud is the joint cloud.
That bar failed, because the clouds are as wide as the gaps between them.

The read that holds came after the result. So this finding is a null on its own terms, and a
support on a read the next pair has to confirm first.

## Table of contents

- [Where the idea came from](#where-the-idea-came-from)
- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. On axes built from the reference clouds, PoE sits low and the correction lifts it](#1-on-axes-built-from-the-reference-clouds-poe-sits-low-and-the-correction-lifts-it)
- [2. The pictures behind the points](#2-the-pictures-behind-the-points)
- [3. The two doses agree on where to go, and not on the exact direction](#3-the-two-doses-agree-on-where-to-go-and-not-on-the-exact-direction)
- [4. Over the 50 steps: the joint run settles first, the plain run wanders](#4-over-the-50-steps-the-joint-run-settles-first-the-plain-run-wanders)
- [5. One seed, frame by frame: the plain run changes animal mid-way](#5-one-seed-frame-by-frame-the-plain-run-changes-animal-mid-way)
- [6. The moving version](#6-the-moving-version)
- [7. The axes, drawn as pictures: x is which animal, y is how many](#7-the-axes-drawn-as-pictures-x-is-which-animal-y-is-how-many)
- [8. The eight held-out pairs: the both-ness bar fails, the pictures compose](#8-the-eight-held-out-pairs-the-both-ness-bar-fails-the-pictures-compose)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## Where the idea came from

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-set-out-to-see) ➡️

**The two posts.**

Dan Kornas's post of 2026-06-06 says "Diffusion models are easier to understand when you can see
the geometry move". Alec Helbling's post of 2025-07-11 is the tool's release: paper, demo and
code links. Both are quoted in
[the Diffusion Explorer entry](../../context/sources.md#2-diffusion-explorer-interactive-exploration-of-diffusion-models)
of the sources register.

The first claims only that watching samples move builds intuition. The second claims nothing; it
is the author's link post. Neither says anything about real image models.

**The tool.**

[Diffusion Explorer](../../context/sources.md#2-diffusion-explorer-interactive-exploration-of-diffusion-models)
(Helbling and Chau, arXiv 2507.01178) animates a 2D toy diffusion model's samples moving from
noise to data under a time slider. Its classifier-free-guidance page draws the conditional and
unconditional predictions as two arrows on one moving sample.

The question the posts raised was whether this project's SDXL sampling could be ported into it:
the joint prompt, each concept alone and the product of experts, compared in the same view.

It cannot. The tool's models are small in-browser networks over two numbers per sample, and an
SDXL latent has 65,536. The paper names extending to real images as open work.

What was taken from it is the design: a slider over precomputed steps, a plane the points move
in, arrows on the current sample. Its UI source came too. The animated page in rung 6 vendors
the tool's `packages/ui` and uses its time slider and legend components unchanged.

**The post.**

An [X post by Arnas Uselis](../../context/sources.md#4-eigenfaces-style-analysis-on-embedding-models-an-x-post)
showed eigenfaces-style walks along the leading principal components of three spaces: pixel
space, DINOv2-B and SigLIP2. Each walk is decoded back to pictures through a
[representation autoencoder](../../context/sources.md#3-diffusion-transformers-with-representation-autoencoders).

Its text: "One can also apply eigenfaces-style analysis to embedding models (SigLIP, DINO).
Pairing them with a decoder (e.g., an RAE) shows what the models primarily care about and what
biases the decoder introduces."

Pixel-space walks change lighting and blur. The DINOv2 walks change what object is there.

That is why this finding's plane lives in DINOv2 space and not over SDXL latents, whose
principal components behave like pixel PCA. It is also why the plan's next rung labels the axes
with decoded pictures rather than a legend.

**The connection to this project.**

The project's mechanism claims are made from norms and cosines of the interaction term. Nothing
showed where the joint prompt, each single concept, product-of-experts and the corrected run
each end up relative to one another.

This finding is that picture, first as endpoints, then as tracks over the run.

## What we set out to see

Navigation: ⬅️ [Previous](#where-the-idea-came-from) | 📋 [TOC](#table-of-contents) | [Next](#what-would-have-counted) ➡️

Five conditions on the same eight held-out seeds of cat × dog, each starting from the same
initial noise: "a cat" alone, "a dog" alone, the joint prompt "a cat and a dog", product-of-experts
with no correction, and product-of-experts plus the rank-32 correction at step 30050, at λ 1.0
and λ 1.2. Each final image embedded with the compose scorer's DINOv2 encoder and drawn as one
point. Then each run's running estimate of its finished image, decoded at 13 saved steps, drawn as
a track through the same plane.

**The hypothesis.** Product-of-experts lands in a region of its own, between the single-animal
clouds or inside one of them, and never inside the joint cloud. The correction moves it toward
the joint cloud along a consistent direction. On the tracks, the corrected run parts from the
plain run inside the first ten steps, which is the window the injection experiments already
named.

## What would have counted

Navigation: ⬅️ [Previous](#what-we-set-out-to-see) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

The plan pre-registered support on two conditions: no PoE render's nearest cloud centroid is the
joint cloud, and the λ 1.0 and λ 1.2 correction arrows agree in direction (cosine above 0.9) on
at least 6 of 8 seeds. It is pinned in the **What has to pass before this runs** section of
[where each condition lands](../../plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md),
written before the reference renders ran.

**Both halves failed.** 5 of 8 PoE renders sit nearest the joint centroid, and 2 of 8 arrow
pairs clear 0.9.

The commit-step and fork-step reads in rung 4 had no bar written before them. They are reported
as post-hoc.

## What was tried, in order

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#1-on-axes-built-from-the-reference-clouds-poe-sits-low-and-the-correction-lifts-it) ➡️

| When (2026-09-05) | What | Where it landed | Outcome |
|---|---|---|---|
| 06:29 | reference renders, first attempt, on the session node mscluster85 | `outputs/showcase/logs/where_each_condition_lands_render.log` | CUDA out of memory: another user's process held 16.2 GB of the 24 GB card, and SDXL's decode needs more than the 7 GB left |
| 06:37 | reference renders, second attempt, `co3` python on a Blackwell card | `outputs/showcase/where_each_condition_lands/solo_a/` (empty) | died without a traceback; the build has no kernels for that card |
| 06:44 to 06:47 | reference renders, `co3_bw`, mscluster110 device 0, PID 397295 | `outputs/showcase/where_each_condition_lands/{solo_a,solo_b,joint}/seed_<n>.png` | 24 of 24, about six seconds each |
| 07:23 | endpoint plot, PCA plane | `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.png` | plane keeps 22% of variance; hulls overlap; nearest-centroid bar fails |
| 07:24 | endpoint plot, cloud axes | `.../cat-x-dog-in-dino-space-cloud-axes.png` | PoE band and corrected band separate |
| 12:00 | dose strip, seeds 10 and 11, from existing PNGs | `.../cat-x-dog-seeds-10-11-dose-strip.png` | what the second dose changes, seen |
| 08:00 to 08:21 | per-step frames, first launch, mscluster111 device 0 (Blackwell, `co3_bw`) | `outputs/showcase/logs/where_each_condition_lands_frames.log` | ran on the CPU without saying so: `torch.cuda.is_available()` is `False` on that card while `nvidia-smi` still lists it, one run took 15 minutes against 16 seconds on a working GPU; killed after one run, relaunched on mscluster109 device 1; recorded as the second form of [poe-launch-002](../../environment/known-failures.md) and the launch script now checks CUDA before starting |
| 08:22 to about 09:15 | per-step frames for all six conditions, 13 saved steps, mscluster109 device 1 (`co3`) | `outputs/showcase/where_each_condition_lands/frames/<cond>/seed_<n>/step_<kk>.png` | one re-render: the windowed LoRA sampler leaves the adapter attached, so references rendered after it carried the correction; the contaminated set is kept as `frames_with_adapter_attached/` and unused |
| midday | frames embedded, axes refit on this run's own final references, scene built | `artifacts/scenes/where-each-condition-lands/`, `.../frames-dino-feats.npz` | the animated page |
| afternoon | both-ness over steps, commit and fork steps, one seed's frame strip | `.../both-ness-over-denoising-steps.png`, `.../commit-and-fork-steps.json`, `.../frames-seed-15-strip.png` | rung 4 and rung 5 below |

Every render shares its seed's initial noise (the training cache's held-out step-0 latents),
50 DDIM steps, guidance 7.5 and 1024 square output; only the prompt and the correction differ.

## 1. On axes built from the reference clouds, PoE sits low and the correction lifts it

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-the-pictures-behind-the-points) ➡️

![The 48 renders on the cloud axes: x from the cat centroid to the dog centroid, y from their midpoint toward the joint-prompt centroid](../../artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-cloud-axes.png)
*Black squares are product-of-experts endpoints, pink diamonds the same seeds with the λ 1.2
correction, one grey-then-pink arrow per seed. What to notice: every arrow ends inside the
joint-prompt hull's height, and no square does.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-1-the-48-renders-on-axes-built-from-the-reference-clouds).

**The number.** Both-ness, the projection onto the unit vector from the midpoint of the two
single-animal centroids to the joint-prompt centroid, in DINOv2 cosine units, mean over seeds 9
to 16: PoE 0.21 (range 0.04 to 0.31), PoE plus λ 1.0 correction 0.41, PoE plus λ 1.2 correction
0.42 (range 0.34 to 0.51), joint prompt 0.52 (range 0.36 to 0.61), single animals 0.00 by
construction. The PoE band and the corrected band do not overlap. From
`artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json`, field
`cloud_axes.both_ness_by_condition`, written by `scripts/showcase/where_each_condition_lands_plot.py`.

## 2. The pictures behind the points

Navigation: ⬅️ [Previous](#1-on-axes-built-from-the-reference-clouds-poe-sits-low-and-the-correction-lifts-it) | 📋 [TOC](#table-of-contents) | [Next](#3-the-two-doses-agree-on-where-to-go-and-not-on-the-exact-direction) ➡️

![Eight seeds by six conditions: cat alone, dog alone, joint prompt, PoE, PoE plus λ 1.0, PoE plus λ 1.2](../../artifacts/results/where-does-each-condition-land/cat-x-dog-contact-sheet.png)
*What to notice: the fourth column is one animal or one fused animal in every row; the fifth and
sixth columns are two animals in every row.*
📊 Drawn in [Figure 3 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-3-the-contact-sheet-eight-seeds-by-six-conditions).

**The number.** Nearest cloud centroid in the full 384-dimensional embedding, cosine distance:
PoE renders nearest the joint cloud for 5 of 8 seeds, nearest cat for 2, nearest dog for 1; λ 1.2
renders nearest the joint cloud for 8 of 8. Centroid gaps: cat to dog 0.80, cat to joint 0.49,
dog to joint 0.54; cloud widths (mean distance to own centroid): 0.28, 0.36, 0.34. Same sidecar,
fields `points[].nearest_cloud`, `centroid_cosine_gaps`, `cloud_stats`.

## 3. The two doses agree on where to go, and not on the exact direction

Navigation: ⬅️ [Previous](#2-the-pictures-behind-the-points) | 📋 [TOC](#table-of-contents) | [Next](#4-over-the-50-steps-the-joint-run-settles-first-the-plain-run-wanders) ➡️

![Seeds 10 and 11: joint prompt, PoE, PoE plus λ 1.0, PoE plus λ 1.2](../../artifacts/results/where-does-each-condition-land/cat-x-dog-seeds-10-11-dose-strip.png)
*What to notice: in both rows the second column is one animal, the third has two, and the fourth
keeps two while changing which one is in front and how much of the frame each takes.*
📊 Drawn in [Figure 4 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-4-the-dose-strip-for-seeds-10-and-11).

**The number.** Cosine between the λ 1.2 arrow and the direction from that seed's PoE point to
the joint centroid: 0.59 to 0.74, positive on all 8 seeds. Cosine between the λ 1.0 arrow and the
λ 1.2 arrow of the same seed: 0.54 to 0.94, above 0.9 on seeds 9 and 13 only. Same sidecar, field
`per_seed`.

## 4. Over the 50 steps: the joint run settles first, the plain run wanders

Navigation: ⬅️ [Previous](#3-the-two-doses-agree-on-where-to-go-and-not-on-the-exact-direction) | 📋 [TOC](#table-of-contents) | [Next](#5-one-seed-frame-by-frame-the-plain-run-changes-animal-mid-way) ➡️

![Both-ness of each run's running estimate against denoising step, one thin line per seed, thick line per condition mean](../../artifacts/results/where-does-each-condition-land/both-ness-over-denoising-steps.png)
*y is both-ness of the running estimate, x is the denoising step, one point per saved frame. What
to notice: the purple mean reaches the dashed joint-prompt centroid line by step 25 and stays;
the pink mean climbs until step 30 and holds under it; the black mean never leaves 0.2 and its
thin lines cross each other all the way to the end.*
📊 Drawn in [Figure 6 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-6-both-ness-over-the-50-denoising-steps).

**The number.** Commit step, the first saved step after which a run's both-ness stays within
0.10 of its final value (`COMMIT_TOL = 0.10` in source), median over 8 seeds: joint prompt 10
(range 5 to 20), PoE plus λ 1.2 correction 15 (range 8 to 35), PoE 22.5 (range 0 to 30). Fork
step, the first saved step at which the PoE run and the corrected run are further apart in the
plane than 0.05 (`FORK_MIN_DIST = 0.05`): median 1, range 0 to 2, so the two running estimates
differ from the first step, which is where the correction is applied. From
`artifacts/results/where-does-each-condition-land/commit-and-fork-steps.json`, fields
`summary` and `per_seed`, written by `scripts/showcase/where_each_condition_lands_dynamics.py`
from the scene's `public/data.json`. No bar was written before these two numbers; they are a
read, not a verdict.

## 5. One seed, frame by frame: the plain run changes animal mid-way

Navigation: ⬅️ [Previous](#4-over-the-50-steps-the-joint-run-settles-first-the-plain-run-wanders) | 📋 [TOC](#table-of-contents) | [Next](#6-the-moving-version) ➡️

![Seed 15, six conditions by seven saved steps: the decoded running estimate of the finished image](../../artifacts/results/where-does-each-condition-land/frames-seed-15-strip.png)
*Rows are the six conditions, columns the saved steps 0, 5, 10, 20, 30, 40 and 50. What to
notice: the fourth row is a cat at steps 5 and 10, a cat with a dog's muzzle at 20 and 30, and a
dog at 40 and 50; the last two rows show two silhouettes from step 5 and two animals from
step 20.*
📊 Drawn in [Figure 7 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-7-seed-15-frame-by-frame).

**The number.** Seed 15's row in `commit-and-fork-steps.json`: PoE final both-ness 0.10, the
lowest of the eight, and its commit step reads 0 because its both-ness never rises, while its
running estimate crosses from the cat side to the dog side of the plane between saved steps 20
and 40 (the x coordinate, visible in the trails screenshot of rung 6); joint prompt final
both-ness 0.67, commit step 20; λ 1.2 final both-ness 0.44, commit step 10.

## 6. The moving version

Navigation: ⬅️ [Previous](#5-one-seed-frame-by-frame-the-plain-run-changes-animal-mid-way) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![The animated page at step 20: every run's running estimate as a point in the cloud-axes plane](../../artifacts/results/where-does-each-condition-land/scene-step-20.png)
*A screenshot of the interactive page at step 20 of 50, all eight seeds. What to notice: the
purple points are already inside their hull, the pink diamonds are climbing toward it, and the
black squares are spread across the middle.*

![The animated page at step 50 with seed 15 selected and trails on](../../artifacts/results/where-does-each-condition-land/scene-seed-15-step-50.png)
*Seed 15 with trails. What to notice: the black trail crosses the plane from left to right
before ending near the dog cloud; the pink trail climbs and ends inside the joint hull. The six
thumbnails on the right are that seed's finished images.*
📊 Drawn in [Figure 5 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-5-the-animated-page-three-screenshots).

**The number.** Same as rung 4; the page draws the same `data.json`. The page itself is the
[where-each-condition-lands scene](../../artifacts/scenes/where-each-condition-lands/README.md), served
per the [scene-serving recipe](../../runbook/looking-at-what-a-run-produced/showing-a-scene-in-the-local-browser.md).

## 7. The axes, drawn as pictures: x is which animal, y is how many

Navigation: ⬅️ [Previous](#6-the-moving-version) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![The cloud-axes plot with seven decoded pictures along each axis](../../artifacts/results/where-does-each-condition-land/axes/cat-x-dog-in-dino-space-cloud-axes-with-axis-pictures.png)
*The plane of rung 1 with a strip of decoder reconstructions on each edge. What to notice: along
the bottom the face turns from cat to dog; up the left edge one animal becomes two side by side.*

![Four renders beside their reconstructions through the representation autoencoder](../../artifacts/results/where-does-each-condition-land/axes/reconstruction_check.png)
*Left to right in pairs: PoE seed 9, PoE seed 10, the joint prompt at seed 9, the corrected seed
15. What to notice: the fused faces come back fused, so the decoder keeps a chimera rather than
cleaning it into one animal.*
📊 Drawn in [Figure 8 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-8-the-axes-drawn-as-pictures).

**The number.** The decoder reads DINOv2-B patch tokens, not the ViT-S class token the plane
uses, so the two axes were rebuilt in the decoder's token space on the same 48 renders and the
walks decoded there. Rank agreement between token-space both-ness and the plane's both-ness over
all 48 renders: Spearman 0.87. Reconstruction error, mean absolute pixel difference on a 0 to
255 scale: PoE seed 9 9.5, PoE seed 10 22.5, joint seed 9 15.3, corrected seed 15 29.3. From
`artifacts/results/where-does-each-condition-land/axes/axes.json`, fields
`spearman_token_vs_class_token_both_ness_over_48` and `reconstruction_mean_abs_error_over_255`,
written by `scripts/showcase/where_each_condition_lands_axes.py`. Every decoded frame is a
reconstruction of a point in the encoder's space and none is an SDXL output.

## 8. The eight held-out pairs: the both-ness bar fails, the pictures compose

Navigation: ⬅️ [Previous](#7-the-axes-drawn-as-pictures-x-is-which-animal-y-is-how-many) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Both-ness of the finished image per pair: PoE, corrected and joint, one dot per seed](../../artifacts/results/where-does-each-condition-land/pairs/bands-across-pairs.png)
*y is both-ness on each pair's own axes, x is the pair, one dot per seed and a bar at the mean.
What to notice: black and pink separate on cat × dog and cow × buffalo, and sit on top of each
other on the five pairs whose two animals look alike.*

![Goose × swan, eight seeds by five arms](../../artifacts/results/where-does-each-condition-land/pairs/a_goose__x__a_swan/contact-sheet.png)
*The pair with the widest overlap. What to notice: the fourth column is one bird in every row,
the fifth is two birds in every row, and yet the pink and black dots above coincide, because a
picture of two white waterfowl embeds next to a picture of one.*
📊 Drawn in [Figure 9 of the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md#figure-9-the-eight-held-out-pairs).

**The number, against the bar.** Counted pairs 7 (elephant × penguin is the named exception).
Pairs whose PoE and corrected both-ness bands do not overlap: 2 of 7 (cat × dog, max PoE 0.28
against min corrected 0.29; cow × buffalo, 0.16 against 0.24). Pairs passing both halves of the
bar (bands apart and corrected median commit step at or before PoE's): 0 of 7. Pairs whose bands
overlap: 5 of 7, above the null threshold of 4. Verdict: **null**. From
`artifacts/results/where-does-each-condition-land/pairs/verdict.json`, fields `n_pass_both_halves`,
`n_band_overlap`, `verdict`, written by `scripts/showcase/where_each_condition_lands_pairs_analyze.py`
with the thresholds copied from the plan.

**The number, post-hoc.** Instance-count compose fraction over 8 seeds (two or more animals
detected, the compose scorer's validated rule) on the same finished images: PoE 0.00 on cat ×
dog, cow × buffalo, goose × swan, leopard × jaguar, seal × walrus and eagle × hawk, 0.12 on frog
× toad, 0.25 on elephant × penguin; corrected 0.75 (cat × dog), 0.88 (leopard × jaguar, seal ×
walrus), 1.00 (the other five); joint prompt 0.62 to 1.00. Same `verdict.json`, field
`per_pair[].instance_count_compose_fraction_post_hoc`. Elephant × penguin, expected to compose
by default, did not: its PoE renders are elephant-penguin fusions in 6 of 8 seeds, and it is the
one pair whose corrected both-ness mean (0.15) sits below its PoE mean (0.16), the surprise the
bar named in advance.

**The commit steps, per pair.** Median commit step (joint / corrected / PoE): cat × dog 10 / 18
/ 15, cow × buffalo 22 / 25 / 8, frog × toad 10 / 12 / 9, goose × swan 20 / 10 / 10, leopard ×
jaguar 15 / 12 / 20, seal × walrus 15 / 15 / 8, eagle × hawk 15 / 5 / 5, elephant × penguin 10
/ 32 / 10. The cat × dog ordering (joint first, PoE last) does not repeat: on four pairs the
PoE run's both-ness settles earliest because it never rises. The fork between the PoE and
corrected running estimates is at saved step 0 to 8 on every seed of every pair.

## What this cannot tell you

Navigation: ⬅️ [Previous](#8-the-eight-held-out-pairs-the-both-ness-bar-fails-the-pictures-compose) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Both-ness does not travel across pairs.** The axis is built per pair from that pair's own
clouds, and when the two animals look alike the joint-prompt cloud sits close to the single-animal
clouds (centroid gap between the two single animals 0.08 on leopard × jaguar, 0.15 on frog ×
toad), so a picture of two of them embeds beside a picture of one. The null in rung 8 is about
that axis. It says nothing against the correction, which the pictures and the instance count
both show composing.

**The axis pictures are the decoder's guesses.** They blur into eigenface-style averages away
from real renders, and they were made in a different space (DINOv2-B tokens) from the plane
they label (DINOv2-S class tokens); the 0.87 rank agreement is what licenses reading one as the
other.

**The axes are built from the clouds they judge.** The joint cloud sitting on the y axis is by
construction. The content is where PoE and the corrected points fall relative to it.

**Every number is a projection, not a distance.** Both planes drop most of each embedding: the
mean off-plane residual is 0.80 of the unit norm, and the PCA plane keeps 22% of the variance.

**Eight seeds show where things land.** They do not support a claim about the shape or overlap
of the distributions.

**A global embedding reads a fused face as "both".** That is why DINOv2 was rejected as the
compose scorer's decision rule, and why the pre-registered nearest-centroid bar failed. The
instance count in [compose rate](../../context/world/compose-rate.md) is the validated rule, and the
contact sheet is what that count sees.

**The commit and fork steps are read off 14 saved frames.** A step between two saved ones is
invisible, and both thresholds were chosen after the curves existed.

**Rungs 1 to 7 are one pair.** Cat × dog is the pair the rank-32 adapter was held out on, not
a pair it trained on; rung 8 is the other seven held-out pairs, eight seeds each, and the
instance-count column there is post-hoc.

**The animated page refits its axes** on its own final reference frames rather than reusing the
endpoint sidecar's. The two agree on the both-ness means to within 0.01.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| Both-ness per condition, nearest clouds, gaps, arrow cosines | `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json`, read 2026-09-05 | verified |
| Commit and fork steps, both-ness per step | `artifacts/results/where-does-each-condition-land/commit-and-fork-steps.json`, computed and read 2026-09-05 from `artifacts/scenes/where-each-condition-lands/public/data.json` | verified |
| The reference renders (cat, dog, joint; seeds 9 to 16) | `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/{solo_a,solo_b,joint}/seed_<n>.png`, `render_manifest.json`, rendered 2026-09-05 on mscluster110 | verified |
| The per-step frames, six conditions | `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/<cond>/seed_<n>/step_<kk>.png`, `frames_manifest.json` | verified |
| The PoE and corrected renders | `/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/renders/full/seed_<n>_lambda_<λ>.png`, `render_run.json` | verified |
| The held-out pair renders, five arms, eight pairs, seeds 9 to 16, frames and trajectories | `/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/pairs/<pair>/<arm>/seed_<n>/`, 320 of 320 counted on mscluster106, rendered 2026-09-05 15:54 to 20:30 | verified |
| The per-pair both-ness, commit steps, instance counts and the verdict | `artifacts/results/where-does-each-condition-land/pairs/verdict.json` and `<pair>/sidecar.json`, computed and read 2026-09-06 | verified |
| The correction | rank-32 adapter, step 30050, `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`, applied over all 50 steps at λ 1.0 and 1.2 | verified |
| The run | tasks **Render the three references** and **Run the plot on the launch node** in [where each condition lands](../../plans/05-when-does-the-outcome-lock-in/plans/figures/06-where-each-condition-lands.md); verdict in [its review file](../../plans/05-when-does-the-outcome-lock-in/review/06-where-each-condition-lands.md) | verified |
| Regenerate with | [reproducing where each condition lands](../../runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md): recipe 1 for the renders and frames on a shared device, recipe 2 for the endpoint figures and sidecar, recipe 3 for the frames' embeddings and the page data, recipe 4 for the page; `where_each_condition_lands_dynamics.py` under `scripts/showcase/` for rung 4's figure | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What product-of-experts composition is and what the joint prompt is the target of:
  [PoE composition](../../context/world/poe-composition.md)
- What the trained correction is and why it never sees the joint prompt:
  [the LoRA corrector](../../context/world/lora-corrector.md)
- What the compose scorer measures and what an embedding cannot:
  [compose rate](../../context/world/compose-rate.md)
- Which python build runs on which card (`co3_bw` on mscluster110 to 112):
  [nodes](../../environment/hpc/nodes.md)
- The three outside sources named above:
  [Diffusion Explorer](../../context/sources.md#2-diffusion-explorer-interactive-exploration-of-diffusion-models),
  [the representation-autoencoder paper](../../context/sources.md#3-diffusion-transformers-with-representation-autoencoders),
  [the X post](../../context/sources.md#4-eigenfaces-style-analysis-on-embedding-models-an-x-post)
- Each picture, read one at a time:
  [the figure explainer](../../artifacts/results/where-does-each-condition-land/figure-explainer.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] A cross-pair measuring tool for the plane. Both-ness failed as one in rung 8; the
      instance count is the validated rule but is a label, not a coordinate. A candidate is the
      count axis decoded in rung 7, defined once on cat × dog and applied to every pair's
      embeddings, which would need its own pre-registered bar before a render.
- [ ] The instance-count support on the unseen pairs is post-hoc for this render. It becomes a
      verdict only when a plan names it as the bar before the next one.
- [ ] Elephant × penguin, listed in the pair pool as composing by default, did not on this
      render (PoE 0.25). The pool's control label needs re-checking against a scored run.
- [ ] The twisted-SMC endpoints joining the same plane once
      [that baseline](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md)
      has a run worth drawing; its first full run is filed as inconclusive in
      [does selecting among PoE proposals compose](../is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md).
