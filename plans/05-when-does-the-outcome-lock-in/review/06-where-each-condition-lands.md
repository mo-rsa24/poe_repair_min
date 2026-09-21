# 🖼️ Review: where does each condition land, and when does each run commit?

**Nothing has been judged yet.** This file judges [the design](../plans/figures/06-where-each-condition-lands.md);
answers land here and nowhere else. Questions below were written at design time, before any
number existed.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```
(For a run that failed and whose failure is worth keeping: `/ingest-error-pattern --from-run-log`.)

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/figures/06-where-each-condition-lands.md) | the five conditions, the two spaces, the four rungs |
| **this file** | **the verdict: what the figures answered, and what they could not** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **cloud**: the eight held-out seeds of one reference condition ("a cat", "a dog", or the joint
  prompt "a cat and a dog") as points in DINOv2 space; drawn as a hull in the plane.
- **centroid distance**: cosine distance from a point to the mean embedding of a cloud, computed
  in the full 384-dimensional space, never in the plane.
- **arrow**: the vector from one seed's PoE endpoint to the same seed's corrected endpoint.
- **commit step**: the first step after which a track's nearest cloud centroid holds for five
  consecutive steps and never changes again (`COMMIT_STABLE_STEPS = 5`).
- **fork step**: the first step at which the PoE and corrected tracks of one seed are further
  apart than `FORK_MIN_COS_DIST = 0.05`.

> **PoE** is product-of-experts: the two single-prompt predictions added and the unconditional
> subtracted. **The correction** is the rank-32 adapter at step 30050, applied at λ 1.2 over all
> 50 steps.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Shows the claim.** A failed bar here does not kill a result elsewhere; it means the figure
cannot be read as designed and the design changes before more seeds or pairs are spent.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| reference renders, cat×dog seeds 9 to 16 | shared-device, mscluster110 device 0, PID 397295 | 2026-09-05 06:44 | 24 renders, about six seconds each | `outputs/showcase/where_each_condition_lands/{solo_a,solo_b,joint}/` | done, 24 PNGs and the manifest counted on the launch node |
| endpoint plot, cat×dog | CPU, mscluster110 | 2026-09-05 07:24 | under a minute | `artifacts/results/where-does-each-condition-land/` (two planes, contact sheet, sidecar, features) | done |
| first attempt, same node | shared-device, `co3` python | 2026-09-05 06:37 | nothing | empty `solo_a/` | died without output: wrong python build for the Blackwell card |
| held-out pairs, first launch, mscluster111 device 0 | shared-device, `co3_bw` | 2026-09-05 14:31 | ten minutes of CPU | one reference cell, discarded | the card reads "GPU requires reset" and torch saw no device, so the sampler ran on the CPU; killed, launcher now asserts CUDA |
| held-out pairs, mscluster106 device 1, PID 2437144 | shared-device, `co3`, RTX 8000 | 2026-09-05 15:54 to about 20:30 | 320 runs, 30 s per reference run and 64 s per adapter run | `outputs/showcase/where_each_condition_lands/pairs/`, 320 images, 320 trajectories, 13 frames each | done, counted on the launch node; one cell re-rendered after the CPU copy was removed |
| per-pair analysis and scorer, mscluster108 device 1 | CPU embedder on GPU, GroundingDINO | 2026-09-06 | about 20 minutes | `artifacts/results/where-does-each-condition-land/pairs/` | done |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ❌ On cat×dog, is every PoE endpoint's nearest centroid one of the single-animal clouds or
      a region of its own, never the joint cloud, and do the λ 1.2 arrows point toward the joint
      centroid seed after seed (per-seed cosine between the λ 1.0 and λ 1.2 arrows above 0.9 for
      at least 6 of 8 seeds)? This is the deciding question because it is what "the failure is a
      place and the fix is a direction" means in numbers.
      **Answer: no on both halves, as the bar was written.** Nearest centroid for the PoE
      endpoints: joint for 5 of 8 seeds (9, 11, 13, 14, 16), cat for seeds 10 and 12, dog for seed
      15. The λ 1.0 versus λ 1.2 arrow cosine clears 0.9 for 2 of 8 seeds (9 at 0.94, 13 at 0.93);
      the other six sit between 0.54 and 0.80. Read from `cat-x-dog-in-dino-space.json`,
      fields `per_seed` and `points[].nearest_cloud`. What did hold, on the same file: every λ 1.2
      endpoint's nearest centroid is the joint cloud (8 of 8), and every λ 1.2 arrow has a positive
      cosine (0.59 to 0.74) with the direction from its PoE point to the joint centroid. The bar
      failed because nearest-centroid is a weak statistic for clouds this wide, which is the
      artefact check below; the "both-ness" read in "Asked after the result" is the one the
      figure should be judged on next time, and it was written after seeing the answer, so it
      cannot replace this question.

- [x] ❌ On the eight held-out pairs rendered by task 4, do the PoE both-ness band and the λ 1.2
      both-ness band fail to overlap on at least 6 of 8 pairs, with the corrected median commit
      step at or before the PoE median (support), or overlap on 4 or more (null), per the bar
      written in the design's **What has to pass before this runs** section on 2026-09-05 before
      the render was launched? Elephant × penguin is the pre-named exception and counts toward
      neither. This is the deciding question for the unseen pairs because it is the cat × dog
      post-hoc read turned into a prediction.
      **Answer: null.** Of the 7 counted pairs, 2 have non-overlapping bands (cat × dog, cow ×
      buffalo), 0 pass both halves, and 5 overlap, above the null threshold of 4. No pair was
      excluded for a solo-centroid gap under 0.05 (the smallest is leopard × jaguar at 0.08).
      The pre-named surprise happened on the exception pair: elephant × penguin's corrected
      both-ness mean (0.15) sits below its PoE mean (0.16). From
      `artifacts/results/where-does-each-condition-land/pairs/verdict.json`, read 2026-09-06.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ How far apart are the three reference clouds (cosine distance between centroids), and how
      wide is each (mean distance to its own centroid)? A gap under 0.05 means the space does not
      separate them and the figure is unreadable.
      **Answer:** centroid gaps cat to dog 0.80, cat to joint 0.49, dog to joint 0.54; widths
      (mean cosine distance to own centroid) cat 0.28, dog 0.36, joint 0.34. The clouds are
      separated but each is roughly as wide as its gap to the joint cloud, so hulls overlap in
      any 2D plane and nearest-centroid flips on a small move. The PCA plane keeps 22% of the
      variance (11% and 11%).
- [x] ✅ Does a chimera render survive reconstruction through the DINOv2-B RAE decoder, or does
      the decoder clean it into one animal? Answered with the saved side-by-side.
      **Answer: it survives.** In `axes/reconstruction_check.png`, PoE seed 9 (a cat with dog
      ears) and PoE seed 10 (one fused face) come back as fused faces. Mean absolute pixel error
      on a 0 to 255 scale: 9.5 and 22.5 for the two chimeras, 15.3 for the joint render at seed
      9, 29.3 for the corrected sketch at seed 15 (`axes/axes.json`,
      `reconstruction_mean_abs_error_over_255`). Decoded frames may label axes and may also
      stand for a point between clouds, always captioned as reconstructions.
- [x] ✅ Does either principal component decode as a count axis (one animal at one end, two at
      the other), and if so, where along it do the PoE endpoints sit?
      **Answer, on the cloud axes rather than principal components** (the walk was built on the
      supervised axes because the PCA plane had already failed the nearest-centroid read):
      the y axis, both-ness, decodes as a count axis. `axes/axis-y-both-ness.png` shows one face
      at −2 sd and two side-by-side figures at +2 sd, with the second figure appearing from about
      +0.67 sd. The x axis decodes from a cat face to a dog face. The walk lives in DINOv2-B token
      space, which agrees with the class-token plane at Spearman 0.87 over the 48 renders. On
      that axis the PoE endpoints sit at token-space both-ness 23 (mean) against 126 for the
      corrected renders and 199 for the joint prompt (`axes.json`,
      `both_ness_means_token_space`; units are token-space projections, not comparable to the
      class-token plane's numbers).
- [x] ✅ On cat×dog seed 9, what are the commit steps of the joint track and the PoE track, and
      does the joint track commit earlier?
      **Answer, over all eight seeds rather than seed 9 alone, from `commit-and-fork-steps.json`
      (commit step = first saved step after which both-ness stays within 0.10 of its final
      value, a threshold chosen after the curves existed):** joint prompt median 10 (range 5 to
      20), PoE plus λ 1.2 correction median 15 (8 to 35), PoE median 22.5 (0 to 30). The joint
      track commits earlier on the medians. The fork between the PoE and corrected running
      estimates is at saved step 0 to 2 on every seed, since the correction is applied from
      step 0. On seed 15 the PoE estimate is a cat until step 10 and a dog by step 40
      (`frames-seed-15-strip.png`).
- [x] ✅ On the unseen pairs, at which step does the corrected track fork from the PoE track, per
      seed, and is that inside the first ten steps?
      **Answer: yes on every seed of every pair.** The first saved step at which the two running
      estimates are further apart than 0.05 in the plane is 0 to 8 across all 64 seed-pair
      cells (`verdict.json`, `per_pair[].fork_step_poe_vs_corrected`), since the correction is
      applied from step 0.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [x] ✅ On the axes defined by the clouds themselves (x: unit vector from the cat centroid to the
      dog centroid; y: unit vector from the midpoint of the two single-animal centroids to the
      joint centroid, made orthogonal to x; both in DINOv2 cosine units), where does each
      condition sit on y, the "both-ness" axis? **Answer, mean over the 8 seeds, from
      `cloud_axes.both_ness_by_condition`:** cat alone 0.00, dog alone 0.00 (by construction, each
      within ±0.16), joint prompt 0.52 (0.36 to 0.61), PoE 0.21 (0.04 to 0.31), PoE plus λ 1.0
      correction 0.41 (0.31 to 0.50), PoE plus λ 1.2 correction 0.42 (0.34 to 0.51). The bands do
      not overlap between PoE and either corrected condition, and every corrected point sits
      above every PoE point of the same seed. The mean off-plane residual is 0.80 of each
      point's norm, so this plane is a 2-axis summary of a mostly off-plane cloud; the numbers
      above are projections, not distances.
- [x] ✅ What do the renders behind the points show? From `cat-x-dog-contact-sheet.png`: the PoE
      column is a single animal or a fused one in every seed (a cat with dog ears at seed 9, one
      fused face at seeds 10 and 12, a single sketched animal at 11, 14 and 15); the λ 1.0 and
      λ 1.2 columns show two animals in 8 of 8 seeds, with seed 14 adding a child holding the
      leash. So "nearest centroid is the joint cloud" for five PoE seeds is the embedding being
      generous to fused faces, and the picture agrees with the both-ness read rather than with
      the nearest-centroid read.
- [x] ✅ On the eight held-out pairs, what does the validated instance count say about the same
      finished images the both-ness bar was judged on? **Answer, compose fraction over 8 seeds
      (two or more animals detected), from `pairs/verdict.json`,
      `instance_count_compose_fraction_post_hoc`:** PoE 0.00 on six pairs, 0.12 on frog × toad,
      0.25 on elephant × penguin; corrected 0.75 on cat × dog, 0.88 on leopard × jaguar and seal
      × walrus, 1.00 on the other five; joint prompt 0.62 to 1.00. The correction composes on
      every unseen pair by the rule the project validated, while the both-ness bar reads null,
      because on pairs whose animals look alike a picture of two embeds beside a picture of one.
      Written after the result; a measuring-tool read, never the deciding question.
- [x] ✅ Does elephant × penguin compose by default, as the pair pool's control label says?
      **Answer: not on this render.** PoE compose fraction 0.25, with elephant-penguin fusions in
      6 of 8 seeds in `pairs/an_elephant__x__a_penguin/contact-sheet.png`; corrected 1.00. The
      pool's label was set on an earlier run and is now open in the finding's Still open list.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

Three fixed checks. Each is answered or explicitly marked not applicable; none is dropped.

- [x] ✅ **Was the comparison fair?** Every point in one figure shares its seed's initial noise,
      sampler, step count and guidance; only the prompt and the correction differ. Confirmed:
      `render_manifest.json` records 50 steps, guidance 7.5, and the training cache's held-out
      step-0 latents; `render_run.json` under `figure_r32_030050` records the same sampler for
      the PoE and corrected renders with the step-30050 checkpoint.
- [x] 🟡 **Was the measuring tool sound?** The DINOv2 recipe matches the compose scorer's. The
      nearest-centroid statistic the deciding question was written against is not sound for
      these clouds: each is as wide as its gap to the joint cloud, so a fused face lands "nearest
      joint". The both-ness projection is the sounder read and is recorded above as a post-hoc
      question, never as the deciding one.
- [x] ✅ **Did the run respect the environment?** Renders on `/datasets`, counted on the launch
      node (24 of 24); the first attempt with the `co3` build died on the Blackwell card and is
      in the Runs table; the rerun used `co3_bw`.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| PoE lands away from the joint cloud and the correction moves it toward it | the variance share of the plane on the figure, and the full-space centroid distances in the appendix |
| the axis strips show what the plane's directions mean | every decoded frame labelled as an RAE reconstruction, with the chimera reconstruction check reported |
| eight seeds per cloud | stated as a look, not a separation claim; a separation claim needs about 32 seeds per condition |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

**Required, even when empty.**

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| The twisted-SMC endpoints (step 51 in the root order) are not yet in this figure | the full twist run finishing and its strips being rendered on seeds 9 to 16 at 50 steps | the "three solutions on one plane" version of the endpoint figure |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run [the design's tasks](../plans/figures/06-where-each-condition-lands.md#tasks), then answer the questions above.
