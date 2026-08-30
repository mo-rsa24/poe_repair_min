# 🔭 The same story from three independent sides

This plan asks whether three checks that share none of the strength series' machinery agree with
its answer anyway.

**Step 7 of 22.** Waits on step 4. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 6 | [when in the run the correction matters](hypothesis-03-when-in-the-run-it-matters.md) | ◑ timing tab owed |
| **7** | **this plan** | **✅** |
| 8 | ~~[what the fix changes inside the model](hypothesis-01-what-the-fix-changes-inside-the-model.md)~~ | ✅ |

Design only. Verdicts live in [this plan's review file](../review/hypothesis-05-the-same-story-from-three-sides.md).

## What this asks, in one line
Three checks that share none of the machinery behind the run across correction strengths, so they
can agree with it independently rather than repeat it.

## Description
Three independent reads:

- **The picture moves out of the blend region.** Place the images from the strength series on the
  existing image-similarity axes and watch them slide as the strength rises. The control is
  the same-sized push in a random direction, which should not slide.
- **The prompt's own arithmetic predicts which pairs are hard.** Ask whether the joined
  prompt is more than the sum of its two parts in text space, and whether a blended picture
  reads back as a blend caption.
- **A blended animal is wrong content, not a bad picture.** Measure image quality on the
  cached broken and working outputs and expect no gap. This removes the objection that the
  correction merely improves quality.

## Purpose
The causal claim shown from three sides (Goal 1's secondary reads). The
language-space tests let the composition types predict which pairs are hard; the
quality check removes the "the correction just improves image quality"
objection. Serves DoD 6.

## Goal
The manifold slide with its random-path control, the caption-match crossover
curve, the additivity-gap and binding-direction results, and the quality-gap
table.

## Environment Facts This Plan Depends On
- L1 and L3 read cached `embeddings.pt` per pair-and-seed run. SDXL's own two text encoders give
  four views: pooled (1280, from text_encoder_2), the CLIP-L 77-token sequence
  (first 768 channels of the cached 2048), the bigG sequence (last 1280), and
  the concatenation cross-attention actually consumes. Concatenation order is
  fixed by `poe_repair/_sdxl/runtime.py`.
- The manifold slide and L2 consume plan 03's images across correction amounts, which are on
  disk: 440 renders under `outputs/interaction_term/dose/pairs`, λ ∈ {0, .25, .5,
  .75, 1}, with 32 runs carrying the `_random` and `_wrong_pair` control rows.
- The quality check reads `poe.png` and `mono.png` from the training cache:
  749 paired runs.
- CLIP embedding and GroundingDINO run in-session on the 3090 (light).
- λ=1 reproduces ε_J exactly, so the picture at the full amount is the joint render
  (measured at 1.9 grey levels of 255). Every reading is therefore taken at the
  largest interior amount, λ=0.75.

## Tasks
- [x] L1 additivity gap per pair (both encoders, pooled and sequence);
      scatter against normalized ‖r_t‖
- [x] L3 binding direction: b = e_J − normalized(e_A+e_B) per pair; cosine
      matrix and SVD across pairs, against a mismatched-solos control
- [x] [chimera](../../../context/world/chimera.md) quality control on cached poe.png/mono.png:
      quality proxies, expect no gap
- [x] manifold slide: embed plan 03's outputs across the λ series on the existing CLIP
      axes; random-direction path as the control
- [x] L2 caption readback on plan 03's images: caption bank including the
      blend caption; crossover curve vs λ

## Success/Failure Outcomes
- **L1 additivity gap**
  - Success: gap computed for all pairs; correlation with ‖r_t‖ reported
    either way (a null is a finding: binding info lives in joint processing,
    not the embedding).

    > A null here means the additivity gap of a pair tells you nothing about how
    > big that pair's correction is: the scatter has no slope.
  - Failure: testing the pooled form only (the sequence form is the one cross-attention
    consumes; both must be reported).

## Next

1. Final figure forms ride plan 10. The quality check and the caption readback are the two
   that carry their own argument; F5 takes the manifold slide.
2. The `wrong_pair` control reaches 44% of the interior travel the pair's own cached true
   correction makes on the manifold slide, against a threshold of 50%. Plan 10 should either
   widen that gap or say plainly that a mis-aimed correction gets you nearly half the slide.
3. L1 and L3 both come back with no effect. If a later plan wants a language-space predictor, the
   place to look is joint processing (cross-attention maps), not the prompt embedding.

**The short version:** the quality check alone. It removes the one standing objection ("the
correction just improves image quality") and runs from cache in minutes.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

# L1 + L3, 75 pairs from cache, ~1 min. Prints a per-pair table, then the
# correlation against the preregistered correction-size measure and the
# shared-direction read against its mismatched-solos control.
$PY scripts/language_probes.py --probe l1 --probe l3

# 749 paired poe.png/mono.png, ~20 min (GroundingDINO on every image).
# Four content-blind quality proxies decide; the compose rate is the positive
# control. Expect flat quality and a large content gap.
$PY scripts/quality_control.py

# 32 runs x 5 correction amounts x 3 rows in CLIP image space, ~3 min. Checks the lambda=1
# endpoint really is the mono render before reading anything off the curves.
$PY scripts/manifold_slide.py

# Same 32 runs against a four-way caption bank, ~3 min.
$PY scripts/caption_readback.py
```

Outputs land in `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`
as `language_probes.json`, `quality_control_cache.json`, `manifold_slide_clip.json`,
`caption_readback.json`, each beside its figure.
