# Can a corrector, or handing the tail to the frozen model, sharpen the adapter's renders?   ⚪ null on both bars · verified 2026-09-06

**The claim**

The rank-32 adapter composes cat × dog on seven of eight held-out seeds and its renders are softer
than the joint prompt's. Two ways of cleaning them up were tested and neither clears its bar.

Corrector steps on the adapter's own corrected score over the last fifteen denoising steps move
sharpness by 8%, inside the 10% band, and redraw fine detail seed by seed in both directions.

Handing those steps to the frozen model instead, with the adapter switched off after step 29, moves
the renders 0.038 nearer the joint prompt's in the scorer's embedding space, against a 0.05 bar,
with composition held. Adding corrector steps on that frozen score moves them back away and adds
grain.

So the softness lives in the adapter's low-noise steps, the hand-off is the lever that touches it,
and settling further into any distribution is not what fixes it.

## Table of contents

- [What would have counted](#what-would-have-counted)
- [1. The corrector on the adapter's own score changes nothing that matters](#1-the-corrector-on-the-adapters-own-score-changes-nothing-that-matters)
- [2. The hand-off to the frozen model is the lever that moves](#2-the-hand-off-to-the-frozen-model-is-the-lever-that-moves)
- [3. The control pair keeps what it already had](#3-the-control-pair-keeps-what-it-already-had)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What would have counted

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-the-corrector-on-the-adapters-own-score-changes-nothing-that-matters) ➡️

Two questions, each with its thresholds as named constants in
`scripts/corrector_window_sweep.py` before its grid ran.

For the corrector on the adapter's tail: support if the mean [Laplacian variance](does-searching-over-the-noise-sharpen-the-corrected-render.md) over the eight
seeds at 20 corrector steps is at least `FIDELITY_MIN_SHARPNESS_RISE = 0.10` above the adapter
alone, with 5 steps in between and the composed count within
`FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1` seed. Null inside that band. Composition breaks if two or
more composed seeds are lost. Inconclusive if the control pair loses two or more.

For the clean tail: the primary read is the cosine distance between a render and that seed's
joint-prompt render in the [compose scorer](../../context/world/compose-rate.md)'s own DINOv2 embedding, lower being nearer the picture
the adapter is trying to reach. Support if a condition moves at least `CLEAN_MIN_MONO_GAIN = 0.05`
nearer than the adapter alone with composition within one seed. Laplacian variance is reported
beside it as a secondary read only, because on the plain-PoE references it is heavy-tailed (mean
152, standard deviation 183 over the eight seeds, three of which render grainy at 275 to 499), so a
band built from it cannot separate conditions. That was measured and written down before this grid
ran.

Both are pinned in
[does the corrector compose in the same window](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md)
and [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md).

## 1. The corrector on the adapter's own score changes nothing that matters

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#2-the-hand-off-to-the-frozen-model-is-the-lever-that-moves) ➡️

![Eight seeds by five columns: the joint prompt, plain PoE, the adapter alone, and the adapter plus 5 and 20 corrector steps on its own score over the last fifteen steps](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-on-adapter-tail-cat-dog-eight-seed-sheet.png)
*Rows are the held-out seeds 9 to 16; each tile carries its instance count and its Laplacian
variance. What to notice: the last three columns are the same picture with fur and edges redrawn,
and none of them reaches the crispness of the first column.*

**The number.** Mean Laplacian variance of the greyscale render over the eight seeds, cat × dog, at
0, 5 and 20 corrector steps on the corrected score inside steps 35 to 49: 57.1, 61.1, 61.7, a rise
of 8.0% against the 10% bar, so the null branch fires; medians 37.2, 43.9, 45.4, and 20 steps is
sharper than none on 4 of the 8 seeds. Composed seeds by the validated instance count: 7, 8, 7 of 8.
The chain moved inside the window (median relative displacement 0.36 at 5 steps and 0.67 at 20 on
seed 9), so this is a null and not a stalled instrument. From
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/tail_fidelity.json`, fields
`summary` and `branch`.

## 2. The hand-off to the frozen model is the lever that moves

Navigation: ⬅️ [Previous](#1-the-corrector-on-the-adapters-own-score-changes-nothing-that-matters) | 📋 [TOC](#table-of-contents) | [Next](#3-the-control-pair-keeps-what-it-already-had) ➡️

![Eight seeds by nine columns: the joint prompt, plain PoE, the adapter alone, then the adapter switched off after step 19 or 29 with 0, 5 or 20 corrector steps on the frozen score](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-clean-tail-cat-dog-eight-seed-sheet.png)
*Rows are the held-out seeds 9 to 16; each tile carries its instance count, its distance to that
seed's joint-prompt render, and its sharpness. What to notice: the sixth column, the adapter
switched off after step 29 with no corrector, is the crispest of the adapter columns on seeds 9,
12, 13 and 16, and the corrector columns beside it are grainier.*

**The number.** Mean cosine distance to the seed's joint-prompt render in the compose scorer's
DINOv2 embedding, over the eight seeds, cat × dog: the adapter alone 0.472, plain
product-of-experts 0.643. The adapter switched off after step 19, with 0, 5 and 20 corrector steps
on the frozen score: 0.462, 0.467, 0.518. Switched off after step 29: 0.434, 0.469, 0.490. The best
is a gain of 0.038 with no corrector at all, under the 0.05 bar, so the null branch fires; the
corrector at 20 steps moves 0.046 and 0.018 the wrong way while raising Laplacian variance from 57
to 112 and 94, which reads as grain. Composed seeds: 5, 6, 7 and 6, 7, 7 of 8 against the adapter's
7. From the same folder's `clean_tail.json`, fields `summary`, `baseline_adapter_alone` and
`branch`, over 96 renders.

## 3. The control pair keeps what it already had

Navigation: ⬅️ [Previous](#2-the-hand-off-to-the-frozen-model-is-the-lever-that-moves) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Eight seeds by nine columns for butterfly and meadow, every frame green](../../artifacts/results/is-the-gap-the-samplers-or-the-models/corrector-clean-tail-butterfly-meadow-eight-seed-sheet.png)
*The same layout on the pair that already composes. What to notice: every tile keeps its butterfly
and its meadow, so no condition here breaks what was already working.*

**The number.** Butterfly × meadow, both concepts detected above confidence 0.30: 8 of 8 in every
one of the six clean-tail conditions and in all three tail conditions, against 8 of 8 for the
adapter alone. Its distance to the joint render moves 0.518 (adapter alone) to 0.440 at the best
condition, and its Laplacian variance rises from 506 to 726 under the corrector on the adapter's own
score, which is the meadow's fine texture returning rather than a composition change. From
`clean_tail.json` and `tail_fidelity.json`, field `summary`.

## What this cannot tell you

Navigation: ⬅️ [Previous](#3-the-control-pair-keeps-what-it-already-had) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**Whether a schedule would clear the bar.** The hand-off tested here is a hard cut: the adapter is
on at full strength through the cutoff step and off after it. A ramp down, or a re-noise and
redenoise of the tail, are different conditions and are not measured.

**The detector disagrees with the eye on two tiles.** On seeds 10 and 11 at the cutoff-20 hand-off
with no corrector, the count reads one animal where the eye sees two, one hidden behind the other,
so that condition's 5 of 8 is 7 of 8 by eye. The counts above are the detector's throughout, and
the disagreement is why the eye read is recorded beside them.

**Two cutoffs, one adapter, one strength.** Steps 19 and 29, the rank-32 adapter at step 30050,
correction strength 1.2 throughout.

**Sharpness is a proxy the references break.** Laplacian variance on the plain-PoE renders spans
12 to 499 across eight seeds of one pair, which is why it is secondary here and why no band was
built from it.

**Mixed on one axis, said plainly.** The clean-tail conditions change the cutoff and the corrector
count together against the adapter-alone baseline, which differs from all of them in both. Reading
the cutoff-30 gain as "the hand-off did it" rests on the corrector columns of the same cutoff being
worse, not on a single-axis comparison.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The corrector on the adapter's tail | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/tail_fidelity.json`, 48 renders, read 2026-09-06 | verified |
| The clean tail | the same folder's `clean_tail.json`, 96 renders, read 2026-09-06 | verified |
| The renders | `corrector/tail/<pair>/seed_<n>/` and `corrector/clean_tail/<pair>/seed_<n>/`, with a per-condition summary beside each | verified |
| The adapter | rank 32, step 30050, `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt`, 210 modules matched and 420 tensors loaded at attach | verified |
| The eye reads | recorded in [the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/04-does-the-corrector-compose-in-the-same-window.md), Claude with veto after | stated |
| The runs | task group 3 and task group 5 of step 27, on mscluster85 device 0, 2026-09-06, commits 87d6cb2 and af7eae6; W&B `prime_lab/poe-repair-animals-compose/s61hldbc` | verified |
| Regenerate with | recipes 5 and 6 of [running the Langevin corrector](../../runbook/running-things-on-the-cluster/running-the-langevin-corrector.md); the adapter stage runs in its own process, per [the windowed sampler note](../../environment/known-failures.md) | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What the trained correction is and why it never sees the joint prompt:
  [the LoRA corrector](../../context/world/lora-corrector.md)
- What a compose rate is, and the detector's known error on this pair:
  [compose rate](../../context/world/compose-rate.md)
- Which checkpoint this adapter is and why softness rose with training:
  [does training longer keep improving the held-out fix](../does-training-longer-help-the-pooled-lora/does-training-longer-keep-improving-the-held-out-fix.md)
- Why an adapter render and a reference render never share a process:
  [the known failures catalog](../../environment/known-failures.md)
- Each picture, read one at a time: [the corrector figures, what these pictures mean](../../artifacts/results/is-the-gap-the-samplers-or-the-models/figure-explainer-the-langevin-corrector.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] The next lever, and the one this finding points at: the correction strength ramped down
      across the hand-off rather than cut, and the re-noise and redenoise tail, both owned by
      [correct early, then clean up](../../plans/01-showcase-the-trained-lora/plans/experiments/14-correct-early-then-clean-up.md).
      Judged on this finding's own bar, the 0.05 gain in distance to the joint render, against this
      finding's adapter-alone baseline of 0.472.
- [ ] Whether the same hand-off helps on a pair the adapter never trained on. Both pairs here are
      the standard failing pair and its composing control.
