# Which checkpoint composes best, and does more correction or an early-only window help?   ✅ 7 of 8 at rank 8 @ 30k, rank 16 @ 50k, rank 32 @ 30k to 90k · ⚪ null on λ above 1 and on the early window · verified 2026-09-05

**The claim**

On the held-out pair cat x dog, the pooled adapter at λ 1 with the correction on every DDIM step
composes 7 of the 8 seeds at rank 8 @ 30k, rank 16 @ 50k, and rank 32 at every checkpoint
probed from 30k to 90k. Plain PoE composes 0 of 8 on the same seeds. Raising λ to 1.2 through 2.0
never adds a seed on a healthy checkpoint and costs the cat-and-dog reading on the degraded
rank 8 checkpoints. Restricting the correction to steps 0 to 9 never reaches the full window's
count. The one seed that never composes, seed 14, has a joint-prompt target that is itself two
cats.

**What would have counted**

No criterion was pre-registered for the λ or window sweeps. They were run to test the session's
hypothesis that the sampler should favour the correction over falling back to PoE, and the read
is descriptive: a seed count that rises with λ or with the early window would have supported the
hypothesis, a flat or falling count is what a saturated correction looks like. The 7 of 8
figure is judged against the instance-count scorer validated in
[can we trust the compose score](../../artifacts/results/can-we-trust-the-compose-score/).

## 1. The three best checkpoints, seed by seed

![Rank 32 at 30k: eight held-out seeds, joint prompt, plain PoE, λ 1.0, λ 1.2](../../artifacts/results/does-training-longer-help-the-pooled-lora/grid-rank32-30k-full-window.png)
*Rows are held-out seeds 9 to 16. Columns: the joint-prompt render, plain PoE, PoE plus the
rank 32 correction at λ 1.0, then at 1.2. What to notice: every PoE tile is one blended animal
and seven of eight corrected tiles are a cat and a dog. Seed 14 is the exception in both
corrected columns.*

The rank 8 @ 30k and rank 16 @ 50k grids are
[grid-rank8-30k-full-window.png](../../artifacts/results/does-training-longer-help-the-pooled-lora/grid-rank8-30k-full-window.png)
and
[grid-rank16-50k-full-window.png](../../artifacts/results/does-training-longer-help-the-pooled-lora/grid-rank16-50k-full-window.png),
same layout, same seed that fails.

**The number.** Seeds the instance-count scorer calls two animals, of 8, λ 1, all 50 steps:
rank 8 @ 30,000: 7; rank 16 @ 50,018: 7; rank 32 @ 30,050: 7. The strict read (a cat box and a
dog box with overlap under 0.5) gives the same 7 in each case. Plain PoE (λ 0): 0 of 8 at every
checkpoint. From `summary.full.1.0.compose_rate` and `rows[].cat_and_dog` in
`/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r8_030000/results.json`,
`figure_r16_050018/results.json` and `figure_r32_030050/results.json`, written by
`scripts/showcase/lambda_boundary_probe.py`.

## 2. More correction does not buy more seeds

![Seeds composing against λ, one line per checkpoint, three ranks, full and early windows](../../artifacts/results/does-training-longer-help-the-pooled-lora/compose-count-vs-lambda-by-checkpoint.png)
*x is λ, y is how many of 8 held-out seeds compose, one line per checkpoint (label = step).
Top row: correction on all 50 steps. Bottom row: correction on steps 0 to 9 only. What to
notice: the top-row lines for rank 16 and rank 32 sit flat between 6 and 8 from λ 1.0 to 2.0,
the rank 8 late checkpoints fall at 2.0, and no bottom-row line reaches its top-row partner.*

**The number.** Full window, λ 1.0 / 1.2 / 1.5 / 2.0: rank 16 @ 50k 7 / 6 / 7 / 7; rank 32 @
60k 7 / 7 / 7 / 7; rank 32 @ 70k 7 / 7 / 7 / 8. Rank 8 @ 240k 5 / 7 / 4 / 3, and at λ 2.0 the
strict cat-and-dog count is 0 of 8 (three tiles counted as two animals are neither a cat nor a
dog); rank 8 @ 280k 4 / – / 4 / 1. Early window (steps 0 to 9), best λ per checkpoint: rank 8
at most 4 of 8, rank 16 at most 4, rank 32 at most 5. From `results.json` in
`lambda_boundary_probe_r16_050018/`, `lambda_boundary_probe_r32_060050/`,
`lambda_boundary_probe_r32_070050/`, `lambda_boundary_probe_240k/`, `lambda_boundary_probe/`
(280k) and `lambda_boundary_probe_300k/` under `/datasets/mmolefe/poe_repair_min/outputs/showcase/`,
fields `rows[].compose` and `rows[].cat_and_dog`.

## 3. What λ 2.0 does to a degraded checkpoint

![Rank 8 at 300k, λ sweep, correction on all 50 steps](../../artifacts/results/does-training-longer-help-the-pooled-lora/grid-rank8-300k-full-window-lambda-sweep.png)
*Rows are seeds 9 to 16, columns λ 0 to 2.0, rank 8 at 300k. What to notice: the tiles are
grey and soft at every λ, and from 1.3 upward several two-animal tiles turn amber, meaning the
scorer sees two animals that are not a cat and a dog.*

**The number.** Rank 8 @ 300k, full window, λ 1.0 / 1.2 / 1.3 / 1.4 / 1.5 / 2.0: two animals
5 / 5 / 4 / 4 / 5 / 5 of 8; strict cat and dog 5 / 5 / 4 / 3 / 4 / 2. DINOv2 drift at λ 1:
+0.014, meaning the corrected renders sit no nearer the joint-prompt image than plain PoE does.
From `lambda_boundary_probe_300k/results.json`, same fields plus `rows[].drift.dino.drift`.

## 4. The sampler is deterministic, so a seed's verdict is a property of the checkpoint

No figure.

**The number.** Rendering seed 9 twice from the rank 8 @ 280k checkpoint on the same device
gives bit-identical PNGs (three renders each, `determinism_repeat_1/` and `determinism_repeat_2/`
under the showcase outputs, log `logs/determinism_repeat.log`). Across GPU generations the same
render differs by about 2 of 255 in mean pixel value, from fp16 kernel differences, and the
compose verdicts agree. The 50-step DDIM sampler runs at η 0 with pinned initial latents from the
training cache, so there is no sampling noise to average over. Mark: stated from the session
record for the cross-device number; the same-device identity is verified from the two folders.

## 5. On the checkpoint the showcase figures use, raising λ changes the scene rather than the count

![Rank 32 at 30050: eight held-out seeds, the joint prompt then λ 0, 1.0, 1.2, 1.3, 1.5 and 2.0, correction on all 50 steps](../../artifacts/results/does-training-longer-help-the-pooled-lora/grid-rank32-30050-full-window-lambda-sweep.png)
*Rows are seeds 9 to 16, columns run from the joint-prompt render to λ 2.0. What to notice: seed
10's two close-up dogs become a distant desert shack with two thumbnail-sized animals from λ 1.3
onward, and at λ 2.0 seeds 9, 11 and 15 change medium, from photograph or pencil sketch to
coloured illustration.*

**The number.** Seeds the instance-count scorer calls two animals, of 8, correction on all 50
steps, λ 0 / 1.0 / 1.2 / 1.3 / 1.5 / 2.0: 0 / 7 / 6 / 6 / 7 / 6. The strict cat-and-dog count is
identical at every λ. DINOv2 drift, where negative means the render sits nearer the joint-prompt
image: +0.531 / −0.122 / −0.162 / −0.134 / −0.135 / −0.124. Five seeds (9, 12, 13, 15, 16)
compose at every λ from 1.0 up; seed 10 stops composing from 1.3, seed 11 fails only at 1.2, and
seed 14 composes only at 1.5. From
`/datasets/mmolefe/poe_repair_min/outputs/showcase/lambda_boundary_probe_r32_030050/results.json`,
fields `rows[].compose`, `rows[].cat_and_dog` and `rows[].drift.dino.drift`, written by
`scripts/showcase/lambda_boundary_probe.py` on 2026-09-06.

**The second number.** The largest per-step correction norm the sampler applied, averaged over
seeds, falls as λ rises: 60.4 at λ 1.0, 51.4 at 1.2, 47.8 at 1.3, 44.8 at 1.5, 31.5 at 2.0
(`rows[].max_delta_norm`, same file). The adapter is reading a trajectory that λ has already
moved, so a larger multiplier does not produce a proportionally larger correction.

## What this cannot tell you

**The count is blind to the scene change.** Seed 10 still has two animals at λ 1.3 and above,
but they are thumbnail-sized in a wide shot and the detector stops counting them, so the drop
from 7 to 6 at those settings reads as a composition failure when the grid shows a framing
change. Any λ conclusion drawn from the count alone inherits this.

**One pair, eight seeds.** Every count here is cat x dog. The other seven held-out pairs were
scored only by the 2-cell tracking set during training and by the fit read in
[is the held-out gap a fit, a drift, or a pair problem](../does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md).

**Rank and step move together across the "best" checkpoints.** Rank 8 @ 30k, rank 16 @ 50k
and rank 32 @ 30k differ on two axes, so nothing here ranks the ranks. Rank 32's flat 7 of 8
from 30k to 90k is the one within-rank series.

**The scorer counts instances, not correctness of species.** The strict cat-and-dog read is a
box-level check with the same detector and is not separately validated.

**Why λ saturates is not measured.** The session's guess is that the correction's magnitude is
already matched to the target (‖Δ̂‖/‖Δ‖ is 0.87 on held-out cells in the fit read), so scaling it
up overshoots rather than pushes further. That is an inference, not a result.

## Where this came from

| What | Source | Mark |
|---|---|---|
| Compose and strict counts per checkpoint, λ and window | `results.json` in the eleven probe folders named in rungs 1 to 3, read 2026-09-05 into `compose-count-vs-lambda-by-checkpoint.png` | verified |
| The grids | `grid_full_window.png` in the same folders, resized copies in `artifacts/results/does-training-longer-help-the-pooled-lora/` | verified |
| Rung 5's counts, drift and correction norms | `/datasets/mmolefe/poe_repair_min/outputs/showcase/lambda_boundary_probe_r32_030050/results.json`, run and read 2026-09-06 on mscluster106 device 1 | verified |
| Rung 5's scene and medium changes | read off `grid-rank32-30050-full-window-lambda-sweep.png` | verified |
| Same-device determinism | `determinism_repeat_1/`, `determinism_repeat_2/`, `logs/determinism_repeat.log` | verified |
| Cross-device pixel difference | session record, 2026-09-02 | stated |
| The checkpoints | rank 8: `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_030000.pt` and `/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints/`; rank 16 and 32: `phase1_r16_100k/checkpoints/`, `phase1_r32_100k/checkpoints/` under the same outputs root | verified |
| The runs that produced them | [train longer](../../plans/01-showcase-the-trained-lora/plans/experiments/08-experiment-a-resume-to-200k.md) for rank 8, the rank 16 and 32 plan in the same scope; W&B `8vl2uzak`, `n1w4fw5b`, `gcej8ib9`, `6xc2l8ix` in `prime_lab/poe-repair-animals-compose` | verified |
| Regenerate with | [render the held-out grid for one checkpoint](../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#1-render-the-8-seed-held-out-grid-for-one-checkpoint) | |

## Depends on

- What the compose rate and the instance-count scorer mean: [the context index](../../context/00-INDEX.md).
- Which sampler and which window: `run_lora_residual_inject_windowed_poe` in `scripts/showcase/lambda_window_grid.py`; off-window steps are plain guided PoE.
- Which nodes and python environments: [the cluster nodes](../../environment/hpc/nodes.md).

## Still open

- [ ] The cross-device pixel difference is quoted from the session, not from a file on disk.
- [ ] The strict cat-and-dog read has no validation set of its own.
- [ ] Seed 14's target is two cats in the cache; whether a re-cached target composes is untested.
- [ ] The scene and medium changes above λ 1.3 in rung 5 are read off the grid by eye. Nothing
      measures them, so there is no number saying how far the framing moves per unit of λ.
