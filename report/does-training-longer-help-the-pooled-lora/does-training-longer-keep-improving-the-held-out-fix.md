# Does training longer keep improving the held-out fix?   ❌ no: fidelity to the joint-prompt image decays from about 70k (rank 32) and 160k (rank 8) while the adapter weights grow without bound · verified 2026-09-05

**The claim**

Past its best checkpoint, more training makes the held-out renders worse, not better. The number
of seeds that compose holds (rank 32) or falls (rank 8), but the corrected renders stop moving
toward the joint-prompt image, then go grey and soft. The adapter's weight norm grows without
bound the whole time, which is what training with weight decay 0 and a constant learning rate on
88 cells does once the loss has flattened. The training pairs do not show the decay; only the
held-out pair does.

**What would have counted**

The rank 8 extension from 100k to 200k was planned in
[train longer](../../plans/01-showcase-the-trained-lora/plans/experiments/08-experiment-a-resume-to-200k.md)
to read whether the tracking set's held-out curves keep improving. No numeric bar was written for
"worse". The read here is descriptive: DINOv2 drift on the 8-seed grid returning to 0 means the
corrected renders are no nearer the joint-prompt image than plain PoE, and that is taken as the
loss of fidelity. The decay criterion is post-hoc and is said so.

## 1. Same seed, later checkpoints

![Rank 32, seeds 9, 11, 16 at 30k, 60k, 70k, 90k](../../artifacts/results/does-training-longer-help-the-pooled-lora/rank32-same-seed-across-checkpoints.png)
*Each row is one held-out seed from the same initial latent, λ 1, correction on all 50 steps.
Columns are the rank 32 checkpoints at 30k, 60k, 70k and 90k. What to notice: 30k and 60k are
near-identical and crisp; at 70k seed 9 has turned into a sketchy group and at 90k its dog's head
is malformed and the colour has drained. All four columns still count as two animals.*

![Rank 8, seeds 9, 12, 16 at 30k, 240k, 280k, 300k](../../artifacts/results/does-training-longer-help-the-pooled-lora/rank8-same-seed-across-checkpoints.png)
*Same layout for the rank 8 lineage. What to notice: by 240k every tile has lost saturation, by
280k seed 9 is one blurred dog, and seed 16 is a haze of fur in the last three columns.*

**The number.** DINOv2 drift, the corrected render's distance to the joint-prompt render minus
its distance to the plain PoE render, mean of 8 seeds at λ 1, all steps (negative = nearer the
joint-prompt image): rank 32 at 30,050 −0.091, 60,050 −0.060, 70,050 −0.003, 90,050 −0.013.
Rank 8 at 30,000 −0.132, 240,000 −0.004, 280,000 +0.041, 300,000 +0.014. Rank 16 at 50,018
−0.102, 60,018 +0.042. The corrected render's absolute distance to the joint-prompt image for
rank 8 grows from 0.551 at 30k to 0.850 at 280k. From `rows[].drift.dino.drift` and
`rows[].drift.dino.d_mono` in the `results.json` of `figure_r32_030050/`,
`lambda_boundary_probe_r32_060050/`, `lambda_boundary_probe_r32_070050/`, `figure_r32_090050/`,
`figure_r8_030000/`, `lambda_boundary_probe_240k/`, `lambda_boundary_probe/`,
`lambda_boundary_probe_300k/`, `figure_r16_050018/` and `lambda_boundary_probe_r16_060018/` under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/`.

## 2. The decay over the whole run, in both instruments

![Held-out DINOv2 drift against training step, three ranks, 8-seed grids and the 2-cell tracking set](../../artifacts/results/does-training-longer-help-the-pooled-lora/held-out-drift-vs-training-step.png)
*x is training step in thousands, y is the DINOv2 drift of rung 1. Bold points are the 8-seed
grids; faint lines are the 2-cell tracking set (cat x dog seeds 9 and 10) each run logged to W&B
every 10k steps. What to notice: all three ranks start near −0.1 and climb to 0; the faint lines
are noisy because two cells is a small sample, but they cross 0 where the bold points do.*

**The number.** Tracking-set drift, mean of the 2 held-out cells, `eval/tracking/embedding_drift/dino/out_out/*`
in W&B: rank 32 (`6xc2l8ix`) −0.264 at 30,050, −0.234 at 60,050, −0.021 at 70,050, +0.010 at
80,050, +0.018 at 90,050, −0.045 at 100,050. The 2 training cells (`in_in`, wolf x husky seeds 1
and 2) stay between −0.05 and −0.15 at every save on every run. Rank 8 (`8vl2uzak`, `n1w4fw5b`)
held-out drift is at or above 0 from 140k on, with excursions to −0.13 at 230k to 240k and 320k.

## 3. The weights never stop growing

![Frobenius norm of the LoRA weights against training step, three ranks](../../artifacts/results/does-training-longer-help-the-pooled-lora/adapter-weight-norm-vs-step.png)
*x is training step, y is the norm of all LoRA weights read from each checkpoint. What to notice:
no curve flattens, and the higher the rank the higher and steeper the curve.*

**The number.** Frobenius norm of `lora_state`: rank 8 35.9 at 10k, 60.1 at 100k, 73.2 at 200k,
93.2 at 450k; rank 16 45.1 at 10k, 72.5 at 100k; rank 32 58.9 at 10k, 68.8 at 30k, 79.0 at 60k,
86.9 at 90k, 89.3 at 100k. The optimizer is AdamW with `weight_decay` 0.0 and a constant
learning rate of 1e-4 (`OptimConfig` in `poe_repair/experiments/one_pair_one_seed/config.py`;
no scheduler in `multi_pair_trainer.py`). Computed 2026-09-05 from the checkpoint files under
`phase1_r8_100k/`, `phase1_r8_200k/`, `phase1_r8_450k/`, `phase1_r16_100k/` and
`phase1_r32_100k/`.

## 4. The compose count is the last thing to go

No figure; the counts are in rung 2 of
[which checkpoint composes best](which-checkpoint-composes-best-and-does-more-correction-help.md).

**The number.** Seeds composing of 8 at λ 1, all steps: rank 32 stays at 7 at 30k, 60k, 70k and
90k while its drift goes from −0.091 to −0.013. Rank 8 goes 7 (30k), 5 (240k), 4 (280k), 5
(300k). So the scorer's count lags the fidelity loss by tens of thousands of steps and is not the
instrument to stop training on.

## What this cannot tell you

**Cause is inferred.** Norm growth, the loss having flattened, and off-trajectory evaluation
(the adapter trains on plain-PoE states and is used on corrected ones) all rise together; no
run varied weight decay or the learning rate, so which of them drives the decay is untested.
The pressure test in
[the idea route](../../artifacts/ideas/improving-the-pooled-lora-run/routes/01-pressure-test-poe-failures.md)
names the same three candidates.

**Two cells is a small tracking set.** The faint W&B lines swing by 0.3 between saves. The
decay reading rests on the 8-seed grids; the tracking set only agrees with them.

**Held-out means cat x dog.** Whether the other seven held-out pairs decay on the same schedule is
not measured.

**The rank 8 lineage crossed three runs and two GPU generations** (A6000 or RTX 8000 to 100k,
Blackwell after). Cross-device fp16 differences are about 2 of 255 in pixel value, so this is
not the decay, but the lineage is not one clean run.

## Where this came from

| What | Source | Mark |
|---|---|---|
| 8-seed drift and counts | `results.json` in the ten probe folders of rung 1, read 2026-09-05 | verified |
| Tracking-set drift | W&B histories of `6xc2l8ix`, `8vl2uzak`, `n1w4fw5b`, `gcej8ib9`, `jii2mtb0` in `prime_lab/poe-repair-animals-compose`, pulled 2026-09-05 with the `wandb` API | verified |
| Weight norms | `lora_state` in every 10k checkpoint, computed 2026-09-05 | verified |
| Optimizer settings | `OptimConfig` in `poe_repair/experiments/one_pair_one_seed/config.py`, read 2026-09-05 | verified |
| The runs | rank 8 100k to 200k (`8vl2uzak`, mscluster110, 6.8 h), 200k to 450k (`n1w4fw5b`, mscluster112, 18.1 h); rank 16 (`jii2mtb0` aborted at 37,418 by the kill criterion after 12.3 h, resumed as `gcej8ib9` on mscluster108, 20.3 h); rank 32 (`le2tp2ti` aborted at 6,500 after 2.3 h, resumed as `6xc2l8ix` on mscluster106, 34.0 h). All finished. Timeline in `training-runs-timeline.png` beside the figures | verified |
| Regenerate with | [render the held-out grid for one checkpoint](../../runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md#1-render-the-8-seed-held-out-grid-for-one-checkpoint), then rerun the chart script named in the artifact card | |

## Depends on

- What the tracking set is and what each curve means: [reading a training run](../../runbook/looking-at-what-a-run-produced/reading-a-training-run.md).
- The kill criterion that aborted the two fresh runs and does not apply to resumed ones: `multi_pair_trainer.py`, the commit-bucket loss must halve from its step-200 value by step 5000; it is not checkpointed.
- Measured step times and eval-pass times per GPU: [throughput](../../environment/hpc/throughput.md).

## Still open

- [ ] A run with weight decay on (the trainer has no flag for it yet; `OptimConfig.weight_decay` is hard-wired to 0.0) and an early stop on held-out drift, to test the inferred cause.
- [ ] The same same-seed strip for one other held-out pair.

## Cross-references

- The mention of **DINOv2 distance to the joint-prompt render** as the read for a checkpoint's fidelity, reused as the bar for the clean tail, in [can a corrector or a clean tail sharpen the adapter's renders](../is-the-gap-the-samplers-or-the-models/can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md).
