# How much rank does the correction need?   ❓ inconclusive: no trend in rank on either comparable cell, and every comparison available is mixed · verified 2026-09-23

**The claim**

Rank 8, 16 and 32 cannot be ordered by the evidence on disk. On the only two cells the old and new
tracking sets share, the middle rank is best on one and worst on the other, and the three runs
differ in loss and in training length as well as in rank. What the axis does settle is the cost:
every adapter in this project holds between 0.19% and 1.38% of the model's parameters.

**What would have counted**

The same measure as the layers question: the DINOv2 drift each run logs, the corrected render's
cosine distance to the joint-prompt render minus its distance to the plain product render,
dimensionless, -2 to +2, negative meaning nearer the target. The compose rate reads 1.0 for every configuration on
every cell where it is scored (here four cells per run) and cannot carry the axis. No bar was pre-registered, so this is a descriptive read.

## 1. Rank against drift, on the two comparable cells

![DINOv2 drift against adapter rank on cat and dog, seeds 9 and 10](../../artifacts/results/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.png)
*Rank on a doubling axis, drift on the vertical axis, one line per cell. What to notice: neither
line falls with rank, and the seed 10 line sits above zero at every rank, which means no checkpoint
of any rank moved that cell nearer its joint-prompt render than the plain product.*

**The number.** DINOv2 drift at the end of training, cross-attention only: cat and dog seed 9 reads
-0.252 at rank 8 (`phase1_r8_200k`, 200,000 steps), -0.516 at rank 16 (`phase1_r16_100k`, 100,018
steps) and -0.408 at rank 32 (`phase1_r32_100k`, 100,050 steps). The same three runs on seed 10
read +0.222, +0.428 and +0.319. Values in
[the figure's sidecar](../../artifacts/results/what-the-adapter-ablations-show/how-much-rank-the-correction-needs.json).

**Why this is mixed, not a result.** The three runs use the epsilon-space loss and differ in
training length (200k, 100k, 100k steps), and they are read on a tracking set whose other cells
differ from the current one. Only cat and dog seeds 9 and 10 appear in both tracking sets, so the
comparison rests on two cells.

## 2. What each rank costs

**The number.** Trainable parameters from each run's `lora_attach.json`, cross-attention only, 210
adapted projections: rank 8 is 4,956,160 of 2,572,419,844 (0.19%), rank 16 is 9,912,320 of
2,577,376,004 (0.38%), rank 32 is 19,824,640 of 2,587,288,324 (0.77%). With self-attention added at
rank 32 the adapter reaches 35,799,040 of 2,603,262,724 (1.38%), and with self-attention at rank 64
it reaches 51,773,440 of 2,619,237,124 (1.98%).

## Still open

- **A clean rank sweep.** Three runs at the same loss, the same pool and the same length, differing
  only in rank, do not exist. Each costs about 14 hours on one 49 GB card.
- **Rank 64 has no read.** `v57-12-x0-self64` was stopped at step 42,500 before an evaluation pass
  was logged, so only its parameter count is reported here.

## Where this came from

| What | How it was established | When |
|---|---|---|
| Every drift value | last logged value in each run's `history.json` on the cluster | 2026-09-23 |
| Every parameter count | `lora_attach.json` in each run's output folder | 2026-09-23 |
| Which cells are comparable | the metric keys present in both eras' `history.json` | 2026-09-23 |
| The figure | `scripts/adapter_ablation_figures.py` | 2026-09-23 |
