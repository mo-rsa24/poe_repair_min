# Does adapting self-attention as well as cross-attention change the result?   ❓ inconclusive: the comparison available today is mixed, and the clean one is still training · verified 2026-09-23

**The claim**

Nothing in the runs on disk shows that covering the self-attention projections as well as the
cross-attention ones makes the correction better or worse. The one adapter that covers both is
better than every cross-attention adapter on one held-out cell and worse than all of them on
another, and it differs from them in two other ways besides its layers, so the gap cannot be read
as the layers' doing.

**What would have counted**

The measure had to have headroom, and the compose rate does not: every configuration here reads
1.0 on the six-cell tracking set, so it cannot separate them. The read is the DINOv2 drift logged
by each run, defined in
[`_inline_sampling.py::embedding_drift`](../../poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py)
as the corrected render's cosine distance to the joint-prompt render minus its distance to the
plain product render. It is dimensionless, runs from -2 to +2, and is negative when the render sits
nearer the joint-prompt target than the product it repairs, so lower is better and 0 means the
correction bought nothing. No bar was pre-registered for this question, so the verdict is a
descriptive read and is said so.

## 1. The four adapters on the same six cells

![DINOv2 drift per tracking cell for three cross-attention adapters and one that also covers self-attention](../../artifacts/results/what-the-adapter-ablations-show/which-layers-carry-the-adapter.png)
*Rows are the six tracking cells, two trained on and four held out. Bars are the four adapters, all
rank 32 on the same pool. What to notice: the red bars (cross and self-attention) are neither
consistently longer nor consistently shorter than the blue ones, and on elephant and penguin at
seed 9 the red bar crosses zero to the wrong side.*

**The number.** DINOv2 drift at step 50,000, cross and self-attention (`v57-11b-x0-self-decay`,
W&B `tiiz0c0c`) against the best cross-attention run (`v57-09-x0-seeds`, W&B `ek4w6lbu`): cat and
dog seed 9, -0.527 against -0.488; elephant and penguin seed 9, +0.174 against -0.234; six-cell
mean -0.269 against -0.329. The other two cross-attention runs are `v57-08-x0-w25` (`a5rv38qy`,
mean -0.288) and `v57-10-x0-contrast` (`boammtzo`, mean -0.271). Every value is in
[the figure's sidecar](../../artifacts/results/what-the-adapter-ablations-show/which-layers-carry-the-adapter.json),
read from each run's `history.json`.

**Why this is mixed, not a result.** `v57-11b-x0-self-decay` also decays its learning rate to zero
and was resumed from a step-5,000 checkpoint, while the three cross-attention runs hold the rate
constant and train from scratch. Two things moved beside the axis, so the comparison is mixed.

## 2. What it costs

**The number.** Trainable parameters, from each run's `lora_attach.json`: cross-attention at rank 32
is 19,824,640 of 2,587,288,324, or 0.77% of the adapted model, over 210 adapted projections. Adding
self-attention takes it to 35,799,040 of 2,603,262,724, or 1.38%, over 420 projections. Holding
cross-attention at rank 32 and self-attention at rank 64 (`v57-12-x0-self64`, W&B `184c4aja`)
reaches 51,773,440 of 2,619,237,124, or 1.98%.

## 3. The cell that does not exist

No run adapts self-attention **instead of** cross-attention. Every adapter in this project covers
the cross-attention projections, alone or with self-attention, so the claim that the correction
needs the layers the prompt enters through has never been tested. The flag that splits the two
ranks, `--lora-rank-self` (commit `e289646`), sets the self-attention rank; a self-attention-only
run needs rank 0 on cross-attention, which the current flag does not express. One run of the same
length costs about 14 hours on one 49 GB card.

## Still open

- **The clean comparison.** `v58-01-plain` (W&B `qx5ffxzp`) and `v58-03-self` (W&B `qtk5gi37`) are
  training on the same pool with the same schedule and differ only in the layer set. When they
  finish, this question gets a clean read and this file gets its verdict.
- **The self-attention-only cell**, above, which needs a flag change and one run.

## Where this came from

| What | How it was established | When |
|---|---|---|
| Every drift value | last logged value in each run's `history.json` on the cluster, extracted by `scripts/local/extract_measures.py` into `measures.json` | 2026-09-23 |
| Every parameter count | `lora_attach.json` in each run's output folder | 2026-09-23 |
| The measure's definition | read from the source of `embedding_drift` | 2026-09-23 |
| The figure | `scripts/adapter_ablation_figures.py` | 2026-09-23 |
