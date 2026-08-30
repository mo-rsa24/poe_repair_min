# 🔬 Does the fix change what a word paints, or where it looks?

This plan asks whether switching the trained fix on changes the content a subject word paints
while leaving the place it looks at alone, measured on 64 pair-and-seed runs the fix never
trained on.

**Step 8 of 22.** Waits on step 3. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 7 | ~~[hypothesis-05-the-same-story-from-three-sides](hypothesis-05-the-same-story-from-three-sides.md)~~ | ✅ |
| **8** | **this plan** | **✅** |
| 9 | [three live curves while training](../../04-does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md) | ⚠️ do this next |

Design only. Findings and run state live in
[../review/hypothesis-01-what-the-fix-changes-inside-the-model.md](../review/hypothesis-01-what-the-fix-changes-inside-the-model.md).

## What this asks, in one line
When the trained fix is switched on, does it change **what** a word paints, or **where** that
word looks? Those are two different mechanisms, and the paper claims the first one.

## Why it matters
Every word in the prompt does two things inside the model: it decides where in the image to look
(the attention weights) and it decides what to write there (the painted content). Our account of
why the fix works says it changes the painted content and leaves the attention weights where they
were. That account currently rests on one seed of one pair, and the paper's mechanism section is
built on it. This plan asks whether it survives 64 pair-and-seed runs the fix never trained on.
It supplies figure **F7**, which the paper has promised and not yet built.

## What gets measured
For each of the pair's two subject words, at three points in the denoising run, capture both maps
(where it looks, what it paints) with the adapter off and again with it on, from the identical
starting state. Then compare how much each map's spatial pattern moved.

## Why the obvious measure gives the wrong answer
This is the load-bearing design decision, because **the obvious comparison gives the opposite
answer.**

The obvious thing is to measure each map's total change, `||on − off|| / ||off||`, and compare.
Under that measure the attention weights move 1.70 times more than the painted content, which
contradicts the hypothesis. That reading is wrong, for two reasons that have nothing to do with
the science:

- The two maps are not on the same footing. Attention weights are rows that sum to one; painted
  content carries raw magnitudes. Their norms are not comparable quantities.
- The adapter dims the attention weights by roughly 25% overall. That is a uniform brightness
  change, and it swamps the spatial-pattern change the hypothesis is actually about.

So the comparison is made scale-free instead. Fit the single best rescaling of the off-map onto
the on-map, then split the change in two: **gain** (how much is just uniform brightness) and
**pattern** (what a rescaling cannot explain). The hypothesis is about pattern, so pattern is
what gets compared. `gain_and_pattern()` in `value_probe.py` computes this for each pair-and-seed
run during capture, so the scoring step reads it rather than re-deriving it.

The full argument, with its guards (a shuffled-map control, a denominator check, the raw sums),
is in `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/measure-fairness.md`.

## Environment Facts This Plan Depends On
- `co3` python at its absolute path. Inference for the single-run test fits the in-session 3090.
  All 64 runs together go to biggpu first, else bigbatch.
- Capture files accumulate: write to `/datasets` with the disk guard, never `/home-mscluster`.
- Checkpoint: `artifacts/scopes/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt`.
  Its 420 adapter tensors sit under `sd["lora_state"]`, not at the top level.

## Tasks
- [x] Derive the token position per pair from the tokenizer, instead of the hardcoded position
      that only works for one-piece animal names.
- [x] Run one pair and one seed in-session first, on a held-out pair, and look at the maps.

      > Held-out means the pair was never used in training, so the fix has not seen it before.
- [x] Run all 64: 8 held-out pairs × seeds 9 to 16, adapter off against on at matched
      steps.
- [x] Compute the comparison table, one row per token and step, using the pattern term described
      above. `scripts/mechanism_study/reprobe_table.py`.
- [x] Record the verdict in the review file against the threshold that was written before the run.
- [x] Decide the figure's statistical entity with `/pair-figure`: one point per pair, median over
      its 8 seeds, seeds shown as a pale spread behind rather than averaged away. F7's caption is
      frozen to the narrower sentence the review file requires.

## Success/Failure Outcomes
- **the first single run**
  - Success: the maps render as recognisable head-and-body shapes, and the token position indexes
    the right word for that pair.
  - Failure: noise or empty maps, meaning the token position missed the words. Fix that before
    launching the other 63, or they measure nothing.
- **the verdict**
  - Replicates: median pattern ratio at least 1.2, and at least 75% of rows above 1.
  - Does not: the paper's mechanism section shrinks to one honest negative paragraph, which is a
    result this plan provides for. Do not loosen the threshold to avoid it.

## Next

1. `/pair-figure` on the mechanism comparison: choose per-seed points or pair-level means, and
   write the choice into the review file's open question.
2. `/design-figure` rides `figure-01-the-seven-paper-figures.md` for F7's final form. Do not
   design it here.

## Illustrations
*(image not yet generated; save under `../assets/` and replace this placeholder)*

**Prompt for image generation:**
> Generate an image of a flowchart showing this experiment: derive the token position per pair,
> try one pair-and-seed run, run all 64 of them, compute the pattern comparison, record the
> verdict. Success path green with checkmark "Completed" pills. Failure path red on the
> single-run stage labeled "token position missed the words, maps are noise" with an X icon and a dashed "Retry
> Stage" callout. Downstream stages muted gray with "Skipped" pills. Glossy, minimalistic,
> modern UI/UX dashboard panel, dark background, rounded rectangle stage cards in a horizontal
> row connected by directional arrows, clean sans-serif labels, generous spacing, no clutter.

## Recommended skill
▶ `/run-experiment` ✅ for the 64 runs; `/pair-figure` ✅ before plotting.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.mechanism_study.value_probe \
  --checkpoint <lora_step_100000.pt> --pair-slug an_eagle__x__a_hawk --seed 9 \
  --steps 10,25,40                              # first short run: expect 3 step files
ls -d /datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe/*/seed_* | wc -l
                                                # expect 64 pair-and-seed runs
$PY scripts/mechanism_study/reprobe_table.py    # table + verdict.json
```
