# 🔬 Does the fix change what a word paints, or where it looks?

**Step 8 of 22.** Waits on step 3. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 7 | ~~[hypothesis-05-the-same-story-from-three-sides](hypothesis-05-the-same-story-from-three-sides.md)~~ | ✅ |
| **8** | **this plan** | **✅** |
| 9 | [instrument-02-three-live-curves-while-training](../../does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md) | ⚠️ do this next |

Design only. Findings and run state live in
[../review/hypothesis-01-what-the-fix-changes-inside-the-model.md](../review/hypothesis-01-what-the-fix-changes-inside-the-model.md).

## What this asks, in one line
When the trained fix is switched on, does it change **what** a word paints, or **where** that
word looks? Those are two different mechanisms, and the paper claims the first one.

## Why it matters
Every word in the prompt does two things inside the model: it decides where in the image to look
(the attention weights) and it decides what to write there (the painted content). Our account of
why the fix works says it changes the second, not the first. That account currently rests on one
seed of one pair, and the paper's mechanism section is built on it. This plan asks whether it
survives 64 cells the fix never trained on. Feeds figure slot **F7**.

## What gets measured
For each of the pair's two subject words, at three points in the denoising run, capture both maps
(where it looks, what it paints) with the adapter off and again with it on, from the identical
starting state. Then compare how much each map's spatial pattern moved.

## Why this measure, and not the obvious one
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
what gets compared. `gain_and_pattern()` in `value_probe.py` computes this per cell during
capture, so the scoring step reads it rather than re-deriving it.

The full argument, with its guards (a shuffled-map control, a denominator check, the raw sums),
is in `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/measure-fairness.md`.

## Environment Facts This Plan Depends On
- `co3` python at its absolute path. Probe inference fits the in-session 3090; the full sweep
  goes to biggpu first, else bigbatch.
- Capture files accumulate: write to `/datasets` with the disk guard, never `/home-mscluster`.
- Checkpoint: `artifacts/scopes/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt`.
  Its 420 adapter tensors sit under `sd["lora_state"]`, not at the top level.

## Tasks
- [x] Derive the token position per pair from the tokenizer, instead of the hardcoded position
      that only works for one-piece animal names.
- [x] Smoke one cell in-session: one held-out pair, one seed, and look at the maps.
- [x] Run the full sweep: 8 held-out pairs × seeds 9 to 16, adapter off against on at matched
      steps.
- [x] Compute the comparison table, one row per token and step, using the pattern term described
      above. `scripts/mechanism_study/reprobe_table.py`.
- [x] Record the verdict against the pre-registered bar in the review file.
- [x] Decide the figure's statistical entity with `/pair-figure`: one point per pair, median over
      its 8 seeds, seeds shown as a pale spread behind rather than averaged away. F7's caption is
      frozen to the narrower sentence the review file requires.

## Success/Failure Outcomes
- **the smoke cell**
  - Success: the maps render as recognisable head-and-body shapes, and the token position indexes
    the right word for that pair.
  - Failure: noise or empty maps, meaning the token position missed the words. Fix that before
    any sweep, or the sweep measures nothing.
- **the verdict**
  - Replicates: median pattern ratio at least 1.2, and at least 75% of rows above 1.
  - Does not: the paper's mechanism section shrinks to one honest negative paragraph, which is a
    result this plan provides for. Do not loosen the bar to avoid it.

## Next

1. `/pair-figure` on the mechanism comparison: choose per-seed points or pair-level means, and
   write the choice into the review file's open question.
2. `/design-figure` rides `figure-01-the-seven-paper-figures.md` for F7's final form. Do not
   design it here.

## Illustrations
*(image not yet generated; save under `../assets/` and replace this placeholder)*

**Prompt for image generation:**
> Generate an image of a flowchart showing this experiment: derive the token position per pair,
> smoke one cell, run the 64-cell sweep, compute the pattern comparison, record the verdict.
> Success path green with checkmark "Completed" pills. Failure path red on the smoke stage
> labeled "token position missed the words, maps are noise" with an X icon and a dashed "Retry
> Stage" callout. Downstream stages muted gray with "Skipped" pills. Glossy, minimalistic,
> modern UI/UX dashboard panel, dark background, rounded rectangle stage cards in a horizontal
> row connected by directional arrows, clean sans-serif labels, generous spacing, no clutter.

## Recommended skill
▶ `/run-experiment` ✅ for the sweep; `/pair-figure` ✅ before plotting.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.mechanism_study.value_probe \
  --checkpoint <lora_step_100000.pt> --pair-slug an_eagle__x__a_hawk --seed 9 \
  --steps 10,25,40                              # smoke: expect 3 step files
ls -d /datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe/*/seed_* | wc -l
                                                # expect 64 cells
$PY scripts/mechanism_study/reprobe_table.py    # table + verdict.json
```
