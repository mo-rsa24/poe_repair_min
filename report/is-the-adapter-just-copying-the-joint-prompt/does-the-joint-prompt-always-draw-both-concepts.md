# Does the joint prompt always draw both concepts?   ❌ no: it drops one on eleven of the thirty-two held-out cells rendered · verified 2026-09-24

**The claim**

The joint prompt is the target the adapter is trained to match, and on some cells it is wrong.
Asked for a chess board and an hourglass it draws a board and no hourglass, on all five seeds
rendered. Asked for a cat and a fox it draws one or two foxes and no cat, on four seeds of six.
Asked for a dog and a dog it puts a person in the frame beside one dog. The plain product it is
meant to beat sometimes contains both concepts where the joint prompt contains one.

**What would have counted**

No bar was pre-registered. The read is descriptive and by eye: a cell counts as a failure when a
concept the prompt names is absent from the render, or when an object nobody asked for is the
subject. The cells are the thirty-two joint-prompt renders on disk for held-out pairs, and the
count below is over those.

## 1. Six cells where the target drops a concept

![The joint prompt, the plain product and the adapter on six cells where the joint prompt is wrong](../../artifacts/results/what-the-main-study-grids-show/where-the-joint-prompt-itself-fails.png)
*Rows are cells, columns are the joint prompt, the plain product and the adapter, each row from
one starting noise. What to notice: the first column is missing the hourglass on both chess rows
and the cat on both fox rows, and on the dog row it draws a person. In the chess row at seed 10
the middle column, the composition this work repairs, holds an hourglass and chess pieces
together, so the product carries what the target dropped.*

**The number.** Eleven failing cells of thirty-two rendered joint prompts: chess board and
hourglass at seeds 9, 10, 11, 12 and 42, five of five, no hourglass in any; cat and fox at seeds
2, 9, 11 and 12, four of six, a fox or two foxes and no cat; dog and dog at seed 1, a person
beside one dog, and at seed 2, one large dog with a malformed small one; elephant and penguin at
seed 9, the elephant alone. Counted by eye over the renders under
`/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/<pair>/seed<NN>/mono.png`, mirrored into
`artifacts/results/which-checkpoint-to-take-each-figure-cell-from/tiles/`.

## 2. What the adapter does on those cells

🖼️ **Figure wanted:** the same six rows with the adapter column filled. Five of the six rows in
the figure above carry no adapter render, because only the dog and dog cell at seed 1 has been
rendered through a v58 checkpoint. On that one row the adapter draws two animals where the joint
prompt drew a person and one dog, though the second animal is dark and horse-like rather than
clearly a dog. Rendering the other five cells is three jobs of the shape already used in
`scripts/figure_candidates.py`, and until they exist this file does not claim the adapter beats
its target.

## What this cannot tell you

- **Whether these cells are representative.** Thirty-two renders over eight pairs and six seeds,
  chosen because they were already on disk for other figures, not sampled.
- **Whether a different phrasing of the joint prompt would fix them.** Every render uses the
  project's standing form, "a chess board and an hourglass", and no alternative was tried.
- **Whether the adapter beats the target here**, which is section 2's empty figure.
- **Why the target fails on these pairs.** The failures cluster on objects and on repeated
  concepts, and nothing here separates a prompt-parsing failure from a rendering one.

## Still open

- **The adapter column for the five unrendered cells**, above.
- **Whether the training pool contains cells whose target is wrong in this way.** The pool's
  cells were curated by eye for v58, but the count was never taken; if a training cell's target
  drops a concept, the adapter is being trained to drop it too.

## Where this came from

| What | How it was established | When |
|---|---|---|
| The eleven failing cells | read by eye from the 32 joint-prompt renders on the cluster | 2026-09-24 |
| The figure | `scripts/main_study_grids.py` | 2026-09-24 |
| The renders | `scripts/figure_candidates.py`, column `mono`, 50 DDIM steps at guidance 7.5 | 2026-09-23 and 2026-09-24 |
