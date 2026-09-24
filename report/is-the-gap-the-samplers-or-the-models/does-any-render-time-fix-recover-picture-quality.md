# Can any render-time fix recover the picture quality the adapter costs?   ❓ inconclusive: the edge measure passes two of them, the pictures do not · verified 2026-09-24

**The claim**

The adapter restores plurality and costs picture quality, and eight fixes that need no retraining
were run on the same eight held-out cells from the same starting noise. Two of them, renoising the
finished picture and repainting it with the adapter switched off at strength 0.25 and 0.40, clear
the 1.10 edge-energy bar the noise-search finding used, at 1.14 and 1.24 times the adapter's own
render. Raising the guidance scale clears it by more, 1.45 at scale 10 and 1.80 at scale 12, while
visibly drawing a different picture: on three of the eight cells it adds or swaps an object that
neither prompt asked for. Applying the correction's implied expert weights explicitly in the
sampler is the one clear harm, at 0.78, with both subjects dissolving into a smeared wash on all
eight cells. No bar was written for this harness before it ran, so every number here is post-hoc
against a bar borrowed from another finding, and no composition count was scored.

## Table of contents

- [Where the idea came from](#where-the-idea-came-from)
- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. What each fix does to the picture](#1-what-each-fix-does-to-the-picture)
- [2. What the edge measure says, and where it disagrees with the eye](#2-what-the-edge-measure-says-and-where-it-disagrees-with-the-eye)
- [3. The reference is broken on one of the two seeds](#3-the-reference-is-broken-on-one-of-the-two-seeds)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## Where the idea came from

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-we-set-out-to-see) ➡️

**The measurement that argues for it.** The part of the correction the two experts could have
supplied by re-weighting their own predictions weights each expert at 1 to 3 where the product
weights them at 7.5, read in
[what the correction is made of](../when-does-the-outcome-lock-in/what-is-the-correction-made-of.md).
Damping of that size is what a lower guidance scale does, so the haze the adapter introduces has a
candidate cause that costs nothing to test: put the contrast back by raising the scale, or let the
undamped frozen weights draw the texture after the layout is settled.

**Image-to-image refinement.** Renoising a finished picture part-way and finishing it with a
different model or setting is standard practice, and it is not what this project tried before.
Switching the adapter off at step 20 keeps one trajectory; this restarts a short run from the
finished picture, so the detail is drawn by weights that were never damped.

## What we set out to see

Navigation: ⬅️ [Previous](#where-the-idea-came-from) | 📋 [TOC](#table-of-contents) | [Next](#what-would-have-counted) ➡️

Eight treatments and two references on four pairs at two held-out seeds each, every cell from the
same starting noise and the same checkpoint, so the only thing that differs between two tiles in a
row is the treatment. The references are the joint prompt and the plain product. The treatments are
the adapter as it ships, guidance raised to 10 and to 12, the experts explicitly weighted 3.0 and
3.0, a repaint pass at strength 0.25 and at 0.40 with the adapter off, the adapter for the first 20
steps only, and eight nearby starting noises.

**The hypothesis.** If the fidelity cost is the correction's in-span part acting as a per-step
guidance damping, then restoring that contrast should restore the texture without touching the
layout, because the layout is settled by the early steps the correction acts in.

## What would have counted

Navigation: ⬅️ [Previous](#what-we-set-out-to-see) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

Nothing was pinned before this harness ran. It was built to be read by eye, and the number came
afterwards. The bar quoted throughout is the one
[does searching over the noise sharpen the corrected render](does-searching-over-the-noise-sharpen-the-corrected-render.md)
was judged against, a median per-cell Laplacian variance 1.10 times the adapter's own render, and
applying it here is post-hoc. That is why the verdict is inconclusive rather than support: two
treatments clear a bar nobody agreed to in advance, on a measure already known to disagree with the
eye.

## What was tried, in order

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#1-what-each-fix-does-to-the-picture) ➡️

| When | What | Where it landed | Outcome |
|---|---|---|---|
| 2026-09-24 | ten jobs, one per column, Slurm 59549 to 59558 on `batch` | `/datasets/mmolefe/poe_repair_min/outputs/fidelity_harness/` | eight tiles each for nine columns, 11 to 19 minutes per job |
| 2026-09-24 | the best-of-8 column, Slurm 59558 | the same tree | stopped at 23 minutes with 2 of 8 cells, when all runs were stopped for a re-assessment |
| 2026-09-24 | sheets and the edge measure | [the evidence folder](../../artifacts/results/which-fidelity-fix-wins/) | eight sheets and `sharpness.json` |

## 1. What each fix does to the picture

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-what-the-edge-measure-says-and-where-it-disagrees-with-the-eye) ➡️

![Tiger and dog, seed 9: the joint prompt and the plain product, then the adapter and seven fixes applied to it, one column each](../../artifacts/results/which-fidelity-fix-wins/a_tiger__x__a_dog__seed09.png)
*The explicit re-weighting, column six, is the only column where both animals stop being animals.*
📊 Drawn in [Figure 1 of the figure explainer](../../artifacts/results/which-fidelity-fix-wins/figure-explainer.md#figure-1-tiger-and-dog-seed-9).

**What the eye reads across all eight cells.** The two repaint passes and the 20-step handover draw
the adapter's picture again, with the same layout and the same subjects. The explicit re-weighting
smears both subjects on every cell and invents a wolf statue on chess board and hourglass at seed 9,
which neither prompt names. Raising the guidance scale changes the scene rather than the surface:
cat and dog at seed 10 gains a toy giraffe at both 10 and 12, tiger and dog at seed 10 gains a third
animal at 10 and replaces the dog with a second tiger at 12, and chess board and hourglass at seed 9
rebuilds the hourglass on a different stand.

## 2. What the edge measure says, and where it disagrees with the eye

Navigation: ⬅️ [Previous](#1-what-each-fix-does-to-the-picture) | 📋 [TOC](#table-of-contents) | [Next](#3-the-reference-is-broken-on-one-of-the-two-seeds) ➡️

**The number.** Median Laplacian variance over the eight cells, greyscale on a 0-to-1 scale, in
units of 1e-4, with the ratio to the adapter's own render beside it. From
`artifacts/results/which-fidelity-fix-wins/sharpness.json`, field `treatments.<name>.median`,
written by `scripts/local/tile_sharpness.py` on 2026-09-24.

| Column | Median | Against the adapter |
|---|---|---|
| guidance 12 | 138.96 | 1.80 |
| guidance 10 | 111.96 | 1.45 |
| repaint at 0.40 | 95.33 | 1.24 |
| the joint prompt | 91.18 | 1.18 |
| repaint at 0.25 | 87.83 | 1.14 |
| adapter for 20 steps | 84.34 | 1.09 |
| the adapter as it ships | 77.08 | 1.00 |
| best of 8, two cells only | 65.11 | 0.85 |
| the plain product | 60.49 | 0.79 |
| experts weighted 3.0 and 3.0 | 59.74 | 0.78 |

**The disagreement.** The measure counts edges, so contrast raises it whether or not any detail was
drawn. Guidance 12 scores 1.80 while drawing a picture with an object that was never asked for.
Both repaint passes cross the joint prompt's own 1.18, which would mean the repaint is sharper than
the target it is trying to look like, and by eye it is not. The same disagreement is on record
twice: the noise-search finding's edge measure picked drawn texture where the eye picked the
photograph, and the sketch seeds dominate any minimum-to-maximum band on cat and dog.

**What survives the disagreement.** The two directions the measure and the eye agree on are the
extremes. Explicit re-weighting at a constant 3.0 and 3.0 is worse than the adapter by both, and it
is worse than the plain product. Nothing tested here reaches the joint prompt's picture.

## 3. The reference is broken on one of the two seeds

Navigation: ⬅️ [Previous](#2-what-the-edge-measure-says-and-where-it-disagrees-with-the-eye) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Cat and dog, seed 10: the leftmost column, the joint prompt, draws two dogs](../../artifacts/results/which-fidelity-fix-wins/a_cat__x__a_dog__seed10.png)
*The target the whole row is compared against has no cat in it.*
📊 Drawn in [Figure 4 of the figure explainer](../../artifacts/results/which-fidelity-fix-wins/figure-explainer.md#figure-4-cat-and-dog-seed-10).

On cat and dog at seed 10 the joint prompt itself renders two dogs, so on that cell "nearer the
joint prompt" is not a statement about composition at all. Four of the eight held-out seeds carry
this, recorded before this harness ran. Two of the eight cells here are seed 10 cells of animal
pairs, so a quarter of this evidence is measured against a target that does not show the pair.

## What this cannot tell you

Navigation: ⬅️ [Previous](#3-the-reference-is-broken-on-one-of-the-two-seeds) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**No composition count was scored.** The validated instance-count scorer was never run over these
87 tiles, so every statement about whether a treatment still draws two things is an eye read off
the sheets. The counts in other findings are not comparable to anything here.

**The bar is borrowed and post-hoc**, as stated above, and the measure is known to rank contrast as
sharpness.

**Best of 8 has two cells of eight**, so its 0.85 is not a result, only the two cells that landed.

**One checkpoint, one pool.** Everything is the v58-05-self-contrast checkpoint at step 15,000. A
different checkpoint could carry a different fidelity cost, and the fixes are applied to this one.

**Eight cells.** Four pairs at two seeds. Two of the four pairs are animal pairs, one is an object
pair, one is a mixed-size animal pair.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The numbers | `artifacts/results/which-fidelity-fix-wins/sharpness.json`, written and read 2026-09-24 | verified |
| The renders | `/datasets/mmolefe/poe_repair_min/outputs/fidelity_harness/`, 87 tiles, rendered 2026-09-24 on `batch` nodes | verified |
| The sheets | the same folder's `sheets/`, built by `scripts/fidelity_harness.py sheets` | verified |
| The checkpoint | `/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-05-self-contrast/checkpoints/lora_step_015000.pt` | verified |
| Regenerate with | `python scripts/fidelity_harness.py submit --checkpoint <pt>` on the cluster, then `sheets`, then `scripts/local/tile_sharpness.py` | |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- What the product of experts is and what the guidance scale does to it: [what product-of-experts composition is](../../context/world/poe-composition.md)
- What the correction is and which part re-weighting could supply: [what is the correction made of](../when-does-the-outcome-lock-in/what-is-the-correction-made-of.md)
- How a render fails, in the words used above: [how a render fails](../../context/world/how-a-render-fails.md)
- Each picture, read one at a time: [the figure explainer](../../artifacts/results/which-fidelity-fix-wins/figure-explainer.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] Score the 87 tiles with the validated instance count, so "still composes" is a number.
- [ ] Write a fidelity bar that a contrast change cannot pass, before the next harness runs.
- [ ] Re-run the re-weighting column with the falling schedule the measurement implies, about 3 early
      to about 1 late, rather than the constant 3.0 tested here.
- [ ] Finish the best-of-8 column's remaining six cells.
- [ ] Repair the joint-prompt references on the four broken held-out seeds, before any cell that
      uses them is judged again.
