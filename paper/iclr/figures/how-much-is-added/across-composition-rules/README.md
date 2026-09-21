# Across composition rules: SuperDiff on the amount axis

Supplementary sheets, not main text. Each one holds SuperDiff (Skreta et al., arXiv 2412.17762,
via the `superdiff/superdiff-sdxl-v1-0` pipeline reimplemented in
`poe_repair/composers/superdiff.py`) at 200 steps, guidance 7.5, with its blending weight
`kappa` fixed per sheet. Rows are seeds 9 to 12. Columns add back a fraction `lambda` of
`r_t^SD = eps_J - eps_M`, so `lambda = 0` is SuperDiff alone and `lambda = 1` is the joint
prompt's prediction exactly. Every tile is the final 1024×1024 render shown at 256px; each
sheet's `.json` sidecar names every tile's source file.

**cat_dog_grid_200_steps_kappa_balanced.png**. Cat and dog with the pipeline computing its own
`kappa` every step, unclamped, its published behaviour. At `lambda = 0` every seed is one
blended animal. Still one at 0.25. At 0.5 all four seeds show a cat and a dog side by side, and
0.75 and 1 are near-identical to each other. The one-to-two switch sits between 0.25 and 0.5 on
every seed, the same shape as the PoE strength grid one folder up.

**cat_dog_grid_200_steps_kappa_050.png**. Same, with `kappa` forced to 0.5. Almost the same
picture: seeds 11 and 12 separate at 0.5, seeds 9 and 10 one column later at 0.75 (seed 9's 0.5
tile is two heads on one body). The pipeline's own `kappa` runs near 0.5 for most of the render
anyway, so forcing it there changes little.

**butterfly_meadow_grid_200_steps_kappa_050.png**. The easy pair, `kappa` forced to 0.5. At
`lambda = 0` the butterfly is present on every seed but dissolved into the meadow, small and
camouflaged. At 0.25 it is a distinct butterfly on all four seeds. From 0.5 it is large and
sharp in the foreground, and 0.5, 0.75 and 1 are near-identical. This pair does not lose its
second concept, it shrinks it, and a quarter of the residual pulls it forward.

**cat_dog_grid_200_steps_kappa_000.png**. `kappa` forced to 0, which makes SuperDiff's blend
pure "a dog" guidance. At `lambda = 0` every seed is a plain dog with no cat anywhere. At 0.25
three seeds already show a second animal, but it is another dog; at 0.5 all four have two
animals and the second has turned into a cat; at 0.75 and 1 it is plainly cat beside dog. The
residual adds a second subject first and fixes its identity second, and this `kappa` reaches two
subjects one column earlier than the balanced sheet.

**cat_dog_grid_200_steps_kappa_050_lora_r8.png**. The rank-8 PoE-trained adapter (checkpoint
`lora_step_410000.pt`, the latest at launch rather than the final 450k) injected into SuperDiff
on every one of its 200 steps at `kappa` 0.5: `eps_M(off) + lambda * (eps_M(on) - eps_M(off))`.
Read beside `cat_dog_grid_200_steps_kappa_050.png`, the same grid with no adapter. Every seed is
degraded at `lambda = 0.25` and washed out from 0.5; none reaches two separate animals at any
`lambda`, where the no-adapter sheet reaches it on all four seeds by 0.75. The correction's
per-step size is at or below SuperDiff's own missing residual throughout, and its per-step cosine
against that residual peaks at a median of +0.34 mid-run and sits near zero at both ends, so this
is a mostly-orthogonal correction accumulating over 200 steps, not an oversized one.

**cat_dog_grid_200_steps_kappa_025.png**. `kappa` forced to 0.25, between the pure-dog blend
and the balanced one. `lambda = 0` is a plain dog on every seed. At 0.25 seed 10 already has a
clear cat beside its dog and seed 12 a faint second animal; at 0.5 seeds 10, 11 and 12 have two
animals; seed 9 needs 0.75. From 0.75 all four are cat beside dog.

**butterfly_meadow_grid_200_steps_kappa_050_lora_r8.png**. The same rank-8 adapter inside
SuperDiff on the easy pair, read beside `butterfly_meadow_grid_200_steps_kappa_050.png`. Same
shape as cat×dog: degraded at `lambda = 0.25`, washed out from 0.5, and no seed's butterfly ever
sharper or larger than the no-adapter sheet already had it. The PoE-trained correction hurts
here too.

**cat_dog_grid_200_steps_kappa_075.png**. `kappa` forced to 0.75, the mirror of the 0.25 sheet:
at `lambda = 0` every seed is a plain cat (high `kappa` leans the blend toward prompt 1, low
toward prompt 2), still one cat at 0.25, two animals on seeds 9, 11 and 12 at 0.5, all four by
0.75. Across the `kappa` sheets the residual is what separates the two animals; `kappa` decides
which single animal appears when there is only one.

**cat_dog_grid_200_steps_kappa_050_lora_r16.png**. The rank-16 PoE-trained adapter (checkpoint
`lora_step_085018.pt`, latest at launch, final is 100k) inside SuperDiff, same recipe as the
rank-8 sheet. Same shape, a shade worse: a noisy chimera at `lambda = 0.25`, heavy wash-out at
0.5, near-uniform mush at 1 on every seed. Doubling the rank did not change the correction's
direction.

**cat_dog_grid_200_steps_kappa_100.png**. `kappa` forced to 1, the pure "a cat" blend and the
mirror of the 0 sheet. Plain cat at `lambda = 0` and 0.25 on every seed. At 0.5 seeds 9, 11 and
12 show two animals, seed 11's second one still cat-like (a dog by 0.75); seed 10 still one cat.
From 0.75 all four are cat beside dog. Read across all six `kappa` sheets: SuperDiff alone never
composes cat×dog on these seeds at 200 steps, its own residual separates the animals on every
seed by `lambda = 0.75` at every `kappa`, and `kappa` only decides which single animal appears
when there is one.

**Where they came from.** Rendered 2026-09-03 by `poe_repair/composers/superdiff.py` with
`kappa_clamp=False`, `kappa_override=<sheet value or None>`, `lam=<column>`, one process on
mscluster85; raw cells under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/lambda_sweep/`.
Assembled from those cells with labels drawn on the sheet. The verdicts above are eye reads;
the validated instance-count detector only covers animal pairs, so the butterfly sheet has no
detector read. Design and verdict live in
`plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md`
and its review file.
