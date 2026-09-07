# Does SuperDiff compose at its own defaults, and what is it missing when it does not?   ⚪ null at 200 steps, ✅ support that the missing piece is the same residual · verified 2026-09-05

**The claim**

SuperDiff (Skreta et al., arXiv 2412.17762, the `superdiff/superdiff-sdxl-v1-0` pipeline
reimplemented in `poe_repair/composers/superdiff.py`) does not compose cat × dog at its own
defaults of 200 steps and guidance 7.5: every seed comes out as one blended animal. The same
pipeline composes both test pairs at 50 steps. Its blending weight `kappa`, clamped or not,
changes nothing. Adding back a fraction λ of SuperDiff's own missing residual,
`r_t^SD = ε_J − ε_M`, on every step turns one animal into two between λ 0.25 and 0.75 on every
seed, the same one-to-two switch this project measures for product-of-experts sampling. So
SuperDiff is missing a residual of the same shape PoE is missing, and the composition rule is
not what decides whether the second concept survives.

**What would have counted**

The plan pre-registered two questions before the eight-cell grid ran: does SuperDiff still
compose at 50 steps, and does clamping `kappa` change the answer, read by the validated
detector on cat × dog and by eye on both pairs. The κ × λ sheets were designed after the grid
came back reversed, so nothing in rung 3 was pre-registered. Both are pinned in
[the plan](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md)
and judged in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/05-what-changes-when-superdiff-leaves-its-own-defaults.md).

## 1. It composes at 50 steps and collapses at 200

![Eight cells: two pairs by two step counts by two kappa-clamp settings, seed 9](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-baseline-eight-cells.png)
*Left half is the pipeline's own 200-step default; right half is 50 steps. What to notice: the
cat × dog row is one cat at 200 steps and a dog-shaped silhouette filled with cat figures at 50;
the butterfly is a faint blur at 200 and a distinct shape at 50. Clamped and unclamped columns
are the same picture.*

**The number.** Instance count from the validated detector on cat × dog, seed 9: 1 (blend) at
200 steps for both clamp settings, 2 (compose) at 50 steps clamped, 1 at 50 steps unclamped
where the eye reads compose and is cited per this project's practice. Butterfly × meadow is
outside the detector's validated contract and is an eye read only: blur at 200, distinct
butterfly at 50, both clamp settings. From the review file's run table for the eight-cell grid,
renders under `corrector/superdiff/parity/grid/`; the detector counts are lifted from the
review, not re-read from the scorer output this session.

## 2. The clamp never fires

**The number.** The `kappa` clamp bound of [−0.5, 1.5] is verified on synthetic input (2.3 → 1.5,
−1.8 → −0.5) and never triggers on a real render of either pair: the pipeline's own per-step
`kappa` sits between roughly 0.4 and 0.7 after the first steps, from the `kappa_raw` field of the
render sidecars. No exemplar is owed for this rung; the eight-cell figure above already shows
clamped beside unclamped.

## 3. Adding its own residual back restores two animals

![SuperDiff alone on cat × dog at κ 0.5, rows seeds 9 to 12, columns λ from 0 to 1](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-alone-cat-dog-kappa-050-lambda-sheet.png)
*Each row is one seed; each column adds back λ of `r_t^SD` on every one of the 200 steps. What to
notice: the λ 0 and 0.25 columns are one animal on every seed, and by λ 0.75 every seed is a cat
beside a dog. The switch sits between 0.25 and 0.75, the same place PoE's switch sits on this
project's strength grid.*

![SuperDiff alone on butterfly × meadow at κ 0.5, same layout](../../artifacts/results/is-the-gap-the-samplers-or-the-models/superdiff-alone-butterfly-meadow-kappa-050-lambda-sheet.png)
*What to notice: the butterfly is present but dissolved into the meadow at λ 0, distinct on every
seed by λ 0.25, and large and sharp from λ 0.5. This pair shrinks its second concept rather than
losing it.*

**The number.** First separating λ per seed, eye read, cat × dog at κ 0.5: seeds 11 and 12 at
0.5, seeds 9 and 10 at 0.75, so 4 of 4 seeds by 0.75 and 0 of 4 at 0.25. Butterfly × meadow at
κ 0.5: 4 of 4 by 0.25. At the pipeline's own unclamped κ the cat × dog switch is at 0.5 on all
four seeds. Across the six κ settings rendered for cat × dog, κ chooses which single animal
appears at λ 0 (a plain dog at κ 0, a plain cat at κ 1) and the residual adds the second
subject before it fixes that subject's identity. From the seven sheets and their `.json`
sidecars in `paper/iclr/figures/how-much-is-added/across-composition-rules/`, 140 renders under
`corrector/superdiff/lambda_sweep/`.

## What this cannot tell you

Why more steps make this pipeline less compositional: the plan's review names one candidate
(the SDE integrator's per-step noise accumulating over 200 steps) and nothing here tests it.
The 50-versus-200 comparison changes the step size and the number of noise injections together,
so it is mixed. The κ × λ reads are eye reads on four seeds, not detector scores, and the
butterfly pair is outside the detector's validated contract at every rung. Nothing here says
whether the SuperDiff residual and the PoE residual point the same way at the same state; the
next finding measures that.

## Where this came from

| What | Source | Mark |
|---|---|---|
| The eight-cell detector counts and eye reads | review file run table, 2026-09-03, renders in `corrector/superdiff/parity/grid/`; lifted this session, scorer output not re-opened | inferred |
| The clamp bound and the per-step `kappa` range | review file tasks 1.3 and 4, sidecar field `kappa_raw` | inferred |
| The first-separating λ per seed | the seven sheets under `paper/iclr/figures/how-much-is-added/across-composition-rules/`, eye read 2026-09-03 and recorded in the review file | stated |
| The renders | eight cells and a 50-step κ sweep on mscluster85 in session; the 140-render κ × λ set by `nohup` on mscluster85, about 5 h at 107 to 136 s per cell, 2026-09-03 | verified |
| Regenerate with | [runbook § render SuperDiff's κ × λ sheets](../../runbook/running-things-on-the-cluster/training-and-watching-a-superdiff-adapter.md#6-render-superdiffs-own-kappa-by-lambda-sheets) | |
| The plan | [what changes when SuperDiff leaves its own defaults](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/05-what-changes-when-superdiff-leaves-its-own-defaults.md), root step 28 | |

## Depends on

- What `r_t`, the interaction term, and a compose rate mean: [context index](../../context/00-INDEX.md)
- The pipeline is a hand-written Euler–Maruyama SDE integrator with no scheduler object, so "DDIM at 50 steps" was never an available setting: the scope [changelog entry of 2026-09-02](../../plans/06-is-the-gap-the-samplers-or-the-models/CHANGELOG.md)
- Which weights: the SuperDiff checkpoint's UNet is byte-identical to SDXL base, cached under `corrector/superdiff/hf_cache` per [the environment index](../../environment/00-INDEX.md)

## Still open

- [ ] The eight-cell detector counts are lifted from the review file; opening the scorer output under `corrector/superdiff/parity/grid/` would make them `verified`.
- [ ] Why 200 steps collapses where 50 composes has no run behind it.
