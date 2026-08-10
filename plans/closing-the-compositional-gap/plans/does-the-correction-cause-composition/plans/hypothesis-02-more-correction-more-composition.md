# 💉 Does more correction give more composition?

Design only. Verdicts and run state live in
[../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md).

## What this asks, in one line
Add the correction in increasing amounts and count how often the picture contains
two separate animals instead of one blended one. If the correction is the reason
composition works, more of it should give more of it.

## Words this plan uses
- **PoE**, the broken way of composing: ask the model about "a cat" and about "a dog"
  separately, then add its two answers. It usually fuses them into one animal.
- **Mono**, the cheat that works: hand the model the joined prompt "a cat and a dog".
  It composes fine, which defeats the point, so it is only ever the target.
- **The correction**, written `r_t`: the step-by-step gap between what Mono predicts
  and what PoE predicts. It is what PoE leaves out.
- **λ (lambda)**: how much of the correction to add back, from 0 (none) to 1 (all).
- **A cell**: one picture, for one animal pair, at one λ, from one starting noise (one
  seed). Every count below counts cells.
- **Compose rate**: the fraction of cells showing two separate animals rather than one
  blended one, decided by the validated scorer and never by eye.

## Description
Take the correction we already have cached, add a fraction of it back into PoE
sampling, and score the picture that comes out. The fraction is λ, and it steps through
0, 0.25, 0.5, 0.75, 1. The starting noise is held fixed, so the only thing changing
across a row is how much correction went in.

Three rows run at every fraction. The real correction for this pair, a random
vector of the same size, and the correction belonging to a different pair. The two
fakes are the controls: they are what tells us the real one is working because of
where it points and not because something was added.

## Purpose
This is the paper's central causal claim: composition fails because this correction is
missing. A rising curve for the real correction, beside two flat curves for the fakes,
is what turns "the fix helps" into "the fix is the cause". Without the flat controls,
adding anything at all might have helped, and the claim would not survive a reviewer.

It is this scope's Goal 1 and the third item of its Definition of Done, both listed in
the scope's `MASTER_PLAN.md`.

## Goal
Register slot **F2**, the paper's headline figure, in two halves that have to be read
together:

- **The number:** three curves on one axis, compose rate against λ, over animal pairs the
  fix never trained on.
- **The picture:** a 3 by 5 grid of real generated cells above them, one row per injected
  vector and one column per strength, same pair and same starting noise throughout.
  `scripts/dose_strip.py` already produces exactly this shape.

The grid is the half that does the persuading. Reading **down** a column at full strength
shows the real correction giving two animals while both fakes still give one blended
animal, so the controls stop being a sentence in the caption and become something the
reader sees. Reading **across** the top row shows the blend separating as the strength
rises.

The scored cells stay on disk, because `hypothesis-04-what-the-cached-runs-already-show`
and `hypothesis-05-the-same-story-from-three-sides` both read them.

## Environment Facts This Plan Depends On
- The corrections are already computed and cached, along with the exact starting noise
  for every cell, under `training_cache`. Nothing here recomputes them.
- One cell runs in-session on this node's 3090. The full set goes to a job, biggpu
  first, else bigbatch. Check `nvidia-smi` for other people's jobs first: the card is
  shared, and a full card means the run dies partway through.
- Large output goes to `/datasets`, and the job aborts if that filesystem is over 90%
  full. The check must look at the filesystem actually being written to, which is the
  bug in the last task below.
- Log the three-panel comparison per cell to W&B (Mono, plain PoE, corrected), so every
  number has a picture beside it.

## Why the two fake rows are a fair comparison
A control only works if it differs from the real thing in exactly one way. If a
fake differs in two ways, and the real correction works while the fake does not,
you cannot say which difference caused it, and the comparison stops being evidence.
So four things are deliberately held equal. These are decisions about the
experiment, not findings from it, which is why they live here and not in the review
file.

- **The fakes are the same size as the real correction, at every step.** Scaled to
  match its length. Otherwise a fake that failed could have failed for being too
  weak rather than for pointing the wrong way, and only direction is supposed to
  differ.
- **The measurements keep describing the real correction, even when a fake is
  injected.** Two numbers are recorded per step. `delta_norm` is how big the real
  correction is. The PMI check is an algebraic test that the correction really equals
  what the joined prompt adds over the two separate ones (PMI is pointwise mutual
  information; `poe_repair/experiments/residual_diagnostics/metrics.py` computes the
  curve). Both describe the pair of concepts, not whatever we chose to inject, so both
  keep measuring the real correction right through a fake run.
- **The full-strength shortcut is switched off during a fake run.** At λ=1 the
  code could skip the arithmetic and use the joined-prompt prediction directly,
  which gives the same answer when the correction is real. During a fake run that
  shortcut would ignore the fake and return the real answer, so every control row
  would have quietly reproduced the real one and all three curves would have
  looked equally good. This is the failure that would have been hardest to notice.
- **The zero-strength row is generated once and shared.** At λ=0 nothing is
  injected, so all three rows are the same picture by construction. Generating it
  three times would only add noise to the one point where the rows must agree.

## Tasks
A plain checkbox here, because a design task either happened or it did not. Whether
the experiment worked is a separate question and it is answered in the review file.

- [x] Write the code that injects a chosen vector, and the two fake rows.
      `scripts/interaction_term_inject.py` over `run_teacher_residual`. Injecting
      the real correction already existed from
      `instrument-01-build-the-measuring-scripts`. Choosing a DIFFERENT vector did not,
      because the sampler always computed its own.
- [x] Prove the harness does not disturb plain PoE when nothing is injected.
      At λ=0 the output must match what the sampler itself saved for plain PoE.
      8 tests, each one shown to fail against a deliberately broken sampler, so
      they are checks and not decoration.
- [x] Run one cell by hand before spending hours: all three rows at full strength
      on a_cat__x__a_dog, seed 9, 20 steps, scored and looked at.
- [x] Run the full set: 8 unseen pairs × 4 seeds × 5 strengths × 3 rows = 480
      pictures, `scripts/mechanism_study/run_dose_sweep.sh`, resumable, about 50
      seconds each.
- [x] Score every picture with the validated composition scorer.
- [ ] Read [../procedures/hypothesis-02-recheck-the-headline-numbers.md](../procedures/hypothesis-02-recheck-the-headline-numbers.md)
      to completion, do what it says, and answer the two open questions in the
      review file. It stops the scorer picking up pictures from older runs, and it
      sets its cutoffs by looking at a picture rather than by choosing a number.
- [ ] Build the three curves and the 3 by 5 grid of cells. The layout is decided in
      `figure-01-the-seven-paper-figures`, not here. The grid is rebuilt by step 6 of the
      procedure above.
- [ ] Move the output off /home-mscluster and repoint `run_dose_sweep.sh`. Its
      `OUT=$REPO/outputs/...` put 3.4GB in the home repo, and its disk check looked
      at /datasets, a filesystem it was not writing to. Make the check follow the
      output.

## Success/Failure Outcomes
- **Does the harness leave plain PoE alone when nothing is injected?**
  - Success: at λ=0, the largest difference against what the sampler saved for
    plain PoE is below 1e-5.
  - Failure: any real difference means our own code is changing the baseline we
    are comparing against, so every later number would be measured from the wrong
    starting point. Stop and fix it before running anything large.
- **Does more correction give more composition, while the fakes stay flat?**
  - Success: 480 scored pictures. The real correction's curve rises as λ rises,
    and both fake rows stay near the bottom at every strength.
  - Failure: either the GPU runs out of memory (move to a bigger node), or the
    scorer and your own eyes disagree about the same picture. The second is the
    unclear case, not a negative one: fix the scorer, score again, and never move
    the threshold to rescue the curve.

## Illustrations
*(image not yet generated)*

**Prompt for image generation:**
> Generate an image of a flowchart showing this experiment: load cached
> correction, check that adding none of it changes nothing, run one cell by hand,
> submit the full set, score the pictures, plot three curves. Success path green with
> checkmark "Completed" pills. Failure path red on the check stage labeled "our own
> code moved the baseline" with an X icon and a dashed "Retry Stage"
> troubleshoot callout. Downstream stages muted gray with "Skipped" pills.
> Glossy, minimalistic, modern UI/UX dashboard panel, dark background, rounded
> rectangle stage cards in a horizontal row connected by directional arrows,
> clean sans-serif labels, generous spacing, no clutter.

## Recommended skill
▶ `/run-experiment` ✅ for the sweep job; `/demonstrate` ✅ for the one-cell strip.

## Recommended Prompts

Not tasks, and not completion criteria: things to run when a term here stops meaning
anything. Each one leaves an artifact you can come back to.

- **When `r_t` is just a symbol.** Derive it once, then make it draggable:
  `/drip --math the correction r_t = the joint-prompt prediction minus the
  Product-of-Experts prediction: derive why PoE drops it, one step per message` →
  `/polish` to file the derivation with its equations properly set →
  `/math-scene` on that file, so you can drag λ and watch the corrected prediction move
  between the PoE point and the Mono target.
- **When the PMI check is a black box.** `/drip --math pointwise mutual information as
  the interaction term a Product-of-Experts composition drops, and why the residual
  should satisfy it algebraically; the implementation is
  poe_repair/experiments/residual_diagnostics/metrics.py`.
- **When you want to see WHERE in the network this happens.** `/deep-learning-scene on
  poe_repair, focused on the cross-attention path the adapter touches` builds a local
  app from the whole model down to the source lines, so "the layer where the prompt
  enters" stops being a phrase.
- **When the experiment's shape is the confusing part**, not the maths:
  `/experiment-atlas this plan: 8 unseen pairs x 4 seeds x 5 strengths x 3 rows, and
  which cells feed register slot F2`.
- **To design F2 itself**, once the confidence cutoff is chosen: `/pair-figure the
  quantitative half is three compose-rate curves against lambda (real correction, random
  vector, other pair's correction) from dose_curves.json; the qualitative half is the 3x5
  grid of real cells from dose_strip.py, one row per injected vector, one column per
  strength, same pair and seed throughout. Decide which pair and seed the grid should
  use, whether the grid sits above or beside the curves, and what the caption may claim
  given that the percentages are provisional until the cutoff is set.`

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.interaction_term.inject \
  --pair a_cat__x__a_dog --seed 9 --lambda 0 --check-canary
# expect: "canary ok, delta < 1e-5"
find outputs/interaction_term/dose/pairs -name "*.png" | wc -l
# expect: 440 images (480 cells, the 40 duplicates being the shared λ=0 row)
$PY scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
# prints per-row compose rate and AUC, writes dose_curves.{json,png}
```
