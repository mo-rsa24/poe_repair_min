# 💉 Does more correction give more composition?

Design only. Verdicts and run state live in
[../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md).

## What this asks, in one line
Add the correction in increasing amounts and count how often the picture contains
two separate animals instead of one blended one. If the correction is the reason
composition works, more of it should give more of it.

## Description
Take the correction we already have cached, add a fraction of it to plain PoE
sampling, and score the picture that comes out. The fraction is λ, and it steps
through 0, 0.25, 0.5, 0.75, 1 with the starting noise held fixed so the only thing
changing is how much correction goes in.

Three rows run at every fraction. The real correction for this pair, a random
vector of the same size, and the correction belonging to a different pair. The two
fakes are the controls: they are what tells us the real one is working because of
where it points and not because something was added.

## Purpose
This is the paper's central causal claim (Goal 1): composition fails because this
correction is missing. A rising curve for the real correction beside two flat
curves for the fakes is what turns "the fix helps" into "the fix is the cause".
Without the flat controls, adding anything at all might have helped, and the claim
would not survive a reviewer. Serves DoD 3.

## Goal
The three-curve figure (compose-rate vs λ) over the held-out animal pairs with
the five-image strip, plus the scored outputs on disk for plans 05 and 06.

## Environment Facts This Plan Depends On
- Cached residuals and pinned init latents per cell (training_cache).
- In-session smoke on the 3090 (SDXL inference fits; check nvidia-smi for
  co-tenants first); full sweep as a job, biggpu first, else bigbatch.
- Disk guard: outputs to /datasets, abort at 90% full. The guard must check the
  filesystem being written to, not a different one.
- W&B: log the Mono vs PoE vs corrected triptych per cell.

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
  injected.** Two numbers are recorded per step: `delta_norm`, how big the real
  correction is, and the PMI identity, a relationship it satisfies. Both describe
  the pair of concepts, not whatever we chose to inject, so both keep reading the
  real correction throughout.
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
      the real correction already existed from plan 00; choosing a different vector
      did not, because the sampler always computed its own.
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
- [ ] Build the three-curve figure and the five-picture strip. Figure candidates
      come from plan 10; the strip is rebuilt by step 6 of the procedure above.
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
> correction, verify the λ=0 canary, smoke one seed, submit the full sweep,
> score images, plot three curves. Success path green with checkmark
> "Completed" pills. Failure path red on the canary stage labeled "harness
> contaminates base path" with an X icon and a dashed "Retry Stage"
> troubleshoot callout. Downstream stages muted gray with "Skipped" pills.
> Glossy, minimalistic, modern UI/UX dashboard panel, dark background, rounded
> rectangle stage cards in a horizontal row connected by directional arrows,
> clean sans-serif labels, generous spacing, no clutter.

## Recommended skill
▶ `/run-experiment` ✅ for the sweep job; `/demonstrate` ✅ for the smoke strip.

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
