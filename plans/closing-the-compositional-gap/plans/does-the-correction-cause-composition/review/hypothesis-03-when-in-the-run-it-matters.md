# Review: when in the run is the correction needed?

**Nothing has run yet.** This file holds the questions, written before the runs so the answers
cannot be chosen after the fact. It judges
[../plans/hypothesis-03-when-in-the-run-it-matters.md](../plans/hypothesis-03-when-in-the-run-it-matters.md),
and its answers fill register slot **F4**, the figure showing when in the denoising run the
correction does its work.

## Words this file uses
- **The correction**: the step-by-step gap between what the joined prompt predicts and what
  adding the two separate prompts predicts. It is what the broken method leaves out.
- **A window**: a stretch of the 50 denoising steps. Inside it the correction is allowed to act;
  outside it, nothing is injected. Sliding the window from start to finish and scoring each
  position is the whole experiment.
- **Two experiments, not one.** One slides a window over the *correction* while the prompt stays
  on throughout. The other slides a window over the *prompt* itself. They answer different
  questions and their peaks may or may not land in the same place.
- **The fork step**: step 16, where the broken path and the working path start pulling apart,
  measured in `hypothesis-04-what-the-cached-runs-already-show`. An independent estimate of the
  same moment, from cached data rather than new runs.

## Run kind
**Tests the claim.** A failure of the bar below closes the plan and opens one follow-on.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| (none yet) | Tests the claim | | `/datasets/.../interaction_term/window/{w1,w2}/` | not started |

## The pre-registered bar

- [ ] ⚠️ Does the compose rate peak at some window position rather than staying flat?
      A peak means there is a moment the correction is needed. Flat means timing does not matter,
      which is a finding and not a failure: the paper would then say the correction is needed
      throughout.

## Written before the run, answered after

- [ ] ⚠️ With the window switched off everywhere, does the output match plain PoE exactly?
      This is the leak check. If switching everything off does not reproduce plain PoE, our own
      harness is altering the baseline, and no curve from it can be read.
- [ ] ⚠️ Do the two experiments peak at the same position?
      The same peak says one mechanism drives both. Different peaks say the prompt and the
      correction are needed at different moments, which is a more interesting paper. Either
      answer goes in the caption.
- [ ] ⚠️ Does the peak land near step 16, the fork step measured from cached data?
      Two independent estimates of one moment. If they agree the claim is stronger. If they
      disagree, diagnose it before either number is printed.
