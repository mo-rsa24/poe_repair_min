# Review: when in the run is the correction needed?

**The leak check passed and the grid is running.** The questions below were written before the
runs, so the answers cannot be chosen after the fact. This file judges
[../plans/hypothesis-03-when-in-the-run-it-matters.md](../plans/hypothesis-03-when-in-the-run-it-matters.md),
and its answers fill register slot **F4**, the figure showing when in the denoising run the
correction does its work.

## Words this file uses
- **The correction**: the step-by-step gap between what the joined prompt predicts and what
  adding the two separate prompts predicts. It is what the broken method leaves out.
- **A window**: a stretch of the 50 denoising steps. Inside it the correction is allowed to act;
  outside it, nothing is injected. Sliding the window from start to finish and scoring each
  position is the whole experiment.
- **The window slides over the correction, not over the prompt.** The prompt stays on at every
  step of every run here. Sliding a window over the prompt itself is a different experiment,
  answering when conditioning is needed at all; it is named in the plan and is not run.
- **The fork step**: step 16, where the broken path and the working path start pulling apart,
  measured in `hypothesis-04-what-the-cached-runs-already-show`. An independent estimate of the
  same moment, from cached data rather than new runs.

## Run kind
**Tests the claim.** A failure of the bar below closes the plan and opens one follow-on.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| Leak check, `--window off --check-identity`, a_cat×a_dog seed 9, 50 steps | Checks the harness | 2026-08-10, in-session mscluster GPU 1 | stdout only | done, passed |
| Smoke, 3 windows (0-10, 15-25, 40-50), a_cat×a_dog seed 9 | Checks the harness | 2026-08-10, in-session GPU 1 | `window/pairs/a_cat__x__a_dog/seed_9/` | done, passed |
| Timing grid, 9 windows × 8 pairs × 4 seeds = 288 cells | Tests the claim | 2026-08-10 19:00, `run_window_sweep.sh` under nohup, GPU 1 | `/datasets/.../interaction_term/window/pairs/` | running |

## The pre-registered bar

- [ ] ⚠️ Does the compose rate peak at some window position rather than staying flat?
      A peak means there is a moment the correction is needed. Flat means timing does not matter,
      which is a finding and not a failure: the paper would then say the correction is needed
      throughout.

## Written before the run, answered after

- [x] ✅ With the window switched off everywhere, does the output match plain PoE exactly?
      Yes, byte-identical on a_cat×a_dog seed 9 at 50 steps. The comparison is against a full-dose
      window placed past the last step, not against `run_cfg_poe`: both runs then batch four UNet
      branches, so only the window logic can differ. Comparing against the three-branch PoE
      sampler would fail for batch-shape reasons alone and would say nothing about leakage.
- [ ] ⚠️ Does the peak sit where the correction is largest?
      It should not, and if it does the reading is confounded. ‖r_t‖ is nearly flat across the
      run (1.8x from smallest step to largest, each fifth carrying 15-22% of the total), so size
      cannot be what picks out a moment. A peak in the compose rate against a flat size curve is
      the interesting result: the correction matters when it lands, not where it is big.
- [ ] ⚠️ Does the peak land near step 16, the fork step measured from cached data?
      Two independent estimates of one moment. If they agree the claim is stronger. If they
      disagree, diagnose it before either number is printed.
