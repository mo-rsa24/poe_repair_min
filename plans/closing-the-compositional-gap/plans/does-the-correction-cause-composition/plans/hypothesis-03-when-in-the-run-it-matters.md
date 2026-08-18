# ⏱️ When in the denoising run is the correction needed?

**Step 6 of 22.** Waits on steps 4 and 5. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 5 | ~~[hypothesis-04-what-the-cached-runs-already-show](hypothesis-04-what-the-cached-runs-already-show.md)~~ | ✅ |
| **6** | **this plan** | **◑ timing tab owed** |
| 7 | ~~[hypothesis-05-the-same-story-from-three-sides](hypothesis-05-the-same-story-from-three-sides.md)~~ | ✅ |

Design only. Verdicts and run state live in
[../review/hypothesis-03-when-in-the-run-it-matters.md](../review/hypothesis-03-when-in-the-run-it-matters.md).

## What this asks, in one line
Let the correction act only inside a narrow window of the 50 denoising steps, slide that
window from start to finish, and measure the compose rate at each position. A peak says
when the correction is needed; a flat curve says it is needed throughout.

## Description
One sliding-window experiment over the same eight held-out pairs and four seeds
the dose sweep used. The base is full guided PoE at every step, with the prompt
on throughout; a width-10 window slid across the 50 steps at stride 5 gates only
the injected r_t. Because conditioning never switches off, the only thing that
changes across the nine positions is when the correction acts.

## Purpose
It measures causally when the interaction term is needed, with no window assumed
in advance. The peak is then compared against the fork step, which is the same
moment estimated a different way, from cached trajectories rather than from new
runs. Serves DoD 4 and Goal 2.

## Goal
Register slot **F4**, in two halves read together: compose rate against window
centre with the fork step drawn on it, and a strip of one cell across all nine
window positions with the scorer's verdict on each.

## Illustrations
*(image not yet generated; save under plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/assets/ and replace this placeholder)*

**Prompt for image generation:**
> Generate an image of a flowchart with two parallel horizontal lanes labeled
> "W1: gates the conditioning" and "W2: gates only the injected correction,
> conditioning always on". Each lane: build harness, verify all-off identity,
> smoke three windows, run full grid, joint figure (shared final card).
> Success path green with checkmark "Completed" pills. Failure path red on
> W2's identity stage labeled "all-off differs from plain PoE, the gating
> leaks" with an X icon and a dashed "Retry Stage" callout, downstream cards
> muted with "Skipped" pills. Glossy, minimalistic, modern UI/UX dashboard
> panel, dark background, rounded rectangle stage cards connected by
> directional arrows, clean sans-serif labels, generous spacing, no clutter.

## How wide the window is, and why not the obvious rule
The width had to be fixed before any window ran. The obvious rule was to take it
from the correction's size: make the window as wide as the narrowest band of
steps carrying half of the total ‖r_t‖. That rule gives 25 steps, half the
trajectory, because ‖r_t‖ barely changes over the run: each fifth of the run
carries 15-22% of the total and the largest step is 1.8x the smallest
(`scripts/window_width.py`, artifact `cache_analyses/window_width.json`). At
width 25 every placement contains the fork step, so the curve would come out
flat whatever the truth is, and the experiment would test nothing.

So the width is set by what the sweep has to be able to resolve, not by where
the correction is large. Width 10 at stride 5 gives nine placements, each a
fifth of the run, with the fork step inside only two of them (10-20 and 15-25).
If timing matters those two win; if nothing peaks, timing does not matter, and
that is a finding rather than a failure. Both numbers live in
`poe_repair/experiments/interaction_term/window_grid.py`, which the runner, the
scorer, the strip, and the inspector all read, so no two of them can disagree
about which grid was run.

## Environment Facts This Plan Depends On
- W2 harness: `scripts/interaction_term_window.py` over
  `run_teacher_residual`'s `correction_window`, which gates only the injected
  r_t and leaves conditioning on at every step. Never reuse W1's
  uncond-outside gating for W2, that confounds losing-the-correction with
  losing-conditioning.
- Largest generation grid in the program: 9 windows x 8 pairs x 4 seeds = 288
  cells at about 31s each, roughly 2.5 hours on one A6000.
- Output goes to /datasets via `POE_REPAIR_OUTPUT_ROOT`, which the sampler
  reads; the disk guard checks the filesystem `$OUT` actually resolves to.
- The scorer refuses to pool runs under 40 steps, because a window at step 20
  of a 20-step run is a different moment from step 20 of a 50-step run.

## Tasks
- [x] W2 harness: window-gated r_t injection on an always-conditioned PoE
      base, and prove the λ=0-outside-window case equals plain PoE exactly.
      `scripts/interaction_term_window.py --window off --check-identity`.
- [x] Fix the window width and the grid in source, from the ‖r_t‖-vs-step
      curve. The section above records what the curve actually supports.
      `scripts/window_width.py`, `window_grid.py`.
- [x] Smoke in-session on one pair, one seed, three windows: early, over the
      fork step, and late. `SMOKE=1 bash scripts/mechanism_study/run_window_sweep.sh`.
- [x] Full W2 grid: 288 cells, resumable, then score every image.
      `bash scripts/mechanism_study/run_window_sweep.sh`, then
      `python scripts/plot_window_curves.py`.
- [x] The curve and the image strips. `scripts/plot_window_curves.py` for the
      curve with the fork step drawn on it, `scripts/window_strip.py` for the
      same cell across all nine windows.
- [ ] Drive it by hand in the inspector: the Correction timing tab, built from
      `scripts/build_window_manifest.py`. Slider moves the window, the picture,
      the verdict and the curve marker together; pin one window to compare two.

## Not run: the conditioning-window half
W1 slides the same fixed-width window over the prompt rather than over the
correction, and would say whether the prompt and the correction are needed at
the same moment. It corroborates but F4 can carry its claim on W2 alone, so it
is not run here and the caption says the conditioning-window comparison is
future work. Running it means adding sliding fixed-width schedules to
`poe_repair/experiments/conditioning_window` and scoring old and new outputs
with the compose scorer.

## Success/Failure Outcomes
- **W2 harness smoke**
  - Success: all-off window reproduces plain PoE byte-identically; a
    mid-trajectory window visibly changes the output.
  - Failure: all-off differs from plain PoE, meaning the gating leaks. Fix
    before any grid.

## Next

1. Drive the Correction timing tab by hand, the one task still open above. The manifest it
   reads is already built; the command is in Engagement Instructions below.
2. F4 goes through plan 10's `/design-figure` pass, carrying two cautions the review
   settled: the caption may not claim the conditioning-window comparison, which was not
   run, and it may not draw step 16 as a band behind the timing curve, because the peak
   and the fork step disagree.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/window

# The leak check: nothing injected must leave the trajectory untouched.
$PY scripts/interaction_term_window.py --pair a_cat__x__a_dog --seed 9 \
    --window off --check-identity        # expect: byte-identical to PoE

# The grid. Resumable, skips any cell whose image exists.
bash scripts/mechanism_study/run_window_sweep.sh
find $OUT/pairs -name '*_w*.png' | wc -l  # expect 288

# Score, then the two halves of F4, then the manifest the inspector reads.
$PY scripts/plot_window_curves.py         # curve + peak band, prints missing windows
$PY scripts/window_strip.py --pair a_cat__x__a_dog --seed 9
$PY scripts/build_window_manifest.py

# Drive it by hand: the Correction timing tab.
bash scripts/run_lora_inspector.sh        # prints the ssh -L line to tunnel with
```
