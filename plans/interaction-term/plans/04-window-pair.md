# ⏱️ The timing pair: when is the correction needed

## Description
Two matched window experiments on the same pairs, seeds, widths, and scorer.
W2 (new): base is full guided PoE at every step; a fixed-width window slid
across the 50 steps gates only the injected r_t. W1 (enhanced): the existing
conditioning_window sweep (guidance on inside the window, unconditional
outside) gains the same fixed-width sliding schedules and a scorer pass.

## Purpose
W2 measures causally when the interaction term is needed, with no assumed
window. W1 answers when conditioning is needed at all. One joint figure says
whether the two windows coincide or differ; either answer is a finding.
Serves DoD 4 and Goal 2.

## Goal
The joint window figure: compose-rate vs window center, both curves, equal
width everywhere, plus peak-window vs tail-window image strips.

## Illustrations
*(image not yet generated; save under plans/interaction-term/assets/ and replace this placeholder)*

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

## Environment Facts This Plan Depends On
- Harness reuse: poe_repair/experiments/conditioning_window/sweep.py (per-step
  mask loop, schedule grid, manifest); never reuse W1's uncond-outside gating
  for W2, that confounds losing-the-correction with losing-conditioning.
- Largest generation grid in the program: runs as jobs, biggpu first.
- Disk guard on /datasets; W&B triptych logging per schedule.

## Tasks
- [ ] ⚠️ W2 harness: window-gated r_t injection on an always-conditioned PoE
      base (adapt the mask loop; the λ=0-outside-window case must equal plain
      PoE exactly)
- [ ] ⚠️ pick the fixed width from the ‖r_t‖-vs-step curve (plan 05) and state
      it in the run config; every window in both experiments uses it
- [ ] ⚠️ W1 enhancement: add sliding fixed-width schedules to the existing
      sweep and score all outputs (old and new) with the compose-scorer
- [ ] ⚠️ smoke both harnesses in-session on one pair, one seed, three windows
- [ ] ⚠️ full grids as jobs: both experiments, shared pairs and seeds
- [ ] ⚠️ joint figure plus the two image strips

## Success/Failure Outcomes
- **W2 harness smoke**
  - Success: all-off window reproduces plain PoE byte-identically; a
    mid-trajectory window visibly changes the output.
  - Failure: all-off differs from plain PoE, meaning the gating leaks. Fix
    before any grid.

## Recommended skill
▶ `/run-experiment` ✅ for the grids; `/demonstrate` ✅ for the smoke.

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -m poe_repair.experiments.interaction_term.window --pair a_cat__x__a_dog \
  --seed 9 --window off --check-identity      # expect byte-identical to PoE
ls /datasets/.../interaction_term/window/{w1,w2}/ | wc -l  # both grids present
$PY scripts/plot_window_curves.py             # joint figure, peak band printed
```
