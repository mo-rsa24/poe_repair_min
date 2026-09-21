# 🧪 Review: does softness track lambda at a fixed checkpoint?

One run complete (jobs 48612/48613/48614, 2026-09-01). This file judges [the design](../plans/experiments/07-experiment-c-lambda-window.md).
Run kind: hypothesis (an intervention on injection strength, no training). Its answer is the
reading key for experiments A and B.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/07-experiment-c-lambda-window.md) | the grid, the softness reads, the strip |
| this file | the verdict and the per-λ numbers |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **λ**: the injection strength on the LoRA's predicted correction; 0 is plain PoE, 1 is the
  shipped configuration.
- **The drift read**: DINOv2/CLIP distance-to-mono minus distance-to-poe per render (plan 06's
  embedding read, computed offline here).

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|
| 2026-09-01 | jobs 48612 (sweep) → 48613 (measure) → 48614 (strip), chained on `afterok`, bigbatch/mscluster76, commit `32d1973`→`c6a470b` (dirty) | `scripts/showcase/lambda_window_grid.py`: the 5-cell × 5-λ grid (25/25 renders), `lambda_softness.json`, `lambda_softness_strip.png` | ~16m32s (sweep 15m38s, measure 49s, strip 5s) | OK, all 25 renders + measurements written; λ=0 identity check passes on all 5 cells (DINOv2/CLIP distance to cached `poe.png` small) |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does the sharpness proxy fall monotonically with λ on the tracked pairs?** The
  threshold, fixed before looking: a monotone decrease across the five λ values on the majority
  of pairs supports the injection account; flat within the pairs' own spread kills it. The per-λ
  table goes here.

  Mean sharpness (Laplacian variance, higher = crisper), averaged across all 5 cells, from
  `lambda_softness.json` (jobs 48612-48614):

  | λ | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
  |---|---|---|---|---|---|
  | mean sharpness | 75.6 | 66.6 | 55.8 | 61.3 | 69.0 |

  0 of 5 cells are strictly monotone-decreasing across the grid (`n_cells_strictly_monotone_decreasing_sharpness: 0` in the sidecar's `aggregate` block): the mean dips through λ=0.5 then rises again, rather than falling steadily. Against the threshold above, this reads as **flat/non-monotone**, not a pass. Eyeball the strip (`lambda_softness_strip.png`) at instruction 2.1 before signing off on the verdict below.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Where along the grid does [compose rate](../../../context/world/compose-rate.md) arrive (the more-correction-more-composition story) relative to where blur
  arrives (the cost story), and do the two leave a usable middle λ?

  Compose rate across the 5 cells, from `lambda_softness.json`:

  | λ | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
  |---|---|---|---|---|---|
  | compose rate | 0.0 | 0.0 | 0.0 | 0.4 | 0.8 |

  Composition arrives late and steeply (λ=0.75 to 1.0), while sharpness never falls steadily
  across the same range (see the question above). The two stories don't cleanly trade off
  against each other the way the design predicted; there isn't an obvious "usable middle λ"
  visible in these numbers alone.
- [x] Did the λ=0 identity hold against the cached PoE render (mode-level, fp16 drift band)?
  Yes, on all 5 cells (DINOv2 / CLIP cosine distance to cached `poe.png`): cat_dog_s9 0.059 /
  0.047, eagle_hawk_s9 0.001 / 0.001, frog_toad_s9 0.002 / 0.002, goose_swan_s9 0.005 / 0.001,
  cat_dog_s1 0.130 / 0.021 — all small, consistent with fp16 drift rather than a structural
  mismatch (cat_dog_s1's distance is the largest of the five but still an order of magnitude
  below the ~0.43 seen when the runner's off-window steps ran unguided instead of full PoE,
  before that was fixed).

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Only λ varied; checkpoint, window, seeds, guidance fixed.
- [ ] ⚠️ **Was the measuring tool sound?** The sharpness proxy sanity-checked on a known-crisp and a
  known-soft render before use.
- [ ] ⚠️ **Did the run respect the environment?** In-session on a free device; outputs on
  `/datasets`.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
