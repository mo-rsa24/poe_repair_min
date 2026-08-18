# 🏷️ The labelled set, and the band it puts on 94%

**No step number: nothing in the paper order waits on this.** This scope runs in its own internal order, and earns numbered steps only on the big-promotion condition its own `MASTER_PLAN.md` sets. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Within this scope | Plan | Status |
|---|---|---|
| 1 of 4 | [gate-01-is-this-hole-already-known](gate-01-is-this-hole-already-known.md) | ⚠️ not started |
| **2 of 4** | **this plan** | **⚠️ not started** |
| 3 of 4 | [idea-01-what-the-current-benchmarks-score](idea-01-what-the-current-benchmarks-score.md) | ⚠️ blocked by gate-01 |

Design only. The verdict lives in [../review/instrument-01-the-three-state-labelled-set.md](../review/instrument-01-the-three-state-labelled-set.md).

## What this asks, in one line
How often does the scorer say "composed" when a person would not, and does that gap get bigger
as λ gets bigger?

## Why this plan exists
94% is what a detector counting animals reports, and nobody has measured how far above the
truth it sits. This plan produces the band. It runs whatever `gate-01` returns, because the
paper needs the band even if the metric idea is dead, and because whatever metric the paper
ends up using has to be certified against something.

## Description
Four steps in a fixed order, and the order is the point. First write down which pairs a person
can even judge, on a rule fixed before any image is opened. Then build a labelling tool that
hides λ. Then check the labelling against the 32 cells already sorted by hand. Only then label
the main set and compute the rates.

The three prerequisites (the rule, the hidden λ, the calibration) are tasks 1 to 3 here rather
than a separate plan, because each is worthless once a label has been written.

## Purpose
Serves Objectives 2 and 4, and Definition-of-Done items 2, 3, 4, 5, 6 and 8.

## Goal
On `/datasets`: the labelled set with counts per λ. In the review file: the false-compose rate
at λ=1 and at λ=0.50 with their denominators, the coverage number, and the judgement against
the three-way bar. In the repo: `evidence/f2-lambda1-audit/README.md` giving the band as 75% to
94%.

## Environment Facts This Plan Depends On
- The GPU is shared and often full. Any detector re-run needs a working `--device cpu` route,
  and the labelling itself needs no GPU at all.
- Large artifacts go to `/datasets` only. The labelled set and any cached crops land there,
  never on `/home-mscluster`.
- Thresholds live as named constants in source, following `MIN_BOX_FRACTION` in
  `poe_repair/experiments/compose_scorer/detection_scorer.py`. The 10% calibration bar and the
  10-point and 5-point λ bars are constants in the labelling and scoring scripts.
- Source images stay where they are. `scripts/plot_dose_curves.py` scores from
  `outputs/interaction_term/dose/pairs/`, so this plan copies and never moves.

## Success/Failure Outcomes
- **"Cannot tell" exceeds 10% on the judgeable pairs.** The judgeable-pair rule was wrong. It
  gets rewritten and the split redone before any rate is computed. A rate over a set where a
  tenth of the images are uncallable is not a rate.
- **Calibration disagrees with the hand sort on more than 10% of the 32 audit cells.** The
  automated pass is discarded and the user labels the main set by hand. The bar sits in the
  script so this cannot be waved through.
- **Fewer than four judgeable pairs survive the rule.** The set is too small to compute a rate
  per λ with a usable denominator. Halt and generate separable pairs before continuing.
- **The obvious failure that has to be designed out:** the labeller seeing λ in the file path
  and labelling high-λ images more generously. Task 2 exists only for this.

## Tasks
- [ ] Write the judgeable-pair rule as a named constant plus a table in source: a pair is
  judgeable if each animal has a visible feature you can name that the other lacks. Apply it to
  all 19 pairs in `outputs/interaction_term/dose/pairs/` and commit the 19-row split BEFORE
  opening a single image. [inferred prerequisite, DoD 2]
- [ ] Build the labelling tool: it strips λ out of the file path, shuffles, shows one image
  with its prompt, and records one of four labels plus the scorer's own call. Any detector call
  it makes takes `--device cpu`. [inferred prerequisite, DoD 3]
- [ ] Calibrate: run the labelling pass over the 32 cells in `evidence/f2-lambda1-audit/`
  only, and score it against `calls.json`. The 10% discard bar is a constant in the script.
  [inferred prerequisite, DoD 4]
- [ ] Label the main set: judgeable pairs × 5 seeds × λ ∈ {0, 0.25, 0.50, 0.75, 1.00} on the
  real-correction row, plus the λ=1 cells of the `_random` and `_wrong_pair` rows. Four labels:
  both requested animals separate; one animal or a fusion; two or more animals but not the two
  asked for; cannot tell. Write to `/datasets`. [DoD 5]
- [ ] Compute the false-compose rate at λ=1 and at λ=0.50, each with its denominator, and the
  coverage number. Judge against the three-way bar, whose 10-point and 5-point thresholds are
  constants in the scoring script. [DoD 6]
- [ ] Rewrite the band in `evidence/f2-lambda1-audit/README.md` to 75% to 94%, stating it
  cannot be narrowed from those images. It currently says 87% to 94%, which counts the 17
  uncallable cells as successes. [DoD 8]

## Engagement Instructions
Five checks, all mechanical:

1. `git log` shows the 19-row split table committed before the first label file. If it does not,
   the rule was written with labels already in hand and the split is void.
2. No label record contains a λ value in its source-path field, and the presentation order does
   not match the on-disk order. Assert both in a test.
3. The calibration disagreement against `calls.json` is at or below 10%. Printed with its
   numerator and denominator, not as a bare pass.
4. "Cannot tell" is at or below 10% of labels on judgeable pairs, printed the same way.
5. Both false-compose rates print with denominators, and the coverage number prints beside them.
   A rate with no denominator does not count as produced.

STOP: check 1 fails → the split is redone from scratch on a rule committed first. Check 3 fails
→ discard the automated pass, the user labels by hand. Check 4 fails → rewrite the rule, redo
the split, do not compute any rate.

## Recommended skill
▶ `/run-experiment` ✅: drives the labelling pass and the rate computation, and stamps the run
   into the review file's Runs table. alt: `/write-tests` for checks 1 and 2, which are the two
   worth having as permanent assertions rather than a one-off look.

## Recommended Prompts
- **Before task 1** (the judgeable-pair rule): `/demonstrate show me the 19 pairs as a contact
  sheet, one representative image each` gives you the pairs to write the rule against without
  seeing any of the images the rule will later be applied to.
- **On task 5** (the two rates): `/pair-figure` before plotting anything. Whether a point is one
  image, one seed, or one pair is a live design question on three other review files in this
  tree, and answering it here differently would make the numbers uncomparable.
- **After task 6** (the corrected band): `/reconcile evidence/f2-lambda1-audit/README.md` to
  check nothing else in that file still assumes the old 87% figure.
