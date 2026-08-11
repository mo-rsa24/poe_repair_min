# Review: how far above the truth does 94% sit, and does the gap grow with λ?

Unanswered. This file judges
[../plans/instrument-01-the-three-state-labelled-set.md](../plans/instrument-01-the-three-state-labelled-set.md),
the labelled set that puts a band on every compose rate the paper prints. Its first question is
the one that decides whether F2's shape is safe.

## Words this file uses
- **Judgeable pair**: each animal has a visible feature you can name that the other lacks
  (stripes, tusks, a trunk), so a person can say from the picture which two animals are there.
  The other pairs cannot be judged at all, because the pool picked them for blending.
- **False-compose rate**: among images the scorer calls compose, how often a person says the two
  requested animals are not both there as separate animals. Not an error rate over all images:
  the denominator is the scorer's own successes, because that is the number the paper prints.
- **Coverage**: how much of the pair pool any "is it there?" metric can be scored on at all.
- **λ**: correction strength, on the grid {0, 0.25, 0.50, 0.75, 1.00} already on disk under
  `outputs/interaction_term/dose/pairs/`.

## Run kind
**Not a run: an instrument.** Judged by whether its checks could have failed, not by what they
found. It changes no claim on its own. What it produces is handed to `gate-02`, which decides
whether anything moves.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| | | | | |

## The questions

- [ ] ⚠️ **The bar.** Is the false-compose rate at λ=1 within 5 points of the rate at λ=0.50, 10
      or more points above it, or between the two?
      *Within 5 points*: the scorer is wrong by about the same amount everywhere, so F2's shape
      is real and only its level is an upper bound. The paper gains one paragraph. *10 points or
      more*: part of F2's slope is the detector becoming easier to please as the correction
      splits one blob into two, so the caption gets capped and a group-1 run is proposed in
      `does-the-correction-cause-composition`. *Between 5 and 10*: report both numbers, add the
      newly generated separable pairs, and do not pick a side. Both rates are quoted with their
      denominators. This is the only question whose answer may move a claim.
- [ ] ⚠️ Was the judgeable-pair rule committed before the first label?
      `git log` decides it, not memory. A rule written after seeing labels can be tuned until
      the rate comes out flattering, and no later check can detect that.
- [ ] ⚠️ Could the labeller see which λ produced the image?
      The tool must strip λ from the source path and shuffle. Two assertions: no label record
      carries a λ in its path field, and the presentation order does not match the on-disk
      order. If either fails, the labels are contaminated by expectation and the set is rebuilt.
- [ ] ⚠️ Did the labelling pass agree with the existing hand sort on at least 90% of the 32
      audit cells?
      Scored against `evidence/f2-lambda1-audit/calls.json`, which the pass never sees. Reported
      as a fraction, not a pass flag. Below 90% the automated pass is discarded and the user
      labels the main set by hand. That bar lives in the labelling script.
- [ ] ⚠️ Is "cannot tell" at or below 10% of labels on judgeable pairs?
      Above that, the judgeable-pair rule was wrong. The rule is rewritten, the split redone, and
      no rate is computed from the old set. A rate over a set where a tenth of the images are
      uncallable is not a rate.
- [ ] ⚠️ How many pairs, of the 19 on disk, can be judged at all?
      The coverage number, reported as a count and a fraction. It bounds every claim this scope
      or any future metric can make, and it is the one number here that is a finding in its own
      right rather than a check.
- [ ] ⚠️ Does the review say plainly what this cannot measure?
      On the pairs chosen because they blend, no detector and no person can say which two animals
      are present. That has to be stated as a limit of the instrument, not left for a reader to
      infer from the coverage number.
- [ ] ⚠️ Does `evidence/f2-lambda1-audit/README.md` now give the band as 75% to 94%?
      It currently says 87% to 94%, which counts the 17 uncallable cells as successes. The
      corrected line must also say the band cannot be narrowed from those images.
