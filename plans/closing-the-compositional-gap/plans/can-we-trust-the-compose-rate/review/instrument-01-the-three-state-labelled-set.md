# 🔬 Review: how far above the truth does 94% sit, and does the gap grow with λ?

**Unanswered.** This file judges
[../plans/instrument-01-the-three-state-labelled-set.md](../plans/instrument-01-the-three-state-labelled-set.md),
the labelled set that puts a band on every compose rate the paper prints. Its first question is
the one that decides whether F2's shape is safe.

## Recommended prompt (when the labelling pass finishes)

```
/analyze-run <labelling pass id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/instrument-01-the-three-state-labelled-set.md) | the labelling rule, the blinding, the three-state scheme |
| **this file** | **the verdict: how far above the truth the printed compose rate sits** |
| [what it supplies](idea-01-what-the-current-benchmarks-score.md) | the judgeable-pair denominator every bake-off score is computed over |
| [what reads it](gate-02-promote-or-close.md) | the decision, which needs the two false-compose rates |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

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

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Not a run: an instrument.** Judged by whether its checks could have failed, not by what they
found. It changes no claim on its own. What it produces is handed to `gate-02`, which decides
whether anything moves.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

The images already exist on disk. The work is labelling them, not generating them.

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| labelling pass over the λ grid at `outputs/interaction_term/dose/pairs/` | Instrument | | labelling only; no generation | the three-state labels, the two false-compose rates, the coverage number | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Is the false-compose rate at λ=1 within 5 points of the rate at λ=0.50, 10
      or more points above it, or between the two?
      *Within 5 points*: the scorer is wrong by about the same amount everywhere, so F2's shape
      is real and only its level is an upper bound. The paper gains one paragraph. *10 points or
      more*: part of F2's slope is the detector becoming easier to please as the correction
      splits one blob into two, so the caption gets capped and a group-1 run is proposed in
      `does-the-correction-cause-composition`. *Between 5 and 10*: report both numbers, add the
      newly generated separable pairs, and do not pick a side. Both rates are quoted with their
      denominators. This is the only question whose answer may move a claim.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

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

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the labelling itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible. Nothing yet: the pass has not run.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Could the labeller see which λ produced the image?
      The tool must strip λ from the source path and shuffle. Two assertions: no label record
      carries a λ in its path field, and the presentation order does not match the on-disk
      order. If either fails, the labels are contaminated by expectation and the set is rebuilt.
      This is the fairness check that matters, because the bar compares two λ values against
      each other.
- [ ] ⚠️ **Was the instrument sound?** Two checks, both required.
      *Was the judgeable-pair rule committed before the first label?* `git log` decides it, not
      memory. A rule written after seeing labels can be tuned until the rate comes out
      flattering, and no later check can detect that.
      *Did the labelling pass agree with the existing hand sort on at least 90% of the 32 audit
      cells?* Scored against `evidence/f2-lambda1-audit/calls.json`, which the pass never sees.
      Reported as a fraction, not a pass flag. Below 90% the automated pass is discarded and the
      user labels the main set by hand. That bar lives in the labelling script.
- [ ] ⚠️ **Did the run respect the environment?** The labelled images must come from the λ grid
      at `outputs/interaction_term/dose/pairs/` and nowhere else, and every λ on the grid must
      have contributed a non-empty set of cells rather than one silently selecting nothing.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| any printed compose rate, including the 94% | the band around it, and both false-compose rates with their denominators. The denominator is the scorer's own successes, not all images |
| F2's shape | whether the slope is partly the detector becoming easier to please as the correction splits one blob into two. That is what the bar decides |
| the coverage number | that on pairs chosen because they blend, no detector and no person can say which two animals are present. This is a limit of the instrument, not a gap in the data |
| `evidence/f2-lambda1-audit/README.md` | the band as 75% to 94%, not the current 87% to 94%, which counts the 17 uncallable cells as successes. The corrected line must also say the band cannot be narrowed from those images |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the labelling pass over the λ grid | [gate-02](gate-02-promote-or-close.md), which needs the two false-compose rates, and [idea-01](idea-01-what-the-current-benchmarks-score.md), which needs the judgeable denominator |
| the band printed in `evidence/f2-lambda1-audit/README.md` | correcting it to 75% to 94% | any reader who takes 87% as the floor, which counts 17 uncallable cells as successes |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Commit the judgeable-pair rule, then run the labelling pass. The commit order is itself one of
the checks above.
