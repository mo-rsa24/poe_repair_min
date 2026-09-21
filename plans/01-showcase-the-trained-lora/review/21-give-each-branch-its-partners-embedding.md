# 🧪 Review: does telling each branch the other animal close the early fit gap?

**Not launched. Waits on the 30k verdicts of plans 15 and 20 for its base setting.** This file judges [the design](../plans/experiments/21-give-each-branch-its-partners-embedding.md).

## Recommended prompt (when the run lands)

```
/analyze-run phase1_r32_partner_40k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/21-give-each-branch-its-partners-embedding.md) | the helper, the three flags, the launcher, the two reads |
| **this file** | **the verdict against the bars below** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The partner embedding**: each single-prompt branch's pooled text vector plus the other prompt's, on the adapter-enabled forward only.
- **The on-cache fit**: cosine between the adapter's correction and the true correction on cached states, per tier and step bucket, no sampling. Branch-blind rank 8 at 30k: 0.985 training pairs, 0.925 cat × dog, early bucket.
- **Drift** and **compose count**: from the 8-seed grid at λ 1, as plans 15 and 20 read them.
- **The same-setting branch-blind run**: whichever of experiment D (raw or EMA weights) plans 15 and 20 leave as the base, so the comparison differs on the partner axis only.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (group 1: one architectural change, trained once). A null closes the direction-side line except the bounded on-policy arm; support changes the adapter's design.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Where | Run id | What ran | Wall time | Outcome |
|---|---|---|---|---|---|
| | | `phase1_r32_partner_40k` | | | not launched |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At 30k, does the early-bucket on-cache fit on cat × dog reach 0.955 or above, and does the 30k grid's drift come out more negative than the same-setting branch-blind run's by at least 0.03 with the compose count within one seed?** Support needs both. Null is an early fit at or below 0.935 (inside the noise of the branch-blind 0.925) whatever the grid says. Between is inconclusive: the fit moved and the render did not, which says the early correction is not what the render's fidelity turns on.
      What would surprise: the late-bucket held-out fit (0.748 today) moving as well, which would say the partner carries state the late correction needs.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ **Does the training-pair fit hold?** At or above 0.95 early; below is the fail criterion in the design.
- [ ] ⚠️ **Is plain PoE unchanged?** The probe's plain-PoE render against the cached `poe.png` within 6 grey levels, the unit check in task 1.1 made visible.

## Asked after the result

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Post-hoc questions with the number and the file.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- **The pooled vector's norm doubled.** Summing two pooled vectors changes the time-embedding input's scale; the design's Error Matrix names averaging as the fallback, and whichever is used is recorded in `config.json` and applied identically at inference.
- **Two axes moved.** Guarded by waiting for the base setting; a launch before plans 15 and 20 report would make every gap a mixed comparison.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The fit table with the partner column beside the branch-blind one.
- One method sentence if supported, and a sentence in the limitations either way on what the branch is and is not told.

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] Sum against average of the two pooled vectors is decided at the smoke, not pre-registered.
- [ ] The token-sequence variant (appending the partner's tokens) is not planned; it changes every cross-attention key and is a different experiment.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

When plans 15 and 20 have their 30k verdicts: task 1.1, the smoke, the launch.
