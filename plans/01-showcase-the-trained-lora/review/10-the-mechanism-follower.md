# 🔬 Review: do the discovered directions actually cause composition?

Nothing has run yet. This file judges [the design](../plans/10-the-mechanism-follower.md). Run
kind: a diagnostic measuring tool, with a causal test attached to every finding.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/10-the-mechanism-follower.md) | the checkpoint watcher, the two reads, the intervention rule |
| this file | per-direction intervention verdicts |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **h-space read**: UNet bottleneck activations, adapter on minus off; where the correction
  lives in the model's own semantic space.
- **Jacobian read**: the denoiser Jacobian's top singular directions at states inside the 0-10
  window; a candidate steering direction with a predictable gain.
- **Intervention**: inject the direction before step 10, score
  [compose rate](../../../context/world/compose-rate.md); the only thing that turns a direction
  picture into a mechanism claim.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Watched run | Node/device | PID | Checkpoints read | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **Does at least one discovered direction move compose rate when injected before step
  10?** The threshold, fixed before looking: moved means above the uncorrected rate by more than
  the binomial spread of those runs at some grid λ. None moving means every mechanism figure
  stays out of the main text, which is the pre-agreed outcome, not a failure to be argued with.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ Does the h-space difference grow, move, or stabilise across first/middle/final
  checkpoints (mechanism formation)?
- [ ] ⚠️ Do the Jacobian's top directions at window states change with training, and does the
  LoRA's r̂ align with any of them?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Adapter on/off computed at identical states; the
  interventions used the same pairs and seeds as the uncorrected runs they are read against.
- [ ] ⚠️ **Was the measuring tool sound?** The dry run on phase1 reproduced sensible reads before
  any live watching.
- [ ] ⚠️ **Did the run respect the environment?** Never on a training device; storage capped to
  pooled activations or a subset of the renders; PIDs recorded.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
