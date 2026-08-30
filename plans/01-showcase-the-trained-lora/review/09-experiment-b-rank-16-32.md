# 🧪 Review: does capacity move what length could not?

Nothing has run yet. This file judges [the design](../plans/09-experiment-b-rank-16-32.md).
Run kind: ablation over the rank axis (one question per rank).

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/09-experiment-b-rank-16-32.md) | the two launches, the matched-steps rule, the null threshold |
| this file | the verdict per rank, read beside the panel of what the cached correction reaches at best |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

- **Matched step counts**: r8, r16 and r32 compared at the same optimizer step; anything else
  is a mixed comparison.
- **The best-case panel**: plan 12's side-by-side of the adapter against the cached true
  correction; the key that says whether a capacity null is a dead end or the answer.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

## Runs

Navigation: ⬅️ [Words](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Date | Node/device | PID | Run id | Rank | Wall time | Outcome |
|---|---|---|---|---|---|---|
| | | | | | | |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**This is the one question whose failure moves the plan.**

- [ ] ⚠️ **At matched 100k, do rank 16 and rank 32 sit inside rank 8's held-out seed-noise
  band?** The threshold, fixed at launch: both inside is a capacity null; either outside by more
  than the band reopens capacity as the lever, with plan 12's panel deciding how hard to push
  it.

## Written before the run, answered after

Navigation: ⬅️ [The bar](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ One question per rank, as an ablation owes: does r16 differ from r8 anywhere in the
  tracking set's reads? Does r32 differ from r16?
- [ ] ⚠️ Does the crispness read move with rank even where
  [compose rate](../../../context/world/compose-rate.md) does not?

## Could the answer be an artefact

Navigation: ⬅️ [Before/after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- [ ] ⚠️ **Was the comparison fair?** Configs diffed: only rank and alpha vary; alpha-to-rank
  ratio constant, so the effective scale is comparable.
- [ ] ⚠️ **Was the measuring tool sound?** Same frozen manifest hash on all three runs' configs.
- [ ] ⚠️ **Did the run respect the environment?** nohup launches recorded with node, device,
  PID; checkpoints on `/datasets`; no second Slurm job.

## Still open

Navigation: ⬅️ [Artefact checks](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents)

Nothing open.
