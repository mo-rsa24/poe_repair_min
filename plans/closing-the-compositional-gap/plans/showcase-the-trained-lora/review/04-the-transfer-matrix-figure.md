# 🔬 Review: does the pooled fix reach pairs sharing no concept with its pool?

**Unanswered.** This file judges [../plans/04-the-transfer-matrix-figure.md](../plans/04-the-transfer-matrix-figure.md). Questions below were written at design time, before any run, per `~/.claude/EXPERIMENT_CONVENTIONS.md`.

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/04-the-transfer-matrix-figure.md) | the group-by-disjoint-pair matrix with its disjointness audit |
| **this file** | **the verdict, once the runs land** |
| [what it feeds](../plans/05-assemble-the-showcase-figures.md) | the transfer matrix ships from here |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Disjoint cell**: no token shared between the training pool's prompts and the evaluation pair's; audited, never assumed.
- **PoE floor**: the un-corrected compose rate on the same cells, the row every reading is against.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

Generalization run: one question per held-out unit (here, per training group).

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run id / log | Config (the varied axis values) | Wall time | Landed where |
|---|---|---|---|
| (none yet) | | | |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **THE BAR.** Per training group: is the mean compose rate over its audited-disjoint cells at least 0.25 absolute above that group's PoE floor? (The floor on blend-prone pairs is near zero, so 0.25 is a quarter of the seeds composing.) Groups passing = transfer at the credible tier; none passing = concept-bound repair, claimed as such. Registered before any cell runs.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

- [ ] ⚠️ How many candidate cells were censused, how many survived the disjointness audit, and which were excluded with which shared token?
- [ ] ⚠️ Is there structure by group (some pools transfer, others not), and does it track the pool's semantic spread?
- [ ] ⚠️ Do the do-no-harm control pairs stay composed under every checkpoint?

## Could the answer be an artefact

Navigation: ⬅️ [Written before the run, answered after](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

Three named checks, each answered before the verdict stands.

- [ ] ⚠️ **Was the comparison fair?** Cells differing in more than the train-group axis are marked mixed and read alone; the floor row comes from the identical cells uncorrected.
- [ ] ⚠️ **Was the instrument sound?** Scorer contract as everywhere; the audit runs before rendering, and a post-hoc shared token voids the cell's reading.
- [ ] ⚠️ **Did the run respect the environment?** Output on `/datasets`, absolute paths on any SSH launch, fp32 upcast before any norm or cosine read, guidance at 7.5 throughout.

## Still open

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Per-group verdicts feed plan 05's matrix figure and the paper's generalization claim tier.
