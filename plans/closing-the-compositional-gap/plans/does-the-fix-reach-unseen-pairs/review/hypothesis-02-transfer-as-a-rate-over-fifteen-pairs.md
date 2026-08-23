# 🧪 Review: does the fix reach pairs it never trained on?

**Nothing has run yet.** This file holds the questions, written before the runs. It judges
[../plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md](../plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md),
the main question of this whole claim, and its answers fill register slot **F8**.

It cannot start until the one-epoch smoke in
[instrument-02](instrument-02-three-live-curves-while-training.md) is green, because fifteen
unattended runs with a broken scorer produce fifteen convincing wrong answers.

## Recommended prompt (when the runs land)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | the leave-one-pair-out sweep, the held-out set, the bar |
| **this file** | **the verdict: transfer as a rate over fifteen held-out pairs** |
| [what gates it](instrument-02-three-live-curves-while-training.md) | the one-epoch smoke that must be green first |
| [what defends it](baseline-01-the-size-matched-control-pool.md) | the size-matched pool that rules out data volume |

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

- **Leave one pair out**: train fifteen adapters, each missing a different pair, and test each one
  on exactly the pair it never saw. Fifteen tests instead of one, so the answer is a rate rather
  than an anecdote.
- **The degradation curve**: compose rate plotted against how much of the pool was held back. Its
  shape is the finding: a gentle slope and a cliff are different papers.
- **At the floor**: a held-out pair that composes no better than the broken method. Two causes,
  and telling them apart is why the extra measures exist: the correction never arrived, or it
  arrived pointing the wrong way.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A failure of the bar below closes the plan and opens one follow-on.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| fifteen runs, one per held-out pair | Tests the claim | | | leaderboard plus degradation curve | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Do most held-out pairs compose on an adapter that never saw them?
      The claim is transfer, and fifteen held-out points turn it into a rate with a spread rather
      than a single yes.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ What shape does the degradation curve take?
      Report it either way. A gentle decline says the fix generalises smoothly; a cliff says
      there is a minimum pool size, which is a different and still publishable claim.
- [ ] ⚠️ For every pair at the floor: did the correction fail to arrive, or arrive pointing
      wrong? Answered from the two direction measures, so a dead run is never misread as a
      transfer failure.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become a bar**, because it was
written with the answer already visible. Nothing yet: the sweep has not started.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Each of the fifteen adapters must differ from the others in
      exactly one thing: which pair was held back. Print the realised training pool of every run
      and confirm the counts match and the held-out pair is genuinely absent from each.
- [ ] ⚠️ **Was the instrument sound?** Each run scored over its own output directory only. With
      fifteen unattended runs writing under one parent, a scorer that collects every folder it
      finds produces fifteen plausible wrong answers, which is the fault this scope has already
      hit once.
- [ ] ⚠️ **Did the run respect the environment?** All fifteen outputs under `/datasets`, the
      hold-out flag confirmed to have selected a non-empty set on every run, and no run silently
      falling back to the full pool.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the fix reaches pairs it never trained on, at rate X over fifteen | the spread, not only the rate. Fifteen points with a spread is the claim; a mean alone hides whether one pair carried it |
| the degradation curve | its shape stated plainly, including when the shape is a cliff. A cliff is a different claim, not a worse one |
| any pair at the floor | which of the two causes it was, from the direction measures. A dead run reported as a transfer failure is a wrong claim |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the fifteen runs | register slot F8, and the transfer argument the paper is built on |
| whether it is safe to launch fifteen unattended runs | the one-epoch smoke in [instrument-02](instrument-02-three-live-curves-while-training.md) going green | the launch itself |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Get the [instrument-02](instrument-02-three-live-curves-while-training.md) smoke green, then
launch the fifteen runs.
