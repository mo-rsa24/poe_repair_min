# 🧪 Review: does one pooled adapter transfer at all?

**Answered yes, strongly, and the read is not finished.** This file judges
[the plan that trains one pooled adapter](../plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md).
It exists to answer one cheap question before fifteen expensive runs are paid for: does the fix
reach unseen pairs at all? It does, so the fifteen runs are warranted.

Three things are still owed, and until they are done the number cannot be quoted in the paper: the
later checkpoints are unscored, the direction measures were wired after this run finished, and the
go-ahead note itself is unwritten.

## Recommended prompt (to finish the read)

```
/analyze-run phase1_r8_100k
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md) | the pooled adapter, the split it is tested on, the level it must beat |
| **this file** | **the verdict: it transfers, at step 60000, with three things still owed** |
| [the fifteen runs this unblocks](hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | transfer as a rate over fifteen held-out pairs |

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

- **Pooled**: one adapter trained on all eleven training pairs at once, rather than one adapter
  per pair.
- **Held-out**: a pair the adapter never trained on. The only kind that tests transfer. Its
  opposite is a pair it did train on.
- **The split it was tested on**: the unseen blend pairs, the known-failure reference pair, and
  the control pair that composes fine with no adapter at all.
- **With its step**: a transfer number is only meaningful beside the checkpoint it came from,
  because the run kept training after it was measured.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim** (does the fix transfer at all; it decides whether the 15 runs go ahead).

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| phase1_r8_100k, pooled rank-8 LoRA on 11 training pairs, 88 pair-and-seed runs | Tests the claim | config `all_groups` | 100000 steps | `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k`, checkpoints to step 100000 | finished; read incomplete |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ Does held-out [compose rate](../../../context/world/compose-rate.md) beat the level plain
      PoE reaches with no fix at all? Emphatically, at step 60000:
      in-distribution 0.96 (n=176), held-out 0.96 (n=128). Per held-out pair: leopard×jaguar 1.0,
      frog×toad 0.94, eagle×hawk 0.94, seal×walrus 0.94, goose×swan 1.0, cow×buffalo 1.0,
      cat×dog 0.875, elephant×penguin (control) 1.0. Every pair well above that level.
      Held-out at 0.96 where vanilla PoE composes ~0 is the scope's core positive signal.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

Nothing beyond the one question above. This run was deliberately cheap and asked one question.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [ ] ⚠️ Do steps 70000 to 100000 change the read? Training ran 40k steps past the last scored
      checkpoint. The citable transfer number must carry its step, and 60000 is currently the
      best-read, not the final word.
- [ ] ⚠️ Does the direction axis agree? The two direction metrics (plan 02) were wired after this
      run's eval; a pair sitting at the no-fix level cannot yet be split delivery-null vs
      no-transfer.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Held-out and in-distribution pairs are scored by the same
      scorer at the same checkpoint, and the only thing differing between them is whether the
      adapter trained on the pair. Confirm the eleven training pairs and the held-out set do not
      overlap, from the realised split rather than the config.
- [ ] ⚠️ **Was the measuring tool sound?** The compose-rate scorer must have read only this run's
      output directory. This is the fault that has already produced convincing wrong numbers in
      this project once, in the dose series.
- [ ] ⚠️ **Did the run respect the environment?** Output under `/datasets`, checkpoints written
      through to step 100000, and no silent fp16 fallback in the eval hook.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| held-out compose-rate reaches 0.96 where vanilla PoE composes near zero | the checkpoint the number came from (step 60000), because the run kept training for another 40k steps after it |
| the pooled fix transfers to unseen pairs | that this is one pooled run, not a rate. The rate comes from the fifteen runs, and the sentence must not read as though this run supplied it |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether steps 70000 to 100000 change the read | scoring the later checkpoints | the citable number, which cannot be quoted without its step |
| whether the direction axis agrees | re-running eval with the two direction metrics from plan 02 wired in | telling a delivery-null pair apart from a no-transfer pair |
| the go/no-go note for the fifteen runs | writing it, with the number and step backing it | nothing technically, but those runs are being launched on an undocumented call. `verdict.json` is a run-health marker, not this call |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Score the checkpoints from step 70000 to 100000, then write the go-ahead note naming the number
and its step.
