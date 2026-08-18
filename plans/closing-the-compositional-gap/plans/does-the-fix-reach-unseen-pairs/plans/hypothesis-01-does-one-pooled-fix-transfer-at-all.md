# 🅰️₀ One pooled LoRA first: does the fix transfer at all?

**Step 10 of 22.** Waits on step 9. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 9 | [instrument-02-three-live-curves-while-training](instrument-02-three-live-curves-while-training.md) | ⚠️ do this next |
| **10** | **this plan** | **◑ read incomplete** |
| 11 | [hypothesis-02-transfer-as-a-rate-over-fifteen-pairs](hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md) | ⚠️ |

Design only. Verdicts live in [../review/hypothesis-01-does-one-pooled-fix-transfer-at-all.md](../review/hypothesis-01-does-one-pooled-fix-transfer-at-all.md).

## What this asks, in one line
Before paying for fifteen training runs, train one LoRA on all the training pairs and test it on
pairs it never saw: if that does not transfer, the fifteen-run version is not worth running.

## Why this plan exists
The 15-LoRA leave-one-pair-out sweep (plan 03) is expensive. Before paying for it,
one pooled LoRA trained on all training pairs and evaluated on a few unseen blend
pairs answers the scope's core question cheaply: does the fix transfer at all? A
positive or ambiguous Phase-1 justifies Phase-2 (the full LOPO); a flat null saves the
15-run cost. The pool file (`pair_pool.yaml`) already documents this two-phase split;
this plan captures it in the tree.

## Description
Train ONE rank-8 cross-attention LoRA on the 11 blend-prone training pairs, then eval
on the held-out split: unseen blend pairs (the transfer test), cat×dog (known-failure
reference), and the compose-by-default control. Read compose-rate for in-distribution
(`in_in`) vs held-out (`out_out`). This is a single-LoRA precursor to plan 03's LOPO,
NOT a replacement.

## Purpose
Serves Objective 2 (Transfer A) as a cheap first cut; gates whether plan 03's 15-run
LOPO is worth running.

## Goal
One pooled LoRA trained, held-out compose-rate read on the best checkpoint, and a
go/no-go call for Phase-2 (the LOPO in plan 03).

## Words this plan uses
- **The adapter**: a small set of extra weights (rank-8, on the layer where the prompt
  enters) trained to add the correction back. One adapter here, trained on all eleven
  training pairs at once, which is what "pooled" means.
- **Compose rate**: the fraction of pictures showing two separate animals rather than
  one blended one, decided by the validated scorer, never by eye.
- **Held-out**: a pair the adapter never trained on. The only kind that tests transfer.
  Its opposite, a pair it did train on, is **in-distribution**.
- **At the floor**: a held-out pair whose compose rate is no better than plain PoE's.
  Two very different things cause it, and telling them apart is why the two extra
  measures exist: either the fix never arrived (it was not delivered), or it arrived
  pointing the wrong way (it did not transfer).

## Tasks
- [x] Train one pooled adapter on the eleven training pairs.
- [x] Score the held-out split for compose rate, in-distribution against held-out.
      The split is the unseen blend pairs, the known-failure reference pair, and the
      control pair that composes fine without any adapter.
- [ ] Score the checkpoints from step 70000 to 100000. Training ran that far; scoring
      stopped at 60000. This is a scoring pass over existing checkpoints, not a rerun.
- [ ] Write the citable transfer number into the review file WITH its checkpoint step.
      The paper never quotes this number without the step it came from.
- [ ] Re-read the floor pairs on the two direction measures, once
      `instrument-02-three-live-curves-while-training`'s smoke run is green, so a floor
      pair can be told apart by which of the two causes above it is.
- [ ] Write the go/no-go note for the fifteen-run sweep into the review file, with the
      number and step backing it. `verdict.json` says only that training finished
      without error; it is not this call.

## Next

1. Score checkpoints 70000 to 100000 with the wired scorer (a narrow scoring pass, not a rerun).
2. Record the citable transfer number WITH its checkpoint step in the review file.
3. Re-read the floor pairs on the direction axis once plan 02's smoke is green.
4. Write the go/no-go note for the 15-run sweep into the review file.

**The short version:** step 2 alone, using step 60000 (held-out 0.96, n=128), stated as
"best-read checkpoint, later steps unscored".

## Engagement Instructions
GATE (pass/fail): the pooled run wrote a compose_rate.json with both `in_in` and
`out_out` populated on the final checkpoint, and a one-line Phase-2 go/no-go is recorded.
A script asserts both quadrant keys are present and non-empty.
STOP: if held-out `out_out` compose-rate is at floor (no better than the vanilla-PoE
fail-rate) AND direction is wrong once wired, mark Phase-1 a no-transfer null and do NOT
launch the 15-run LOPO — the cheap pass already answered it.

## Recommended skill
▶ `/run-experiment` ✅: drives the final-checkpoint scoring pass. alt: `/analyze-run` to
   sweep the pooled run's checkpoints for the converged held-out read.
