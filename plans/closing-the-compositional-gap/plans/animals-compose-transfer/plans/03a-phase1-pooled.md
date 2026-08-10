# 🅰️₀ One pooled LoRA first: does the fix transfer at all?

Design only. Verdicts live in [../review/03a-phase1-pooled.md](../review/03a-phase1-pooled.md).

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

## Tasks
- [x] Train one pooled rank-8 cross-attention LoRA on the 11 training pairs.
 
- [x] Eval the held-out split (unseen blends + cat×dog + control) for compose-rate,
  in-distribution vs held-out.
- [ ] Score compose-rate for steps 70000–100000 (60000 is read and strong: out_out
  0.96; the run continued 40k further steps that are still unscored — narrow remaining
  slice, not from scratch).
      - [ ] record the citable transfer number with its checkpoint in the Status
        block (consumed by plans/closing-the-compositional-gap/plans/interaction-term Goal 7: the paper never cites
        the number without its step)
- [ ] Read Phase-1 on both axes (add direction-cosine + distance-reached, dep: plan 02
  direction metrics) so a floor pair is split delivery-null vs no-transfer.
- [ ] Record the Phase-2 go/no-go call (LOPO warranted or not) with the held-out
  number backing it — verdict.json is a run-health marker, not this call; the numbers
  (out_out 0.96 @ step 60000) support a go but the note itself is unwritten.

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
