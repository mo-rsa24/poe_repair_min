# 🅰️₀ Phase-1: one pooled LoRA, held-out transfer read (cheap first pass)

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

## Status (2026-08-03)
RUN DONE, READ NEAR-COMPLETE. `outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k`:
rank-8 LoRA on attn2.to_{q,k,v}, 11 training pairs, trained to step 100000 (latest.json).
`verdict.json` records `{"verdict": "ok", "reason": null, "epoch": 2000, "optimizer_step":
100000}` — a run-health marker (training completed without error), NOT the Phase-2
go/no-go call; that call is still a separate open task below.

Compose-rate now scored at steps 10000/20000/30000/40000/50000/60000. Best read (step
60000): in-distribution `in_in` 0.96 (n=176), held-out `out_out` 0.96 (n=128) — both
quadrants converged to the same rate. Per-held-out-pair at step 60000: leopard×jaguar
1.0, frog×toad 0.9375, eagle×hawk 0.9375, seal×walrus 0.9375, goose×swan 1.0,
cow×buffalo 1.0, cat×dog 0.875, elephant×penguin (control) 1.0. Every held-out pair is
well above floor, including cat×dog. Held-out transfer at 0.96 is a strong positive
signal → Phase-2 (plan 03) is warranted, pending the formal go/no-go note. Remaining:
steps 70000–100000 are still unscored (training reached 100000, compose-rate stops at
60000), and the direction axis (plan 02) is still unwired.

Unplanned-but-related: a triptych comparison viz (Mono | PoE | LoRA per eval step) was
added to the training loop's image logging (`_inline_sampling.py::compose_triptych`,
wired in `train_pooled.py`). Improves qualitative readability of this same compose-rate
data; does not touch direction-cosine/distance-reached.

## Tasks
- [x] ✅ Train one pooled rank-8 cross-attention LoRA on the 11 training pairs.
  ✓ verified (checkpoints to step 100000, config all_groups, dataset_meta 88 cells).
- [x] ✅ Eval the held-out split (unseen blends + cat×dog + control) for compose-rate,
  in-distribution vs held-out. ✓ verified (compose_rate.json: in_in 0.96 / out_out 0.96
  at step 60000).
- [ ] ⚠️ Score compose-rate for steps 70000–100000 (60000 is read and strong: out_out
  0.96; the run continued 40k further steps that are still unscored — narrow remaining
  slice, not from scratch).
      - [ ] record the citable transfer number with its checkpoint in the Status
        block (consumed by plans/interaction-term Goal 7: the paper never cites
        the number without its step)
- [ ] ⚠️ Read Phase-1 on both axes (add direction-cosine + distance-reached, dep: plan 02
  direction metrics) so a floor pair is split delivery-null vs no-transfer.
- [ ] ⚠️ Record the Phase-2 go/no-go call (LOPO warranted or not) with the held-out
  number backing it — verdict.json is a run-health marker, not this call; the numbers
  (out_out 0.96 @ step 60000) support a go but the note itself is unwritten.

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
