# LoRA-Mechanism-Study

## Mission
When Mono-free LoRA correction visibly fixes a PoE composition failure, does it work through
the same channel test-time attention optimization would, or a different one, and does that
answer explain why some pairs fail and others don't?

## Objectives
1. **Instrument** — wire existing attention-capture machinery (`_CrossAttnRecorder`, already
   working on the `teacher_residual`/oracle composer) into the actual LoRA inference path
   (`run_lora_residual_inject`), which has never had capture wired to it.
2. **Baseline-Compare** — build the minimal Attend-and-Excite-equivalent test-time attention
   intervention (new code, reuses the recorder's existing `keep_grad=True` path), so the
   LoRA's attention behavior has an independent comparison point.
3. **Read the Mechanism** — on cat×dog's 12 seeds (8 train {1-8} + 4 held-out {9-12}, from
   `outputs/cross_seed_lora_pooling/seed_pool.yaml`), determine whether the LoRA's attention
   shift, where it visually succeeds, matches, diverges from, or is absent relative to the
   Attend-and-Excite-equivalent's shift.

## Goals
1. **Instrument**: attention `.pt` files exist for the LoRA path (λ=0 and λ=1) on all 12
   cat×dog seeds, in the same schema as the existing `veracity_attn` cache.
   [checkpoint: file count + a 12×50 sanity table of attention mass per seed/timestep]
2. **Baseline-Compare**: attention `.pt` files exist for the Attend-and-Excite-equivalent
   intervention on the same 12 seeds, same schema.
   [checkpoint: same sanity-table format as Goal 1]
3. **Read the Mechanism** (three-way, pressure-test-upgraded rule):
   - *support* if ≥3 visually-successful seeds show Δ_attn(LoRA) matching Δ_attn(AAE) in sign
     and within 2x magnitude (commitment-window-restricted, window from `G02`/
     `residual-diagnostics.md`)
   - *null* if the majority of visually-successful seeds show sign-mismatch or >4x magnitude
     divergence
   - *inconclusive* if fewer than 3 seeds visually succeed (known ~40% delivery ceiling from
     rung 1), or the AAE baseline itself is noisy → triggers the parked "Widen if Needed"
     item, does not loosen the threshold on cat×dog alone
   - Prerequisite check gating this read: does Δ_attn(LoRA) correlate with the visual
     composition label at all — if null, the AAE comparison is moot.

## Expected Outcome
Either (a) evidence the trained Mono-free LoRA recovers the same attention-reallocation
behavior an expensive test-time method would, meaning it learned a cheap substitute for a
known mechanism, or (b) evidence it fixes composition through a different channel, itself a
finding worth explaining. Either way, a concrete numeric account of what "the LoRA
understands compositionality" cashes out to, paired with qualitative attention-map examples.

## Definition of Done
1. ⚠️ Attention capture wired into `run_lora_residual_inject`.
2. ⚠️ 12-seed capture run complete for plain PoE (λ=0) + LoRA (λ=1).
3. ⚠️ Attend-and-Excite-equivalent baseline built and run on the same 12 seeds.
4. ⚠️ Δ_attn metric computed (commitment-window-restricted), paired with visual composition
   labels per seed.
5. ⚠️ Prerequisite check read before the headline comparison is read.
6. ⚠️ Headline LoRA-vs-AAE comparison read against the three-way rule; verdict recorded
   either way.

## Other mechanism candidates, not pursued here
Two other mechanism checks came out of the same framing session but are deliberately out of
this scope's DoD:
- **Embedding-space direction consistency + LoRA-vs-Mono attention comparison** — parked in
  `PARKING_LOT.md` with a `/frame-hypothesis` refinement prompt (needs its own falsification
  rule before it's plan-ready).
- **Denoising-timing overlay against the commitment window** — not a separate check, folded
  as a side-effect figure once DoD-4's per-timestep data exists; no new capture needed.

## Sub-Scopes
(none yet)

## Plans
- ⚠️ plans/01-instrument-attention-capture.md — wire attention capture into the LoRA path, run λ=0/λ=1 on 12 seeds (DoD 1, 2)
- ⚠️ plans/02-attend-and-excite-baseline.md — build + run the Attend-and-Excite-equivalent baseline on the same 12 seeds (DoD 3)
- ⚠️ plans/03-read-the-mechanism.md — compute Δ_attn, prerequisite check, headline LoRA-vs-AAE three-way read (DoD 4, 5, 6)
