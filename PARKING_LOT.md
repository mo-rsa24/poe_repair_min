# Parking Lot

Deferred and not-yet-ripe items for the LoRA-Fixes-PoE program. Promoted into the plan tree when their blocker clears.

## Defer

### ~~Delivery is the first-order limiter, not transfer~~ RESOLVED 2026-08-04
From: socratic compile (transfer-vs-overfit argument), 2026-07-21
~~The trained cell only reaches ~40% of the PoE→Mono distance and plateaus, before any transfer is asked.~~ **Resolved by evidence:** the pooled animals run `phase1_r8_100k` reaches full correction magnitude (delta_hat_norm 27.66 vs delta_target_norm 27.32) and 0.96 held-out compose-rate at step 60k. The 40% plateau was a property of the old taxonomy cells, not of the method. No candidate fix needed.

### Old taxonomy training work (shelved 2026-08-04)
From: triage-plan (interaction-term routing), 2026-08-04
G1–G3 single-seed trainings, G1–G4 pooled runs, and the all-groups crossbar resume (`2em6frqv`) are shelved under the ICLR interaction-term anchor. Plan files moved to `plans/shelved/rungs/`. The old taxonomy's pairs still serve the program as cached cells for the composition-type scatter (no training). Revive only if a reviewer-facing gap demands taxonomy-wide training evidence.

### Attend-and-Excite baseline (left behind from mechanism-study)
From: mechanism-study plan 02 (never run), shelved 2026-08-04
Test-time attention-steering baseline for the "why not just re-aim attention" question. The mechanism section answers it descriptively (attention barely moves); running AAE is held in reserve. Trigger: a reviewer asks for the direct comparison. Plan content preserved in `plans/shelved/mechanism-study/plans/02-attend-and-excite-baseline.md`.

### SuperDiff AND as a full baseline
From: triage-plan (interaction-term routing), 2026-08-04
The paper carries one defended sentence (rebalancing cannot manufacture a correction outside the span of the two predictions; gets a /pressure-test pass before print). Running SuperDiff AND as an actual baseline is held in reserve with the AAE item above, same trigger: a reviewer asks "why not just rebalance".

### SLERP-merge the Plan-09 per-pair LoRAs as a second transfer route
From: augment compile (strengthened transfer case), 2026-07-21
Instead of only training a group-pool LoRA, spherically interpolate (SLERP) the single-pair LoRAs already trained in Plan 09 into a group corrector, and test transfer to held-out pairs. Cheap (reuses existing artifacts), and SLERP keeps the correction's direction better than plain averaging. Risk: merging corrections can dilute strength. Revisit if the pooled-LoRA line ever underperforms. Ref: LoRA Soups (arxiv 2410.13025).

### ~~Degradation-curve evidence shape (rate vs fraction held out)~~ PROMOTED 2026-07-28
From: augment compile (strengthened transfer case), 2026-07-21
~~Report transfer as a curve, recognisable-composition rate vs the fraction of pairs held out, instead of a per-group pass/fail.~~ **Promoted:** subsumed by the `animals-compose-transfer` scope's leave-one-pair-out design. Ref arxiv 2508.20783 still applies.

### Widen the mechanism study to a second pair (updated 2026-08-04)
From: hypothesis-to-scope → triage-plan (mechanism-study scope), 2026-07-26
Extend the mechanism read to a second pair from the compose-by-default bucket. Trigger updated: fires if the `interaction-term` scope's mechanism re-probe (the cross-seed, cross-pair re-run of the value-channel finding on `lora_step_100000.pt`) reads inconclusive. The re-probe's held-out pairs already span multiple pairs, so this may self-resolve; check before promoting. Bundled with the taxonomy-relabel item below.

### Two-bucket taxonomy relabel (compose-by-default vs fails-by-default)
From: hypothesis-to-scope → triage-plan (mechanism-study scope), 2026-07-26
Re-sort the existing taxonomy (G1–G6) into two buckets using the visual reads from the old rung 1. Infrastructure for the "Widen" item above; only needed if its trigger fires. Note: the `interaction-term` composition-type scatter assigns regime labels to the same cached pairs by measured ‖r_t‖, which may supersede this eyeball relabel entirely.

### LoRA-vs-Mono attention comparison (trimmed 2026-08-04)
From: hypothesis-to-scope → triage-plan (mechanism-study scope), 2026-07-26
Original entry had two halves. (a) Cross-seed direction-consistency: now covered by the `interaction-term` spectrum analysis (stacked cached targets, per-pair blocks). (b) Does LoRA-corrected attention resemble Mono's actual attention: still parked, still no three-way falsification rule, needs Mono-side attention capture confirmed.
▶ `/frame-hypothesis "does LoRA-corrected cross-attention resemble Mono's actual attention on the same held-out cells, once the interaction-term mechanism re-probe has landed"`

### Text-space intervention (add/subtract the binding vector)
From: language-probes discussion, 2026-08-03
The passive probes L1–L3 measure the binding vector b = e_J − normalized(e_A+e_B). The intervention (adding b to a compose-by-default pair's conditioning, or subtracting it) is a new experimental surface with its own failure modes. Parked until the passive probes land.
▶ `/frame-hypothesis "does adding the measured text-space binding direction b to a regime-1 pair's conditioning shift its λ=0 composition behavior, given L1–L3 results"`
