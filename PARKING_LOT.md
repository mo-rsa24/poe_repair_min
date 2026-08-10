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

### Results and experiments prose (blocked on the numbers)
From: triage-plan (paper-iclr routing), 2026-08-05
The results section cannot be written in phase 1 because most numbers do not exist. `interaction-term` plans 01–11 are unrun (only plan 00, the instruments, is complete); `animals-compose-transfer` has the pooled 03a read (out_out 0.96 at step 60k) but owes the leave-one-pair-out run, the mixed-pool contrast, and the 70k–100k scoring. Phase 1 writes section skeletons with named placeholders instead. Trigger: promote per figure as each run lands, not as one block. The figure layout from `plans/paper-iclr/plans/02-figure-layout.md` sets which numbers are needed first.

### Figure production stays in the result scopes
From: triage-plan (paper-iclr routing), 2026-08-05
The new `paper-iclr` scope owns figure *layout* (which figure goes in which section, in what order) but not figure *production*. F2–F5 stay with `animals-compose-transfer/plans/05-figures.md`; the seven-figure cascade stays with `interaction-term/plans/10-figures.md`. Revisit only if the split makes "where are my figures" genuinely hard to answer in practice, which is the one cost of this decision.

### Finishing either result scope before drafting
From: triage-plan (paper-iclr routing), 2026-08-05
Dropped, recorded so it is not re-litigated. `interaction-term`'s Definition of Done has 11 items including SD 1.5 / SD 2.1 replication, a sampler sweep, two /pressure-test passes and the Inspector tabs; that is a paper's worth of work on its own. The two scopes are also mutually dependent (interaction-term DoD item 7 says the 100k transfer number is owned by animals plan 03a), so "finish one then the other" was never available. The draft proceeds against placeholders, and the ranked figure set decides the run order.

### Seven more skills the design/review split would eventually touch
From: the plan-tree conventions session, 2026-08-10
Four skills already know about the split: `PLAN_TREE_FORMAT.md`, `populate-plans`,
`verify-plan`, `sync-plan-tree`. Seven more would need it eventually, and all seven are
deliberately deferred until a second scope has been converted by hand and proved the format:

- `init-master-plan`, `decompose-plan`: a new or child scientific scope inherits the
  conventions pointer and the two folders.
- `integrate-plans`: a task added to a scientific scope may owe a review question.
- `frame-hypothesis`: it runs inside the follow-on plan, never in the plan that just failed.
- `experiment-planner`, `hypothesis-to-scope`: emit the run kind and the review questions at
  design time, which is the only time they can honestly be written.
- `analyze-run`, `training-analyst`: their output lands as answered review questions, not prose.
- `task-graph`: a plan blocked on an unanswered review question is blocked.
- `execute-plan-tree`: it must not tick a plan whose pre-registered question is unanswered.

The reason for waiting: six files describe this convention and one plan uses it. Spreading
instructions that have never been executed to seven more skills makes seven more places for it
to drift. Convert `animals-compose-transfer` by hand first and see whether `populate-plans`
produces what plan 03 has.
