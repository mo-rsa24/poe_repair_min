# Parking Lot

Deferred and not-yet-ripe items for the LoRA-Fixes-PoE program. Promoted into the plan tree when their blocker clears.

## Defer

### Delivery is the first-order limiter, not transfer
From: socratic compile (transfer-vs-overfit argument), 2026-07-21
The trained cell only reaches ~40% of the PoE→Mono distance and plateaus, before any transfer is asked. So the binding constraint is delivery, not memory. Candidate fixes (deferred; the plans already defer architecture sweeps): longer training / higher rank; a λ-schedule weighting the commitment window; manifold-aware correction (the Plan 15-latent LSO framing). Revisit AFTER a transfer cell is measured with the two-tier bar (04-group-wise), so we know whether direction or delivery is the actual failure. (Partly addressed: the Attend-and-Excite grafts B1/B2 in 04-group-wise are a separate delivery fix now in the plans.)

### SLERP-merge the Plan-09 per-pair LoRAs as a second transfer route
From: augment compile (strengthened transfer case), 2026-07-21
Instead of only training a group-pool LoRA, spherically interpolate (SLERP) the single-pair LoRAs already trained in Plan 09 into a group corrector, and test transfer to held-out pairs. Cheap (reuses existing artifacts), and SLERP keeps the correction's direction better than plain averaging. Risk: merging corrections can dilute strength (the ~40% ceiling again). Revisit as an alternative to the pooled-LoRA line if Plan-16 group-pooling underperforms, or as a quick parallel check once the Plan-09 LoRAs for G1–G3 exist. Ref: LoRA Soups (arxiv 2410.13025).

### Degradation-curve evidence shape (rate vs fraction held out)
From: augment compile (strengthened transfer case), 2026-07-21
Report transfer as a curve — recognisable-composition rate vs the fraction of pairs held out — instead of a per-group pass/fail. Stronger, reviewer-recognised evidence (matches the compositional-generalisation eval literature, arxiv 2508.20783). Needs more held-out cells than the current 3-pairs-per-group plan produces. Revisit once the Plan-16 within-group runs land, so there is data to plot; may need widening the held-out sweep.
