# 03 · Read the Mechanism — LoRA vs Attend-and-Excite attention comparison

[⌂ Index](00-INDEX.md) · [← prev](02-attend-and-excite-baseline.md)

## Reference while you do it
- 📄 Plan: plans/mechanism-study/plans/03-read-the-mechanism.md
- 📄 Spec: EXPERIMENTS.md (EXP-06), docs/results-archive/residual-diagnostics.md (commitment window)
- 📄 Parking lot (if inconclusive): PARKING_LOT.md ("Widen the mechanism study to a second pair")

## Section context (paste into the Todoist section)
**Description:** Depends on Plans 01 and 02 both being done — this task only computes and reads, it runs no new generation. Two-stage read: first the prerequisite check (does the LoRA's attention shift correlate with visual success at all), then the headline comparison (does the LoRA's shift match the Attend-and-Excite-equivalent's shift) — the pressure-test-upgraded, actually-novel claim. Metric is Δ_attn, commitment-window-restricted, using the AAE-canon aggregation already in `_CrossAttnRecorder`.
**Objective:** Serves Objective 3 (Read the Mechanism) and Definition-of-Done items 4–6 — the payload of the whole scope, where "does the LoRA understand compositionality" becomes a recorded, falsifiable verdict instead of an assertion.
**Goal:** A recorded three-way verdict (support / null / inconclusive) on whether the LoRA's attention shift resembles the Attend-and-Excite-equivalent's shift, plus the prerequisite-check verdict, both backed by the headline scatter figure (Δ_attn(LoRA) vs Δ_attn(AAE), 12 points, colored by visual success).
**Verify (whole leaf):**
``bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
cat /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/verdict.json
# expect {"prerequisite": "support|null|inconclusive", "headline": "support|null|inconclusive", "n_visual_success": <int>}
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/figures/delta_attn_scatter.png
``
**▶ Recommended prompt:** — custom; no skill fits. This is a local analysis/read task over already-captured `.pt` files (no GPU job, no queue) — `run-experiment` explicitly excludes one-off local commands with no GPU or queue involved.

## Tasks (one at a time)
- [ ] Compute Δ_attn(method, seed) for both LoRA and AAE-equivalent, commitment-window-restricted (window from `G02`), against each seed's own λ=0 baseline.
- [ ] Read the prerequisite check: does Δ_attn(LoRA) correlate with the visual composition label across the 12 seeds? Record support/null/inconclusive. If null, flag explicitly that the headline comparison below is moot rather than proceeding silently.
- [ ] Read the headline comparison against the three-way rule: support if ≥3 visually-successful seeds show Δ_attn(LoRA) matching Δ_attn(AAE) in sign and within 2x magnitude; null if the majority diverge in sign or exceed 4x; inconclusive if fewer than 3 seeds visually succeed or the AAE baseline is noisy (→ triggers the parked "Widen if Needed" item in `PARKING_LOT.md`, not a looser threshold on the same 12 seeds).
- [ ] Render the headline scatter figure + 2-3 representative per-seed three-trajectory plots (PoE/LoRA/AAE over timesteps), per EXP-06's figure list.
