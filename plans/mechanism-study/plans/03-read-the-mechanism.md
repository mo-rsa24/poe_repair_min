# 📊 Read the Mechanism — LoRA vs Attend-and-Excite attention comparison

## Description
Depends on Plans 01 and 02 both being done — this plan only computes and reads, it runs no
new generation. Two-stage read, per the master plan's Goal 3: first the prerequisite check
(does the LoRA's attention shift correlate with visual success at all, the original
pre-pressure-test hypothesis), then the headline comparison (does the LoRA's shift match the
AAE-equivalent's shift), which is the pressure-test-upgraded, actually-novel claim. Metric is
`Δ_attn`, commitment-window-restricted, using the same AAE-canon aggregation already in
`_CrossAttnRecorder` (query_len ≤ 32², resized to 16×16) — not a fresh design choice, adopting
what plans 01/02 already produced.

## Purpose
Serves Objective 3 (Read the Mechanism) and Definition-of-Done items 4–6. This is the payload
of the whole scope — the point where "does the LoRA understand compositionality" becomes a
recorded, falsifiable verdict instead of an assertion.

## Goal
A recorded three-way verdict (support / null / inconclusive) on whether the LoRA's attention
shift resembles the Attend-and-Excite-equivalent's shift, plus the prerequisite-check verdict,
both backed by the headline scatter figure (Δ_attn(LoRA) vs Δ_attn(AAE), 12 points, colored by
visual success).

## Tasks
- [ ] ⚠️ **[publishable-bar]** Compute Δ_attn(method, seed) for both LoRA and AAE-equivalent,
  commitment-window-restricted (window from `G02`), against each seed's own λ=0 baseline.
- [ ] ⚠️ **[publishable-bar]** Read the prerequisite check: does Δ_attn(LoRA) correlate with
  the visual composition label across the 12 seeds? Record support/null/inconclusive. If null,
  flag explicitly that the headline comparison below is moot (no attention signature to
  compare) rather than proceeding silently.
- [ ] ⚠️ **[publishable-bar]** Read the headline comparison against the three-way rule:
  support if ≥3 visually-successful seeds show Δ_attn(LoRA) matching Δ_attn(AAE) in sign and
  within 2x magnitude; null if the majority diverge in sign or exceed 4x; inconclusive if
  fewer than 3 seeds visually succeed or the AAE baseline is noisy (→ triggers the parked
  "Widen if Needed" item in `PARKING_LOT.md`, not a looser threshold on the same 12 seeds).
- [ ] ⚠️ **[publishable-bar]** Render the headline scatter figure + 2-3 representative
  per-seed three-trajectory plots (PoE/LoRA/AAE over timesteps), per EXP-06's figure list.

## Recommended skill
— custom; no skill fits. This is a local analysis/read task over already-captured `.pt`
  files (no GPU job, no queue) — `run-experiment` explicitly excludes "one-off local commands
  with no GPU or queue involved."

## Engagement Instructions
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
# after the read script runs, expect a verdict.json with {"prerequisite": "support|null|inconclusive",
#   "headline": "support|null|inconclusive", "n_visual_success": <int>}
cat /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/verdict.json
# and the scatter figure:
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/figures/delta_attn_scatter.png
```
