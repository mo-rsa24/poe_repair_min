# 02 · Baseline-Compare — build the Attend-and-Excite-equivalent intervention

[⌂ Index](00-INDEX.md) · [← prev](01-instrument-attention-capture.md) · [next →](03-read-the-mechanism.md)

## Reference while you do it
- 💻 Code artifacts: poe_repair/methods/_sampling.py (`_CrossAttnRecorder` with `keep_grad=True`)
- 📄 Plan: plans/mechanism-study/plans/02-attend-and-excite-baseline.md
- 📄 Spec: EXPERIMENTS.md (EXP-06), residual-diagnostics.md (commitment window)

## Section context (paste into the Todoist section)
**Description:** No test-time attention-optimization baseline exists in this repo yet. `_CrossAttnRecorder` already supports `keep_grad=True` (built for FOCUS's velocity correction), so the gradient path is not new — only the intervention loop is. At the commitment-window steps, compute `L = Σ_tokens max(0, 1 − max_spatial_attn(token))` on plain PoE's attention, backprop through the latent, take 1-2 gradient steps per intervention step. Fixed step count, not swept — this is a comparison baseline, not the paper's method.
**Objective:** Serves Objective 2 (Baseline-Compare) and Definition-of-Done item 3 — this is the independent comparison point Plan 03's headline read depends on; per the pressure-test verdict, that comparison (not the LoRA's shift in isolation) is the actual novel claim.
**Goal:** Attention `.pt` files exist for the AAE-equivalent intervention, all 12 cat×dog seeds, same schema as Plan 01's output, confirmed with the same 12×50 sanity-table format.
**Verify (whole leaf):**
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/aae_equiv/a_cat__x__a_dog/ | wc -l   # expect 12 seed dirs
# spot-check one seed's decoded image shows visible intervention (attention boosted vs plain PoE)
```
**▶ Recommended prompt:** `/run-experiment` — same shape as Plan 01: inference-scale compute with a gradient-step intervention, no training loop, but still worth the GPU preflight/smoke test given the backward-through-latent path is new code. (No existing skill authors the intervention loop itself — that part is custom, project-specific method code.)

## Tasks (one at a time)
- [ ] Implement the intervention loop (~80-120 lines): reuse `_CrossAttnRecorder(unet, keep_grad=True)` on plain PoE (no LoRA adapter active), apply at the commitment-window steps identified in `G02`, 1-2 gradient steps per step, decode + re-encode the updated latent back into the sampling loop.
- [ ] Run the intervention on all 12 cat×dog seeds, capturing attention the same way as Plan 01 (same `aggregate_token_map` call, same 16×16 `agg_resolution`), plus the resulting decoded image for the visual composition label.
    - [ ] Output: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/aae_equiv/a_cat__x__a_dog/seed_<N>/`
- [ ] Run the same 12×50 sanity table from Plan 01 against this output, confirming the AAE-equivalent's capture pipeline behaves identically before Plan 03 reads it.
