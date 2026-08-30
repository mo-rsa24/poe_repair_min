# 01 · Instrument — wire attention capture into the LoRA inference path

[⌂ Index](00-INDEX.md) · [next →](02-attend-and-excite-baseline.md)

## Reference while you do it
- 💻 Code artifacts: poe_repair/methods/_sampling.py (`_CrossAttnRecorder`, `run_lora_residual_inject`)
- 📄 Plan: plans/mechanism-study/plans/01-instrument-attention-capture.md
- 📄 Spec: report/experiments-log.md (EXP-06)

## Section context (paste into the Todoist section)
**Description:** `_CrossAttnRecorder` already exists and already works — it captured attention for the `teacher_residual` (oracle Mono−PoE) composer. It has never been wired to `run_lora_residual_inject`, the actual LoRA deployment path. This plan adds that wiring and runs capture across cat×dog's full seed set at both λ=0 (plain PoE) and λ=1 (LoRA-corrected).
**Objective:** Serves Objective 1 (Instrument) and Definition-of-Done items 1–2 — without this, there is no LoRA attention data to compare against anything; every downstream plan in this scope depends on it.
**Goal:** Attention `.pt` files exist for both λ=0 and λ=1, all 12 cat×dog seeds, same schema as the existing `veracity_attn` cache, plus a 12×50 sanity table (seed × timestep) confirming the capture pipeline behaves.
**Verify (whole leaf):**
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/plain_poe/a_cat__x__a_dog/ | wc -l   # expect 12 seed dirs
ls /datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/lora_lambda1/a_cat__x__a_dog/ | wc -l   # expect 12 seed dirs
$PY -c "import torch; sd=torch.load('<one .pt path>', weights_only=True); print(sd.keys())"   # loads, has 'map'/'spec'/'step_index'/'timestep'
```
**▶ Recommended prompt:** `/run-experiment` — GPU preflight + smoke test, then the capture runs across 12 seeds (inference-only, no training loop, but still a cluster job worth the mandatory smoke test before committing all 12 seeds).

## Tasks (one at a time)
- [ ] Run capture on plain PoE (λ=0), all 12 seeds — zero new code, `run_lora_residual_inject` already supports `disable_adapters()`. This is the cheapest possible first step and should run before touching the λ=1 wiring.
    - [ ] Prompt: reuse the existing `attn_capture_dir`/`attn_token_indices` args already threaded through `teacher_residual`'s driver code (`poe_repair/methods/_sampling.py:428-496`), calling `run_lora_residual_inject` with the adapter disabled and the same token-index spec used by `teacher_residual` (cat/dog token indices, branch mapping updated for the 3-branch {A, B, ∅} case since there's no J branch here).
    - [ ] Output: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/plain_poe/a_cat__x__a_dog/seed_<N>/`
- [ ] Build a small script that loads the 12 seeds' `.pt` files and prints/plots `A_missing(λ=0, seed, t)` as a 12×50 table — the checkpoint that confirms the capture pipeline works before any new wiring or code is written.
- [ ] Wire `attn_capture_dir`/`attn_token_indices` into `run_lora_residual_inject`'s LoRA-enabled (λ=1) forward pass — mechanical addition, wrap the `_three_branch_forward()` call in the same `_CrossAttnRecorder(unet, keep_grad=False)` context manager `teacher_residual` already uses (`_sampling.py:1351`, used at line 469).
- [ ] Run capture on the LoRA-corrected pass (λ=1, checkpoint `lora_step_062500.pt`), all 12 seeds.
    - [ ] Output: `/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism/lora_lambda1/a_cat__x__a_dog/seed_<N>/`
