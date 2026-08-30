# 🔁 Replication on other models and samplers

This plan asks whether more correction still gives more composition when the model and the
sampler are swapped for different ones.

**No step number: nothing waits on this.** Background, listed in the background-experiments pool of the [repo root MASTER_PLAN.md](../../../MASTER_PLAN.md), which also holds the one `## Running order` table. This plan asks the same result on another model and sampler. Its scaffold waits at `artifacts/plans/parked/cross-model-replication`.

## What this asks, in one line
Does more correction still give more composition on a different model and a different sampler?
This is background work: a reviewer is likely to ask for it, and the paper makes no claim that
depends on it.

## Description
Repeat the causal core cheaply, beyond the one model and one sampler it was found on.
Three pieces:

- **Other models.** Cache the corrections, then generate at every correction strength in one go,
  on SD 1.5 and SD 2.1 as well as SDXL.
- **Other samplers.** DDIM, DDPM and Euler, testing one specific thing: that the window
  sits at fixed noise levels rather than at fixed step numbers.
- **The density traces** per concept, on the runs that use a random sampler.

## Purpose
Goal 4 claims the effect is universal. If the same story holds on three models and across
samplers, the [interaction term](../../../context/world/interaction-term.md) is a property of
text-to-image diffusion in general rather than something peculiar to SDXL. Serves DoD 8.

## Goal
One compose-rate-against-λ curve per model, the window-in-SNR overlay across samplers, the SDE
density traces, and an answer to whether added noise on its own ever escapes
the blend.

## Environment Facts This Plan Depends On
- Per-model caches are new infrastructure (new pinned latents, config parity):
  jobs on biggpu first, else bigbatch, disk guard on /datasets.
- Anonymous HF pulls have sufficed for SD 1.5/2.1.
- Starts after the SDXL core (plans 03-05) lands.

## Tasks
- [ ] build the cross-model replication  → decomposed: see
      `artifacts/plans/parked/cross-model-replication/MASTER_PLAN.md`

## Environment Facts note
All further facts live in the sub-scope's own plans.

## Recommended skill
▶ `/run-experiment` ✅ inside the sub-scope's tasks.

## Engagement Instructions
```bash
# done when the sub-scope's DoD is done; spot check:
ls /datasets/mmolefe/poe_repair_min/outputs/interaction_term/replication/
# expect one compose-rate curve per model + sampler window overlay
```
