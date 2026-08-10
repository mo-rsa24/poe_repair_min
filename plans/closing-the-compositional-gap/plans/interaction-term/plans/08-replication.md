# 🔁 Replication on other models and samplers

## What this asks, in one line
Does the dose result survive a different model and a different sampler? Background: a likely reviewer ask, not a claim the paper makes.

## Description
Repeat the causal core cheaply beyond SDXL and DDIM: residual caches and dose
tests on SD 1.5 and SD 2.1, a sampler sweep (DDIM, DDPM, Euler) testing that
the window sits at fixed noise levels, and the full per-concept density traces
on the stochastic-sampler runs.

## Purpose
The universality claim (Goal 4): if the same story holds on three models and
across samplers, the interaction term is a property of text-to-image
diffusion, not of SDXL. Serves DoD 8.

## Goal
One dose curve per model, the window-in-SNR overlay across samplers, the SDE
density traces, and the stochastic-rescue read (does noise alone ever escape
the blend).

## Environment Facts This Plan Depends On
- Per-model caches are new infrastructure (new pinned latents, config parity):
  jobs on biggpu first, else bigbatch, disk guard on /datasets.
- Anonymous HF pulls have sufficed for SD 1.5/2.1.
- Starts after the SDXL core (plans 03-05) lands.

## Tasks
- [ ] build the cross-model replication  → decomposed: see
      `cross-model-replication/MASTER_PLAN.md`

## Environment Facts note
All further facts live in the sub-scope's own plans.

## Recommended skill
▶ `/run-experiment` ✅ inside the sub-scope's tasks.

## Engagement Instructions
```bash
# done when the sub-scope's DoD is done; spot check:
ls /datasets/mmolefe/poe_repair_min/outputs/interaction_term/replication/
# expect per-model dose curves + sampler window overlay
```
