# 🧪 The dog × dog null probe

**Step 32 in the root running order. Waits on: nothing. Next: [03-the-lora-dose-sweep](03-the-lora-dose-sweep.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 31 (previous) | [01-read-the-plateau-curves](01-read-the-plateau-curves.md) | The free curve-read framing A and B |
| **32 (current)** | **02: the-dog-x-dog-null-probe** | The null-input control on the trained LoRA |
| 33 (next) | [03-the-lora-dose-sweep](03-the-lora-dose-sweep.md) | The causal dose curve; this probe is its zero cell |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis:** *the LoRA learned a state-dependent rule for the interaction term, not a plurality prior ("always add a second animal").*

**The probe:** run PoE with both experts set to the same concept, C1 = C2 = "a dog", with the LoRA active at λ=1 over steps 0 to 10, on existing seeds. Because the experts agree, the true correction is near zero, and language space agrees (the L1 additivity gap for an agreeing pair is near zero).

**If true (supports):** the corrected run still shows one dog, and per-step ‖r̂‖ is small on the cross-pair scale.

**If false (falsifies):** two dogs appear. The LoRA carries a plurality prior, and the "learned a rule" caption dies as stated.

**Inconclusive:** one dog with large ‖r̂‖: the output is right but the mechanism read is not settled.

**Preflight, before reading anything:** verify PoE(A,A) reduces to Mono(A) within fp16 drift, copying the `run_cfg_masked(all_off)` identity pattern. Without this identity the probe is not interpretable.

**Associated materials:**
- **Review questions:** [../review/02-the-dog-x-dog-null-probe.md](../review/02-the-dog-x-dog-null-probe.md)
- **The ledger entry:** [decisions-taken-here.md § The dog x dog probe](../decisions-taken-here.md#the-dog-x-dog-probe-pre-registered)
- **Checkpoint:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` (rank 8, alpha 8, attn2 q/k/v)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** a handful of renders (one identity preflight cell, then the probe over the existing seeds): under an hour on one device at 50 DDIM steps. Measure it and record the wall time in the review file.

**Prerequisites:** the checkpoint above; the training cache's seeds; the validated instance-count scorer (`scorer_validated.json`).

**GPU:** in-session on the node's free device is enough; no Slurm needed at this size, per [execution-protocol](../../../../../environment/hpc/execution-protocol.md).

**W&B project:** `prime_lab/poe-repair-animals-compose`.

**Guidance is never raised to buy crispness** (off-manifold trap; ledger).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; large outputs to `/datasets` only; the disk guard checks the filesystem written to ([environment/overview.md](../../../../../environment/overview.md)).
- fp16 renders are mode-reproducible, not byte-reproducible; the identity preflight is judged within fp16 drift, not equality.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A judged null-input control: the LoRA on an agreeing pair either leaves the image alone (rule) or invents plurality (prior), decided against the pre-registered outcomes.** It matters now because the "learned a rule, not a vector" caption is the showcase's central sentence and this is its cheapest falsification attempt.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**The problem.** Every showcase figure leans on the claim that the LoRA computes a state-dependent correction. A model that merely learned "add a second animal" would pass most held-out cells and the claim would be wrong.

**The solution.** Feed it a pair whose true correction is near zero and watch what it adds.

**Key insight.** The probe doubles as the dose sweep's zero-interaction control cell; the two share one harness (ledger).

---

## Description: what to build

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The identity preflight**: PoE("a dog","a dog") with the LoRA off equals Mono("a dog") within fp16 drift, one cell, copying `run_cfg_masked(all_off)` from the window machinery (`scripts/interaction_term_window.py`).
2. **The probe cells**: the same pair with the LoRA on, λ=1, window 0-10, over the existing held-out seeds; per-step ‖r̂‖ logged; renders scored with the instance-count scorer.
3. **The verdict inputs**: instance count per render, and ‖r̂‖ per step against the cross-pair scale (the norms logged in `history.json`, `train/delta_target_norm` ≈ 29.7 as the reference scale).

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves master-plan objective 2 and goal 2. Checkable outcomes:

1. The preflight identity holds and is recorded.
2. Every probe cell is rendered, scored, and its ‖r̂‖ curve saved with a sidecar.
3. The review file's pre-registered questions are answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/02-the-dog-x-dog-null-probe.md`**

▶ **Next: [task 1.1](#1--build-and-run-the-probe)**.

### 1. 🧪 Build and run the probe

◀ **Needs: nothing from other plans** — checkpoint and cache exist.

- [ ] **1.1 Write the probe runner** extending the existing injection harness (`scripts/interaction_term_inject.py` pattern): same-concept experts, LoRA attach per `lora_attach.json`, λ and window as flags, per-step ‖r̂‖ dumped to the sidecar.
  - Location: `scripts/showcase/dog_x_dog_probe.py`
- [ ] **1.2 Run the identity preflight** (LoRA off). Completion is observable: the printed max-abs pixel difference between PoE(A,A) and Mono(A), and it sits within the fp16 drift band recorded for the λ-0 canary.
- [ ] **1.3 Run the probe cells** (LoRA on, λ=1, window 0-10, existing held-out seeds). Output to `/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/`.
- [ ] **1.4 Score the renders** with the validated scorer; write `probe_scores.json` (per cell: `n_instances`, per-step ‖r̂‖, seed).

▶ **Next: [instruction 2.1](#2--judge-the-probe)** (eyeball the renders and record the verdict).

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

### 2. 👁️ Judge the probe

◀ **Needs: [tasks 1.1 to 1.4](#1--build-and-run-the-probe)** done, so renders and scores exist.

2.1 **Open the probe renders** under `/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/` (file browser or `feh`/VS Code over SSH). Per seed: ✅ one dog = supports; ❌ two dogs in any cell = falsifies as stated; 🟡 one dog but the ‖r̂‖ curve is not small on the 29.7 scale = inconclusive.

2.2 **Write the verdict** into the [review file](../review/02-the-dog-x-dog-null-probe.md) with the counts and the ‖r̂‖ summary.

▶ **Next: the engagement gate.**

---

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> The "learned a rule" caption is the showcase's spine. A falsified probe does not stop the scope; it rewrites what the figures may say, which is cheaper now than after assembly.

**Pass criteria:**
- Preflight identity within fp16 drift, recorded.
- Every probe cell rendered and scored; the review file's bar answered.

**Fail criteria:**
- Preflight identity fails: stop; the harness is wrong, and no probe render means anything.

**When you get results, answer the open questions in the [review file](../review/02-the-dog-x-dog-null-probe.md).**

---

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | What it shows | Save to |
|--------|------|---------------|---------|
| (none pre-drawn) | — | — | — |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| probe strip | — | one row of renders (per seed) + the ‖r̂‖-per-step curve beside them | task 1.3/1.4 | ⏳ |

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.2](#2--judge-the-probe)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (only after a red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/`**

▶ **Next: [03-the-lora-dose-sweep](03-the-lora-dose-sweep.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/interaction_term_window.py` — the `run_cfg_masked(all_off)` identity pattern the preflight copies.
**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/lora_attach.json` — the attach targets (attn2 to_q/to_k/to_v) the runner must reproduce.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (in-session GPU; the probe's bar is pre-registered):

```
/run-experiment plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/plans/02-the-dog-x-dog-null-probe.md — identity preflight PoE(A,A) vs Mono(A) first; stop and report if it fails.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[03-the-lora-dose-sweep](03-the-lora-dose-sweep.md): the causal dose curve on r̂, with this probe as its zero-interaction cell.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
