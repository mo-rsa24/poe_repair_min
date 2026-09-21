# 🧪 The dog × dog same-prompt check

**This plan asks one question: when both experts are given the same animal, does the LoRA leave
the picture alone, or does it add a second dog anyway?**

**Step 32 in the root running order. Waits on: nothing. Next: [03-the-correction-amount-series](03-the-correction-amount-series.md).**

## Recommended prompt (after run completes)

```
/ingest-error-pattern --from-run-log
```

---

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| 31 (previous) | [01-read-where-the-curves-flatten](../reading/01-read-where-the-curves-flatten.md) | The free curve-read framing A and B |
| **32 (current)** | **02: the-dog-x-dog-null-probe** | The same-prompt check on the trained LoRA |
| 33 (next) | [03-the-correction-amount-series](03-the-correction-amount-series.md) | Compose rate against the amount of correction; this test is its zero-interaction run |

---

## Table of contents
- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Recommended skill](#recommended-skill)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: where you are

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis:** *the LoRA learned a state-dependent rule for the [interaction term](../../../../context/world/interaction-term.md), not a plurality prior ("always add a second animal").*

**What runs:** a 4-seed by 3-column grid on the same agreeing pair, C1 = C2 = "a dog": a Mono column ("a dog and a dog", the literal joint-prompt template, no PoE, no LoRA), a PoE column (the LoRA off), and a LoRA column (the LoRA active at λ=1 over steps 0 to 10). Because the experts agree, the true correction is near zero, and language space agrees (the L1 additivity gap for an agreeing pair is near zero).

**If true (supports):** the LoRA column still shows one dog in every seed, and per-step ‖r̂‖ is small on the cross-pair scale. This reads stronger where the Mono column already shows two dogs, since it then means the LoRA is correcting a tendency the base model already has, not the reverse.

**If false (falsifies):** two dogs appear in the LoRA column. The LoRA carries a plurality prior, and the "learned a rule" caption dies as stated.

**Inconclusive:** one dog with large ‖r̂‖: the output is right but the mechanism read is not settled.

**The identity check, before reading anything:** verify PoE(A,A) reduces to Mono(A) within fp16 drift, copying the `run_cfg_masked(all_off)` identity pattern. Without this identity nothing the run produces can be interpreted.

**Associated materials:**
- **Review questions:** [the review file](../../review/02-the-dog-x-dog-same-prompt-check.md)
- **The ledger entry:** [decisions-taken-here.md § The dog x dog test](../../decisions-taken-here.md#the-dog-x-dog-test-pre-registered)
- **Checkpoint:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` (rank 8, alpha 8, attn2 q/k/v)

---

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Expected runtime:** a handful of renders (one identity-check render, then the 4-seed by 3-column grid, 12 renders): under an hour on one device at 50 DDIM steps. Measure it and record the wall time in the review file.

**Prerequisites:** the checkpoint above; the training cache's seeds; the validated instance-count scorer (`scorer_validated.json`).

**GPU:** in-session on the node's free device is enough; no Slurm needed at this size, per [execution-protocol](../../../../environment/hpc/execution-protocol.md).

**W&B project:** `prime_lab/poe-repair-animals-compose`.

**Guidance is never raised to buy crispness**, since raising it pushes the sample off the data manifold, per [the ledger](../../decisions-taken-here.md#the-dog-x-dog-test-pre-registered).

---

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

- The `co3` python; large outputs to `/datasets` only; the disk guard checks the filesystem written to ([environment/overview.md](../../../../environment/overview.md)).
- fp16 renders are mode-reproducible, not byte-reproducible; the identity check is judged within fp16 drift, not equality.

---

## The claim

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**A judged same-prompt check: the LoRA on an agreeing pair either leaves the image alone (rule) or invents plurality (prior), decided against the pre-registered outcomes.** It matters now because "the LoRA learned a rule, not a stored vector" is the showcase's central sentence and this is the cheapest way to try to break it.

---

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** Every showcase figure leans on the claim that the LoRA computes a state-dependent correction. A model that merely learned "add a second animal" would pass most held-out runs and the claim would be wrong.

> Held-out means the pairs were never shown during training, so the number says how well the
> adapter does on animals it has not seen.

**The solution.** Feed it a pair whose true correction is near zero and watch what it adds, against a Mono baseline that shows whether a redundant prompt already pulls the base model toward two subjects on its own.

**Key insight.** This doubles as the zero-interaction control in the run across correction amounts next door; the two share one runner (ledger).

---

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

**Gate.** LoRA off, PoE("a dog","a dog") must equal Mono("a dog") within fp16 drift. If it does not, the runner is wrong and nothing below is judged.

**Render.** Once the gate passes, the same agreeing pair is rendered as a grid across the held-out seeds: Mono ("a dog and a dog"), PoE with the LoRA off, and PoE with the LoRA on at λ=1 over steps 0 to 10.

**Score.** Every cell gets an instance count. The LoRA column also gets a per-step ‖r̂‖ reading.

**Verdict.** The LoRA column shows one dog in every seed, with small ‖r̂‖: supports the rule reading. Any LoRA-column render with two dogs: falsifies it as stated. One dog with a large ‖r̂‖: inconclusive.

The pending subject-lane diagram in the [Figure Catalog](#figure-catalog) draws this: the two agreeing expert cards, the near-zero correction the adapter should add, and the green/red outcome split.

---

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The identity check**: PoE("a dog","a dog") with the LoRA off equals Mono("a dog") within fp16 drift, one render, copying `run_cfg_masked(all_off)` from the window machinery (`scripts/interaction_term_window.py`). Gates everything below; nothing downstream is judged if this fails.
2. **The dog × dog grid**: four held-out seeds by three columns. Mono column: the literal joint prompt "a dog and a dog", no PoE, no LoRA. PoE column: PoE("a dog","a dog") with the LoRA off. LoRA column: the same pair with the LoRA on, λ=1, window 0-10. Twelve renders total; per-step ‖r̂‖ logged for the LoRA column only, since ‖r̂‖ is the correction the LoRA adds.
3. **The verdict inputs**: instance count per grid cell, all three columns and all four seeds, plus the LoRA column's ‖r̂‖ per step against the cross-pair scale (the norms logged in `history.json`, `train/delta_target_norm` ≈ 29.7 as the reference scale).

---

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Serves master-plan objective 2 and goal 2. Checkable outcomes:

1. The identity check holds and is recorded.
2. Every cell of the 4-seed by 3-column grid is rendered and scored; the LoRA column's ‖r̂‖ curve saved with a sidecar.
3. The review file's pre-registered questions are answered.

---

## Tasks

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 0. ✅ Verify this plan first

- [ ] **0.1 Run the following prompt: `/verify-plan plans/01-showcase-the-trained-lora/plans/tests/02-the-dog-x-dog-same-prompt-check.md`**

▶ **Next: [task 1.1](#1--build-and-run-the-test)**.

### 1. 🧪 Build and run the test

◀ **Needs: nothing from other plans** — checkpoint and cache exist.

- [ ] **1.1 Write the probe runner** extending the existing injection runner (`scripts/interaction_term_inject.py` pattern): a `--mode {mono,poe,lora}` flag selecting the literal joint prompt ("a dog and a dog"), PoE with the LoRA off, or PoE with the LoRA on; λ and window as flags for `lora` mode; per-step ‖r̂‖ dumped to the sidecar for `lora` mode only, since ‖r̂‖ is the correction the LoRA adds.
  - Location: `scripts/showcase/dog_x_dog_probe.py`
- [ ] **1.2 Run the identity preflight** (LoRA off, one seed). Completion is observable: the printed max-abs pixel difference between PoE(A,A) and Mono(A), and it sits within the fp16 drift band recorded for the λ=0 check that must pass before anything below runs.
- [ ] **1.3 Render the Mono column**: "a dog and a dog", one render per held-out seed, no PoE, no LoRA. Output to `/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/mono/`.
- [ ] **1.4 Render the PoE column**: PoE("a dog","a dog") with the LoRA off, one render per held-out seed (the same seeds as 1.3; this extends the single identity-check render in 1.2 to the full seed set). Output to `.../dog_x_dog_probe/poe/`.
- [ ] **1.5 Render the LoRA column** (LoRA on, λ=1, window 0-10, existing held-out seeds). Output to `.../dog_x_dog_probe/lora/`.
- [ ] **1.6 Score every cell** with the validated scorer; write `probe_scores.json` (per render: mode, seed, `n_instances`; per-step ‖r̂‖ for the LoRA column).

▶ **Next: [instruction 2.1](#2--judge-the-renders)** (eyeball the grid and record the verdict).

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#what-has-to-pass-before-this-runs) ➡️

### 2. 👁️ Judge the renders

◀ **Needs: [tasks 1.1 to 1.6](#1--build-and-run-the-test)** done, so the full grid and scores exist.

2.1 **Open the grid** under `/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/` (file browser or `feh`/VS Code over SSH), four seeds by three columns (Mono, PoE, LoRA). Per seed: ✅ LoRA column shows one dog = supports; ❌ LoRA column shows two dogs = falsifies as stated; 🟡 one dog but the ‖r̂‖ curve is not small on the 29.7 scale = inconclusive. Also note whether the Mono column already shows two dogs, since that changes what a passing LoRA column means (see [Quick context](#quick-context-where-you-are)).

2.2 **Write the verdict** into the [review file](../../review/02-the-dog-x-dog-same-prompt-check.md) with the counts per column and the ‖r̂‖ summary.

▶ **Next: what has to pass before this runs.**

---

## What has to pass before this runs

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> The "learned a rule" caption holds the showcase together. A falsified result does not stop the scope; it rewrites what the figures may say, which is cheaper now than after assembly.

**Pass criteria:**
- The identity check comes out within fp16 drift, recorded.
- Every cell of the 4-seed by 3-column grid rendered and scored; the review file's threshold question answered.

**Fail criteria:**
- The identity check fails: stop; the runner is wrong, and no render means anything.

**When you get results, answer the open questions in the [review file](../../review/02-the-dog-x-dog-same-prompt-check.md).**

---

## Figure Catalog

⬅️ [Previous](#what-has-to-pass-before-this-runs) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

### Pending

| Figure | Lane | Prompt file | What it shows | Save to |
|--------|------|-------------|----------------|---------|
| the same-prompt check | subject | [diagram-prompts.md#prompt-2-subject-the-same-prompt-check-planned](../../diagram-prompts.md#prompt-2-subject-the-same-prompt-check-planned) | two agreeing expert cards, the adapter's near-zero correction, and the rule-vs-plurality outcome split | `diagrams/the-same-prompt-check.png` |

### Generated during plan execution

| Figure | Lane | Description | Generated by | Status |
|--------|------|-------------|--------------|--------|
| the dog × dog grid | — | four seeds by three columns (Mono: "a dog and a dog"; PoE: LoRA off; LoRA: λ=1) + the LoRA column's ‖r̂‖-per-step curve beside it | tasks 1.3 to 1.6 | ⏳ |

### Organization workflow

1. Render the pending subject-lane piece via `/render-diagrams` once the verdict is recorded, so the outcome split it draws matches the real result.
2. Run the plan's Tasks and Instructions.
3. Confirm the render grid and `probe_scores.json` land at the paths tasks 1.3 to 1.6 name.
4. Link the rendered diagram and the grid here once both exist.

---

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Extract errors | `/ingest-error-pattern --from-run-log` | task 3.1 | new patterns into the catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | auto | this file's matrix regenerated |
| Close out | `/sync-plan-tree` | task 3.2 | statuses aggregated up |

### 3. 🧹 Close out

◀ **Needs: [instruction 2.2](#2--judge-the-renders)** done.

- [ ] **3.1 Run the following prompt: `/ingest-error-pattern --from-run-log`** (only after a red run).
- [ ] **3.2 Run the following prompt: `/sync-plan-tree plans/01-showcase-the-trained-lora/`**

▶ **Next: [03-the-correction-amount-series](03-the-correction-amount-series.md).**

---

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#recommended-skill) ➡️

**File:** `scripts/interaction_term_window.py` — the `run_cfg_masked(all_off)` identity pattern the identity check copies.
**File:** `artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/lora_attach.json` — the attach targets (attn2 to_q/to_k/to_v) the runner must reproduce.

---

## Recommended skill

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

▶ Paste to run this plan (in-session GPU; the threshold is pre-registered):

```
/run-experiment plans/01-showcase-the-trained-lora/plans/tests/02-the-dog-x-dog-same-prompt-check.md — identity check PoE(A,A) vs Mono(A) first; stop and report if it fails.
```

---

## Next step

⬅️ [Previous](#recommended-skill) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[03-the-correction-amount-series](03-the-correction-amount-series.md): compose rate against the amount of correction on r̂, with this test as its zero-interaction run.

---

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

### From global catalog

(empty until `/ingest-error-pattern` populates)

### From project catalog

(empty until `/ingest-error-pattern` populates)
