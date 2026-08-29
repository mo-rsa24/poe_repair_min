# 🧪 The free bound on the model's share

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md — <one line on what the cached numbers said>
```

## Recommended skill

▶ `/analyze-run the correction size as the run approaches zero noise, from paper/iclr/figures/correction-size-over-the-denoising-run.json` ✅ reads a finished result and writes the verdict against a pre-registered bar, which is exactly this plan's shape.
   alt: `/eda-synthesize` if the two JSON files turn out to need reshaping before they can be read together.

## Position in the plan tree

**Step 24 of 30.** Waits on nothing. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 6 | [hypothesis-03: when-in-the-run-it-matters](../../03-does-the-correction-cause-composition/plans/hypothesis-03-when-in-the-run-it-matters.md) ◑ | found the cliff at steps 0 to 10, which is also where a sampler artifact would live |
| **24 (current)** | **hypothesis-01: the-free-bound-on-the-models-share** ⚠️ | **reads the correction's size as the run reaches zero noise off files already on disk, and turns it into a floor under the model's share** |
| 25 | [instrument-01: the-corrector-and-the-step-size-it-runs-at](instrument-01-the-corrector-and-the-step-size-it-runs-at.md) ⚠️ | builds the corrector this bound caps the value of |
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](hypothesis-02-what-is-left-once-the-chain-settles.md) ⚠️ | the gate, which reads its own result against this floor |

Design only. Verdicts and run state live in
[the paired review file](../review/hypothesis-01-the-free-bound-on-the-models-share.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to build](#description-what-to-build)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Is the correction still bounded away from zero at the last denoising step, in the numbers already
sitting on disk? The [sampler's share of the error](/home-mscluster/mmolefe/goal-setting/learning/sampler-correctors-for-composition/plans/15-the-two-gaps.md) is defined to vanish as the noise goes to zero,
and the model's share is not. So anything left at the end of the run is already a floor under the
model's share, bought with no GPU at all.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The hypothesis.** The correction `r_t = eps_J - eps_PoE` does not fall to zero as the run
approaches zero noise. What is left there is the model's disagreement with the product, not the
sampler's.

**If true.** The model's share has a floor before any corrector is built, and the whole `k` grid at
step 26 becomes a sizing exercise on a quantity already known to be nonzero.

**If false.** A size near zero at the last step, honestly measured, is the first evidence that the
correction is mostly the sampler's, and it raises the stakes on everything downstream rather than
lowering them.

**Rationale.** Summing two diffused scores gives the score of the product of two diffused
marginals. At `t = 0` that product is exactly the product of experts the method asked for, so the
sampler's contribution to `r_t` is zero there by construction. The remainder is the model.

**What this reads.** Two files that already exist. `correction-size-over-the-denoising-run.json`
under `paper/iclr/figures/`, which carries 50 steps over 3 seeds and two pairs. And
`step_collapse.json` beside `snr_collapse.json` under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`, whose `peak_at_edge`
field speaks to how concentrated the correction is at high noise.

**Associated materials.** [The review questions](../review/hypothesis-01-the-free-bound-on-the-models-share.md),
[the whole corrector design](../source/the-whole-corrector-design.md) this plan came out of, and
[the scope's direction](../MASTER_PLAN.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**Expected runtime: minutes, and no GPU.**

This plan reads files. It is the one thing in the scope that costs nothing, which is why it runs
first and why its answer is allowed to change what the rest is worth.

**The cached measure is a ratio, and this question needs a numerator.**

What is on disk is `‖r_t‖/‖eps_PoE‖`, computed at
[correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) line 101. The
question here is about the unnormalised `‖r_t‖`. Either recover it or say plainly that it is
unrecoverable from what was saved. A ratio falling because its denominator grew says nothing about
the model's share.

**Which trajectory the numbers were cached along decides what they can mean.**

At λ=0 the cached path is the plain product-of-experts path, so the answer is about that path and
no other. Say which path in the answer or the number is uninterpretable.

**Both arms must have been evaluated at the same latent at each step.**

If they were not, the late-step values carry accumulated path difference as well as rule
difference, and they cannot be read at all. This is a property of how the cache was written, not
something this plan can fix.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**The correction's size at the last denoising step, on the uncorrected path, is a floor under the
model's share of the error, and it can be read today from files already on disk.**

**Independent variable.** The denoising step `t`, read at the low-noise end of the run. Nothing is
varied and nothing is run.

**Dependent variables.** The correction's size at the last steps, in whichever form the cache
actually supports: the unnormalised `‖r_t‖₂` if it is recoverable, and the ratio
`‖r_t‖/‖eps_PoE‖` beside it either way, with the denominator reported separately.

**Falsify condition.** This is a read, not a run, so the bar is on the honesty of the answer rather
than on its value. The answer counts only if it states four things: the number with its unit, the
seeds it came from, the trajectory it was cached along, and the unnormalised `‖r_t‖` beside the
ratio. An answer missing any of the four is not an answer and the question stays open.

**Why this matters right now.** It costs nothing and it caps everything below it. A large floor
means the model's share is already big and the `k` grid at step 26 is measuring the size of
something known to exist. A floor near zero makes the corrector work more urgent, not less.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The scope is about to spend roughly 670 plain-render equivalents of GPU time
separating two errors, and nobody has checked what the cached numbers already say about one of
them.

**The approach.** Read the low-noise end of a curve that already exists. The sampler's share is
zero there by construction, so the value is a lower bound on the model's share with no new
sampling and no new code.

**Key insights.**

1. A bound that costs nothing is worth having even when it is loose, because it is the only number
   in this scope available before anything is built.
2. The three caveats attached to the read (ratio against numerator, which trajectory, same latent
   or not) are what make the difference between a bound and a plausible-looking number. They are
   pre-registered here so they cannot be skipped once a number is on screen.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  correction size
  |
  |  \
  |   \___                 the cached curve, already on disk
  |       \____
  |            \_______
  |                    \________________
  |  - - - - - - - - - - - - - - - - - -  <- if it lands here, above zero,
  |                                          that height is a floor under
  +--------------------------------------    the MODEL's share
     step 0                        step 49
     high noise                    low noise
     sampler + model               sampler is zero by construction
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

Nothing is built. Three things are read and one thing is written.

1. **The low-noise end of the cached size curve.** From
   `paper/iclr/figures/correction-size-over-the-denoising-run.json`, 50 steps, 3 seeds, two pairs.
   Read the last steps rather than the whole curve.
2. **The high-noise concentration figure.** From
   `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/step_collapse.json`
   and `snr_collapse.json` beside it, whose `peak_at_edge` field says how concentrated the
   correction is at the noisy end. This is context for the number, not the number.
3. **Whether the unnormalised numerator survives in the cache.** Read
   [correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) at line 101,
   where the ratio is formed, and follow what was written out against what was discarded.

The output is prose plus numbers in
[the review file](../review/hypothesis-01-the-free-bound-on-the-models-share.md). No figure, no
new file under `/datasets`, no code.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 1 of [the scope's direction](../MASTER_PLAN.md): bound the model's share for free,
off cached files, before any corrector exists, so everything downstream is capped before it is paid
for.

**Goals**

1. A number with its unit and its meaning for the correction's size at the last denoising step.
2. A stated answer to whether the unnormalised numerator is recoverable from the cache, or a stated
   admission that it is not.
3. One line in the review file saying what the number does to the rest of the scope.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- **Every command below runs from the repo root**,
  `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`, because the paths under
  `paper/iclr/figures/` and `scripts/` are relative to it. A relative path run from anywhere else
  resolves against `$HOME` and fails silently, which is catalogued here as `poe-launch-001` in
  [environment/known-failures.md](../../../environment/known-failures.md).
- The cache lives under `/datasets/mmolefe/poe_repair_min/outputs/`, not under `/home-mscluster`,
  per [environment/storage.md](../../../environment/storage.md). This plan only reads, so no disk
  guard is needed and nothing new is written there.
- The cached predictions are fp16. Any norm recomputed here upcasts to fp32 first, since the
  differences being read are small enough that fp16 accumulation shows up in the third digit.
- No GPU and no queue. This runs in-session and needs neither Slurm nor a `nohup` launch.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Re-read [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md).
      Confirm the three numbers this scope quotes are still what it says. They are 0.656 at steps 0
      to 10, 0.000 from steps 20 to 30 onward, and the correction about 2.7 times larger late than
      early.
  - Done when: each of the three is found in that file, or the difference is written into
    [the review file](../review/hypothesis-01-the-free-bound-on-the-models-share.md). If any has
    moved, the threat this scope answers has changed shape and the claim needs rewriting before
    anything else runs.

▶ **Next: [task 1.1](#1--read-the-cached-size-at-the-low-noise-end)**, the read that costs no GPU.

### 1. 📊 Read the cached size at the low-noise end

◀ **Needs: [task 0.2](#0--preflight-check-this-plan-before-working-from-it)**, so the numbers this
plan is read against are known to be current.

- [ ] **1.1** Read the per-step correction size as the run approaches zero noise.

    ```bash
    cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    CA=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses
    $PY -c "import json;d=json.load(open('paper/iclr/figures/correction-size-over-the-denoising-run.json'));print(json.dumps(d,indent=2)[:4000])"
    $PY -c "import json;d=json.load(open('$CA/step_collapse.json'));print(list(d)[:20])"
    $PY -c "import json;d=json.load(open('$CA/snr_collapse.json'));print({k:v for k,v in list(d.items())[:5]})"
    ```

  - Produces: the last-step values per seed and per pair, plus the `peak_at_edge` field from the
    collapse analyses.
  - **The seeds here are 4, 42 and 123**, on `a_cat__x__a_dog` and
    `a_butterfly__x__a_flower_meadow`. [The gate at step 26](hypothesis-02-what-is-left-once-the-chain-settles.md)
    runs seed 9 on the same two pairs, so reading its remainder against this floor is a cross-seed
    comparison. Say so when the two numbers are put beside each other.
  - **Done when:** the review file names the last-step value for each of the two pairs, with its
    unit and its seeds, not when the commands have run.
- [ ] **1.2** State the three things that make the read worth having, in the review file.
  - The cached measure is the ratio `‖r_t‖/‖eps_PoE‖` and this question needs the unnormalised
    numerator. **It is recoverable, and here is where.** The JSON stores only the ratio, but
    `correction_size_over_the_run.py` forms that ratio at line 101 from raw tensors still on disk:
    `delta` and `eps_poe` in 50 per-step `.pt` files at
    `/datasets/mmolefe/poe_repair_min/outputs/veracity/pairs/<pair>/seed_<n>/teacher_residual_const_lam000/residuals/step_NNN.pt`,
    for both pairs at all three seeds. Read `‖delta‖` from those and report it beside the ratio.
  - Which trajectory the residuals were cached along. At λ=0 that is the plain product-of-experts
    path and the answer is about that path only.
  - Whether both arms were evaluated at the same latent at each step. If they were not, the
    late-step values carry accumulated path difference as well as rule difference and cannot be
    read at all.
  - **Done when:** all three appear as sentences in the review file, each with an answer rather
    than a hedge.
  - 💡 `/plain-speak --gloss` on the finished paragraph if the three caveats end up reading as
    jargon; they have to survive a cold read by someone judging the number months later.
- [ ] **1.3** Write the floor into the review file as a number with its unit and its meaning, and
      say what it does to the claim.
  - A large floor means the model's share is already big and the `k` grid at step 26 is a sizing
    exercise. A floor near zero, honestly measured, is the first evidence for the sampler's share
    and raises the stakes on everything below.
  - **Done when:** the first pre-registered question in the review file is ticked with the number
    that answered it beside it.

▶ **Next: [instruction 2.1](#2--judge-whether-the-number-is-actually-a-bound)**, the eye read that
decides whether the number is a bound or an artefact of how the cache was written.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.

▶ **Next: [the engagement gate](#the-engagement-gate).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 2. 👁️ Judge whether the number is actually a bound

◀ **Needs: [task 1.2](#1--read-the-cached-size-at-the-low-noise-end)**, the three stated caveats.

- [ ] **2.1** Open the review file and read the three caveat sentences.
  - Expected result: each says what is true, not what is likely.
  - ✅ If all three are answered, the number is a bound and the rest of the scope reads against it.
  - ❌ If the "same latent at each step" answer is no, or unknown, the late-step values are not a
    bound at all. Record that, and mark the first pre-registered question 🟡 rather than ✅: the
    run finished and the question still cannot be answered.
- [ ] **2.2** Decide in one line whether this changes the `k` grid at step 26.
  - A floor above roughly a fifth of the `k=0` value makes the grid a sizing exercise on a known
    quantity. A floor near zero makes the grid the whole argument.
  - Record your line in the review file's `## Still open` or against the bar, whichever it belongs
    to. This is a judgement the code cannot make.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 25, the corrector and the step size it runs at](instrument-01-the-corrector-and-the-step-size-it-runs-at.md),
which builds the corrector this bound caps the value of.

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** every plan after this one spends GPU time. This is the only one
> that does not, and its answer is what says how much that time is worth.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
CA=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses
RES=/datasets/mmolefe/poe_repair_min/outputs/veracity/pairs

# The three files this plan reads. All must exist before anything is claimed.
ls -l paper/iclr/figures/correction-size-over-the-denoising-run.json
ls -l "$CA/step_collapse.json" "$CA/snr_collapse.json"

# Where the ratio is formed, which is what task 1.2 unpicks.
sed -n '95,110p' scripts/correction_size_over_the_run.py

# The raw tensors the unnormalised numerator comes from: 50 per seed, 6 pair-seeds.
ls "$RES/a_cat__x__a_dog/seed_42/teacher_residual_const_lam000/residuals" | wc -l   # expect 50
```

**Pass criteria**

- The last-step correction size is written into the review file with its unit, for both pairs.
- All three caveats are answered rather than hedged.
- One line says what the number does to the rest of the scope.

**Fail criteria (STOP)**

- Any of the three files is missing or unreadable. Nothing downstream is blocked by this, but the
  bound is then unavailable and step 26 has to be read without it, which the review file records.

**Partial pass guidance**

- The unnormalised numerator being unrecoverable is a partial pass, not a failure. Record the ratio
  with its denominator beside it and say the numerator is gone. What is not acceptable is reporting
  the ratio as if it were the numerator.

**When you get results, answer**
[the review file](../review/hypothesis-01-the-free-bound-on-the-models-share.md).

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The bar every figure in this scope is held to is
[in the scope's MASTER_PLAN](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| none | — | this plan produces numbers and prose, not a figure. The cached curve it reads is already a built figure owned by the causal scope | — | — | reading an existing figure's sidecar is not producing a figure |

### Organization workflow

Nothing to organise. The output of this plan is answers in its review file.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the floor is measured | [the idea map's claim 2](../../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md), whose "checks outstanding" row names this exact read |
| the floor is measured | [the gate plan](hypothesis-02-what-is-left-once-the-chain-settles.md)'s reading of its own curve, which is judged against this number |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/hypothesis-01-the-free-bound-on-the-models-share.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [scripts/correction_size_over_the_run.py](../../../scripts/correction_size_over_the_run.py) | where `‖r_t‖/‖eps_PoE‖` is computed today, at line 101, which task 1.2 has to unpick to answer whether the numerator survived |
| `paper/iclr/figures/correction-size-over-the-denoising-run.json` | the cached curve, 50 steps over 3 seeds and two pairs |
| `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/step_collapse.json` | the step-wise collapse analysis, read beside `snr_collapse.json` for the `peak_at_edge` field |
| `/datasets/mmolefe/poe_repair_min/outputs/veracity/pairs/<pair>/seed_<n>/teacher_residual_const_lam000/residuals/` | the raw per-step tensors the ratio was formed from, 50 files per seed for both pairs at seeds 4, 42 and 123. `delta` is the unnormalised `r_t` and `eps_poe` is the denominator, so the numerator this plan wants is read here rather than recovered from the JSON |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 25, the corrector and the step size it runs at](instrument-01-the-corrector-and-the-step-size-it-runs-at.md).
It does not wait on this plan's answer, so it can start in parallel; what waits is the reading of
step 26's curve, which is judged against this floor.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>1 catalogued failure and its fix</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. Seeded with the one
failure mode this plan is exposed to, so it is recognised rather than rediscovered.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 the ratio read as if it were the numerator

**When it happens:** reading `correction-size-over-the-denoising-run.json` without checking what
was actually stored.
**What you see:** a plausible falling curve and a plausible last-step value.
**Why:** the cached measure is `‖r_t‖/‖eps_PoE‖`, and a ratio can fall because its denominator
rose. The model's share is a statement about the numerator.
**How to fix:** report the numerator and denominator separately, or state that the numerator is
unrecoverable, per task 1.2.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
