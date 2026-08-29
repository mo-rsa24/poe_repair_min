# 🔌 SuperDiff at this repo's fifty steps

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md — <does it still compose at 50 steps>
```

## Recommended skill

▶ `/replicate https://github.com/necludov/super-diffusion` ✅ wiring a published method into this
   repo's interface, at this repo's settings, with a parity check against the authors' defaults, is
   what that skill is for.
   alt: `/learn-codebase` on the vendored source first if the pipeline's step-count handling is not
   obvious from a read.

## Position in the plan tree

**Step 28 of 30.** Waits on step 26. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](hypothesis-02-what-is-left-once-the-chain-settles.md) ⚠️ | the gate. A null there turns this half into a baselines table rather than a diagnosis, and the framing of every caption changes with it |
| **28 (current)** | **baseline-01: superdiff-at-this-repos-fifty-steps** ⚠️ | **wires a published composition rule into this repo, matched to 50 steps at guidance 7.5, and checks whether matching broke it** |
| 29 | [baseline-02: three-rules-on-one-dose-axis](baseline-02-three-rules-on-one-dose-axis.md) ⚠️ | needs the per-step prediction this plan exposes |

Design only. Verdicts and run state live in
[the paired review file](../review/baseline-01-superdiff-at-this-repos-fifty-steps.md).

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

Wire SuperDiff into this repo as a composer and run it at SDXL base, DDIM, 50 steps and guidance
7.5, so its numbers are comparable to everything else measured here. Then check whether cutting its
step count from its default 200 down to 50 breaks it.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The system.** `poe_repair/composers/superdiff.py`, a third composer beside plain
product-of-experts and the Langevin corrector, wrapping the pipeline from
[superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0), source at
[necludov/super-diffusion](https://github.com/necludov/super-diffusion).

**What it does.** SuperDiff is a composition rule derived from the continuity equation rather than
from naive score addition. Skreta et al.,
[arXiv 2412.17762](https://arxiv.org/abs/2412.17762), ICLR 2025 Spotlight. It is a published
alternative to the rule this paper is about, and a reviewer will ask about it whichever way
[the gate](hypothesis-02-what-is-left-once-the-chain-settles.md) came back.

**Key components.** The composer wrapper, the step-count match, and a hook exposing the per-step
prediction `eps_M` so that `r_t^SD = eps_J - eps_M` can be formed. That last one is what
[step 29](baseline-02-three-rules-on-one-dose-axis.md) needs and is easy to leave out.

**Testing approach.** One parity check, and it is the whole point of the plan: render the same pair
and seed at 200 steps and at 50 steps and compare. If the 50-step render stops composing, every
later comparison against SuperDiff is between a working rule and a crippled one.

**Associated materials.** [The review questions](../review/baseline-01-superdiff-at-this-repos-fifty-steps.md),
[the whole corrector design](../source/the-whole-corrector-design.md), and the register row for
2412.17762 in [the reading register](../../standing/literature/reading-register.md), at abstract
level since 2026-08-12.

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**The default is 200 steps and this repo runs 50.**

Everything in this project is measured at 50 DDIM steps at guidance 7.5. Comparing a 200-step
SuperDiff against a 50-step product-of-experts would be comparing two things at once. Matching the
step count is the right call and it has a cost that this plan measures rather than assumes.

**A baseline freezes on landing.**

Per this project's run conventions, a baseline may not change any claim. What it can do is give the
comparison a floor, and that only works if the floor is a fair one.

**The per-step prediction is the deliverable that is easiest to skip.**

Rendering a picture with SuperDiff is satisfying and insufficient. Step 29 needs `eps_M` at every
step, and a wrapper that only returns the final image cannot supply it.

**The cache cannot be used.** SuperDiff follows its own trajectory.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**SuperDiff runs in this repo at SDXL base, DDIM, 50 steps and guidance 7.5, exposes its per-step
prediction, and either still composes at that step count or is recorded as not doing so.**

**Independent variable.** The number of inference steps, 200 against 50, on one pair and one seed.
Nothing else differs.

**Dependent variable.** Whether the render composes, by the detector and by eye.

**Falsify condition.** The bar is on the parity check, not on SuperDiff's quality.

- **Pass.** The 50-step render composes on the tested pair and seed. SuperDiff is comparable and
  step 29 proceeds with it as an ordinary row.
- **Fail.** The 50-step render does not compose while the 200-step one does. This does not stop the
  plan: it puts a sentence in every caption that compares against SuperDiff, saying the comparison
  is between a working rule and one run outside its intended settings. What is not acceptable is
  discovering this inside the dose grid, where it would read as SuperDiff being weak.
- **Inconclusive.** Neither render composes on the tested pair. Then the pair is wrong for this
  check, not the method. Try one more pair before recording anything.

**Why this matters right now.** It is a reviewer's first question about a paper proposing a
composition fix, and it is asked whether or not the gate came back with a split.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The problem.** The paper compares its own composition rule against nothing published. That is a
gap a reviewer closes for you, unfavourably.

**The approach.** Wire one published rule at this repo's own settings, and measure what matching
the settings cost before any comparison is drawn.

**Key insights.**

1. Matching the step count is what makes the comparison fair on the axis that matters, and it is
   also the thing most likely to break the method. Both facts are handled by measuring rather than
   choosing.
2. Exposing `eps_M` per step is what turns a picture into a row on a dose axis. Without it,
   SuperDiff can be shown but not compared.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-build) ➡️

```
  same pair, same seed, only the step count differs

  200 steps (the authors' default)     50 steps (this repo's setting)
  ┌───────────────┐                    ┌───────────────┐
  │               │                    │               │
  │   composes?   │       vs           │   composes?   │
  │               │                    │               │
  └───────────────┘                    └───────────────┘
        yes                                  ?
                                             │
                            yes ─────────────┴───────────── no
                             │                               │
                    an ordinary baseline row      a caption sentence on every
                    at step 29                    comparison: working rule
                                                  against a crippled one
```

## Description: what to build

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **`poe_repair/composers/superdiff.py`.** The pipeline from
   [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0) wrapped in this
   repo's composer interface, so it is callable the same way
   [poe.py](../../../poe_repair/composers/poe.py) is.
2. **The step-count match.** SDXL base, DDIM, 50 steps, guidance 7.5, and the parity render at 200
   steps beside it.
3. **The per-step prediction hook.** `eps_M` exposed at each step so `r_t^SD = eps_J - eps_M` can be
   formed, which is what the generalised dose axis at
   [step 29](baseline-02-three-rules-on-one-dose-axis.md) needs.

Renders write under
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`.

## Purpose and goal

⬅️ [Previous](#description-what-to-build) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 5 of [the scope's direction](../MASTER_PLAN.md): wire SuperDiff at this repo's
settings and check it still works there, before three rules are compared along one dose axis.

**Goals**

1. `poe_repair/composers/superdiff.py` exists and renders at 50 DDIM steps at guidance 7.5.
2. The 200-against-50 parity check is recorded, with the verdict and both renders.
3. `eps_M` is available per step, verified by forming `r_t^SD` on one cell.

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- `co3` python at its absolute path, `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`.
  Never a bare `python`.
- Renders and weights write under
  `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`, with a disk
  guard on `/datasets`, per [environment/storage.md](../../../environment/storage.md). The
  SuperDiff checkpoint is a model download and must not land on `/home-mscluster`.
- biggpu allows one job per user. The two parity renders are short enough to run in-session; the
  `nohup`-outside-Slurm rule bites at [step 29](baseline-02-three-rules-on-one-dose-axis.md). Read
  [environment/hpc/execution-protocol.md](../../../environment/hpc/execution-protocol.md) before
  launching onto a node this session is not on.
- **The cached trajectories cannot be used.** SuperDiff follows its own path.
- The models run in fp16, and anything normed upcasts to fp32 first.
- SDXL base, DDIM, 50 steps, guidance 7.5, latents 4×128×128 at 1024².

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Confirm which branch [the gate](hypothesis-02-what-is-left-once-the-chain-settles.md)
      fired, and record it in this plan's review file.
  - A null there turns this half of the scope into a baselines table rather than a diagnosis, which
    changes the framing of every caption this plan and step 29 produce. The runs are the same
    either way; the sentences around them are not.
  - **Done when:** the branch is quoted in the review file.

▶ **Next: [task 1.1](#1--wire-the-pipeline)**.

### 1. 🔌 Wire the pipeline

◀ **Needs: [task 0.2](#0--preflight-check-this-plan-before-working-from-it)**, so the captions are
framed correctly from the start.

- [ ] **1.1** Wire the pipeline into `poe_repair/composers/superdiff.py`.
  - Source: [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0), code at
    [necludov/super-diffusion](https://github.com/necludov/super-diffusion).
  - Follow this repo's composer interface, the one
    [poe.py](../../../poe_repair/composers/poe.py) implements.
  - **Done when:** one render completes and lands under
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/`.
- [ ] **1.2** Expose `eps_M` per step so `r_t^SD = eps_J - eps_M` can be formed.
  - **Done when:** `r_t^SD` is formed on one cell and its per-step norm printed for all 50 steps,
    proving the hook returns a prediction and not a placeholder.

▶ **Next: [task 2.1](#2--the-step-count-parity-check)**.

### 2. 🚀 The step-count parity check

◀ **Needs: [task 1.1](#1--wire-the-pipeline)**, so there is something to render with.

- [ ] **2.1** Render the same pair and seed at 200 steps and at 50 steps, changing nothing else.
  - Output goes to:
    `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/parity/`
  - **Done when:** both renders exist, and the scored verdict for each is written beside them.
- [ ] **2.2** Record whether the 50-step render still composes, in the review file, with the
      detector verdict and the eye verdict side by side.
  - If matching the step count breaks the method, the comparison is between a working rule and a
    crippled one, and that sentence goes in the caption of every figure comparing against
    SuperDiff rather than being discovered inside the grid.
  - **Done when:** the review file's pre-registered question is ticked with both verdicts.

▶ **Next: [instruction 3.1](#3--look-at-the-two-renders-yourself)**, since the detector's word on a
published method is not enough to put in a caption.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md — <one line>`
  - Done when: statuses, the running order and the Error Matrix match reality.
- [ ] **Promote the register row.** Move
      [the reading register](../../standing/literature/reading-register.md)'s row for 2412.17762
      from abstract level to full read, since wiring the method required reading it.
  - Done when: the row records the promotion, dated, as a promotion of the 2026-08-12 row rather
    than a first read.

▶ **Next: [the engagement gate](#the-engagement-gate).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 3. 👁️ Look at the two renders yourself

◀ **Needs: [task 2.1](#2--the-step-count-parity-check)**, both parity renders.

- [ ] **3.1** Open the 200-step render and the 50-step render side by side.
  - Expected result: two pictures of the same pair at the same seed, differing only in how many
    steps produced them.
  - ✅ If both show two separate things, the step-count match is free and SuperDiff is an ordinary
    row at step 29.
  - ❌ If the 50-step one blends and the 200-step one does not, record it. It is a caption sentence
    from here on, not a footnote.
- [ ] **3.2** If neither composes, try one more pair before writing anything down.
  - A published method failing on one pair at its own default step count is more likely the pair
    than the method, and recording it as a method failure would be unfair and wrong.
- [ ] **3.3** Write your eye verdict into the review file beside the detector's.
  - Where they disagree, the eye is the one cited, following the practice
    [the timing verdict](../../03-does-the-correction-cause-composition/review/hypothesis-03-when-in-the-run-it-matters.md)
    set for this project.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, then
[step 29](baseline-02-three-rules-on-one-dose-axis.md).

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** step 29 puts SuperDiff on a shared axis with this project's own
> rule. If the step-count match crippled it, that axis compares a method against a handicapped one
> and every conclusion drawn from it is wrong in the paper's favour, which is the worst direction
> for an error to point.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SD=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff

ls -l "$SD/parity/"          # both renders, 200 steps and 50 steps
# the per-step prediction hook: 50 norms, not one
$PY -c "import json;d=json.load(open('$SD/parity/eps_m_norms.json'));print(len(d['steps']),'steps')"
```

**Pass criteria**

- `superdiff.py` renders at 50 DDIM steps at guidance 7.5.
- Both parity renders exist and are scored.
- `eps_M` is available at all 50 steps, verified by forming `r_t^SD` on one cell.
- Instruction 3.3 has recorded the eye verdict beside the detector's.

**Fail criteria (STOP)**

- `eps_M` cannot be exposed per step. Step 29 cannot form the generalised dose axis for this row,
  and the plan is re-scoped to a picture-only comparison rather than proceeding as if the axis
  existed.

**Partial pass guidance**

- The 50-step render not composing is a partial pass, and it is recorded rather than worked around.
  SuperDiff still runs at step 29; every caption comparing against it carries the sentence.

**When you get results, answer**
[the review file](../review/baseline-01-superdiff-at-this-repos-fifty-steps.md).

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The bar every figure in this scope is held to is
[in the scope's MASTER_PLAN](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| the two parity renders | — | the same pair and seed at 200 steps and at 50 steps, side by side | task 2.1 | ⏳ | **Not a paper figure.** It measures whether the instrument was set up fairly, not the phenomenon, so it stays in the review file. If the 50-step render fails, the pair of images is worth keeping as the evidence behind the caption sentence, filed under `paper/iclr/figures/when-the-correction-arrives/superdiff/` with a `README.md` entry |

### Organization workflow

1. Render both parity cells under `/datasets`.
2. Keep them in the review file unless the check failed, in which case file the pair under
   `paper/iclr/figures/when-the-correction-arrives/superdiff/` with its sidecar and README entry,
   because a caption sentence needs its evidence reachable.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the parity verdict | every caption at [step 29](baseline-02-three-rules-on-one-dose-axis.md) that compares against SuperDiff |
| the full read happens | [the reading register](../../standing/literature/reading-register.md)'s row for 2412.17762 moves from abstract level to full read, as a promotion of the 2026-08-12 row |
| a figure lands in `superdiff/` | that folder's `README.md` gains an entry naming the algorithm and what produced it |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/06-is-the-gap-the-samplers-or-the-models/plans/baseline-01-superdiff-at-this-repos-fifty-steps.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Code references

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Path | Why it is read |
|---|---|
| [poe_repair/composers/poe.py](../../../poe_repair/composers/poe.py) | this repo's composer interface, which `superdiff.py` implements so it is callable the same way |
| [necludov/super-diffusion](https://github.com/necludov/super-diffusion) | the authors' implementation, and where the 200-step default lives |
| [superdiff-sdxl-v1-0](https://huggingface.co/superdiff/superdiff-sdxl-v1-0) | the working SDXL pipeline and its weights |
| [the reading register](../../standing/literature/reading-register.md) | the row for 2412.17762, at abstract level since 2026-08-12, promoted by this plan's close out |

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

[Step 29, three rules on one dose axis](baseline-02-three-rules-on-one-dose-axis.md). It needs the
per-step prediction this plan exposes, and it is where the comparison is actually drawn.

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>2 catalogued failures and their fixes</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🔴 the wrapper returns an image but not a per-step prediction

**When it happens:** wrapping a published pipeline by its top-level call rather than its denoising
loop.
**What you see:** SuperDiff renders fine, and step 29 cannot form `r_t^SD` for its row.
**Why:** the generalised dose axis needs `eps_M` at every step, which a top-level wrapper discards.
**How to fix:** task 1.2 verifies the hook by printing 50 norms before anything else is built on it.

#### 🟡 the checkpoint downloads to `/home-mscluster`

**When it happens:** a Hugging Face pipeline pulled without setting the cache directory.
**What you see:** a full home filesystem, which has already once silently killed checkpointing here
(`poe-disk-001`).
**Why:** the default cache is under `$HOME`.
**How to fix:** point the cache at `/datasets` before the first pull. See
[environment/storage.md](../../../environment/storage.md).

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
