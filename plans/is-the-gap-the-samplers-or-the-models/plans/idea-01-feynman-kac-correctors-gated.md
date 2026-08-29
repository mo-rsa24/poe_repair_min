# 🔍 Feynman-Kac correctors, gated

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md — <built or cited, and why>
```

## Recommended skill

▶ `/unpack-paper https://arxiv.org/abs/2503.02819` ✅ a full read of one paper, ending in what would
   have to be implemented and at what cost, is exactly the deliverable this plan wants.
   alt: `/paper-scout` first, if the search turns up a usable implementation and the read becomes a
   comparison rather than a from-scratch cost estimate.

## Position in the plan tree

**Step 30 of 30.** Waits on steps 26 and 29. The one order is the `## Running order` table in the
[repo root MASTER_PLAN.md](../../../MASTER_PLAN.md).

| Step | Plan | What it does |
|------|------|-------------|
| 26 | [hypothesis-02: what-is-left-once-the-chain-settles](hypothesis-02-what-is-left-once-the-chain-settles.md) ⚠️ | the gate. A null there closes this plan unrun |
| 29 | [baseline-02: three-rules-on-one-dose-axis](baseline-02-three-rules-on-one-dose-axis.md) ⚠️ | says whether the corrector rows are worth extending to a second corrector family |
| **30 (current)** | **idea-01: feynman-kac-correctors-gated** ⚠️ | **a full read of arXiv 2503.02819, and a recorded decision to build it or cite it** |

Design only. Verdicts and run state live in
[the paired review file](../review/idea-01-feynman-kac-correctors-gated.md).

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [What this asks, in one line](#what-this-asks-in-one-line)
- [Quick context: where you are](#quick-context-where-you-are)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What happens (visual)](#what-happens-visual)
- [Description: what to do](#description-what-to-do)
- [Purpose and goal](#purpose-and-goal)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks) — things for Claude to execute
- [Instructions](#instructions) — things for you to do manually
- [The engagement gate](#the-engagement-gate)
- [Figure Catalog](#figure-catalog)
- [Orchestration: keeping catalogs and plan files in sync](#orchestration-keeping-catalogs-and-plan-files-in-sync)
- [Reading list](#reading-list)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## What this asks, in one line

⬅️ [Previous](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Next](#quick-context-where-you-are) ➡️

Read Skreta et al.'s [Feynman-Kac correctors](/home-mscluster/mmolefe/goal-setting/learning/sampler-correctors-for-composition/plans/26-feynman-kac-correctors.md) in full and decide, on the record, whether this project
implements a second corrector family or cites one, with the cost of building it written down either
way.

## Quick context: where you are

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

**The topic.** *Feynman-Kac Correctors in Diffusion*,
[arXiv 2503.02819](https://arxiv.org/abs/2503.02819), ICML 2025 Spotlight. A sequential
Monte Carlo corrector for sampling correctly from product distributions, which is the exact problem
this scope is about.

**Learning goal.** Enough of the method to say what implementing it here would take and what it
would buy, rather than enough to have an opinion about it.

**Why it matters.** The Langevin corrector this scope builds is the simplest member of a family.
If the family matters, a reviewer will ask why the simplest member is the only one tried. The
answer is either "we tried a second one" or "here is the cost, and here is why it was not worth it",
and both are acceptable. Silence is not.

**Time budget.** A full read plus a written cost estimate. No GPU, no queue, no implementation
unless the decision comes back build, in which case that build is a sub-scope rather than a task
here.

**Sources.** The paper, its register row at abstract level since 2026-08-12 in
[the reading register](../../standing/literature/reading-register.md), and its parked mention in
[the draft map](../../../paper/iclr/DRAFT_MAP.md) under section 2.

**Associated materials.** [The review questions](../review/idea-01-feynman-kac-correctors-gated.md),
[the whole corrector design](../source/the-whole-corrector-design.md).

## Considerations

⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claim) ➡️

**This plan is allowed to close unrun, and that is a real outcome.**

It tries an idea, so it may not change any claim. If [the gate](hypothesis-02-what-is-left-once-the-chain-settles.md)
returned a null, the corrector arm moved nothing and a second corrector family is a related-work
paragraph. Closing unrun with the reason recorded is a completed plan, not an abandoned one.

**No usable implementation was found when the scope was designed.**

That is why this is a read rather than a wiring job, and it is also the first thing the read should
recheck: an implementation appearing since would change the cost estimate by an order of magnitude.

**Building it is a sub-scope, not a task.**

A second corrector family needs its own instrument, its own leak checks, and its own step-size
equivalent. If the decision is build, it is decomposed rather than appended here.

**Time budget: one read.** Nothing here competes for GPU.

**Known issues.** See [Error Matrix](#error-matrix).

## The claim

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#why-this-plan-exists) ➡️

**Feynman-Kac correctors are either built here or cited here, and which one is recorded with the
reason and the cost.**

**What is varied.** Nothing. This is a read and a decision.

**What is produced.** A cost estimate for implementing the method in this repo, and a
built-or-cited decision written into the review file.

**Falsify condition.** There is no bar, and the plan says so plainly rather than inventing one. It
tries an idea, so its outcome may propose an experiment and may not change a claim. What would make
it a failure is closing with no decision recorded, or with a decision whose reason is not written
down.

**Why this matters right now.** It is the last open question in the scope, and it is cheap.

## Why this plan exists

⬅️ [Previous](#the-claim) | 📋 [TOC](#table-of-contents) | [Next](#what-happens-visual) ➡️

**The gap.** The scope tries one corrector and draws conclusions about what correctors can do. A
second family either supports or bounds that, and nobody has read the paper closely enough to say
which.

**The approach.** Read it fully, cost the implementation, decide, and write the decision down where
the paper's related-work section can reach it.

**Key insights.**

1. The decision matters more than the implementation. "Cited, because implementing it costs X and
   the gate showed Y" is a defensible sentence. "Not tried" is not.
2. Gating the read on steps 26 and 29 stops it happening before it can be answered usefully. Its
   cost estimate depends on what the simplest corrector already showed.

## What happens (visual)

⬅️ [Previous](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Next](#description-what-to-do) ➡️

```
   step 26 gate            step 29 dose grids
        │                        │
        └────────┬───────────────┘
                 ▼
      did the corrector arm move anything?
                 │
        no ──────┴────── yes
        │                 │
        ▼                 ▼
   close unrun,      full read of 2503.02819
   record the             │
   reason           ┌─────┴─────┐
                    ▼           ▼
                 "cited,      "built": becomes its own
                  cost X"      sub-scope, not a task here
```

## Description: what to do

⬅️ [Previous](#what-happens-visual) | 📋 [TOC](#table-of-contents) | [Next](#purpose-and-goal) ➡️

1. **The read.** Promote [arXiv 2503.02819](https://arxiv.org/abs/2503.02819) from its
   abstract-level row in [the reading register](../../standing/literature/reading-register.md) to a
   full read.
2. **The cost estimate.** What would have to be implemented in this repo, against what already
   exists after [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md), and what it
   would buy given what the gate returned.
3. **The decision.** Built or cited, in the review file, with the reason.

No code, no renders, no output under `/datasets`.

## Purpose and goal

⬅️ [Previous](#description-what-to-do) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Purpose**

Serves objective 6 of [the scope's direction](../MASTER_PLAN.md): decide whether Feynman-Kac
correctors are built or cited, with the reason recorded either way.

**Goals**

1. The register row for 2503.02819 moves from abstract level to full read, as a promotion of the
   2026-08-12 row rather than a first read.
2. A written cost estimate for implementing it here.
3. A built-or-cited decision in the review file, with its reason, including when the answer is
   "not run and why".

## Environment Facts This Plan Depends On

⬅️ [Previous](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- No GPU, no queue, no output under `/datasets`. This plan reads and writes markdown.
- Network access is needed to fetch the paper. If the fetch fails, the arXiv id is enough to
  request the PDF another way; do not substitute the abstract for the full read and call it done.
- No system LaTeX here, so nothing in this plan compiles anything, per
  [environment/paper.md](../../../environment/paper.md).

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Preflight: check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete, before acting on it.
  - Paste: `/verify-plan @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Read the gate's branch and the `λ=1` classification, and decide whether this plan runs
      at all.
  - Run it only if the corrector arm moved something. If
    [the gate](hypothesis-02-what-is-left-once-the-chain-settles.md) returned a null, the
    Feynman-Kac read is a related-work paragraph and this plan closes unrun.
  - **Done when:** the review file records either "running, because the gate returned X" or
    "closed unrun, because the gate returned a null", with the branch named.

▶ **Next: [task 1.1](#1--read-it-and-cost-it)** if the gate opened, otherwise
[the close out](#close-out--record-what-this-plan-taught) with the reason recorded.

### 1. 📖 Read it, and cost it

◀ **Needs: [task 0.2](#0--preflight-check-this-plan-before-working-from-it)**, the gate's branch.

- [ ] **1.1** Run the following prompt: `/unpack-paper https://arxiv.org/abs/2503.02819`
  - Produces: the full read of *Feynman-Kac Correctors in Diffusion*, Skreta et al., ICML 2025
    Spotlight, at the depth this project's reading register calls a full read.
  - Output goes to: the register row in
    [the reading register](../../standing/literature/reading-register.md), promoted from abstract
    level, plus the notes that skill files.
  - **Done when:** the register row reads as a full read and names what the method does at each
    noise level, not only what it claims.
- [ ] **1.2** Recheck whether a usable implementation exists now.
  - None was found when this scope was designed, which is the whole reason this is a read. An
    implementation appearing since changes the cost estimate by an order of magnitude.
  - **Done when:** the review file names what was searched and what was found, including "nothing".
- [ ] **1.3** Write the cost estimate: what would have to be implemented here, against what already
      exists after [step 25](instrument-01-the-corrector-and-the-step-size-it-runs-at.md), and what
      it would buy given the gate's branch.
  - **Done when:** the estimate names the pieces and gives a rough size for each, so the
    built-or-cited decision is made against a number rather than a feeling.

▶ **Next: [instruction 2.1](#2--make-the-built-or-cited-call)**, which is a judgement, not a
computation.

### Close out. 🔄 Record what this plan taught

◀ **Needs:** every group above attempted, including the ones that went red, and including the case
where the plan closed unrun.

- [ ] **Capture the failures this plan hit**, while they are still fresh.
  - Paste: `/ingest-error-pattern --from-run-log @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md`
  - Done when: each failure has a catalog entry, or there were none to record.
- [ ] **Bring the tree current** with what actually happened.
  - Paste: `/sync-plan-tree @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md — <built, cited, or closed unrun>`
  - Done when: statuses, the running order and the Error Matrix match reality.
- [ ] **Answer every pre-registered question in the review file**, including the ones whose answer
      is "not run and why".
  - Done when: no question in that file is left at ⚠️ without a stated reason.

▶ **Next: [the engagement gate](#the-engagement-gate).**

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-engagement-gate) ➡️

**For you to follow manually.** Do these yourself, interleaved with the Tasks rather than after
them.

### 2. 💭 Make the built-or-cited call

◀ **Needs: [task 1.3](#1--read-it-and-cost-it)**, the cost estimate.

- [ ] **2.1** Read the cost estimate against what the gate returned.
  - Expected result: a decision that can be defended in one sentence in the related-work section.
  - ✅ **Cited** if the cost is real and the gate's branch does not make a second corrector family
    load-bearing. Write the sentence the paper will use.
  - ✅ **Built** if the gate showed a large sampler share and this family plausibly reaches further.
    Then say so and stop here: the build is its own sub-scope, decomposed rather than appended to
    this plan.
- [ ] **2.2** Write the decision and its reason into the review file.
  - The reason is the part that matters. A decision with no reason is unreviewable in three months
    and gets remade from scratch.

▶ **Next: [the close out](#close-out--record-what-this-plan-taught)**, and with it the scope's
recall gallery, per criterion 12 of [the scope's Definition of Done](../MASTER_PLAN.md#definition-of-done).

## The engagement gate

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

> **Why this checkpoint matters:** this is the last plan in the scope, so its close-out is also the
> scope's. A question left at ⚠️ here is a question nothing downstream will ever come back for.

**Pass criteria**

- The register row for 2503.02819 reads as a full read, or the plan is recorded as closed unrun
  with the gate's branch named.
- The built-or-cited decision is in the review file with its reason.
- Every pre-registered question in the review file is answered, including with "not run and why".

**Fail criteria (STOP)**

- A decision recorded with no reason. That is the one outcome this plan cannot deliver, because the
  reason is the deliverable.

**Partial pass guidance**

- Closing unrun is a full pass when the gate returned a null and that is recorded. It is not a
  partial anything.

**When you get results, answer**
[the review file](../review/idea-01-feynman-kac-correctors-gated.md).

## Figure Catalog

⬅️ [Previous](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Next](#orchestration-keeping-catalogs-and-plan-files-in-sync) ➡️

The bar every figure in this scope is held to is
[in the scope's MASTER_PLAN](../MASTER_PLAN.md#the-figure-bar-every-plan-here-is-held-to).

### Pending: to be generated from prompts

None. This scope carries no `diagram-prompts.md`, so there is no illustrated map to draw from.

### Generated during execution

| Item | Lane | Description | Generated by | Status | Details |
|---|---|---|---|---|---|
| none | — | this plan produces a read, a cost estimate and a decision | — | — | a decision is prose, and drawing it would add nothing a sentence does not carry |

### Organization workflow

The read lands in the reading register. The decision lands in the review file. Nothing is filed
under `paper/iclr/figures/`.

## Orchestration: keeping catalogs and plan files in sync

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#reading-list) ➡️

| What changes | Where it has to be reflected |
|---|---|
| the full read happens | [the reading register](../../standing/literature/reading-register.md)'s row for 2503.02819 moves from abstract level to full read, as a promotion of the 2026-08-12 row |
| the decision | [the draft map](../../../paper/iclr/DRAFT_MAP.md)'s parked mention under section 2, which currently holds the paper without a verdict |
| the decision is build | a new sub-scope, via `/decompose-plan`, never a task appended here |
| the plan's status | the scope [MASTER_PLAN.md](../MASTER_PLAN.md) and the root running order, both by `sync-plan-tree` rather than by hand |

| Step | Command | Triggered by | Outcome |
|------|---------|--------------|---------|
| Check the plan | `/verify-plan @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md` | **task 0.1**, before any work | Conformance and thin instructions reported |
| Capture patterns | `/ingest-error-pattern --from-run-log @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md` | **the close out**, after any red run | Errors added to catalogs |
| Update Error Matrix | `/sync-plan-tree --update-error-matrices` | Auto (by ingest-error-pattern) | This plan file's Error Matrix regenerated |
| Bring the tree current | `/sync-plan-tree @plans/is-the-gap-the-samplers-or-the-models/plans/idea-01-feynman-kac-correctors-gated.md` | **the close out** | Statuses, running order and Error Matrix match reality |

## Reading list

⬅️ [Previous](#orchestration-keeping-catalogs-and-plan-files-in-sync) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Item | Author | Link | Status | Notes |
|---|---|---|---|---|
| Feynman-Kac Correctors in Diffusion | Skreta et al., ICML 2025 Spotlight | [arXiv 2503.02819](https://arxiv.org/abs/2503.02819) | 📖 abstract level since 2026-08-12, promoted by this plan | a sequential Monte Carlo corrector for sampling correctly from product distributions |
| Reduce, Reuse, Recycle | Du et al., ICML 2023 | [arXiv 2302.11552](https://arxiv.org/abs/2302.11552) | 📖 abstract level, vendored code in this repo | the simplest corrector family, and the one this scope actually built |
| Catastrophic Compositional Generation | Soiffer et al. | [arXiv 2606.23920](https://arxiv.org/abs/2606.23920) | 📖 the deepest engagement of the four | says existing correctors, Feynman-Kac included, reduce the gap without closing it. Read it before deciding to build |

## Next step

⬅️ [Previous](#reading-list) | 📋 [TOC](#table-of-contents) | [Next](#error-matrix) ➡️

None inside this scope: this is its last plan. Its close-out triggers the scope's recall gallery,
`/recap-plan-tree @plans/is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md`, per criterion 12 of
[the Definition of Done](../MASTER_PLAN.md#definition-of-done).

## Error Matrix

⬅️ [Previous](#next-step) | 📋 [TOC](#table-of-contents)

<details>
<summary>1 catalogued failure and its fix</summary>

Auto-updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`.

### From global catalog

(Patterns applicable across all projects.) None yet.

### From project catalog

#### 🟡 the abstract stands in for the full read

**When it happens:** the paper is hard to fetch, and the abstract already says what the method
claims.
**What you see:** a register row marked full read that cannot answer what the method does at each
noise level.
**Why:** an abstract carries claims and not mechanics, and the cost estimate needs mechanics.
**How to fix:** request the PDF another way. A read that cannot cost the implementation has not
happened, and saying so is better than a row that overstates itself.

---

**Auto-update note:** regenerated by `/sync-plan-tree` after new errors are added to the catalogs.
Do not edit manually.

</details>
