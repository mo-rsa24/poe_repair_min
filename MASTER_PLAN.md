# LoRA-Fixes-PoE

## Where things stand

This block is a snapshot; the live version prints at session start, or on demand with
`python3 scripts/plan_pulse.py --brief`.

- **What is going on:** step 9's one-epoch smoke run is in flight on a shared biggpu device.
- **The last thing we did:** taught the environment the shared-device launch path (an allocated
  biggpu node with an idle second GPU, reached over SSH) and launched step 9's smoke run with it.
- **Do this next:** step 9's one-epoch smoke run,
  [instrument-02-three-live-curves-while-training](plans/04-does-the-fix-reach-unseen-pairs/plans/instrument-02-three-live-curves-while-training.md),
  is cooking on a shared biggpu device; it gates the whole transfer chain (steps 10, 11, 12 and 14
  wait behind it, and so does register slot F8). While it runs, write steps 16 or 18, which need
  no GPU and no queue. When the run finishes, judge its three curves against the review file's
  bar.

## The paper: what has to land

One order across every plan in every scope. **Step numbers are permanent:** a finished plan keeps
its number and stays struck through, so the count tells you where you are rather than how much is
left in a queue that renumbers itself. A plan's own file carries this same number under its title.

| Step | Plan | What it does | Status | Waits on |
|---|---|---|---|---|
| 1 | ~~03-does-the-correction-cause-composition/instrument-01-build-the-measuring-scripts~~ | the thirteen measuring scripts, built and smoked | ✅ |  |
| 2 | ~~03-does-the-correction-cause-composition/instrument-02-fix-the-size-measure-before-any-result~~ | how the correction's size is expressed, fixed before any result was read | ✅ |  |
| 3 | ~~04-does-the-fix-reach-unseen-pairs/instrument-01-the-clean-pair-pool~~ | the pool of pairs that blend by default, confirmed by the scorer | ✅ |  |
| 4 | ~~03-does-the-correction-cause-composition/hypothesis-02-more-correction-more-composition~~ | the headline: more correction, more composition, controls flat | ✅ |  |
| 5 | ~~03-does-the-correction-cause-composition/hypothesis-04-what-the-cached-runs-already-show~~ | the analyses needing no GPU and no queue | ✅ | 1 |
| 6 | 03-does-the-correction-cause-composition/hypothesis-03-when-in-the-run-it-matters | when the correction matters: the cliff is at the start | ◑ driving the timing tab by hand | 4, 5 |
| 7 | ~~03-does-the-correction-cause-composition/hypothesis-05-the-same-story-from-three-sides~~ | the independent checks: two image-side yes, two language-side null | ✅ | 4 |
| 8 | ~~03-does-the-correction-cause-composition/hypothesis-01-what-the-fix-changes-inside-the-model~~ | the fix changes what a word paints, not where it looks | ✅ | 3 |
| 9 | 04-does-the-fix-reach-unseen-pairs/instrument-02-three-live-curves-while-training | the one-epoch smoke confirming three live curves | ◑ run in flight; verdict lands in the review file |  |
| 10 | 04-does-the-fix-reach-unseen-pairs/hypothesis-01-does-one-pooled-fix-transfer-at-all | finish the read: steps 70k to 100k unscored, go/no-go note owed | ◑ run done, read incomplete | 9 |
| 11 | 04-does-the-fix-reach-unseen-pairs/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs | fifteen adapters, leaderboard, degradation curve | ⚠️ | 10 |
| 12 | 04-does-the-fix-reach-unseen-pairs/baseline-01-the-size-matched-control-pool | the size-matched mixed pool against animals-only | ⚠️ | 11 |
| 13 | 03-does-the-correction-cause-composition/figure-01-the-seven-paper-figures | the figures this scope owes the paper | ◑ F1 to F5b, F7a, D1 to D4 built; F6 needs a decision | 6, 7, 8 |
| 14 | 04-does-the-fix-reach-unseen-pairs/figure-01-the-transfer-figures | the transfer evidence figures | ◑ F8a and F8b built; F8 waits on the sweep | 11, 12 |
| 15 | 03-does-the-correction-cause-composition/gate-01-two-literature-checks-before-print | the two `/pressure-test` passes before anything is written | ⚠️ | 13 |
| 16 | 07-writing-the-paper/writing-01-make-the-template-build | tectonic build, de-stub, the figure-path rule | ◑ figure-path rule written; the title is still a stub |  |
| 17 | 07-writing-the-paper/writing-02-the-title-and-the-section-spine | the claim in one line, section order | ⚠️ | 16 |
| 18 | 07-writing-the-paper/writing-05-the-results-skeleton | placeholders, not prose: empty tables and XX numbers, one per register slot | ⚠️ |  |
| 19 | 07-writing-the-paper/writing-03-where-each-figure-goes | which figure goes where, and the run order that implies | ⚠️ | 13, 14 |
| 20 | 07-writing-the-paper/writing-04-method-and-introduction | method and intro prose | ⚠️ | 17 |
| 21 | 07-writing-the-paper/writing-06-mechanism-and-limitations | the mechanism section, honest about what did not replicate | ⚠️ | 15, 26 |
| 22 | 07-writing-the-paper/writing-07-the-abstract-written-last | written last, from the spine and the method | ⚠️ | 20, 21 |
| 23 | [is-the-gap-the-samplers-or-the-models](plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md) | how much of the correction is the sampler's and how much the model's, plus three composition rules on one dose axis. Grew into a scope of its own; its seven plans are steps 24 to 30 | ⚠️ its own scope | see 24 to 30 |
| 24 | 06-is-the-gap-the-samplers-or-the-models/hypothesis-01-the-free-bound-on-the-models-share | the correction's size as the run reaches zero noise, read off cached files. A floor under the model's share for no GPU | ⚠️ |  |
| 25 | 06-is-the-gap-the-samplers-or-the-models/instrument-01-the-corrector-and-the-step-size-it-runs-at | the Langevin corrector composer, both leak checks, and the step size fixed before any curve is read | ⚠️ |  |
| 26 | 06-is-the-gap-the-samplers-or-the-models/hypothesis-02-what-is-left-once-the-chain-settles | the gate: the correction's size per step against corrector count k, with the three-way bar in source | ⚠️ | 24, 25 |
| 27 | 06-is-the-gap-the-samplers-or-the-models/hypothesis-03-does-the-corrector-compose-in-the-same-window | the nine-position window sweep rerun with the corrector in place of the injected correction | ⚠️ | 26 |
| 28 | 06-is-the-gap-the-samplers-or-the-models/baseline-01-superdiff-at-this-repos-fifty-steps | SuperDiff wired and matched to 50 steps at guidance 7.5, with its own 200-against-50 check | ⚠️ | 26 |
| 29 | 06-is-the-gap-the-samplers-or-the-models/baseline-02-three-rules-on-one-dose-axis | the two dose grids putting product-of-experts, SuperDiff and the corrector on one axis | ⚠️ | 25, 28 |
| 30 | 06-is-the-gap-the-samplers-or-the-models/idea-01-feynman-kac-correctors-gated | the full read of arXiv 2503.02819 and a built-or-cited decision. Closes unrun if the gate came back null | ⚠️ | 26, 29 |
| 31 | 01-showcase-the-trained-lora/01-read-the-plateau-curves | the free curve-read that re-scopes how A and B read; informs, does not gate | ⚠️ |  |
| 32 | 01-showcase-the-trained-lora/02-the-dog-x-dog-null-probe | the null-input control with its baseline identity preflight | ⚠️ |  |
| 33 | 01-showcase-the-trained-lora/03-the-lora-dose-sweep | the causal dose curve for the shipped LoRA, four control rows | ⚠️ | 32 |
| 34 | 01-showcase-the-trained-lora/04-the-transfer-matrix-figure | group-pooled LoRAs on audited concept-disjoint pairs | ⚠️ |  |
| 35 | 01-showcase-the-trained-lora/05-assemble-the-showcase-figures | the figure set into paper/iclr/figures under the ledger's standard | ⚠️ | 32, 33, 34 |
| 36 | 01-showcase-the-trained-lora/06-extend-the-tracking-set | the four adopted curves wired into instrument-02's live logging, smoke-proven | ⚠️ | 31 |
| 37 | 01-showcase-the-trained-lora/07-experiment-c-lambda-window | lambda-times-window sweep on existing checkpoints; builds the shared injection harness | ⚠️ | 36 |
| 38 | 01-showcase-the-trained-lora/08-experiment-a-resume-to-200k | length axis: rank 8 resumed from 100k to 200k, sbatch on biggpu | ⚠️ | 36 |
| 39 | 01-showcase-the-trained-lora/09-experiment-b-rank-16-32 | rank axis at 100k: two fresh runs over the SSH idle-node path | ⚠️ | 36 |
| 40 | 01-showcase-the-trained-lora/10-the-mechanism-follower | h-space and Jacobian reads per broad checkpoint during A and B, ending in interventions | ⚠️ | 38, 39 started |
| 41 | 01-showcase-the-trained-lora/11-the-counted-joint-prompt-figure | score the joint-prompt renders; three bars per pair plus the repair strip | ⚠️ |  |
| 42 | 01-showcase-the-trained-lora/12-close-f8a-and-the-oracle-panel | score the 70k-100k samples; the oracle-ceiling panel that keys experiment B | ⚠️ |  |
| 43 | 01-showcase-the-trained-lora/13-revalidate-the-scorer-off-animals | the label-pass validation that opens tier-three captions | ⚠️ |  |
| 44 | 05-when-does-the-outcome-lock-in/01-basins-by-hand | one cell, three probed steps, nine endings: proves basins and a ridge exist for the composed flow, or stops the scope for one afternoon's cost | ⚠️ |  |
| 45 | 05-when-does-the-outcome-lock-in/02-the-free-probe | posterior-mean drift per cell from the cache alone, judged against the pre-registered ordering (settling at or before divergence) | ⚠️ |  |
| 46 | 05-when-does-the-outcome-lock-in/03-wire-the-oracle | LCM-SDXL downloaded to /datasets, the adapter with its three asserts, one smoke against a teacher ending | ⚠️ | 44 |
| 47 | 05-when-does-the-outcome-lock-in/04-calibrate-the-instrument | 240 states, oracle against teacher per family, two bars in code, verdict: adopt, shrink, or fall back | ⚠️ | 46 |
| 48 | 05-when-does-the-outcome-lock-in/05-the-grid-and-the-figures | both sweeps with stability copies, the speciation table, the three-timestamp figure and the compose-rate curves | ⚠️ | 45, 47 |

Steps 16 and 18 wait on nothing and need no GPU, so they are what to write while a run cooks.
Steps 24 and 25 wait on nothing either: 24 needs no GPU at all, and 25 is a build.

## Reading, in the background

A pool, not an order. Pull from it when a claim needs backing or a method needs a source.
Found with `/paper-scout`, read with `/unpack-paper` or `/drip --paper`, registered in
[plans/standing/literature/](plans/standing/literature). No row here blocks a row above.

| Paper | Why it matters to us | Which claim it touches | Read |
|---|---|---|---|
| (the 7 already reconciled on the does-the-correction-cause-composition question) | establishes that the residual IS the term PoE drops | does-the-correction-cause-composition, the causal claim | ✅ back-fill owed into the register |

## Experiments running in the background

A pool. Every row is a run that tries an idea, so no row here may change a claim: a striking
number earns the right to propose an experiment and nothing more. Results land as a new row or
task in the plan tree directly, never in a separate holding file.

| Run | What it would earn | State |
|---|---|---|
| 03-does-the-correction-cause-composition/idea-01-does-it-hold-for-attribute-pairs | whether the correction behaves the same for attribute pairs as for object pairs, which would widen the claim's reach | ⚠️ not started |
| 03-does-the-correction-cause-composition/generalization-01-other-models-and-samplers | the same result on a second model and sampler, which is a likely reviewer ask but not a claim we make | ⚠️ not started |
| 02-can-we-trust-the-compose-rate/gate-01-is-this-hole-already-known | whether "presence metrics miss fusion, count metrics miss a repeat" is already published. Already-known cancels the two rows below and leaves a methods paragraph | ⚠️ not started |
| 02-can-we-trust-the-compose-rate/instrument-01-the-three-state-labelled-set | the band on every compose rate the paper prints, and whether the scorer's error grows with λ. A 10-point growth caps F2's caption | ⚠️ not started, runs whatever gate-01 says |
| 02-can-we-trust-the-compose-rate/idea-01-what-the-current-benchmarks-score | whether any published metric agrees with people where ours does not, at 95% against our 85% | ⚠️ not started, blocked by gate-01 |
| 02-can-we-trust-the-compose-rate/gate-02-promote-or-close | the decision that moves this scope into the paper table with a step number, or closes it | ⚠️ not started, blocked by all three |

## Standing jobs

No order and no end. Re-entered rather than closed.

- [plans/standing/literature/plans/01-reading-register.md](plans/standing/literature/plans/01-reading-register.md): keep the reading table above current, and make sure every idea-trying run names the paper it came from.

Artifact reconciliation (`artifacts/plans/parked/artifact-reconciliation/`) is shelved, not standing: the
two-filesystem catalogue-and-integrity-check job it did is not currently worth the overhead it
costs to keep current. Revive by moving it back to `plans/standing/` if lost or untrustworthy
artifacts become a real problem again.

## The scopes, and the state of each

What each folder under `plans/` is and whether to open it. Live means it has rows in the lists
above. Standing means it is re-entered, never finished. Done means its output exists and is in
use. Nothing here is ambiguous on purpose: a scope that cannot say its state in one line gets
shelved until it can.

The listing itself says the state: a numbered folder is live and its number is its place in the
reading order below, and everything else sits in a container named for its state. The numbers
order the scopes for a reader arriving cold; they are not the step order, which stays in the one
`## Running order` table above and interleaves across scopes.

| Folder | State | One line |
|---|---|---|
| `01-showcase-the-trained-lora/` | live | the trained adapter `phase1_r8_100k` and the paper figures measured from its own output, under the decision ledger the two design walks settled. First because the adapter is the artifact the paper ships and the other scopes measure against it |
| `02-can-we-trust-the-compose-rate/` | live | the instrument: whether the printed compose rates can be trusted at all. Second because every number in scopes 03 and 04 leans on the scorer this scope audits |
| `03-does-the-correction-cause-composition/` | live | the causal claim: the correction exists, causes composition, has a timing window, and is learnable |
| `04-does-the-fix-reach-unseen-pairs/` | live | the transfer claim: one pooled LoRA composes pairs it never trained on. Depends on `compose-scorer`'s `scorer_validated.json` |
| `05-when-does-the-outcome-lock-in/` | live, nothing started | the commitment probe: at which step the final image is decided, which is what explains the gap between the injection window and the divergence step. Adopted the basin-oracle walk's decision ledger |
| `06-is-the-gap-the-samplers-or-the-models/` | live, nothing started | the one threat to the paper's framing: how much of the correction a Langevin corrector removes, and how much no corrector touches. Then three composition rules on one dose axis |
| `07-writing-the-paper/` | live | the ICLR manuscript in `paper/iclr/`. No GPU, no queue. Runs nothing and consumes everything the six scopes above produce |
| `standing/literature/` | standing | the reading register: what the field already knows, and the source behind every idea-trying run |
| `retrofit-poe-repair-min.md` | standing | the one-name-per-thing sweep across the repo, executed by `/retrofit-repo`; sits beside the tree because it touches every scope |

Everything finished, parked or cold has left `plans/` for `artifacts/plans/`, which is where to
look for it:

| Where it went | What is there |
|---|---|
| `artifacts/plans/completed/compose-scorer/` | the reusable instrument: delivered `scorer_validated.json`, the cross-scope contract scopes 03 and 04 both read |
| `artifacts/plans/parked/` | `artifact-reconciliation`, `composition-type-cells`, `cross-model-replication`, `inspector-interaction-term`, `mechanism-study`; each opens with a `## Parked` block saying what was last done, why it stopped, and what to read on return |
| `artifacts/plans/archived/` | `phases` and `rungs`, the two superseded work breakdowns; `closing-the-compositional-gap` and `showcase-the-trained-lora`, the emptied shells of scopes whose contents were absorbed |

### Every filename says its run kind

Inside any scope, `ls plans/` sorts the work by run kind, because the kind is the filename's first
word. The rule attached to each kind is what that plan answers to.

| Prefix | Run kind | Its rule |
|---|---|---|
| `hypothesis-NN-` | tests the core claim | seen to the end unless the science is wrong; a pre-registered bar in its review file |
| `baseline-NN-` | a competitor to beat | frozen the moment it lands |
| `figure-NN-` | draws a settled result for the paper | the caption may claim no more than its register slot's sentence |
| `idea-NN-` | tries a new technique | names the paper it came from; a result becomes a deferred task, never a claim |
| `generalization-NN-` | other models, datasets, samplers | instances chosen before any runs; a failure bounds the claim's scope |
| `instrument-NN-` | not a run: a tool or a choice fixed before results | judged by whether it can fail, not by what it found |
| `gate-NN-` | not a run: a literature check before print | a `/pressure-test` verdict |
| `writing-NN-` | not a run: manuscript prose | consumes the register and the review files |

### Figures first, then the writing

The figure register, [the eight slots and the claim each will make](paper/iclr/figures.md), is the
scoreboard, and a reserved slot is not idle: it carries the claim its caption will make, written
before the experiment runs. Work in these scopes means resolving slots.

The trigger to start writing in earnest is 5 to 10 slots resolved (built, fillable, or honestly
downgraded with the boundary stated). The skeleton plus five real figures is a draft; eight perfect
figures with no prose is not. A background result that earns a slot moves into the paper table and
gets a step number, which is how a promotion becomes visible instead of being a quiet field change.

## One plan, one table

Every live plan appears in exactly one of the four lists above, and all four live in this file.
No scope keeps a list of its own. When a background experiment starts feeding the paper it
**moves** into the paper table and gets a step number, which is how a promotion becomes visible
instead of being a quiet field change. `plan_pulse --checks=8` fails if a live plan is in none
of them or in more than one.

## Mission
Does a LoRA make PoE co-occur like Mono, and does that fix carry to unseen pairs?
When SDXL composes two concepts by Product-of-Experts it usually fails (chimera /
single concept / noise). We train a rank-8 cross-attention LoRA on the cached
guided residual r_t = ε̃_J − ε̃_PoE so that, at inference and without ever
encoding the joint prompt (Mono-free), the corrected PoE prediction moves toward
the Mono ceiling, far enough to separate the concepts by eye on the beachhead
cell (~40% of the PoE→Mono distance). The program asks how far that fix reaches:
one cell, one pair, one difficulty class, or the whole studied taxonomy.

## Objectives
(The five pyramid rungs — direction. Each widens the held-out set.)
1. **Overfit** — a rank-8 cross-attn LoRA on the cached residual makes PoE
   co-occur like Mono at inference (Mono-free), and the mechanism is
   pair-generic, not concept-collision-specific.
2. **Survive-Noise** — the fix survives seed variation: one LoRA pooled over
   seeds generalises to held-out seeds, per group.
3. **Cross-Pair** — a pair-trained fix transfers to an unseen sibling pair of the
   same group (the cheap within-group transfer probe).
4. **Group-Wise** — "group" is a deployable pooling unit: one LoRA per difficulty
   class, trained on within-group pairs, generalises to held-out pairs.
5. **Scale** — a single LoRA spans the studied taxonomy held out on BOTH pair and
   seed axes (the deployment crossbar) — or the deployable artefact is the
   per-group catalogue, established with evidence.

## Goals
(Checkpoints — measurable. Status from report/decision-timeline.md.)
1. Overfit: cat×dog seed 42 — λ=0 byte-identical to PoE, λ=1 two distinct animals
   by ~ep600 [✅ G04]; one representative pair per G1–G4+G6 closes the gap
   single-seed [◑ only G4+G6 trained; G1–G3 owed].
2. Survive-Noise: pooled LoRA composes on ≥3/4 held-out seeds for the
   representative pair, per group [G6: pool trained to convergence ✅ (verdict
   ok), BUT composes-on-held-out-seeds ⧗ pending (enactment generating, job
   recap_g6); ◑ G1–G4 part-trained, no verdicts].
3. Cross-Pair: a group-G LoRA composes on a held-out sibling on ≥2/4 held-out
   seeds [⚠️ code ready (Plan 12), not run].
4. Group-Wise: within-group 7-pair pool composes on the held-out 3 pairs,
   matching or beating the single-pair sibling smoke [◑ g6 smoke only; g1–g4 not
   started].
5. Scale: one LoRA on 5 pairs × 8 seeds — held-pair×held-seed quadrant composes
   with per-group structure, OR documented fallback to per-group catalogue
   [⏸ trained to step 30k, crossbar never evaluated].

## Expected Outcome
A deployable, Mono-free PoE corrector whose reach is characterised: at minimum a
per-pair/per-group catalogue backed by evidence, at most a single
taxonomy-spanning LoRA. The alternatives that do NOT work (external correctors,
PoE-internal forces) are documented as negative controls, and every landing is
recorded in the decision timeline.

## Definition of Done
1. ⚠️ Overfit read across G1–G4+G6 single-seed (gap closed by eyeball + MDS bend).
2. ⚠️ Survive-Noise: per-group pooled LoRAs have held-out-seed verdicts.
3. ➖ Cross-Pair (OPTIONAL smoke — downgraded 2026-07-22, not a publication gate): single-pair→sibling transfer is confounded; the reviewer-credible transfer test is DoD-4 (Group-Wise) with concept-disjoint pairs. See report/experiments-log.md EXP-03.
4. ⚠️ Group-Wise: within-group pooled LoRAs read for G1–G4+G6 (or the honest subset).
5. ⚠️ Scale: four-quadrant crossbar evaluated; held-pair×held-seed classified;
   deployment unit chosen (single LoRA vs per-group catalogue).
6. ✅ Negative controls (group-A, internal-force) reported; Mono-free property holds
   at every λ; report/decision-timeline.md reflects each landing.
7. ✅ G5 (entanglement) explicitly deferred with rationale.

## Sub-Scopes

Seven live scopes, in the reading order their numbers fix. Each is standalone: it links to the
others where it depends on them, and no scope holds an order of its own.

- ⚠️ [01-showcase-the-trained-lora](plans/01-showcase-the-trained-lora/MASTER_PLAN.md) — the trained adapter's own measured output as the paper's results figures, under the scope's decision ledger. Root running-order steps 31 to 35
- ⚠️ [02-can-we-trust-the-compose-rate](plans/02-can-we-trust-the-compose-rate/MASTER_PLAN.md) — whether the printed compose rates survive an audit; earns numbered steps only on the promotion condition its own master plan sets
- ⚠️ [03-does-the-correction-cause-composition](plans/03-does-the-correction-cause-composition/MASTER_PLAN.md) — the causal claim. Steps 1, 2, 3, 8, 9, 10, 19
- ⚠️ [04-does-the-fix-reach-unseen-pairs](plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md) — the transfer claim. Steps 4 to 7 and 11. Depends on `compose-scorer`'s `scorer_validated.json`
- ⚠️ [05-when-does-the-outcome-lock-in](plans/05-when-does-the-outcome-lock-in/MASTER_PLAN.md) — the commitment probe, adopting the basin-oracle walk's ledger. Not yet in the running order
- ⚠️ [06-is-the-gap-the-samplers-or-the-models](plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md) — the sampler's share of the correction against the model's, then three composition rules on one dose axis. Nothing started
- ⚠️ [07-writing-the-paper](plans/07-writing-the-paper/MASTER_PLAN.md) — the manuscript. Steps 12 to 18
- ⚠️ [standing/literature](plans/standing/literature/MASTER_PLAN.md) — standing: what the field already knows, and the source behind every idea-trying run
- ✅ [compose-scorer, now completed and out of the live tree](artifacts/plans/completed/compose-scorer/MASTER_PLAN.md) — the reusable instrument that tells a two-animal composition from a chimera blend; emits `scorer_validated.json`

## Plans
(One plan file per pyramid rung, grouped under `plans/rungs/`. Detailed phase
files are archived under `artifacts/plans/archived/phases/` and referenced from each rung plan;
`artifacts/plans/archived/phases/PHASE_MAP.md` is the retired 8-phase orchestrator.)
- ⚠️ rungs/01-overfit.md — beachhead + taxonomy breadth + negative controls (DoD 1, 6)
- ⚠️ rungs/02-survive-noise.md — seed-pooled LoRA, held-out seeds, per group (DoD 2)
- ⚠️ rungs/03-cross-pair.md — held-out-pair transfer probe (DoD 3)
- ⚠️ rungs/04-group-wise.md — within-group pooling, is "group" a unit (DoD 4)
- ⚠️ rungs/05-scale.md — one LoRA, four-quadrant crossbar, or catalogue fallback (DoD 5)
- ⚠️ rungs/06-supervisor-briefing.md — communicate current state + name next moves, plain-speak'd (delivery, not a rung)

## Environment Context
See `environment/00-INDEX.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in any scope. `CLAUDE.md` at the repo
root holds the rules for how runs are classified, recorded, and allowed to move
a plan.

## Context
What this project is about in the real world lives in `context/`. Start at
`context/00-INDEX.md`. A plan that names a column, a code, an ID or a real-world object (`r_t`,
`arm`, `pair_slug`, a chimera, an animal pair) links its entry there rather than explaining it.

## Runbook
How to do the recurring things here by hand lives in `runbook/`. Start at
`runbook/00-INDEX.md`. Check there before writing out a command sequence or click path for
something done here regularly.

## 🖥️ Viewing results (web apps)

The fastest way to *see* the results is the **LoRA Inspector** — a Flask app with four
tabs: **CFG-mask ablation** (no-LoRA floor) · **LoRA residual** (the epoch × λ morph +
MDS trajectory) · **MDS large** · **LoRA + CFG-mask**. Pair dropdown top-right.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/build_lora_manifest.py        # (re)build the manifest the inspector reads
bash scripts/run_lora_inspector.sh        # serves 127.0.0.1:5050 and prints the exact tunnel line
#   from your laptop:  ssh -L 5050:localhost:5050 <cluster-node>   (it prints the node)
#   then open:         http://localhost:5050
```

Live checkpoint viewer (group-A student runs, writes side-by-side PoE|Mono|student PNGs):

```bash
$PY -m scripts.watch_and_visualize --ckpt-dir <ckpt-dir> --pair "a cat|a dog" --seed 42
```

## Glossary

The kept terms, one plain line each. The plan prose uses these; the commands and Δ/ε notation stay exact.

- **PoE (Product-of-Experts):** composing "a cat and a dog" by adding the model's separate opinions about each prompt. It usually fails and fuses them.
- **Chimera:** that failure. One animal with parts of both, instead of two animals.
- **Mono / the ceiling:** the cheat that works — give the model the literal joined prompt "a cat and a dog". It composes fine, but it defeats the point, so we only use it as the target.
- **Mono-free:** at test time the LoRA never sees the joined prompt. That is the whole point.
- **The oracle:** the correction computed from the joined prompt, the thing a trained LoRA is imitating. It is the target and not a shippable method: at full dose it reproduces the Mono render by construction.
- **The residual (`r_t = ε̃_J − ε̃_PoE`, also `Δ_t`):** the step-by-step correction, the gap from broken-PoE toward the Mono target.
- **LoRA (rank-8, cross-attention / `attn2`):** a small set of extra weights bolted onto the layer where the text prompt enters. What we train.
- **λ (lambda), `PoE+λ·R`:** the dial for how much of the correction to add. 0 is off (identical to plain PoE), 1 is full.
- **Seed vs pair:** a seed re-rolls only the starting noise; a pair swaps the two concepts. A new pair is the harder test.
- **Cell:** one (pair, seed) training or evaluation point.
- **Crossbar / `in_in` / `out_in` / `out_out`:** the 2×2 test grid — pair seen/unseen crossed with seed seen/unseen. `out_out` (both new) is the hardest.
- **Task D:** the check of whether the LoRA's correction points along the shared group direction (a cosine), separate from whether the picture composes.
- **MDS:** a 2-D plot of the denoising paths, used to see the corrected path bend toward the joint target.
