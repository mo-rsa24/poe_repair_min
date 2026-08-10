# Closing the Compositional Gap

The paper, and everything it needs. If you have not opened this window in a while, this file
is where you start, and you should not need to hold anything in your head that is not written
below.

## Start here, cold

1. **`python3 scripts/plan_pulse.py --brief`** (the session-start hook already ran it): what is
   running, the last thing that finished, the single next action, what to launch so it cooks
   while you write, and what to write meanwhile. That is today's answer.
2. **The figure register, `paper/iclr/figures.md`**: eight slots, each the claim a figure will
   make. What is not yet built or fillable there is what the paper still cannot show, and it is
   the whole reason any experiment below runs.
3. **The paper table in the repo root `MASTER_PLAN.md`**: the ordered steps and what each waits
   on. This scope keeps no competing copy of it.

## The three sub-scopes, and which group of work lives in each

| Open this | When you are doing | Its runs, by group |
|---|---|---|
| `plans/interaction-term/` | the causal claim: the correction exists, causes composition, has a timing window, is learnable | group 1 hypothesis runs (the dose sweep, the window pair), group 3 figure runs (slots F2 to F7), group 8 robustness (the seed spread inside each sweep) |
| `plans/animals-compose-transfer/` | the transfer claim: one LoRA composes pairs it never saw | group 1 hypothesis runs (the pooled read, the 15-run leave-one-pair-out), group 5 the size-matched baseline pool, group 3 figure runs (slot F8) |
| `plans/paper-iclr/` | the writing: spine, sections via `/draft-section`, layout, the build | no runs; it consumes the register and the review files |

## The work, by group

Every plan in this scope, filed under the group whose rules it answers to. The files live with
their claim (that is what keeps a design beside its verdict); this listing is how you navigate
by group. Statuses roll up from the review files on every `/sync-plan-tree` pass.

**Hypothesis runs: test the core claim.** Seen to the end unless the foundation or the science
is wrong. Careful design is critical; they answer to benchmarks and faithfulness.

| Plan | Tests | Status |
|---|---|---|
| `interaction-term/plans/03-dose-response.md` | more correction, more composition, two flat controls | ◑ re-score owed |
| `interaction-term/plans/04-window-pair.md` | when in the denoising run the correction matters | ⚠️ |
| `interaction-term/plans/05-cache-analyses.md` | size follows noise; the paths fork; low-rank enough to learn | ◑ one decision left |
| `interaction-term/plans/06-corroborations.md` | the same story from three independent sides | ⚠️ |
| `interaction-term/plans/02-mechanism-reprobe.md` | the fix changes what a word paints, not where it looks | ◑ one decision left |
| `animals-compose-transfer/plans/02-wire-scorer-eval-hook.md` | the three live curves (the sweep's safety gate) | ⚠️ smoke owed |
| `animals-compose-transfer/plans/03a-phase1-pooled.md` | does the fix transfer at all | ◑ read incomplete |
| `animals-compose-transfer/plans/03-run-A-leave-one-pair-out.md` | transfer as a rate: 15 held-out points | ⚠️ |

**Idea runs: try new techniques to improve results.** Each one names the paper it came from.
Results to `PARKING_LOT.md`; a striking number proposes a hypothesis run, never rewrites one.

| Plan | Would earn | Status |
|---|---|---|
| `interaction-term/plans/07-composition-type.md` | the claim widened to attribute pairs | background, not started |
| (candidates parked with their sources) | see `PARKING_LOT.md` | |

**Figure runs: generate paper figures from settled results only.** The caption can claim no
more than the figure slot's sentence. The register (`paper/iclr/figures.md`) is the scoreboard.

| Plan | Slots | Status |
|---|---|---|
| `interaction-term/plans/10-figures.md` | F1 to F7 | ⚠️ F2 fillable after the re-score |
| `animals-compose-transfer/plans/05-figures.md` | F8, via its A2 to A5 internals | ⚠️ waits on the sweep |

**Reproduction runs: confirm or break an existing number against the original config and
seed.** No standing plans; a rerun lands on the existing claim's evidence tag.

**Baseline runs: competitors to beat.** Frozen the moment they land.

| Plan | Competitor | Status |
|---|---|---|
| `animals-compose-transfer/plans/04-run-B-contrast.md` | the size-matched mixed pool | ⚠️ |
| (held in reserve, trigger written down) | Attend-and-Excite; SuperDiff AND | `PARKING_LOT.md` |

**Robustness and generalization** (groups 8 and 9): the seed spread lives inside each sweep and
is reported in its review answers; other models and samplers are
`interaction-term/plans/08-replication.md`, background, scaffold shelved. **Ablations of our
method** (group 7): not yet planned; they enter as review questions when `/experiment-planner`
next runs on a result scope.

**The structural separation** rides underneath all of it: reading lives in
`standing/literature/`, designs in each scope's `plans/`, verdicts in each scope's `review/`,
and the two print gates in `interaction-term/plans/09-print-gates.md` guard the wording.
The writing itself is `paper-iclr/`, which runs nothing and consumes everything.

## How this scope works: figures first, then the writing

The foreground is group 3: the register is the scoreboard, and a reserved slot is not idle, it
is a placeholder carrying the claim its caption will make, written before the experiment runs.
Work in this scope means resolving slots.

**The trigger to start writing in earnest is 5 to 10 slots resolved** (built, fillable, or
honestly downgraded with the boundary stated). Do not wait for all eight: the skeleton plus
five real figures is a draft; eight perfect figures with no prose is not. The results skeleton
(paper step 16) starts today, because it writes XX against every slot by design.

Groups 1, 2, 5 and 9 run in the background, blocking nothing. A background result that earns a
slot MOVES into the paper table, visibly.

## Mission
Publish the claim: Product-of-Experts composition fails because a specific correction term is
missing; injecting that term causally restores composition (dose response with flat controls);
a rank-8 LoRA learns it Mono-free and transfers to pairs it never trained on.

## Sub-Scopes
- ⚠️ plans/interaction-term/ — the causal claim. Paper steps 1, 2, 3, 8, 9, 10, 19.
- ⚠️ plans/animals-compose-transfer/ — the transfer claim. Paper steps 4 to 7 and 11.
- ⚠️ plans/paper-iclr/ — the manuscript. Paper steps 12 to 18.

## Expected Outcome
A submitted rough draft whose every number traces to an answered review question, every figure
to a register slot, and every citation to the reading register.

## Definition of Done
The eight register slots resolved (built, or honestly downgraded with the boundary stated), the
sections compiled through `/draft-section`, and the two `/pressure-test` print gates passed.

## Running order
This scope keeps no order of its own. The single order is the paper table in the repo root
`MASTER_PLAN.md`; the sub-scopes' rows carry their step numbers there.

## Environment Context
See `docs/ENVIRONMENT.md`. Read before drafting or checking any plan in this scope.
