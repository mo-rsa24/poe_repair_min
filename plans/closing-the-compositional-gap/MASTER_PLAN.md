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

## Where every run group lives in this tree

The six-plus-three groups (`~/.claude/EXPERIMENT_CONVENTIONS.md`), located, so "which group am
I working in right now" always has an answer:

- **1, hypothesis:** the two result sub-scopes' `plans/`, judged in their `review/` files.
- **2, new ideas:** the background pool in the root master plan; results to `PARKING_LOT.md`;
  each names its source paper in the reading register.
- **3, figures:** the register, plus `interaction-term/plans/10-figures.md` and
  `animals-compose-transfer/plans/05-figures.md`. The foreground group, per the cadence below.
- **4, reproduction:** no standing plans; a rerun lands on the existing claim's evidence tag.
- **5, baselines:** `animals-compose-transfer/plans/04-run-B-contrast.md` (the size-matched
  pool), plus Attend-and-Excite and SuperDiff held in reserve in `PARKING_LOT.md` with their
  trigger written down.
- **7, ablations of our method:** not yet planned; they enter as review questions when
  `/experiment-planner` next runs on a result scope.
- **8, robustness:** the seed axes inside the existing sweeps; the spread is reported in each
  review answer rather than as separate plans.
- **9, generalization:** `interaction-term/plans/08-replication.md` (other models and
  samplers), background, its scaffold shelved at `plans/shelved/cross-model-replication`.

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
