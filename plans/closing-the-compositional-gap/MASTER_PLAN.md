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

## The three folders, in plain words

| Open this | When you are doing | Its runs, by group |
|---|---|---|
| `does-the-correction-cause-composition/` | the causal claim: the correction exists, causes composition, has a timing window, is learnable | group 1 hypothesis runs (the dose sweep, the window pair), group 3 figure runs (slots F2 to F7), group 8 robustness (the seed spread inside each sweep) |
| `does-the-fix-reach-unseen-pairs/` | the transfer claim: one LoRA composes pairs it never saw | group 1 hypothesis runs (the pooled read, the 15-run leave-one-pair-out), group 5 the size-matched baseline pool, group 3 figure runs (slot F8) |
| `writing-the-paper/` | the writing: spine, sections via `/draft-section`, layout, the build | no runs; it consumes the register and the review files |

## Every filename says its run kind

Inside any of the three folders, `ls plans/` sorts your work by run kind, because the kind is the
filename's first word:

| Prefix | Run kind | Its rule |
|---|---|---|
| `hypothesis-NN-` | tests the core claim | seen to the end unless the science is wrong; a pre-registered bar in its review file |
| `baseline-NN-` | a competitor to beat | frozen the moment it lands |
| `figure-NN-` | draws a settled result for the paper | the caption may claim no more than its register slot's sentence |
| `idea-NN-` | tries a new technique | names the paper it came from; result to `PARKING_LOT.md`, never to a claim |
| `generalization-NN-` | other models, datasets, samplers | instances chosen before any runs; a failure bounds the claim's scope |
| `instrument-NN-` | not a run: a tool or a choice fixed before results | judged by whether it can fail, not by what it found |
| `gate-NN-` | not a run: a literature check before print | a `/pressure-test` verdict |
| `writing-NN-` | not a run: manuscript prose | consumes the register and the review files |

## The work, by kind

Every plan, filed under the kind whose rules it answers to. Statuses roll up from the review
files on each `/sync-plan-tree` pass.

**Hypothesis runs (`hypothesis-*`): test the core claim.** Seen to the end unless the foundation or the science
is wrong. Careful design is critical; they answer to benchmarks and faithfulness.

| Plan | Tests | Status |
|---|---|---|
| `correction/hypothesis-02-more-correction-more-composition.md` | more correction, more composition, two flat controls | ◑ re-score owed |
| `correction/hypothesis-03-when-in-the-run-it-matters.md` | when in the denoising run the correction matters | ⚠️ |
| `correction/hypothesis-04-what-the-cached-runs-already-show.md` | size follows noise; the paths fork; low-rank enough to learn | ◑ one decision left |
| `correction/hypothesis-05-the-same-story-from-three-sides.md` | the same story from three independent sides | ⚠️ |
| `correction/hypothesis-01-what-the-fix-changes-inside-the-model.md` | the fix changes what a word paints, not where it looks | ◑ one decision left |
| `transfer/instrument-02-three-live-curves-while-training.md` | the three live curves (the sweep's safety gate) | ⚠️ smoke owed |
| `transfer/hypothesis-01-does-one-pooled-fix-transfer-at-all.md` | does the fix transfer at all | ◑ read incomplete |
| `transfer/hypothesis-02-transfer-as-a-rate-over-fifteen-pairs.md` | transfer as a rate: 15 held-out points | ⚠️ |

**Idea runs (`idea-*`): try new techniques to improve results.** Each one names the paper it came from.
Results to `PARKING_LOT.md`; a striking number proposes a hypothesis run, never rewrites one.

| Plan | Would earn | Status |
|---|---|---|
| `correction/idea-01-does-it-hold-for-attribute-pairs.md` | the claim widened to attribute pairs | background, not started |
| (candidates parked with their sources) | see `PARKING_LOT.md` | |

**Figure runs (`figure-*`): generate paper figures from settled results only.** The caption can claim no
more than the figure slot's sentence. The register (`paper/iclr/figures.md`) is the scoreboard.

| Plan | Slots | Status |
|---|---|---|
| `correction/figure-01-the-seven-paper-figures.md` | F1 to F7 | ⚠️ F2 fillable after the re-score |
| `transfer/figure-01-the-transfer-figures.md` | F8, via its A2 to A5 internals | ⚠️ waits on the sweep |

**Reproduction runs: confirm or break an existing number against the original config and
seed.** No standing plans; a rerun lands on the existing claim's evidence tag.

**Baseline runs (`baseline-*`): competitors to beat.** Frozen the moment they land.

| Plan | Competitor | Status |
|---|---|---|
| `transfer/baseline-01-the-size-matched-control-pool.md` | the size-matched mixed pool | ⚠️ |
| (held in reserve, trigger written down) | Attend-and-Excite; SuperDiff AND | `PARKING_LOT.md` |

**Robustness and generalization** (groups 8 and 9): the seed spread lives inside each sweep and
is reported in its review answers; other models and samplers are
`correction/generalization-01-other-models-and-samplers.md`, background, scaffold shelved. **Ablations of our
method** (group 7): not yet planned; they enter as review questions when `/experiment-planner`
next runs on a result scope.

**The structural separation** rides underneath all of it: reading lives in
`standing/literature/`, designs in each scope's `plans/`, verdicts in each scope's `review/`,
and the two print gates in `correction/gate-01-two-literature-checks-before-print.md` guard the wording.
The writing itself is `writing-the-paper/`, which runs nothing and consumes everything.

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
- ⚠️ plans/does-the-correction-cause-composition/ — the causal claim. Paper steps 1, 2, 3, 8, 9, 10, 19.
- ⚠️ plans/does-the-fix-reach-unseen-pairs/ — the transfer claim. Paper steps 4 to 7 and 11.
- ⚠️ plans/writing-the-paper/ — the manuscript. Paper steps 12 to 18.

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

In the tables above, `correction/` and `transfer/` are shorthand for the two claim folders'
`plans/` directories.
