# Closing the Compositional Gap

The paper. One parent scope holding everything the manuscript needs, so the folder listing
under `plans/` says what this project is currently for.

## Mission
Publish the claim: Product-of-Experts composition fails because a specific correction term is
missing; injecting that term causally restores composition (dose response with flat controls);
a rank-8 LoRA learns it Mono-free and transfers to pairs it never trained on.

## Sub-Scopes
- ⚠️ plans/interaction-term/ — the causal claim: the correction exists, causes composition,
  has a timing window, and is low-rank enough to learn. Paper steps 1, 2, 3, 8, 9, 10, 19.
- ⚠️ plans/animals-compose-transfer/ — the transfer claim: one LoRA composes pairs it never
  saw, against a size-matched control pool. Paper steps 4 to 7 and 11.
- ⚠️ plans/paper-iclr/ — the manuscript itself: spine, sections via /draft-section, the figure
  register and layout, the build. Paper steps 12 to 18.

## Expected Outcome
A submitted rough draft whose every number traces to an answered review question, every figure
to a register slot, and every citation to the reading register.

## Definition of Done
The eight register slots resolved (built, or honestly downgraded with the boundary stated), the
sections compiled through /draft-section, and the two /pressure-test print gates passed.

## How this scope works: figures first, then the writing

The cadence, in the six groups' terms:

**The foreground is group 3: runs that generate figures towards the paper.** The figure
register (`paper/iclr/figures.md`) is the scoreboard: eight slots, each either built, fillable,
or reserved. A reserved slot is not idle, it is a placeholder carrying the claim its caption
will make, written before the experiment runs, which is the expectation on record. Work in this
scope means resolving slots.

**The trigger to start writing: 5 to 10 slots resolved**, where resolved means built, fillable,
or honestly downgraded with its boundary stated. At that point the writing plans start in
earnest (`/draft-section` on method and results against the register); the results skeleton
(step 16) can start today because it writes XX against every slot by design. Do not wait for
all eight: the skeleton plus five real figures is a draft, eight perfect figures with no prose
is not.

**The background, in parallel and blocking nothing: groups 1, 2, and 5.** Hypothesis runs
beyond the register's needs, new-idea runs and ablations (each naming the paper it came from,
results to `PARKING_LOT.md`), and the baselines held in reserve with their trigger written
down. The root master plan's background pool is their home; a background result that earns a
slot MOVES into the paper table, visibly.

Opening this folder cold: read the register first, then the root's paper table. The register
says what the paper still cannot show; the table says what to do about it today.

## Running order

This scope keeps no order of its own. The single order is the paper table in the repo root
`MASTER_PLAN.md`; the three sub-scopes' rows carry their step numbers there.

## Environment Context
See `docs/ENVIRONMENT.md`. Read before drafting or checking any plan in this scope.
