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

## Running order

This scope keeps no order of its own. The single order is the paper table in the repo root
`MASTER_PLAN.md`; the three sub-scopes' rows carry their step numbers there.

## Environment Context
See `docs/ENVIRONMENT.md`. Read before drafting or checking any plan in this scope.
