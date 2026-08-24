# Literature: what the field already knows, kept current

## Mission
Keep a standing, current answer to two questions that every other scope asks and
none of them owns: has someone already done this, and is the method we are about
to try sound? Reading is continuous and belongs to no single claim, so it lives
here instead of being re-done inside each experiment scope.

## Objectives
1. **Coverage.** Every claim the paper makes has a read set behind it: what the
   nearest prior work is, and where our claim differs.
2. **Soundness.** Every run that tries an idea (a sampler, a method, an
   algorithm) answers to a paper, not to a hunch.
3. **Currency.** New work in Product-of-Experts composition, compositional
   diffusion, and LoRA-based correction is seen while it still matters.

## Goals
1. A reading register with one entry per paper: what it claims, what it proves,
   what we borrow, and which of our plans it touches.
2. Every "tries an idea" run in `PARKING_LOT.md` names the paper it came from.
3. A standing `/pressure-test` verdict on the paper's headline claim, re-run
   whenever the claim changes.

## Expected Outcome
The related-work section writes itself from the register, and no experiment is
designed against a question the field already settled.

## Definition of Done
This scope is never done. It is a standing scope, the same recurring shape as
`plans/standing/artifact-reconciliation/`. Its plan is re-entered, not closed.

## Plans
- ⚠️ 01-reading-register.md (standing / recurring)

## Running order

This scope keeps no order of its own. The single flat order across every scope
and level is the `## Running order` table in the repo root `MASTER_PLAN.md`.

## Environment Context
See `environment/00-INDEX.md` for this project's environment/architecture facts.
Reading needs no GPU and no queue, so nothing in this scope is submitted as a
job. Network access is needed for paper fetches, which the compute nodes have.
