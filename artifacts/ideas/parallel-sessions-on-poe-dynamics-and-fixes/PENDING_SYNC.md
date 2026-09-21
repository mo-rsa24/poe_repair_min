# Edits to shared plan-tree files, waiting for one sync

Sessions running in parallel append here instead of editing the root running order, a scope's
plans table, or existing reading-register rows. One `/sync-plan-tree` run at the end reads this
list, applies each line, and empties the file.

One line per edit: which session, which file, what to add.

| Session | File | What to add |
|---|---|---|
| E | `MASTER_PLAN.md` (root), `## The paper: what has to land` | a row: step 56, `06-is-the-gap-the-samplers-or-the-models/baseline-07-zero-order-search-over-the-initial-noise`, "one round of zero-order search over the cached initial noise (Ma et al., arXiv 2501.09732) at N 8, σ 0.1 and 0.3, plain PoE, the validated compose scorer as verifier on the final image and on x0-hat at step 10; random search at N 8 read off the existing renders", waits on nothing |
| E | `plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md`, `## Plans` table | a row: 56, [zero-order search over the initial noise](plans/baselines/11-zero-order-search-over-the-initial-noise.md), establishes a baseline, the same one-line description, waits on nothing; and the scope's opening note "steps 24 to 30, 49 to 51 and 55" gains "and 56" |
| E | `MASTER_PLAN.md` (root), `## The paper: what has to land` | a row: step 57, `06-is-the-gap-the-samplers-or-the-models/baseline-08-noise-search-on-top-of-the-adapter`, "the same eight-candidate noise search with the rank-32 step-30050 adapter attached at λ 1.2, the verifier picking by compose count then sharpness; per-seed sharpness ratios against the adapter pivot, plain PoE and Mono", waits on 56 |
| E | `plans/06-is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md`, `## Plans` table | a row: 57, [noise search on top of the adapter](plans/baselines/12-noise-search-on-top-of-the-adapter.md), establishes a baseline, the same one-line description, waits on 56; the opening note gains "and 57" |
