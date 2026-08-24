# Evidence: what is here and what the paper does with it

One row per item. A folder whose name starts with a slot id (F2, F6, F7, F8) feeds that row of
the figure register, `paper/iclr/figures.md`. A folder without a prefix is supporting or
superseded and is visibly not load-bearing. **Caption owed** means the result is good and its
caption or writeup has to be rewritten fresh and plain before the paper may use it.

Moved here from `docs/evidence/INDEX.md` during the retrofit sweep; every item path below is
updated to where the retrofit moved that item.

## Feeds the paper

| Item | Register slot | State |
|---|---|---|
| `paper/iclr/figures/compose-rate-as-correction-rises.png` | F2, the headline dose figure | result stands (oracle 7% to 93%, controls flat); the exact percentages wait on the re-score; **caption owed** |
| `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/` (`QUERY.md`, `geometry_vs_transfer.png`, `result.json`) | F6, why it is learnable | the held-out projection licenses nothing about transfer. `QUERY.md` is the argument the caption must not exceed |
| `artifacts/results/which-way-the-correction-points/what-the-spectrum-measures/` (`QUERY.md`, `result.json`) | F6, why it is learnable | **the spectrum's floor does not control for ‖r_t‖ spread, and against one that does the pooled stack is 1.4x at k=8 rather than 10.7x.** What direction structure survives is within single runs (4.8x at k=8), not across pairs (1.2x). F6's shared-structure argument does not stand and the slot needs a decision |
| `artifacts/results/residual-dynamics/content-change-relative-to-attention-change/` (`measure-fairness.md`, `RUN_ON_THIS_NODE.md`) | F7, the mechanism panel | replicated, median 1.52x over 64 cells. `measure-fairness.md` is why the obvious measure gives the opposite answer, which the appendix needs; **caption owed** |
| `artifacts/results/does-the-fix-reach-unseen-pairs/hard-vs-easy-transfer/` (`demo.py`, `hard_vs_easy_transfer.png`) | F8, transfer | supporting: the hard-vs-easy split behind the transfer read; **caption owed, and the figure regenerates from `demo.py`** |
| `artifacts/results/how-much-correction-is-needed/plausibility_climb.png` | corroborates F3/F5 | keep; the corrected sign reading lives in the cache-analyses review |
| `artifacts/results/when-the-correction-must-arrive/commitment-step/` (`QUERY.md`, `result.json`, `commitment-step-per-pair.png`) | no slot yet, EXP-01 of `report/experiments-log.md` | the step where a run's estimate settles varies by pair (medians 18 to 36 over 162 cells), but every pair settles after the correction has stopped working. EXP-04 then showed it predicts nothing about the window, so this measure is not tracking the decision |
| `artifacts/results/when-the-correction-must-arrive/window-vs-commitment/` (`QUERY.md`, `result.json`, `window-vs-commitment.png`) | no slot yet, EXP-04 of `report/experiments-log.md` | **all 8 pairs peak at the same window, steps 0 to 10, while their settling steps span 13.** The window does not move with the pair, so the adapter's schedule cannot be misplaced per pair and composition is decided inside the first fifth of the run. Left-censored: the best window is the earliest the grid holds |

## Supporting, not drawn from

| Item | Why it stays |
|---|---|
| `artifacts/_quarantine/results-archive/group-a-failure.md`, `internal-force-failure.md` | the negative controls behind the paper's "external correctors and internal forces fail" sentence (DoD 6) |
| `artifacts/_quarantine/results-archive/{lora-success,residual-diagnostics,conditioning-window}.md` | May-era writeups, superseded by the review files; history only |
| `report/instrument_smoke.md`, `report/normalization_preregistration.md` | instrument provenance: what was committed before results were read |
| `report/decision-timeline.md` (repo root), `report/RESULTS_SUMMARY.md` | the running record; RESULTS_SUMMARY defers to the review files where they disagree |
| the regeneration code for F6/F8, now merged beside each figure's evidence folder under `artifacts/results/which-way-the-correction-points/` | the throwaway code that produced `F6`'s and `F8`'s images; regeneration path, never cited |
| `plans/shelved/artifact-reconciliation/inventory/` | the artifact-reconciliation scope's four output documents |

## Not paper material

`dl-scene/`, `learning-captures/`, `recap/` (already archived), `artifacts/notes/batch-shape-nondeterminism/`
(formerly `show-me/`), `todoist-staging/`: learning and tooling artifacts. They serve sessions, not
the manuscript.
