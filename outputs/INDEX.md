# outputs/: who owns each folder and what state it is in

**The rule this file used to state, and no longer does:** "folder names are not renamed: scripts
write into these exact paths." That rule is gone. Every path in this project now comes from one
module, `poe_repair/paths.py`, so a rename is one edit there plus a move on disk, not an edit to
159 call sites. Renamed on 2026-08-24 as stage 4 of `plans/retrofit-poe-repair-min.md`.

States: **live** (a register figure reads from it), **supporting** (cited, not drawn from),
**cold** (superseded; kept, never deleted).

| Was in `outputs/` | Now at | Size | Owner | State |
|---|---|---|---|---|
| `interaction_term/` | `outputs/interaction_term/` (unchanged; split across this filesystem and the mount, see `poe_repair/paths.py`'s module docstring) | 71M here, 6.3G+ on the mount | does-the-correction-cause-composition plans 03/04/05 | **live**: F2 cells, fork paths, window grids land here |
| `animals_compose_transfer/` | `artifacts/results/does-the-fix-reach-unseen-pairs/` | 5.6G | animals scope plans 01/03a | **live**: the pool, fail rates, the pooled run feeding F8 |
| `compose_scorer/` | `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/` | 15M | completed/compose-scorer | **live**: the contract `scorer_validated.json` and the F1 evidence |
| `group_a_failure/` | `artifacts/results/residual-dynamics/correction-outside-the-unet/` | 6.5G | the negative controls (DoD 6) | **supporting**, irreplaceable: training runs, not cheaply regenerable |
| `residual_diagnostics/` | `artifacts/results/residual-dynamics/residual-between-mono-and-poe/` | 857M | shelved phases 02 | **supporting**: F3's precursor reads |
| `poe/` | `artifacts/results/poe-blends-instead-of-composing/poe-baseline-samples/` | 186M | base PoE references | **supporting**: λ=0 exemplars for F1 |
| `manifold_cache/` (symlink into `artifacts/`) | unchanged | | does-the-correction-cause-composition plan 06 | **supporting**: the CLIP axes for the manifold slide |
| `conditioning_window/`, `conditioning_window_lora/` | `artifacts/results/when-the-correction-must-arrive/cfg-window-without-lora/`, `.../cfg-window-with-lora/` | 6.1G | shelved rungs | **cold**: superseded by plan 04's W1/W2 design |
| `cross_pair_lora_pooling/`, `cross_seed_lora_pooling/`, `lora/` | unchanged (config only, not renamed by the walk) | ~32K | shelved rungs | **cold** |
| `presentation/` | unchanged, held for a look pass before any verdict | 6M | old slides | **cold**, undecided |

**`artifacts/rung1-overfit/lora/` and `artifacts/rung2-survive-noise/cross_seed/` are not listed
here**: they were never under `outputs/`. `rung1-overfit/lora` moved to
`artifacts/results/can-lora-learn-a-residual-that-corrects-poe/one-pair-one-seed/` in the same
sweep. `rung2-survive-noise/cross_seed` did **not** move: the mount holds a second, disjoint
four-seed pooled-run set under that same old name, and the sweep's safety floor forbade touching
`/datasets` bytes, so renaming only this filesystem's copy would have broken `paths.resolve()`'s
one-name-both-filesystems design. See `poe_repair/paths.py`'s `HELD_OUT_SEEDS` constant for the
full note and what finishes it.

The rule that keeps this honest: a folder feeding a register slot is live; when its slot is
built and the paper freezes, it drops to supporting. Nothing here is deleted; cold folders are
candidates for the /datasets migration if home fills.
