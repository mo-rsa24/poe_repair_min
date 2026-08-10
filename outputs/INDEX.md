# outputs/: who owns each folder and what state it is in

Everything here is data, ~22GB on /home-mscluster, all regenerable-or-not as marked. States:
**live** (a register figure reads from it), **supporting** (cited, not drawn from),
**cold** (superseded; kept, never deleted). Folder names are not renamed: scripts write into
these exact paths.

| Folder | Size | Owner | State |
|---|---|---|---|
| `interaction_term/` | 3.4G | interaction-term plans 03/04/05 | **live**: F2 cells, fork paths, window grids land here |
| `animals_compose_transfer/` | 5.6G | animals scope plans 01/03a | **live**: the pool, fail rates, the pooled run feeding F8 |
| `compose_scorer/` | 15M | completed/compose-scorer | **live**: the contract `scorer_validated.json` and the F1 evidence |
| `group_a_failure/` | 6.5G | the negative controls (DoD 6) | **supporting**, irreplaceable: training runs, not cheaply regenerable |
| `residual_diagnostics/` | 857M | shelved phases 02 | **supporting**: F3's precursor reads |
| `poe/` | 186M | base PoE references | **supporting**: λ=0 exemplars for F1 |
| `manifold_cache/` (symlink into `artifacts/`) | | interaction-term plan 06 | **supporting**: the CLIP axes for the manifold slide |
| `conditioning_window/`, `conditioning_window_lora/` | 6.1G | shelved rungs | **cold**: superseded by plan 04's W1/W2 design |
| `cross_pair_lora_pooling/`, `cross_seed_lora_pooling/`, `lora/` | ~32K | shelved rungs | **cold** |
| `presentation/` | 6M | old slides | **cold** |

The rule that keeps this honest: a folder feeding a register slot is live; when its slot is
built and the paper freezes, it drops to supporting. Nothing here is deleted; cold folders are
candidates for the /datasets migration if home fills.
