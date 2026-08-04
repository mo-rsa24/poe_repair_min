# Integrity Check and Disposition (lora / cross_seed / cross_pair + training_cache)

**Generated**: 2026-07-21
**Roots**: `R:` repo (`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`) · `D:` datasets (`/datasets/mmolefe/poe_repair_min/outputs`)
**Method**: read-only. Checkpoints load-tested with `torch.load(weights_only=True)` (CPU); caches checked by shard-count vs the per-cell contract plus a spot-open of 12 random cells; run dirs checked for internal JSON consistency. Nothing was written, moved, or deleted.
**Re-run**: `inventory/scripts/03_integrity.py` (checkpoints), `inventory/scripts/03b_cache_check.py` (caches).
**Chains from**: `01-artifact-inventory.md` (W&B status) and `02-two-root-classified.md` (two-root scope).

## Headline

- **Every checkpoint tested loads cleanly** (16/16 PASS: final checkpoint of each kept run dir plus all four suspects). All are rank-8, 420 LoRA keys, sample shape `(8, 640)`. **No file corruption anywhere.**
- The four W&B-flagged suspects are **suspect by run completeness, not by bytes**: their checkpoints on disk are intact and loadable. `d5b2706v` (the benign sync-fatal) is confirmed intact at step 16510.
- The only real data gaps are: **stale top-level cache manifests**, an **incomplete `manifold_cache`** (0/7 complete by its own inventory), the **un-evaluated `all_groups/main`** run, and the **known 0-byte stubs**. None is silent corruption.
- One nuance that would otherwise read as a failure: **24 cache cells hold a single residual shard by design** (`build_eval_cache.py` writes one `step_000.pt` with zeroed eps; only `x_t` is read for held-out-pair eval). These are correct, not truncated.

## Checkpoint load test (all PASS)

| Artifact | Load | LoRA keys / shape | W&B status | Suspect? | Decision | Reason |
|---|---|---|---|:--:|---|---|
| R: `lora/cat_dog/.../lora_step_062500.pt` (+ 080000) | ✓ PASS | 420 · (8,640) | worked (local) | no | **keep** | Flagship; both README-headline and final checkpoint load. |
| R: `lora/a_typewriter__x__a_cactus/.../133632/lora_step_080000.pt` | ✓ PASS | 420 · (8,640) | `wag4z592` finished | no | **keep** | The real (post-false-start) run; intact. |
| R: `lora/a_typewriter__x__a_cactus/.../131542/lora_step_000000.pt` | ✓ PASS | 420 · (8,640) | `lu7g7svh` false-start | **yes** | **discard** | Whole `131542` dir is the aborted false-start (only a step-0 init). Superseded by `133632`. Byte-valid but useless. |
| R: `lora/a_camel__x__a_desert_landscape/.../lora_step_080000.pt` | ✓ PASS | 420 · (8,640) | `8p1spi5b` finished | no | **keep** | Finished run. Exploratory pair (not a plan-09 representative), but a valid artifact. |
| R: `cross_seed/.../task_b_learning_curve/k01_pick1__ep1600/lora_step_080000.pt` | ✓ PASS | 420 · (8,640) | `hbpotmnk` finished, `verdict.json="ok"` | no | **keep** | Finished, verdict ok (ep 1600 / step 80000). |
| R: `cross_seed/.../task_c_per_seed_ceiling/per_seed_s9__ep1600/lora_step_016510.pt` | ✓ PASS | 420 · (8,640) | `d5b2706v` benign sync-fatal | **yes** | **keep** | Checkpoint intact at step 16510 as promised. `verdict.json` is **absent** (W&B shutdown died before it was written) — cosmetic only; the weights are fine. |
| R: `cross_pair/all_groups/main/lora_step_030000.pt` | ✓ PASS | 420 · (8,640) | `0y9un0o4` died-early (after `2em6frqv`) | **yes** | **re-run** | Bytes fine, but the run **died early and was never crossbar-evaluated** (see caches/results below). Resume-to-finish + run `sample_crossbar` if the all-groups claim is wanted. |
| R: `cross_pair/all_groups/dryrun/lora_step_000005.pt` | ✓ PASS | 420 · (8,640) | smoke | no | **discard** | 5-step smoke; superseded by `main`. |
| R: `cross_pair/within_group/g6/main/lora_step_030000.pt` | ✓ PASS | 420 · (8,640) | `ow1jo0xq` train died-early; eval finished | **yes** | **keep** | Train stopped at 30k, but its crossbar **eval at step 20000 completed (43/43 cells)** — a usable evaluated artifact exists. |
| D: `cross_seed/a_dog__x__oil_painting_style/.../k04__ep2000/lora_step_100000.pt` | ✓ PASS | 420 · (8,640) | (bank) | no | **keep** | Seed bank, reaches 100k. |
| D: `cross_seed/a_dolphin__x__an_ocean_wave/.../k04__ep2000/lora_step_100000.pt` | ✓ PASS | 420 · (8,640) | (bank) | no | **keep** | Seed bank, reaches 100k. |
| D: `cross_seed/a_mailbox__x__a_snowfield/.../k04__ep2000/lora_step_080000.pt` | ✓ PASS | 420 · (8,640) | (bank) | no | **keep** (note) | Only reaches **80k** (siblings reach 100k). Load-valid; confirm the short stop was intended, else resume. |
| D: `cross_seed/a_typewriter__x__a_cactus/.../k04__ep2000/lora_step_100000.pt` | ✓ PASS | 420 · (8,640) | (bank) | no | **keep** | Seed bank, reaches 100k. |
| D: `cross_seed/task_b_learning_curve/k04__ep2000_resumed/lora_step_100000.pt` | ✓ PASS | 420 · (8,640) | finished, `verdict.json="ok"` | no | **keep** | The G6 cross-seed headline (ep 2000 / step 100000, verdict ok). |
| D: `cross_seed/task_b_learning_curve/k04__ep200/lora_step_010000.pt` | ✓ PASS | 420 · (8,640) | short run | no | **discard candidate** | Early ep200 config, superseded by `k04__ep2000_resumed`. Keep only if you want the ep200 point. |

Sampling note: I load-tested the final checkpoint per kept run dir plus every suspect (16 of ~230 `.pt` files). Intermediate checkpoints were not each opened; given uniform PASS and identical structure, per-file corruption in the untested ones is unlikely but not proven. `03_integrity.py` can be pointed at the full list if you want exhaustive coverage.

## Cache completeness

| Artifact | Shards / manifest | Status | Decision | Reason |
|---|---|---|---|---|
| D: `training_cache/train/` (8 pairs, 35 cells) | 50/50 shards every cell; 12-cell spot-open OK | ✓ PASS | **keep** | Full cells intact (`embeddings.pt` + `meta.json` + 50 `residuals/step_*.pt`). |
| D: `training_cache/heldout/` (51 pairs, 610 cells) | 586 full + **24 single-shard** | ⚠️ by-design | **keep** | The 24 single-shard cells are the sibling held-out-pair **eval stubs** (`build_eval_cache.py`: one `step_000.pt`, eps zeroed, only `x_t` read). Exactly the 6 siblings (charcoal, drum_set/snowman, fire_hydrant/snowfield, lion/dog, polar_bear/iceberg, wolf/husky) × seeds 9-12. Correct, not truncated. |
| D: `training_cache/{manifest.json, manifest_all/A/B/H.json}` | declare **3 train / 0 heldout** vs **645 on disk** | ❌ stale | **keep data, regenerate manifest** | Manifest drift: the top-level manifests are a May-6/11 snapshot and do not describe the current 645-cell cache. Data is fine; the index is wrong. Rebuild it before trusting any manifest-driven loader that reads these files. |
| R: `manifold_cache/` (vs `inventory.json`) | `n_total=7, n_complete=0, missing_eps=7` | ❌ incomplete | **re-run or discard** | The cache reports itself 0/7 complete: all 7 cells miss their eps trajectories. Only `seed_42_phase1` (n=1 figure) and a partial `a_cat__x__a_dog` bank exist. Backfill via `scripts/manifold/sample_with_trajectory.py` if the latent-manifold figure matters; otherwise discard as an abandoned partial. Orphan relative to the three experiments. |

## Result-vs-manifest consistency

| Run dir | Check | Status | Decision |
|---|---|---|---|
| `cross_pair/within_group/g6/.../eval_crossbar/step_020000` | manifest `n_cells_planned = n_cells_sampled = 43`; `cells.jsonl` = 43 rows; 45 PNGs (43 samples + baseline arms); quadrants `[in_in, out_in]` | ✓ PASS | **keep** — the 43-cell claim is internally consistent. |
| `cross_pair/all_groups/main` | `dataset_meta.json` = 40 cells (5 pairs × 8 seeds); checkpoints to `lora_step_030000.pt`; `samples/` holds **only `per_epoch/`**, **no `cells.jsonl`** | ⚠️ un-evaluated | **re-run eval** — the flagship all-groups LoRA was trained (partially, died-early) but the four-quadrant crossbar (`sample_crossbar` → `contact_sheet` → `task_d_bridge`) was never run. No `out_out` result exists. |
| `cross_seed` task_b (k01, G6-resumed) | `verdict.json = "ok"`, step/epoch match filenames | ✓ PASS | **keep**. |
| `cross_seed` task_c (`per_seed_s9`) | `verdict.json` **absent** | ⚠️ | **keep** — missing verdict is a symptom of the `d5b2706v` sync-fatal shutdown, not a training failure; checkpoint intact. Optionally re-emit the verdict. |

## "0-byte stubs" — CORRECTED 2026-07-21 (they are symlink views, not empty)

The four items below were read as 0-byte by `du`/`find -type f`, which do not
traverse symlinks. On inspection during the reorg (`04`), all four contain
symlinks. Two decisions changed:

| Path | Actual contents | Decision | Reason |
|---|---|---|---|
| R: `outputs/lora/a_dog__x__oil_painting_style` | `seed_42/results` → cross_seed seed bank (10 ckpts) | **keep** | Live alias view onto the pair's cross_seed bank; resolves post-reorg. NOT empty. |
| R: `outputs/lora/a_dolphin__x__an_ocean_wave` | `seed_42/results` → cross_seed seed bank (10 ckpts) | **keep** | Live alias view; resolves. |
| R: `outputs/lora/a_mailbox__x__a_snowfield` | `seed_42/results` → cross_seed seed bank (8 ckpts) | **keep** | Live alias view; resolves. |
| D: `outputs/training_cache_overfit_catdog` | symlink view (5 train + 1 heldout cat_dog seeds) → a **nonexistent** repo `outputs/training_cache` path | **discard (quarantined)** | Symlink view whose links were **already broken before the reorg** (target never existed in this checkout). Moved to `artifacts/_quarantine/broken-symlink-view/`, not deleted. |

## Disposition summary

- **Keep (intact, evaluated or usable):** all `lora` finished runs (cat_dog, typewriter-133632, camel), both `cross_seed` task_b headlines (k01, G6-resumed, both verdict ok), `cross_seed` task_c s9 (intact, verdict cosmetically missing), the four datasets-root seed banks, `cross_pair/within_group/g6` (train + 43/43 eval), the full `training_cache` (train + heldout, including the 24 by-design eval stubs).
- **Re-run:** `cross_pair/all_groups/main` — finish training and run the crossbar eval (bytes fine, result missing). `manifold_cache` — backfill eps or drop.
- **Regenerate (not re-train):** the stale top-level `training_cache` manifests.
- **Discard:** the `lu7g7svh` typewriter-131542 false-start dir, the `all_groups/dryrun` smoke, the three 0-byte `lora` stubs, the `training_cache_overfit_catdog` empty dir, and (optionally) the superseded `k04__ep200` short run.

## What was not checked

- Intermediate checkpoints (only the final per run dir + suspects were opened). Structure is uniform, so this is a sampling decision, not a gap in the kept-headline set.
- Byte-level cross-root duplication (checksums). `02-two-root-classified.md` already showed the repo vs datasets `cross_seed` pair dirs are different-stage artifacts (eval samples vs seed banks), not copies.
- Live W&B server state. All run statuses are inherited from segment 1's **local** run-dir reconciliation.
- Semantic correctness of the weights (whether each LoRA actually composes). This check verifies the artifacts load and are structurally complete, not that they produce good samples.

---

## Re-sweep 2026-08-04: the eight previously-unfiled top-dirs

Method as above: read-only, `torch.load(weights_only=True)` on CPU. Destinations
come from the scope call in `inventory/sweeps/2026-08-04-scope-call.md`.

| Artifact | Load | Structure | Decision | Destination | Reason |
|---|---|---|---|---|---|
| R: `animals_compose_transfer/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` | ✓ PASS | `lora_state` 420 keys, rank-8; `step=100000`, `epoch=2000` | **keep** | `artifacts/scopes/animals-compose-transfer/` | The transfer headline. Run `1d3qy31e` finished 2026-07-30. Cite always with its checkpoint. |
| R: same dir, `lora_step_005000.pt`, `lora_step_055000.pt` | ✓ PASS | identical envelope | **keep** | (same) | 20 checkpoints on disk; three load-tested across the range. |
| R: `compose_scorer/` | n/a (no `.pt`) | `scorer_validated.json` records `pass: true`, method `instance_count` | **keep** | `artifacts/scopes/compose-scorer/` | The validated scorer contract the animals scope depends on. |
| R: `poe/pairs/` | n/a (no `.pt`) | per-pair baseline sample grids | **keep** | `artifacts/scopes/poe-baselines/` | Baseline reference grids. |
| R: `residual_diagnostics/delta_structure_unguided/tensors.pt` | ✓ PASS | `delta`, `eps_poe`, `eps_mono`, `timesteps`, `seeds` | **keep** | `artifacts/diagnostics/residual_diagnostics/` | The cached correction term itself. Interaction-term scope reads this. |
| R: `residual_diagnostics/delta_structure/.../teacher_residual_*.pt` | ✓ PASS | 13 keys incl. `x_t`, `timestep`, `seq_a`, `pool_a` | **keep** | (same) | 402 `.pt` files; first and last opened. |
| R: `group_a_failure/.../direct_eps_overfit_catdog/best.pt` | ✓ PASS | `student`, `step`, `val_metrics`, `guidance_scale`, `target_kind` | **keep (reference)** | `artifacts/diagnostics/group_a_failure/` | Recorded architecture failure. The paper's negative claims lean on it. |
| R: `group_a_failure/latent_unet/a_cat__x__a_dog/seed_42/...` | ✓ PASS | `delta_hat`, `eps_poe`, `x_t`, `tweedie_x0`, `timestep` | **keep (reference)** | (same) | 1031 `.pt` files, 7G. Revisit the 7G once the negative claims are written and cited. |
| R: `conditioning_window/` | n/a (no `.pt`) | 6 pair dirs, figures and grids | **keep** | `artifacts/diagnostics/conditioning_window/` | Nothing to load-test. |
| R: `conditioning_window_lora/` | n/a (no `.pt`) | 2 pair dirs, figures and grids | **keep** | `artifacts/diagnostics/conditioning_window_lora/` | Nothing to load-test. |
| R: `presentation/heldout_summary/` | n/a | 5 cross-pair held-out summary PNGs | **keep** | `artifacts/_shared/presentation/` | Reads from several rungs, owned by none. |

**Envelope mismatch worth flagging.** `lora_step_100000.pt` does not match the flat
420-key layout every earlier row in this file records. Its tensors live under
`sd["lora_state"]`. Loaders written against the old contract will fail on it.

Sampling note: first and last `.pt` per directory group plus three points across
the headline run. The 1031 files in `group_a_failure` and 402 in
`residual_diagnostics` were not each opened.
