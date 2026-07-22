# Two-Root Classified Inventory (lora / cross_seed / cross_pair)

**Roots swept**: `/datasets/mmolefe` and `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min` (repo)
**Generated**: 2026-07-21
**Scope**: artifacts owned by the three experiments `lora`, `cross_seed_lora_pooling`, `cross_pair_lora_pooling`, plus the caches they read. Everything else in `/datasets/mmolefe` is a different project or an older experiment family (listed under "Out of scope").

Legend for **type**: `dataset` | `lora-ckpt` (LoRA checkpoints + the run's saved figures/samples) | `cache:training` | `cache:group` | `cache:eval` | `results` (saved metrics/figures/verdicts). `R:` = repo root, `D:` = `/datasets/mmolefe`.

## Main table: path → type → group → owning experiment → size → last-modified

| Path | Type | Group | Owning experiment | Size | Last-mod |
|---|---|---|---|---|---|
| R: `outputs/lora/cat_dog` | lora-ckpt + results | lora | lora | 3.4G | 2026-05-18 |
| R: `outputs/lora/a_typewriter__x__a_cactus` | lora-ckpt + results | lora | lora | 1.5G | 2026-05-20 |
| R: `outputs/lora/a_camel__x__a_desert_landscape` | lora-ckpt + results | lora | lora | 945M | 2026-05-20 |
| R: `outputs/lora/a_dog__x__oil_painting_style` | (empty stub) | lora | lora | **0B** | 2026-05-26 |
| R: `outputs/lora/a_dolphin__x__an_ocean_wave` | (empty stub) | lora | lora | **0B** | 2026-05-26 |
| R: `outputs/lora/a_mailbox__x__a_snowfield` | (empty stub) | lora | lora | **0B** | 2026-05-26 |
| R: `outputs/cross_seed_lora_pooling/task_b_learning_curve/k01_pick1__ep1600` | lora-ckpt + results | cross_seed | cross_seed_lora_pooling | 284M | 2026-05-19 |
| R: `outputs/cross_seed_lora_pooling/task_c_per_seed_ceiling/per_seed_s9__ep1600` | lora-ckpt + results | cross_seed | cross_seed_lora_pooling | 66M | 2026-05-19 |
| R: `outputs/cross_seed_lora_pooling/step0_prescreen` | results | cross_seed | cross_seed_lora_pooling | 66M | 2026-05-19 |
| R: `outputs/cross_seed_lora_pooling/step0_prescreen_seed42` | results | cross_seed | cross_seed_lora_pooling | 19M | 2026-05-19 |
| R: `outputs/cross_seed_lora_pooling/trajectory_diagram/seed_42` | results (mono.pt/poe.pt) | cross_seed | cross_seed_lora_pooling | 79M | 2026-05-19 |
| R: `outputs/cross_seed_lora_pooling/{a_cat__x__a_dog, a_dog…, a_dolphin…, a_mailbox…, a_typewriter…}` | results (heldout eval samples) | cross_seed | cross_seed_lora_pooling | 4–79M ea | 2026-05-25 |
| D: `poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/{k04__ep200, k04__ep2000_resumed}` | lora-ckpt (training banks) | cross_seed | cross_seed_lora_pooling | 3.1G | 2026-05-20 |
| D: `poe_repair_min/outputs/cross_seed_lora_pooling/a_dolphin__x__an_ocean_wave` | lora-ckpt (seed bank) | cross_seed | cross_seed_lora_pooling | 1.4G | 2026-05-23 |
| D: `poe_repair_min/outputs/cross_seed_lora_pooling/a_dog__x__oil_painting_style` | lora-ckpt (seed bank) | cross_seed | cross_seed_lora_pooling | 1.3G | 2026-05-23 |
| D: `poe_repair_min/outputs/cross_seed_lora_pooling/a_mailbox__x__a_snowfield` | lora-ckpt (seed bank) | cross_seed | cross_seed_lora_pooling | 996M | 2026-05-23 |
| D: `poe_repair_min/outputs/cross_seed_lora_pooling/a_typewriter__x__a_cactus` | lora-ckpt (seed bank) | cross_seed | cross_seed_lora_pooling | 949M | 2026-05-23 |
| R: `outputs/cross_pair_lora_pooling/all_groups` | lora-ckpt + eval | cross_pair | cross_pair_lora_pooling | 477M | 2026-05-26 |
| R: `outputs/cross_pair_lora_pooling/within_group` | lora-ckpt + eval crossbar | cross_pair | cross_pair_lora_pooling | 402M | 2026-05-27 |
| D: `poe_repair_min/outputs/training_cache/train/` (8 pair dirs) | dataset + cache:training | shared | cross_pair + cross_seed | part of 22G | 2026-05-11 |
| D: `poe_repair_min/outputs/training_cache/{manifest_A,manifest_B,manifest_H}.json` | cache:group | shared | cross_pair + cross_seed | small | 2026-05-06 |
| D: `poe_repair_min/outputs/training_cache/heldout/` | cache:eval | shared | cross_pair + cross_seed | part of 22G | 2026-05-27 |
| R: `outputs/manifold_cache/` | cache (eps/z banks) | — | scripts/manifold (not one of the 3) | 838M | 2026-05-26 |

Shared cache total: `D: training_cache` = **22G** (built by `scripts/build_training_cache.py`; group split by `build_group_cache.py`, eval split by `build_eval_cache.py`). Read by `cross_pair_lora_pooling/train_pooled.py`, `sample_crossbar.py`, `eval_crossbar_wandb.py`, `task_d_bridge.py` and `cross_seed_lora_pooling/train_pooled.py`, `step0_prescreen.py`, `trajectory_diagram.py`.

## Duplicates flagged

1. **cross_seed pair-name collision across roots** — `a_dog__x__oil_painting_style`, `a_dolphin__x__an_ocean_wave`, `a_mailbox__x__a_snowfield`, `a_typewriter__x__a_cactus` exist under `cross_seed_lora_pooling/` in **both** roots. **Not byte-duplicates**: datasets copies are heavy seed banks (949M–1.4G, dated 2026-05-23), repo copies are light held-out eval samples (4–79M, dated 2026-05-25). Same pair, different stage. Safe to keep both; do not dedup blindly.
2. **task_b_learning_curve in both roots** — datasets has `k04__ep200` + `k04__ep2000_resumed` (3.1G), repo has `k01_pick1__ep1600` (284M). Different sweep configs, not a duplicate.
3. **lora pair names also reappear** — the three empty repo `lora/` pairs share names with datasets `cross_seed/` pairs, but they are different experiments (per-pair LoRA vs seed-pool). Name overlap only.

No exact byte-for-byte duplicate directories were found across the two roots for these three experiments.

## Orphans (no owning experiment among the three)

- R: `outputs/lora/{a_dog__x__oil_painting_style, a_dolphin__x__an_ocean_wave, a_mailbox__x__a_snowfield}` — **0-byte stubs**, no run, no checkpoint. Namespaced to `lora` but nothing owns them.
- D: `poe_repair_min/outputs/training_cache_overfit_catdog` — **0-byte** empty dir.
- R: `outputs/manifold_cache/` (838M) — a real cache but owned by `scripts/manifold/*`, not by lora/cross_seed/cross_pair. Orphan relative to the three.

## Out of scope (different project or older experiment family, not orphans of the three)

- **D: `poe_repair_min/outputs/`** other families: `synthesizer` (63G), `veracity*` (~10G across variants), `idea1/2/5a/5b`, `e_*`, `sweep_*`, `mcmc_*`, `_archive`, `diag*`, `students`, `seed_preview`. These belong to earlier/other experiments in the same project, not to the three named here.
- **D: other roots**: `checkpoints/` (chest-xray/cardio `.safetensors`, `augcase`, `o2*`, `arm1_cleancardio`), `coco2017`, `vinbigdata`, `nih`, `chexmask`, `chess`, `mini_sd3*`, `eccv2026/`, `neurips2026/`, `backup/` (full home-dir backup). Separate projects (medical imaging etc.).
- **R: other experiments**: `outputs/{group_a_failure, conditioning_window, conditioning_window_lora, residual_diagnostics}` — catalogued in `01-artifact-inventory.md`, outside the three named here.

## Coverage notes

- Sizes via `du -sh` per directory; the 22G `training_cache` was not re-walked in full (recursive `du` on it times out) — its 22G figure and subdir mtimes are from `ls`/`stat`.
- "Duplicate" = same directory name across roots; content was compared by size and mtime, not by checksum. A byte-level `rsync -n`/hash diff would confirm, but the size/date gaps already show the collisions are different-stage artifacts, not copies.
- No `.safetensors`/`.bin` LoRA adapters in the repo; all repo checkpoints are `.pt`. The `.safetensors` under `/datasets/mmolefe/checkpoints` belong to a different (medical-imaging) project.
