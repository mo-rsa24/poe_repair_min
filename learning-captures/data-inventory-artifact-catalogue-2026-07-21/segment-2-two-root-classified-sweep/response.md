# Segment 2 — Response (full)

## Method / scope
Swept both roots: `/datasets/mmolefe` and the repo. Most of `/datasets/mmolefe`
(~148 GB: coco2017, vinbigdata, nih, chexmask, chess, backup, eccv2026,
neurips2026, mini_sd3, a separate `checkpoints/` of chest-xray/cardio
`.safetensors`) is a different project. The load-bearing shared location is
`/datasets/mmolefe/poe_repair_min/outputs`, which is **not** a mirror of the repo
outputs but a separate, larger output tree (100+ GB: `synthesizer` 63G,
`training_cache` 22G, `veracity*`, `idea*`, `e_*` families).

Of the three named experiments, only `cross_seed_lora_pooling` (7.6G) and
`training_cache` (22G) appear under `/datasets/mmolefe`. No `lora` or
`cross_pair_lora_pooling` there.

Cache ownership confirmed by scoped grep:
- `training_cache` read by `cross_pair_lora_pooling/{train_pooled,sample_crossbar,
  eval_crossbar_wandb,contact_sheet,task_d_bridge,_inline_sampling}.py` and
  `cross_seed_lora_pooling/{train_pooled,step0_prescreen,trajectory_diagram,
  _inline_sampling}.py`. Built by `scripts/build_training_cache.py`; group split
  `build_group_cache.py`; eval split `build_eval_cache.py`.
- `manifold_cache` read by `scripts/manifold/*` only (not the three experiments).

## Main table: path → type → group → owning experiment → size → last-mod
`R:` = repo, `D:` = `/datasets/mmolefe`.

| Path | Type | Owning exp | Size | Last-mod |
|---|---|---|---|---|
| R `outputs/lora/cat_dog` | lora-ckpt + results | lora | 3.4G | 2026-05-18 |
| R `outputs/lora/a_typewriter__x__a_cactus` | lora-ckpt + results | lora | 1.5G | 2026-05-20 |
| R `outputs/lora/a_camel__x__a_desert_landscape` | lora-ckpt + results | lora | 945M | 2026-05-20 |
| R `outputs/lora/a_dog__x__oil_painting_style` | empty stub | lora | **0B** | 2026-05-26 |
| R `outputs/lora/a_dolphin__x__an_ocean_wave` | empty stub | lora | **0B** | 2026-05-26 |
| R `outputs/lora/a_mailbox__x__a_snowfield` | empty stub | lora | **0B** | 2026-05-26 |
| R `cross_seed/task_b_learning_curve/k01_pick1__ep1600` | lora-ckpt + results | cross_seed | 284M | 2026-05-19 |
| R `cross_seed/task_c_per_seed_ceiling/per_seed_s9__ep1600` | lora-ckpt + results | cross_seed | 66M | 2026-05-19 |
| R `cross_seed/step0_prescreen` | results | cross_seed | 66M | 2026-05-19 |
| R `cross_seed/step0_prescreen_seed42` | results | cross_seed | 19M | 2026-05-19 |
| R `cross_seed/trajectory_diagram/seed_42` | results (mono.pt/poe.pt) | cross_seed | 79M | 2026-05-19 |
| R `cross_seed/{a_cat×a_dog, a_dog×oil, a_dolphin×ocean, a_mailbox×snowfield, a_typewriter×cactus}` | results (heldout eval samples) | cross_seed | 4–79M ea | 2026-05-25 |
| D `…/cross_seed/task_b_learning_curve/{k04__ep200, k04__ep2000_resumed}` | lora-ckpt (train banks) | cross_seed | 3.1G | 2026-05-20 |
| D `…/cross_seed/a_dolphin__x__an_ocean_wave` | lora-ckpt (seed bank) | cross_seed | 1.4G | 2026-05-23 |
| D `…/cross_seed/a_dog__x__oil_painting_style` | lora-ckpt (seed bank) | cross_seed | 1.3G | 2026-05-23 |
| D `…/cross_seed/a_mailbox__x__a_snowfield` | lora-ckpt (seed bank) | cross_seed | 996M | 2026-05-23 |
| D `…/cross_seed/a_typewriter__x__a_cactus` | lora-ckpt (seed bank) | cross_seed | 949M | 2026-05-23 |
| R `cross_pair/all_groups` | lora-ckpt + eval | cross_pair | 477M | 2026-05-26 |
| R `cross_pair/within_group` | lora-ckpt + eval crossbar | cross_pair | 402M | 2026-05-27 |
| D `…/training_cache/train/` (8 pair dirs) | dataset + cache:training | cross_pair + cross_seed (shared) | part of 22G | 2026-05-11 |
| D `…/training_cache/manifest_{A,B,H}.json` | cache:group | shared | small | 2026-05-06 |
| D `…/training_cache/heldout/` | cache:eval | shared | part of 22G | 2026-05-27 |
| R `outputs/manifold_cache/` | cache (eps/z banks) | scripts/manifold (not one of 3) | 838M | 2026-05-26 |

`training_cache/train/` 8 pairs: a_bear×a_salmon, a_butterfly×a_flower_meadow,
a_cat×a_lion, a_dog×a_duck, a_dog×a_horse, a_lion×a_dog, a_tiger×a_dog,
a_wolf×a_husky.

## Duplicates flagged
1. **cross_seed pair-name collision across both roots** (`a_dog×oil`,
   `a_dolphin×ocean`, `a_mailbox×snowfield`, `a_typewriter×cactus`) — **not
   byte-dups**: datasets = heavy seed banks (~1G, 2026-05-23), repo = light
   heldout eval (4–79M, 2026-05-25). Different stage; don't dedup blindly.
2. **task_b_learning_curve in both roots** — datasets `k04__ep200` +
   `k04__ep2000_resumed` (3.1G) vs repo `k01_pick1__ep1600` (284M). Different
   sweep configs, not a copy.
3. **lora pair names reappear** as datasets cross_seed pair names — different
   experiments, name overlap only.
No exact byte-for-byte duplicate directories found across roots.

## Orphans (no owning experiment among the three)
- R `outputs/lora/{a_dog×oil, a_dolphin×ocean, a_mailbox×snowfield}` — 0-byte
  stubs, no run.
- D `…/outputs/training_cache_overfit_catdog` — 0-byte empty dir.
- R `outputs/manifold_cache/` (838M) — real cache owned by `scripts/manifold/`,
  not the three.

## Out of scope (other work, not orphans)
- D `poe_repair_min/outputs/`: `synthesizer` (63G), `veracity*` (~10G), `idea*`,
  `e_*`, `sweep_*`, `mcmc_*`, `_archive` — older experiment families.
- D other roots: `checkpoints/` (chest-xray/cardio `.safetensors`), `coco2017`,
  `vinbigdata`, `nih`, `chexmask`, `chess`, `mini_sd3*`, `eccv2026`,
  `neurips2026`, `backup/` — separate projects.
- R other experiments: `outputs/{group_a_failure, conditioning_window,
  conditioning_window_lora, residual_diagnostics}` — outside the three.

## Deliverable written
- `inventory/02-two-root-classified.md`

## Caveats stated
- The 22G `training_cache` was not re-walked in full (recursive `du` times out);
  size and subdir dates from `ls`/`stat`.
- "Duplicate" judged by name + size + mtime, not checksums. Offer made to run a
  hash / `rsync -n` diff of the cross_seed pair dirs across roots — not executed.
