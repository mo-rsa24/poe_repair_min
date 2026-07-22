# Plan 12 — Held-out-pair evaluation of the cross-seed `cat × dog` LoRA

> Parent: [10-cross-seed-lora-per-group.md](10-cross-seed-lora-per-group.md).
> This is the within-group-transfer probe Plan 10 calls
> "`--heldout-pair`", scoped to the *already-trained* G6 checkpoint
> `pueuo7bl` (wandb `prime_lab/poe-repair-cross-seed/pueuo7bl`,
> name `k04__ep2000_resumed`).

## Question

The pueuo7bl LoRA was trained on `a_cat__x__a_dog` with seeds {1..4}
pooled. Does it produce recognisable composition on **sibling pairs**
from the same group (G6 concept-collision) — e.g. `a_wolf__x__a_husky`,
`a_lion__x__a_dog` — on seeds the LoRA never saw at training time?

Two reads, one figure per (held-out pair, seed):

1. **Final-checkpoint sample** (`lora_step_100000.pt`, ep 2000). Does
   the converged LoRA compose on the sibling pair?
2. **Per-epoch trajectory.** Replay the per-epoch checkpoints
   (`lora_step_012500.pt` … `lora_step_100000.pt`, ~36 checkpoints) on
   the sibling pair. When during training does the within-group transfer
   appear, and does it persist or collapse?

Output: `summary_seed_NN.png` figures matching the layout already
produced by [render_seed_summary.py](../scripts/cross_seed_lora_pooling/render_seed_summary.py)
(PoE | per-epoch grid | Mono) but for the held-out pair.

## Checkpoint under test

- Run dir: `/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed/`
- Final: `checkpoints/lora_step_100000.pt` (epoch 2000, verdict `ok`)
- Per-epoch: every 2,500 optimizer steps (50 epochs), step 12,500 → 100,000
- Config: rank-8, α=8, `attn2.{to_q,to_k,to_v}`, lr=1e-4, fp16, SDXL
  base 1.0, DDIM 50 steps, CFG 7.5
- Train-pool actually used: seeds {1,2,3,4} (k=4, not the full
  `train_pool={1..8}` from `seed_pool.yaml`)
- Original training pair: `a_cat__x__a_dog` (G6)

## Held-out pairs (proposed)

| Slug | Prompts | Note |
|---|---|---|
| `a_wolf__x__a_husky` | "a wolf", "a husky" → "a wolf and a husky" | Plan-10's canonical G6 sibling |
| `a_lion__x__a_dog` | "a lion", "a dog" → "a lion and a dog" | Probes "concept-collision among canids/felids" — `dog` is shared with the training pair, only one slot is held-out |
| `a_fox__x__a_rabbit` | "a fox", "a rabbit" → "a fox and a rabbit" | Optional — fully held-out canid/lagomorph pair, neither side seen at training |

Configurable via CLI; nothing pinned in code.

## Seeds

- Held-out: `{9, 10, 11, 12}` — the four seeds reserved by the run's
  `seed_pool.held_out`. These are the **headline** seeds (the LoRA has
  never seen these seeds, nor the held-out pair).
- Train-pool: `{1, 2}` — included as a sanity check (the LoRA *has*
  seen these seeds on the training pair). If it still composes here
  on a held-out pair, the LoRA has learned the *task* (within-group
  composition) more than the seed-specific noise.

Total: 6 seeds × N pairs cells per held-out pair. For 3 held-out
pairs that's 18 (pair, seed) cells.

## Plan-10 acceptance bar (per pair)

- **Good**: composition recognisable on ≥ 2 of 4 held-out seeds.
- **Mixed**: composition on 1 of 4.
- **Bad**: vanilla-PoE-equivalent on all 4.

We classify per held-out pair, then look for per-pair structure (is
`wolf×husky` easier than `lion×dog`?).

## What this plan does *not* do

- **Train anything.** Read-only on pueuo7bl.
- **k=8 retrain.** Even though pueuo7bl is k=4, this plan does not
  re-launch a k=8 sibling. Plan 10's per-group rerun is the place for
  that.
- **G1–G5 sibling pairs.** Out of scope — G1–G5 LoRA checkpoints don't
  exist yet.
- **Cross-group transfer.** Held-out pairs are restricted to G6.

## Code

| Path | Status | Role |
|---|---|---|
| `scripts/build_eval_cache.py` | **new** | Build a minimal cell at `$POE_REPAIR_TRAINING_CACHE/heldout/<pair>/seed_N/`: `meta.json`, `embeddings.pt`, `residuals/step_000.pt` (init latent only — eps tensors zeroed for `load_step_raw` compat), optional `poe.png` + `mono.png` for the figure's left panel. ~75s per cell with PoE+Mono refs; ~1s per cell with `--skip-refs`. |
| `poe_repair/experiments/cross_seed_lora_pooling/sample_heldout.py` | **patched** | New CLI flags `--heldout-pair`, `--prompt-a`, `--prompt-b`, `--joint-prompt`. When `--heldout-pair` differs from the checkpoint's training pair, the eval pair's cache cells are read and the override prompts are encoded. |
| `scripts/cross_seed_lora_pooling/render_per_epoch.py` | **patched** | New flag `--pair-slug-override`. When set, init latents and PoE/Mono ref columns are pulled from the held-out pair's cache rather than the checkpoint's training pair. Prompt overrides (`--prompt-a/-b/--joint-prompt`) already exist. |
| `scripts/cross_seed_lora_pooling/render_seed_summary.py` | **unchanged** | Already accepts `--pair-slug` + `--per-epoch-subdir`. Point it at `samples/per_epoch_heldout/<pair>` and the held-out pair's cache. |

## Per-epoch output layout

Mirrors the existing `samples/per_epoch/epoch_NNNN_step_MMMMMM/`
convention so `render_seed_summary.py` works unchanged:

```
<run-dir>/samples/per_epoch_heldout/<pair-slug>/
    epoch_0250_step_012500/sample_seed_09.png
    epoch_0250_step_012500/sample_seed_10.png
    …
    epoch_2000_step_100000/sample_seed_12.png
    manifest.json
    strips/strip_seed_NN.png   # auto, from render_per_epoch
    contact_sheet.png
```

## Commands

(See the assistant's last message for the exact ready-to-paste sequence,
or run the steps in this section in order.)

### Step A — eval-only caches

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min

for SEED in 9 10 11 12 1 2; do
  $PY -m scripts.build_eval_cache \
      --prompt-a "a wolf" --prompt-b "a husky" \
      --joint-prompt "a wolf and a husky" --seed $SEED
  $PY -m scripts.build_eval_cache \
      --prompt-a "a lion" --prompt-b "a dog" \
      --joint-prompt "a lion and a dog" --seed $SEED
done
```

### Step B — final-checkpoint sample (sanity)

```bash
CKPT=/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed/checkpoints/lora_step_100000.pt
RUN=/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed

for PAIR_SLUG in a_wolf__x__a_husky a_lion__x__a_dog; do
  case $PAIR_SLUG in
    a_wolf__x__a_husky) A="a wolf"; B="a husky"; J="a wolf and a husky" ;;
    a_lion__x__a_dog)   A="a lion"; B="a dog";   J="a lion and a dog" ;;
  esac
  $PY -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
      --checkpoint $CKPT \
      --heldout-pair $PAIR_SLUG --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J" \
      --seeds 9,10,11,12 \
      --out-dir $RUN/samples/heldout_pair/$PAIR_SLUG/final
done
```

### Step C — per-epoch sweep (the headline)

```bash
for PAIR_SLUG in a_wolf__x__a_husky a_lion__x__a_dog; do
  case $PAIR_SLUG in
    a_wolf__x__a_husky) A="a wolf"; B="a husky"; J="a wolf and a husky" ;;
    a_lion__x__a_dog)   A="a lion"; B="a dog";   J="a lion and a dog" ;;
  esac
  $PY -m scripts.cross_seed_lora_pooling.render_per_epoch \
      --run-dir $RUN \
      --pair-slug-override $PAIR_SLUG \
      --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J" \
      --seeds 9,10,11,12,1,2 \
      --out-subdir samples/per_epoch_heldout/$PAIR_SLUG
done
```

### Step D — seed-summary figures

```bash
for PAIR_SLUG in a_wolf__x__a_husky a_lion__x__a_dog; do
  for SEED in 9 10 11 12 1 2; do
    $PY -m scripts.cross_seed_lora_pooling.render_seed_summary \
        --seed $SEED \
        --run-dir $RUN \
        --per-epoch-subdir samples/per_epoch_heldout/$PAIR_SLUG \
        --pair-slug $PAIR_SLUG \
        --out-dir $RUN/samples/per_seed_summary_heldout/$PAIR_SLUG
  done
done
```

## How to read the result

For each held-out pair, eyeball the 4 held-out-seed summaries:

- **All 4 compose** → strong evidence the LoRA learned a G6-level
  composition rule, not the `cat×dog` pair specifically. Reportable as
  "within-G6 transfer."
- **2-3 of 4 compose** → Plan-10 `Good` bar met. Still reportable.
- **0-1 of 4 compose** → the LoRA is largely pair-specific on the seed
  axis it was trained on. Re-running with k=8 (full train pool) before
  drawing a strong conclusion is a fair follow-up.

Compare across held-out pairs to see whether sharing one concept with
the training pair (`a_lion__x__a_dog` shares "dog") makes transfer
easier than fully held-out (`a_wolf__x__a_husky`, `a_fox__x__a_rabbit`).

## Estimated GPU time

- Step A: 6 seeds × 2 pairs × ~75s = ~15 min
- Step B: 4 seeds × 2 pairs × ~25s = ~3 min
- Step C: 36 ckpts × 6 seeds × 2 pairs × ~25s = ~3 h (dominant cost)
- Step D: CPU only, < 1 min

Adding `a_fox__x__a_rabbit` adds ~50 % to Steps A and C.

## Status — 2026-05-25

| Item | Done | To do |
|---|:---:|:---:|
| pueuo7bl checkpoint located on disk | ✅ | |
| `scripts/build_eval_cache.py` | ✅ | |
| `sample_heldout.py` `--heldout-pair` patch | ✅ | |
| `render_per_epoch.py` `--pair-slug-override` patch | ✅ | |
| Step A — eval-only caches built | | ⬜ |
| Step B — final-checkpoint sanity samples | | ⬜ |
| Step C — per-epoch sweep on held-out pairs | | ⬜ |
| Step D — seed-summary figures | | ⬜ |
| Per-pair classification (Good/Mixed/Bad) | | ⬜ |
