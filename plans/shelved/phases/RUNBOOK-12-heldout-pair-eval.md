# Runbook — Plan 12 — Held-out-pair eval of pueuo7bl

Companion to [12-cross-seed-lora-heldout-pair-eval.md](12-cross-seed-lora-heldout-pair-eval.md).
This file is the operational ready-to-paste command sequence — design
rationale and pass/fail criteria live in the plan.

Goal: take the finished `cat × dog` cross-seed LoRA from wandb run
`prime_lab/poe-repair-cross-seed/pueuo7bl` (`k04__ep2000_resumed`) and
evaluate it on G6 sibling pairs (`a_wolf__x__a_husky`,
`a_lion__x__a_dog`), producing per-seed summary figures that mirror the
[render_seed_summary.py](../scripts/cross_seed_lora_pooling/render_seed_summary.py)
layout (PoE | per-epoch grid | Mono).

## What this runbook touches

| Artefact | Path |
|---|---|
| Repo | `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min` |
| Cache root | `/datasets/mmolefe/poe_repair_min/outputs/training_cache` |
| Run dir (read-only) | `/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed` |
| Final checkpoint | `<run>/checkpoints/lora_step_100000.pt` (ep 2000) |
| Per-epoch checkpoints | `<run>/checkpoints/lora_step_{012500..100000..2500}.pt` (~36 of them) |
| New eval-pair caches | `<cache>/heldout/a_wolf__x__a_husky/seed_{1,2,9,10,11,12}/`<br>`<cache>/heldout/a_lion__x__a_dog/seed_{1,2,9,10,11,12}/` |
| Output figures | `<run>/samples/per_seed_summary_heldout/<pair>/summary_seed_NN.png` |

## Code under the hood

| File | Status | Why we need it |
|---|---|---|
| [scripts/build_eval_cache.py](../scripts/build_eval_cache.py) | new | Writes minimal `(meta.json + embeddings.pt + residuals/step_000.pt + poe.png + mono.png)` per `(pair, seed)`. The sampler only needs init latents; PoE/Mono PNGs feed the figure's left panel. |
| [poe_repair/experiments/held_out_seeds/sample_heldout.py](../poe_repair/experiments/held_out_seeds/sample_heldout.py) | patched | New `--heldout-pair` flag: load checkpoint trained on pair P_train, sample on pair P_eval's cache cells. Prompts overridable via `--prompt-a/-b/--joint-prompt`. |
| [scripts/cross_seed_lora_pooling/render_per_epoch.py](../scripts/cross_seed_lora_pooling/render_per_epoch.py) | patched | New `--pair-slug-override` flag: per-epoch sweep reads init latents + PoE/Mono refs from the override pair's cache. |
| [scripts/cross_seed_lora_pooling/render_seed_summary.py](../scripts/cross_seed_lora_pooling/render_seed_summary.py) | unchanged | Already accepts `--pair-slug` + `--per-epoch-subdir`. |

## 0. One-time shell setup

Paste once per shell session before any step. All later steps assume
`$PY`, `$RUN`, `$CKPT`, and `$POE_REPAIR_TRAINING_CACHE` are set, and
`cwd` is the repo root.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export CUDA_VISIBLE_DEVICES=1
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
RUN=/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed
CKPT=$RUN/checkpoints/lora_step_100000.pt
```

Quick sanity check that the checkpoint is where we think it is:

```bash
ls -lh $CKPT && cat $RUN/verdict.json && cat $RUN/pooled_meta.json
```

Expected: `verdict = "ok"`, `epoch = 2000`, `seeds_used = [1,2,3,4]`.

## 1. Smoke-test one cell first

Single-cell build to confirm the SDXL load + cache-write path is sane
before launching the whole batch. About 75 s on RTX 8000.

```bash
$PY -m scripts.build_eval_cache \
    --prompt-a "a wolf" --prompt-b "a husky" \
    --joint-prompt "a wolf and a husky" --seed 9
ls $POE_REPAIR_TRAINING_CACHE/heldout/a_wolf__x__a_husky/seed_9
```

Should produce `meta.json embeddings.pt mono.png poe.png residuals/`,
and `residuals/step_000.pt`. Eyeball `poe.png` and `mono.png` — if they
look like SDXL output for the prompts, you're good.

## 2. Build all eval caches (Step A)

~15 min total: 6 seeds × 2 pairs × ~75 s. Idempotent — re-running skips
existing cells. Add `--skip-refs` to drop per-cell time to <1 s if you
don't care about the PoE/Mono panels in the figure.

```bash
for SEED in 9 10 11 12 1 2; do
  $PY -m scripts.build_eval_cache \
      --prompt-a "a wolf" --prompt-b "a husky" \
      --joint-prompt "a wolf and a husky" --seed $SEED
  $PY -m scripts.build_eval_cache \
      --prompt-a "a lion" --prompt-b "a dog" \
      --joint-prompt "a lion and a dog" --seed $SEED
done
```

Verify all 12 cells are present:

```bash
for PAIR in a_wolf__x__a_husky a_lion__x__a_dog; do
  echo "=== $PAIR ==="
  ls $POE_REPAIR_TRAINING_CACHE/heldout/$PAIR
done
```

## 3. Sanity: final-checkpoint sample on each held-out pair (Step B)

Cheapest read. About 3 min. If this produces nothing recognisable, the
per-epoch sweep (Step C) is unlikely to surface much either — abort
early and re-examine assumptions instead of burning 3 GPU-hours.

```bash
# wolf x husky
$PY -m poe_repair.experiments.held_out_seeds.sample_heldout \
    --checkpoint $CKPT \
    --heldout-pair a_wolf__x__a_husky \
    --prompt-a "a wolf" --prompt-b "a husky" --joint-prompt "a wolf and a husky" \
    --seeds 9,10,11,12 \
    --out-dir $RUN/samples/heldout_pair/a_wolf__x__a_husky/final

# lion x dog
$PY -m poe_repair.experiments.held_out_seeds.sample_heldout \
    --checkpoint $CKPT \
    --heldout-pair a_lion__x__a_dog \
    --prompt-a "a lion" --prompt-b "a dog" --joint-prompt "a lion and a dog" \
    --seeds 9,10,11,12 \
    --out-dir $RUN/samples/heldout_pair/a_lion__x__a_dog/final
```

Inspect:

```bash
ls $RUN/samples/heldout_pair/a_wolf__x__a_husky/final
ls $RUN/samples/heldout_pair/a_lion__x__a_dog/final
```

Eyeball the 8 PNGs. Decision point: if 0–1 of 8 show plausible
composition, stop here and reconsider before launching Step C.

## 4. Per-epoch sweep on held-out pairs (Step C — the headline)

~3 GPU-hours. 36 checkpoints × 6 seeds × 2 pairs × ~25 s. Best to
launch in the background and check on it later.

```bash
mkdir -p $RUN/logs

# wolf x husky
nohup $PY -m scripts.cross_seed_lora_pooling.render_per_epoch \
    --run-dir $RUN \
    --pair-slug-override a_wolf__x__a_husky \
    --prompt-a "a wolf" --prompt-b "a husky" --joint-prompt "a wolf and a husky" \
    --seeds 9,10,11,12,1,2 \
    --out-subdir samples/per_epoch_heldout/a_wolf__x__a_husky \
    > $RUN/logs/per_epoch_wolf_husky.log 2>&1 &

# lion x dog
nohup $PY -m scripts.cross_seed_lora_pooling.render_per_epoch \
    --run-dir $RUN \
    --pair-slug-override a_lion__x__a_dog \
    --prompt-a "a lion" --prompt-b "a dog" --joint-prompt "a lion and a dog" \
    --seeds 9,10,11,12,1,2 \
    --out-subdir samples/per_epoch_heldout/a_lion__x__a_dog \
    > $RUN/logs/per_epoch_lion_dog.log 2>&1 &

# wait + watch
jobs -l
tail -f $RUN/logs/per_epoch_wolf_husky.log
```

> Note: launching both backgrounded sweeps simultaneously will share
> the single visible GPU and run roughly half-speed each (no net
> wall-clock saving). Sequential is fine; choose based on whether you
> want to inspect one pair's results before committing to the second.

When each sweep completes, check the manifest and contact sheet:

```bash
for PAIR in a_wolf__x__a_husky a_lion__x__a_dog; do
  ls $RUN/samples/per_epoch_heldout/$PAIR
  cat $RUN/samples/per_epoch_heldout/$PAIR/manifest.json | head -20
done
```

## 5. Render the seed-summary figures (Step D)

CPU only, < 1 min. Produces the screenshot-style figure (PoE on top-
left, Mono bottom-left, 4×5 epoch grid on the right) for each
(held-out pair, seed).

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

Resulting files:

```
$RUN/samples/per_seed_summary_heldout/
├── a_wolf__x__a_husky/
│   ├── summary_seed_01.png   ← train-pool seed sanity
│   ├── summary_seed_02.png   ← train-pool seed sanity
│   ├── summary_seed_09.png   ← held-out seed × held-out pair  ← HEADLINE
│   ├── summary_seed_10.png   ← held-out seed × held-out pair  ← HEADLINE
│   ├── summary_seed_11.png   ← held-out seed × held-out pair  ← HEADLINE
│   └── summary_seed_12.png   ← held-out seed × held-out pair  ← HEADLINE
└── a_lion__x__a_dog/
    └── …  (same shape)
```

The four `seed_{09..12}` figures for each held-out pair are the read.

## 6. Optional — add `a_fox__x__a_rabbit` as a third pair

Fully held-out canid/lagomorph pair (neither side seen at training).
Extends Steps A/B/C/D by ~50 %.

```bash
# A
for SEED in 9 10 11 12 1 2; do
  $PY -m scripts.build_eval_cache \
      --prompt-a "a fox" --prompt-b "a rabbit" \
      --joint-prompt "a fox and a rabbit" --seed $SEED
done

# B
$PY -m poe_repair.experiments.held_out_seeds.sample_heldout \
    --checkpoint $CKPT --heldout-pair a_fox__x__a_rabbit \
    --prompt-a "a fox" --prompt-b "a rabbit" --joint-prompt "a fox and a rabbit" \
    --seeds 9,10,11,12 \
    --out-dir $RUN/samples/heldout_pair/a_fox__x__a_rabbit/final

# C
nohup $PY -m scripts.cross_seed_lora_pooling.render_per_epoch \
    --run-dir $RUN --pair-slug-override a_fox__x__a_rabbit \
    --prompt-a "a fox" --prompt-b "a rabbit" --joint-prompt "a fox and a rabbit" \
    --seeds 9,10,11,12,1,2 \
    --out-subdir samples/per_epoch_heldout/a_fox__x__a_rabbit \
    > $RUN/logs/per_epoch_fox_rabbit.log 2>&1 &

# D
for SEED in 9 10 11 12 1 2; do
  $PY -m scripts.cross_seed_lora_pooling.render_seed_summary \
      --seed $SEED --run-dir $RUN \
      --per-epoch-subdir samples/per_epoch_heldout/a_fox__x__a_rabbit \
      --pair-slug a_fox__x__a_rabbit \
      --out-dir $RUN/samples/per_seed_summary_heldout/a_fox__x__a_rabbit
done
```

## Caveats when reading the output

1. **The LoRA is k=4, not k=8.** `seeds_used = [1,2,3,4]` from the full
   `train_pool = {1..8}`. If transfer looks weak, that's a confound — a
   k=8 retrain (Plan 10's actual headline run) is the natural follow-up,
   not a verdict on the mechanism itself.
2. **`a_lion__x__a_dog` shares "a dog"** with the training pair. If
   lion×dog transfers but wolf×husky doesn't, the LoRA is probably
   leaning on the shared concept rather than learning G6-level
   composition. Note this in the writeup.
3. **Eval-only cells must not be trained on.** `step_000.pt`'s `eps_*`
   fields are zero tensors (placeholders for `load_step_raw` shape
   compat). If they get fed into a trainer the loss target is
   arbitrary. They're read-only for the sampler.
4. **The two background sweeps in Step C share one GPU.** No net
   wall-clock saving from launching both at once — pick sequential if
   you want to inspect one pair's results before committing to the
   next.

## Quick reference — paths

```
$RUN  = /datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed
$CKPT = $RUN/checkpoints/lora_step_100000.pt
cache = /datasets/mmolefe/poe_repair_min/outputs/training_cache/heldout/<pair_slug>/seed_<N>/

new eval pairs:
  a_wolf__x__a_husky
  a_lion__x__a_dog
  a_fox__x__a_rabbit  (optional)

seeds:
  9, 10, 11, 12   ← held-out (headline)
  1, 2            ← train-pool sanity
```
