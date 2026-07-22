#!/usr/bin/env bash
# Plan 16 — within-group cross-pair × all-seed driver.
# Per group in $PLAN16_GROUPS: leak-guard pools → train pooled LoRA (7 train
# pairs × 12 seeds = 84 cells) → sample crossbar with quadrants
# {in_in, out_in} and out_in_train_seeds=12 (so held pairs × all 12
# seeds) → render contact sheets → run Task D Δ̄_t bridge.
#
# Reuses Plan 15's cross_pair_lora_pooling code verbatim; only the
# pair_pool.yaml and the sampler quadrant flags change.
#
# Training runs in the foreground — output streams live to the terminal.
# Use tmux (or screen) so a dropped SSH session doesn't kill the run;
# W&B captures metrics online so the run is also visible at
# https://wandb.ai/<entity>/<project>/runs.
#
# Override via env vars:
#   PLAN16_GROUPS="g6 g1"     # default: "g1 g2 g3 g4 g6"
#   EPOCHS=2400               # matches Plan 15's verified recipe
#   EPOCH_SIZE=50
#   LORA_RANK=8
#   LORA_ALPHA=8
#   LR=1e-4
#   DTYPE=float16
#   RUN_ID=main               # disk path: $GROUP_DIR/$RUN_ID
#   WANDB_MODE=online         # online | offline | disabled
#   WANDB_PROJECT=poe-repair-cross-pair
#   PY=...
#   CUDA_VISIBLE_DEVICES=0
#   POE_REPAIR_TRAINING_CACHE=/datasets/.../training_cache
#   SKIP_TRAIN=1              # only sample/contact-sheet/task-d an existing run
#   SKIP_SAMPLE=1
#   SKIP_CONTACT=1
#   SKIP_TASKD=1
#   DRY_RUN=1                 # echo commands only
set -euo pipefail

PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
PLAN16_GROUPS="${PLAN16_GROUPS:-g1 g2 g3 g4 g6}"
EPOCHS="${EPOCHS:-2400}"
EPOCH_SIZE="${EPOCH_SIZE:-50}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LR="${LR:-1e-4}"
DTYPE="${DTYPE:-float16}"
RUN_ID="${RUN_ID:-main}"
LOG_EVERY="${LOG_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-200}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-poe-repair-cross-pair}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

WITHIN_ROOT="outputs/cross_pair_lora_pooling/within_group"
SEED_POOL="$WITHIN_ROOT/seed_pool.yaml"
PAIR_PROMPTS="outputs/cross_pair_lora_pooling/pair_prompts.yaml"

for f in "$SEED_POOL" "$PAIR_PROMPTS"; do
  [[ -f "$f" ]] || { echo "[err] missing $f" >&2; exit 2; }
done

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry] %s\n' "$*"
  else
    printf '[run] %s\n' "$*"
    eval "$@"
  fi
}

# Leak guard on the shared seed pool (cheap, fail-fast).
run "$PY -m poe_repair.experiments.cross_pair_lora_pooling.seed_pool \
      --seed-pool-path '$SEED_POOL' --check-only"

for g in $PLAN16_GROUPS; do
  GROUP_DIR="$WITHIN_ROOT/$g"
  PAIR_POOL="$GROUP_DIR/pair_pool.yaml"
  RUN_DIR="$GROUP_DIR/$RUN_ID"
  # Per-group W&B run name keeps each group as a separate run in the
  # project view; disk layout stays $GROUP_DIR/$RUN_ID per group.
  WANDB_NAME="${RUN_ID}_${g}"

  if [[ ! -f "$PAIR_POOL" ]]; then
    echo "[err] $g: missing $PAIR_POOL" >&2; exit 2
  fi

  echo "===== $g ====="

  # 1. Leak-guard the per-group pair pool.
  run "$PY -m poe_repair.experiments.cross_pair_lora_pooling.pair_pool \
        --pair-pool '$PAIR_POOL' --check-only"

  # 2. Train (skippable). Foreground — output streams to the terminal
  # (use tmux to keep the session alive; W&B captures metrics online).
  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    # W&B run name encoded via WANDB_NAME env var (read by the wandb
    # client at init time); --wandb-project/--wandb-mode go through CLI.
    run "WANDB_NAME='$WANDB_NAME' $PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
          --pair-pool '$PAIR_POOL' \
          --seed-pool-path '$SEED_POOL' \
          --pair-prompts '$PAIR_PROMPTS' \
          --total-epochs $EPOCHS \
          --epoch-size $EPOCH_SIZE \
          --train-batch-size $TRAIN_BATCH_SIZE \
          --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lr $LR \
          --dtype $DTYPE \
          --ckpt-every-epochs $CKPT_EVERY \
          --log-every-epochs $LOG_EVERY \
          --output-root '$GROUP_DIR' \
          --run-id '$RUN_ID' \
          --wandb-mode $WANDB_MODE \
          --wandb-project '$WANDB_PROJECT'"
  else
    echo "[skip] training for $g (SKIP_TRAIN=1)"
  fi

  # 3. Resolve the most recent checkpoint for sampling.
  CKPT=""
  if [[ -d "$RUN_DIR/checkpoints" ]]; then
    CKPT="$(ls -t "$RUN_DIR"/checkpoints/lora_step_*.pt 2>/dev/null | head -1 || true)"
  fi

  # 4. Crossbar sampling — Plan 16 uses --quadrants in_in,out_in with
  #    --out-in-train-seeds 12 so the held pairs are sampled at every
  #    seed in train_pool (which is all 12 seeds under this plan's
  #    seed_pool). in_out and out_out are empty by construction.
  if [[ "${SKIP_SAMPLE:-0}" != "1" ]]; then
    if [[ -z "$CKPT" ]]; then
      echo "[err] $g: no checkpoint under $RUN_DIR/checkpoints — skipping sample/contact/task-d" >&2
      continue
    fi
    run "$PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \
          --checkpoint '$CKPT' \
          --pair-pool '$PAIR_POOL' \
          --seed-pool-path '$SEED_POOL' \
          --pair-prompts '$PAIR_PROMPTS' \
          --out-dir '$RUN_DIR/samples' \
          --quadrants in_in,out_in \
          --out-in-train-seeds 12"
  else
    echo "[skip] sampling for $g (SKIP_SAMPLE=1)"
  fi

  # 5. Contact sheets (skippable).
  if [[ "${SKIP_CONTACT:-0}" != "1" ]]; then
    run "$PY -m poe_repair.experiments.cross_pair_lora_pooling.contact_sheet \
          --pooled-run '$RUN_DIR' \
          --pair-pool '$PAIR_POOL' \
          --seed-pool-path '$SEED_POOL'"
  else
    echo "[skip] contact sheet for $g (SKIP_CONTACT=1)"
  fi

  # 6. Δ̄_t bridge (skippable).
  if [[ "${SKIP_TASKD:-0}" != "1" ]]; then
    CELLS="$RUN_DIR/samples/cells.jsonl"
    if [[ -f "$CELLS" ]]; then
      run "$PY -m poe_repair.experiments.cross_pair_lora_pooling.task_d_bridge \
            --pooled-run '$RUN_DIR' \
            --pair-pool '$PAIR_POOL' \
            --seed-pool-path '$SEED_POOL' \
            --cells '$CELLS'"
    else
      echo "[warn] $g: $CELLS missing — skipping task D" >&2
    fi
  else
    echo "[skip] task D for $g (SKIP_TASKD=1)"
  fi

  echo "[done] $g  →  $RUN_DIR"
done
