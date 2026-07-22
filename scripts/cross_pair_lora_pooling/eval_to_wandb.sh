#!/usr/bin/env bash
# Plan 16 — mid-training crossbar eval.
# Loads the latest lora_step_*.pt under $GROUP_DIR/$RUN_ID/checkpoints,
# samples in_in + out_in (training pairs + held pairs, both at all 12
# train seeds), writes PNGs + cells.jsonl + contact sheets under
# $RUN_DIR/eval_crossbar/step_<NNNNNN>/, and pushes everything to W&B
# under the eval_crossbar/* namespace.
#
# Required:
#   WANDB_RUN_ID=<the existing training run id to attach to>
#
# Override via env vars:
#   GROUP=g6                  # which within-group run to evaluate
#   RUN_ID=main               # disk path: outputs/.../within_group/$GROUP/$RUN_ID
#   QUADRANTS=in_in,out_in
#   OUT_IN_TRAIN_SEEDS=12
#   NUM_STEPS=50              # DDIM inference steps for eval
#   HEIGHT=1024
#   WIDTH=1024
#   GUIDANCE=7.5
#   LAMBDA=1.0                # 1.0 = full LoRA; 0.0 = frozen PoE
#   DTYPE=float16
#   THUMB=256                 # contact-sheet thumbnail px
#   WANDB_PROJECT=poe-repair-cross-pair
#   WANDB_ENTITY=...
#   WANDB_MODE=online         # online | offline | disabled
#   WANDB_NAMESPACE=eval_crossbar
#   PY=/path/to/python
#   CUDA_VISIBLE_DEVICES=0
#   POE_REPAIR_TRAINING_CACHE=/datasets/.../training_cache
#   DRY_RUN=1                 # echo command only

set -euo pipefail

PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
GROUP="${GROUP:-g6}"
RUN_ID="${RUN_ID:-main}"
QUADRANTS="${QUADRANTS:-in_in,out_in}"
OUT_IN_TRAIN_SEEDS="${OUT_IN_TRAIN_SEEDS:-12}"
NUM_STEPS="${NUM_STEPS:-50}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
GUIDANCE="${GUIDANCE:-7.5}"
LAMBDA="${LAMBDA:-1.0}"
DTYPE="${DTYPE:-float16}"
THUMB="${THUMB:-256}"
WANDB_PROJECT="${WANDB_PROJECT:-poe-repair-cross-pair}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAMESPACE="${WANDB_NAMESPACE:-eval_crossbar}"

if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  echo "[err] set WANDB_RUN_ID=<existing training run id> to attach to a run" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

WITHIN_ROOT="outputs/cross_pair_lora_pooling/within_group"
GROUP_DIR="$WITHIN_ROOT/$GROUP"
RUN_DIR="$GROUP_DIR/$RUN_ID"
PAIR_POOL="$GROUP_DIR/pair_pool.yaml"
SEED_POOL="$WITHIN_ROOT/seed_pool.yaml"
PAIR_PROMPTS="outputs/cross_pair_lora_pooling/pair_prompts.yaml"

for f in "$PAIR_POOL" "$SEED_POOL" "$PAIR_PROMPTS"; do
  [[ -f "$f" ]] || { echo "[err] missing $f" >&2; exit 2; }
done
[[ -d "$RUN_DIR/checkpoints" ]] || {
  echo "[err] no checkpoints dir at $RUN_DIR/checkpoints" >&2
  exit 2
}

ENTITY_ARG=""
if [[ -n "$WANDB_ENTITY" ]]; then
  ENTITY_ARG="--wandb-entity '$WANDB_ENTITY'"
fi

CMD="$PY -m poe_repair.experiments.cross_pair_lora_pooling.eval_crossbar_wandb \
      --run-dir '$RUN_DIR' \
      --pair-pool '$PAIR_POOL' \
      --seed-pool-path '$SEED_POOL' \
      --pair-prompts '$PAIR_PROMPTS' \
      --quadrants '$QUADRANTS' \
      --out-in-train-seeds $OUT_IN_TRAIN_SEEDS \
      --guidance-scale $GUIDANCE \
      --num-inference-steps $NUM_STEPS \
      --height $HEIGHT --width $WIDTH \
      --lambda $LAMBDA \
      --dtype $DTYPE \
      --thumb $THUMB \
      --wandb-run-id '$WANDB_RUN_ID' \
      --wandb-project '$WANDB_PROJECT' \
      --wandb-mode '$WANDB_MODE' \
      --wandb-namespace '$WANDB_NAMESPACE' \
      $ENTITY_ARG"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry] %s\n' "$CMD"
else
  printf '[run] %s\n' "$CMD"
  eval "$CMD"
fi
