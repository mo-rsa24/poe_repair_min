#!/usr/bin/env bash
# Task B — Pooled learning curve in k.
#
# Trains pooled LoRAs at k ∈ {1, 4, 8} (k=16 disabled until cache cells
# 13–16 are generated). For k=1 we train two single-seed picks to avoid
# lucky-seed reads. After each training, samples the four held-out seeds
# with that LoRA, leaving PNGs under .../samples/heldout/.
#
# Override CUDA_VISIBLE_DEVICES, EPOCHS, or KSET via env vars.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
CUDA=${CUDA_VISIBLE_DEVICES:-1}
EPOCHS=${EPOCHS:-1600}
KSET=${KSET:-"1a 1b 4 8"}
WANDB_MODE_=${WANDB_MODE_:-online}
WANDB_PROJECT_=${WANDB_PROJECT_:-poe-repair-cross-seed}
BATCH=${BATCH:-4}
XFORMERS=${XFORMERS:-1}
OUT_ROOT="$REPO_ROOT/outputs/cross_seed_lora_pooling/task_b_learning_curve"
mkdir -p "$OUT_ROOT"

cd "$REPO_ROOT"

train_and_sample () {
    local label="$1"; shift
    local k="$1"; shift
    local extra_train_flags="$1"; shift

    local run_dir
    local xflag=""
    [ "$XFORMERS" = "1" ] && xflag="--xformers"
    CUDA_VISIBLE_DEVICES=$CUDA "$PY" -m \
        poe_repair.experiments.cross_seed_lora_pooling.train_pooled \
        --k "$k" --total-epochs "$EPOCHS" \
        --output-root "$OUT_ROOT" \
        --run-id "k${label}__ep${EPOCHS}" \
        --wandb-mode "$WANDB_MODE_" \
        --wandb-project "$WANDB_PROJECT_" \
        --train-batch-size "$BATCH" \
        $xflag \
        $extra_train_flags

    run_dir="$OUT_ROOT/k${label}__ep${EPOCHS}"
    local ckpt
    ckpt=$(jq -r .path "$run_dir/checkpoints/latest.json")
    echo "==> sampling held-out seeds with $ckpt"
    CUDA_VISIBLE_DEVICES=$CUDA "$PY" -m \
        poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
        --checkpoint "$ckpt" \
        --out-dir "$run_dir/samples/heldout" \
        --record-eps
}

for tag in $KSET; do
    case "$tag" in
        1a) train_and_sample "01_pick1" 1 "--single-seed-pick 1" ;;
        1b) train_and_sample "01_pick5" 1 "--single-seed-pick 5" ;;
        4)  train_and_sample "04"       4 "" ;;
        8)  train_and_sample "08"       8 "" ;;
        16) train_and_sample "16"      16 "" ;;
        *)  echo "unknown KSET tag: $tag" >&2; exit 2 ;;
    esac
done

echo "==> Task B complete. Build contact sheet:"
echo "   $PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet --task B"
