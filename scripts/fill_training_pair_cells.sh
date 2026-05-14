#!/usr/bin/env bash
# Fill the 20 missing CO3 cells in dataset/ — and replace pilot-sourced
# PoE/Mono symlinks on the four training pairs with in-repo SDXL outputs.
#
# Strategy: two e1_held_out workers, each pinned to a different CUDA
# device, each handling half the training-pair grid. Each worker uses
# ~20-25 GB (SDXL loaded once by our pipeline + once by CO3's own
# pipeline), well under the 48 GB on each RTX 8000. With both workers
# busy, wall-clock drops from ~70 min serial to ~35 min.
#
# Pair partition:
#   GPU 0 : a cow × a horse, a lion × a horse  (the "horse" pairs)
#   GPU 1 : a lion × a dog,  a tiger × a dog   (the "dog" pairs)
# Each worker = 2 pairs × 5 seeds × 5 methods = 50 sampler calls.
# CO3 caches one instance per (prompt_orig, prompt_concat) so 2 pairs/
# worker means 2 CO3 rebuilds per worker.
#
# After both workers finish, scripts/build_dataset.py is re-run so
# dataset/cells/ symlinks pick up the new outputs/e1_held_out/ paths.
#
# Usage:
#   scripts/fill_training_pair_cells.sh                  # cached run
#   scripts/fill_training_pair_cells.sh --overwrite      # force regen
#
# Extra args are passed through to both workers.

set -euo pipefail

PYTHON="${PYTHON:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"

cd "$(dirname "$0")/.."

LOG_DIR="logs/fill_training_pair_cells"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_GPU0="${LOG_DIR}/${TS}_gpu0_horse_pairs.log"
LOG_GPU1="${LOG_DIR}/${TS}_gpu1_dog_pairs.log"

echo "[fill] launching GPU 0 (horse pairs)  -> $LOG_GPU0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m poe_repair.experiments.e1_held_out \
    --pairs "a cow|a horse" "a lion|a horse" \
    --seeds 1 2 3 4 42 \
    "$@" >"$LOG_GPU0" 2>&1 &
PID_GPU0=$!

echo "[fill] launching GPU 1 (dog pairs)    -> $LOG_GPU1"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m poe_repair.experiments.e1_held_out \
    --pairs "a lion|a dog" "a tiger|a dog" \
    --seeds 1 2 3 4 42 \
    "$@" >"$LOG_GPU1" 2>&1 &
PID_GPU1=$!

echo "[fill] waiting on workers (pids $PID_GPU0, $PID_GPU1)"
EXIT=0
if ! wait "$PID_GPU0"; then
    echo "[fill] GPU 0 worker FAILED — see $LOG_GPU0"
    EXIT=1
fi
if ! wait "$PID_GPU1"; then
    echo "[fill] GPU 1 worker FAILED — see $LOG_GPU1"
    EXIT=1
fi

if [[ $EXIT -ne 0 ]]; then
    echo "[fill] one or more workers failed; logs in $LOG_DIR"
    exit 1
fi

echo "[fill] both workers done; refreshing dataset/ symlinks"
"$PYTHON" scripts/build_dataset.py

echo "[fill] done"
