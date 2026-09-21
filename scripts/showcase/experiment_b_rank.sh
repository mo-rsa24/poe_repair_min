#!/bin/bash
# Plan 09 (experiment-b-rank-16-32), task 1.1/1.2: clone of
# scripts/animals_compose_transfer/train_phase1.sh at a different rank.
# Only --lora-rank/--lora-alpha, --run-id and CUDA_VISIBLE_DEVICES change;
# every other flag is copied verbatim from the r8 run this compares against
# (artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/config.json),
# so the three runs are a matched-steps, single-axis comparison.
#
# Shared-device path (execution-protocol.md): biggpu already holds this
# user's one Slurm job, so this runs directly over SSH on a pinned free
# device on an allocated-but-half-used node, invisible to squeue. Launch
# the two ranks on two separate nodes/devices, e.g.:
#   ssh <node-a> 'nohup bash /abs/path/experiment_b_rank.sh 16 1 > /abs/path/logs/expB_r16.log 2>&1 &'
#   ssh <node-b> 'nohup bash /abs/path/experiment_b_rank.sh 32 1 > /abs/path/logs/expB_r32.log 2>&1 &'
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

RANK="${1:?usage: experiment_b_rank.sh <rank: 16|32> <cuda_device_index>}"
DEVICE="${2:?usage: experiment_b_rank.sh <rank: 16|32> <cuda_device_index>}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
RUN_ID="phase1_r${RANK}_100k"

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE rank=$RANK === experiment_b (${RUN_ID}) ==="

# --- Guards (execution-protocol.md, mandatory) -----------------------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE")
if [ "$USED" -gt 1024 ]; then
  echo "ABORT: device $DEVICE has ${USED}MiB in use (>1024MiB guard) — not free" >&2
  exit 1
fi
DISK_PCT=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9')
if [ "$DISK_PCT" -ge 90 ]; then
  echo "ABORT: /datasets at ${DISK_PCT}% (>=90% guard)" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ABORT: co3_bw python not found at $PY" >&2
  exit 1
fi
if ! nvidia-smi -i "$DEVICE" >/dev/null 2>&1; then
  echo "ABORT: no GPU visible at device $DEVICE" >&2
  exit 1
fi
if [ "$RANK" != "16" ] && [ "$RANK" != "32" ]; then
  echo "ABORT: rank must be 16 or 32 (rank 8 is already phase1_r8_100k), got $RANK" >&2
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] guards passed: device ${DEVICE} used=${USED}MiB, /datasets=${DISK_PCT}%"

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank "$RANK" --lora-alpha "$RANK" --lr 1e-4 \
    --total-epochs 2000 --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 10 \
    --sample-every-epochs 200 \
    --sample-cells-per-train-pair 8 --sample-cells-per-heldout-pair 8 \
    --sample-num-inference-steps 25 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUN_ID" \
    --output-root "$SCOPE/pooled_lora"

echo "[$(date -u +%H:%M:%S)] DONE ${RUN_ID}."
