#!/bin/bash
# Plan 06 (extend-the-tracking-set), task 1.3: the one-epoch smoke run proving
# the seven metric families log non-null and the frozen manifest hash reaches
# W&B config. Fresh LoRA (no --resume-from): this is a disposable proof run,
# not a step in the real A/B training lineage.
#
# Shared-device path (execution-protocol.md): biggpu already holds this
# user's one Slurm job, so this runs directly over SSH on a pinned free
# device on an allocated-but-half-used node, invisible to squeue.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

DEVICE="${1:?usage: tracking_smoke.sh <cuda_device_index>}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE === tracking_smoke ==="

# --- Guards (execution-protocol.md, mandatory) -----------------------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE")
if [ "$USED" -gt 1024 ]; then
  echo "ABORT: device $DEVICE has ${USED}MiB in use (>1024MiB guard) — not free" >&2
  exit 1
fi
# poe-launch-002: a fault reports near-zero memory too, so the memory guard
# alone passes a broken device as readily as a healthy idle one.
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$DEVICE")
if [[ "$UTIL" == *"N/A"* || "$UTIL" == *"reset"* ]]; then
  echo "ABORT: device $DEVICE reads '$UTIL' for utilization — hardware fault, not free (poe-launch-002)" >&2
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
echo "[$(date -u +%H:%M:%S)] guards passed: device ${DEVICE} used=${USED}MiB, /datasets=${DISK_PCT}%"

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --total-epochs 1 --epoch-size 50 \
    --ckpt-every-epochs 1 --log-every-epochs 1 \
    --sample-every-epochs 1 \
    --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2 \
    --sample-train-pairs a_wolf__x__a_husky \
    --sample-heldout-pairs a_cat__x__a_dog \
    --sample-num-inference-steps 50 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id tracking_smoke \
    --output-root "$OUTROOT"

echo "[$(date -u +%H:%M:%S)] DONE tracking_smoke."
