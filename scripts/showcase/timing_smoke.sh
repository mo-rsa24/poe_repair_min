#!/bin/bash
# Real per-stage timing for experiments A and B, before launching either
# (07-writing-the-paper... no — 01-showcase-the-trained-lora, plans 08/09).
# Fresh LoRA at the given rank (resume doesn't change per-step compute, so
# this stands in for A's resumed rank-8 timing too). Matches the production
# cadence: checkpoint every 5k steps (--ckpt-every-epochs 100, epoch-size
# 50), tracking-set eval every 10k steps (--sample-every-epochs 200) — the
# same cadence train_phase1.sh's non-dry mode uses, so the ONE eval pass
# this captures is apples-to-apples with what a real run pays per checkpoint.
#
# Tracking scope: one in-pair (a_wolf__x__a_husky), one out-pair
# (a_cat__x__a_dog), 50 DDIM steps — plan 06's cost-driven decision, carried
# forward here on the assumption the real A/B launchers adopt it too.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

DEVICE="${1:?usage: timing_smoke.sh <cuda_device_index> <rank> <alpha> <max_epochs> [train_batch_size]}"
RANK="${2:?rank required (8, 16, or 32)}"
ALPHA="${3:?alpha required (equals rank, per plan 09)}"
MAX_EPOCHS="${4:?max epochs required (200 = 10k steps for the full rank-8 measurement, 20 = 1k steps for the rank comparison)}"
BATCH="${5:-1}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase
RUNID="timing_smoke_r${RANK}_b${BATCH}"

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE rank=$RANK max_epochs=$MAX_EPOCHS batch=$BATCH === $RUNID ==="

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
    --lora-rank "$RANK" --lora-alpha "$ALPHA" --lr 1e-4 \
    --train-batch-size "$BATCH" \
    --total-epochs "$MAX_EPOCHS" --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 1 \
    --sample-every-epochs 200 \
    --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2 \
    --sample-train-pairs a_wolf__x__a_husky \
    --sample-heldout-pairs a_cat__x__a_dog \
    --sample-num-inference-steps 50 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUNID" \
    --output-root "$OUTROOT"

echo "[$(date -u +%H:%M:%S)] DONE $RUNID."
