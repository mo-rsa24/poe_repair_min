#!/bin/bash
# Scope 01 plan 15 (experiment D): the rank-32 pooled run again with AdamW weight decay on,
# everything else copied verbatim from experiment_b_rank.sh at rank 32 (the run that made
# phase1_r32_100k, config at /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/config.json).
# Two flags differ from that run, both named in the review file's fairness check:
#   --weight-decay <wd>            the one axis under test
#   --kill-halve-after-steps 1e9   the kill criterion aborted the fresh rank-32 run at 6,500 steps
#                                  (loss started low and could not halve); a resume cannot be
#                                  killed by it, so the baseline effectively ran without it too.
#
# Shared-device path (environment/hpc/execution-protocol.md): runs over SSH on a pinned device
# of an allocated biggpu node, invisible to squeue. Example:
#   ssh mscluster109 'nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_d_weight_decay.sh 0.1 0 > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_d_wd0.1.log 2>&1 &'
set -euo pipefail

WD="${1:?usage: experiment_d_weight_decay.sh <weight_decay e.g. 0.1> <cuda_device_index> [python]}"
DEVICE="${2:?usage: experiment_d_weight_decay.sh <weight_decay e.g. 0.1> <cuda_device_index> [python]}"
# co3 on the A6000 / RTX 8000 nodes (106, 108, 109); co3_bw on the Blackwell nodes (110 to 112).
PY="${3:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase
RUN_ID="phase1_r32_wd${WD}_100k"

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE wd=$WD === experiment_d (${RUN_ID}) ==="

# --- Guards (execution-protocol.md) -----------------------------------------
# The memory guard is 4096 MiB here, not the usual 1024: on 2026-09-06 the only device with
# 0% utilisation held another user's idle 2.9 GB process on a 49 GB card. The utilisation
# guard below is what actually protects the run's speed.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE")
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$DEVICE")
if [ "$USED" -gt 4096 ]; then
  echo "ABORT: device $DEVICE has ${USED}MiB in use (>4096MiB guard), not free" >&2; exit 1
fi
if [ "$UTIL" -gt 5 ]; then
  echo "ABORT: device $DEVICE at ${UTIL}% utilisation, someone is computing on it" >&2; exit 1
fi
DISK_PCT=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9')
if [ "$DISK_PCT" -ge 90 ]; then
  echo "ABORT: /datasets at ${DISK_PCT}% (>=90% guard)" >&2; exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ABORT: python not found at $PY" >&2; exit 1
fi
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ABORT: torch.cuda.is_available() is False on device $DEVICE (mscluster111-style dead GPU)" >&2; exit 1
fi
if [ -e "$OUT_ROOT/$RUN_ID" ]; then
  echo "ABORT: $OUT_ROOT/$RUN_ID already exists; pick another wd or move it aside" >&2; exit 1
fi
echo "[$(date -u +%H:%M:%S)] guards passed: device ${DEVICE} used=${USED}MiB util=${UTIL}%, /datasets=${DISK_PCT}%, cuda ok"

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 32 --lora-alpha 32 --lr 1e-4 \
    --weight-decay "$WD" \
    --kill-halve-after-steps 1000000000 \
    --total-epochs 2000 --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 10 \
    --sample-every-epochs 200 \
    --sample-cells-per-train-pair 8 --sample-cells-per-heldout-pair 8 \
    --sample-num-inference-steps 25 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUN_ID" \
    --output-root "$OUT_ROOT"

echo "[$(date -u +%H:%M:%S)] DONE ${RUN_ID}."
