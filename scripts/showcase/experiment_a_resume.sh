#!/bin/bash
# Experiment A (01-showcase-the-trained-lora, plan 08): resume phase1_r8_100k
# from lora_step_100000.pt to 200k steps, nothing else changed — rank 8,
# alpha 8, batch size 1, matching the original 0-100k recipe exactly. Per
# throughput.md's measurement, batch size does not reduce wall-clock to a
# fixed step count on this hardware (overhead-bound at batch=1), so there is
# no speed reason to change it, and changing it would confound the length
# axis this experiment exists to isolate.
#
# New checkpoints go to /datasets under a new run-id (phase1_r8_200k), not
# back into the existing phase1_r8_100k directory: that directory sits on
# /home-mscluster, which this project's own environment facts flag as the
# filesystem that once hit 100% and killed checkpointing. The resume source
# is read from there (read-only); nothing new is written there.
#
# Tracking scope: one in-pair (a_wolf__x__a_husky), one out-pair
# (a_cat__x__a_dog), 50 DDIM steps, cfg.probe buckets — plan 06's cost-driven
# decision, adopted here for the first time in a real (non-smoke) run.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

DEVICE="${1:?usage: experiment_a_resume.sh <cuda_device_index>}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
RESUME_FROM="$SCOPE/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt"
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase
RUNID="phase1_r8_200k"

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE === $RUNID (resume from 100k) ==="

# --- Guards (execution-protocol.md, mandatory) -----------------------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE")
if [ "$USED" -gt 1024 ]; then
  echo "ABORT: device $DEVICE has ${USED}MiB in use (>1024MiB guard) — not free" >&2
  exit 1
fi
# poe-launch-002: a fault reports near-zero memory too.
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
if [ ! -f "$RESUME_FROM" ]; then
  echo "ABORT: resume checkpoint not found at $RESUME_FROM" >&2
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] guards passed: device ${DEVICE} used=${USED}MiB, /datasets=${DISK_PCT}%, resume checkpoint present"

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --resume-from "$RESUME_FROM" \
    --total-epochs 4000 --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 20 \
    --sample-every-epochs 200 \
    --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2 \
    --sample-train-pairs a_wolf__x__a_husky \
    --sample-heldout-pairs a_cat__x__a_dog \
    --sample-num-inference-steps 50 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUNID" \
    --output-root "$OUTROOT"

echo "[$(date -u +%H:%M:%S)] DONE $RUNID."
