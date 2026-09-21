#!/bin/bash
#SBATCH --job-name=r32_wd_ema
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/r32_wd_ema-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/r32_wd_ema-%j.err
# Plan 20 (01-showcase-the-trained-lora): experiment D again with an EMA of the LoRA weights.
# Every flag is experiment D's launch (scripts/showcase/experiment_d_weight_decay.sh: rank 32,
# weight decay 0.1, the kill criterion off) plus --ema-decay, so the raw weights replicate D's
# trajectory and the EMA weights are the one new arm. 800 epochs x 50 steps = 40,000 steps, a
# checkpoint every 100 epochs (5,000 steps), the tracking set every 200 epochs.
# Runs on bigbatch (RTX 3090, 24 GB, co3) through Slurm because every biggpu device is busy;
# gradient checkpointing is on because the plain run needs about 25 GB (job 50339 ran out of memory).

set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
mkdir -p logs
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase

RANK=32
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-800}"
RUN_ID="${RUN_ID:-phase1_r32_wd${WEIGHT_DECAY}_ema${EMA_DECAY}_40k}"

USE_PCT=$(df --output=pcent "$OUT_ROOT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT_ROOT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
if [ -e "$OUT_ROOT/$RUN_ID" ]; then
  echo "ABORT: $OUT_ROOT/$RUN_ID already exists" >&2
  exit 1
fi
[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }
$PY -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'" \
  || { echo "torch.cuda not available" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) " \
     "dirty=$(git status --porcelain | wc -l) disk_use=${USE_PCT}% run_id=$RUN_ID " \
     "weight_decay=$WEIGHT_DECAY ema_decay=$EMA_DECAY total_epochs=$TOTAL_EPOCHS"

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank "$RANK" --lora-alpha "$RANK" --lr 1e-4 \
    --weight-decay "$WEIGHT_DECAY" --ema-decay "$EMA_DECAY" \
    --kill-halve-after-steps 1000000000 --gradient-checkpointing \
    --total-epochs "$TOTAL_EPOCHS" --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 10 \
    --sample-every-epochs 200 \
    --sample-cells-per-train-pair 8 --sample-cells-per-heldout-pair 8 \
    --sample-num-inference-steps 25 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUN_ID" \
    --output-root "$OUT_ROOT"

echo "[$(date -u +%H:%M:%S)] DONE ${RUN_ID}."
