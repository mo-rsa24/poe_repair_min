#!/bin/bash
# Step 9 smoke (instrument-02-three-live-curves-while-training): 1 epoch of pooled
# LoRA training on the 11 animal train pairs, then one full inline-sampling eval pass
# (11 train pairs x 8 train seeds in_in + 8 held-out pairs x 8 held-out seeds out_out),
# logging the three live curves to W&B: eval/compose_rate, eval/direction_cosine,
# eval/frac_distance_reached. Same recipe as train_phase1.sh except epochs=1 and
# sampling every epoch.
#
# Runs DIRECTLY on a biggpu node over SSH (shared-node path in docs/ENVIRONMENT.md),
# never via sbatch. All paths on the launch line must be ABSOLUTE (SSH starts in $HOME;
# see docs/EXPERIMENT_ERROR_CATALOG.md poe-launch-001):
#   ssh <node> 'GPU=<free device> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/animals_compose_transfer/smoke_live_curves.sh > /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/step-09-smoke-<node>.log 2>&1 &'
set -uo pipefail

GPU="${GPU:?set GPU=<free device index>; verify with nvidia-smi that it is free first}"
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/live_curves_smoke_run
cd "$REPO"

# Device guard: abort if the chosen device is not actually free RIGHT NOW.
# The preflight minutes earlier is not enough; the device can be claimed in between.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ -z "$USED" ] || [ "$USED" -gt 1024 ]; then
  echo "FATAL: GPU $GPU on $(hostname -s) has ${USED:-unreadable}MiB in use; refusing to share a busy device" >&2
  exit 1
fi

# Disk guard on the filesystem this run actually writes to (/datasets, not /home-mscluster).
PCT=$(df /datasets | tail -1 | awk '{print int($5)}')
if [ "$PCT" -gt 90 ]; then
  echo "FATAL: /datasets at ${PCT}% capacity" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE=$REPO/outputs/animals_compose_transfer
mkdir -p "$OUT"

RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)"
echo "=== step-09 smoke launch header ==="
echo "run_id=$RUN_ID"
echo "node=$(hostname -s) gpu_device=$GPU pid=$$"
echo "commit=$(git rev-parse --short HEAD) dirty_files=$(git status --porcelain | wc -l)"
echo "gpu_state_at_launch: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)"
echo "==================================="

WANDB_NAME="$RUN_ID" exec "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --total-epochs 1 --epoch-size 50 \
    --ckpt-every-epochs 1 --log-every-epochs 1 \
    --sample-every-epochs 1 \
    --sample-cells-per-train-pair 8 --sample-cells-per-heldout-pair 8 \
    --sample-num-inference-steps 25 --sample-thumb 256 \
    --wandb-mode online --wandb-project poe-repair-animals-compose \
    --run-id "$RUN_ID" \
    --output-root "$OUT"
