#!/bin/bash
# Phase-1: train ONE pooled rank-8 cross-attn LoRA on the 11 animal train pairs to 100k
# steps (matching the reference run), logging to W&B: loss curves + train/held-out
# samples over training steps (all 8 seeds each). Run DIRECTLY on node 110 (inside the
# existing allocation), NOT via sbatch (biggpu 1-job limit).
set -uo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs

MODE="${1:-run}"   # "dry" for a 1-epoch validation (wandb off), else full 100k
if [ "$MODE" = "dry" ]; then
  EXTRA="--dry-run"; WANDB="disabled"; SAMPLE_EVERY=1
else
  EXTRA=""; WANDB="online"; SAMPLE_EVERY=200
fi

$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool      "$SCOPE/pair_pool.yaml" \
    --pair-prompts   "$SCOPE/pair_prompts.yaml" \
    --seed-pool-path "$SCOPE/seed_pool.yaml" \
    --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --total-epochs 2000 --epoch-size 50 \
    --ckpt-every-epochs 100 --log-every-epochs 10 \
    --sample-every-epochs "$SAMPLE_EVERY" \
    --sample-cells-per-train-pair 8 --sample-cells-per-heldout-pair 8 \
    --sample-num-inference-steps 25 --sample-thumb 256 \
    --wandb-mode "$WANDB" --wandb-project poe-repair-animals-compose \
    --run-id phase1_r8_100k \
    --output-root "$SCOPE/pooled_lora" \
    $EXTRA
