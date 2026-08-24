#!/bin/bash
#SBATCH --job-name=animals_train
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster108
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=results/animals_compose_transfer/slurm_train_%j.log
# Phase-1: train ONE pooled rank-8 cross-attn LoRA on the 11 animal TRAIN pairs, then
# sample the held-out pairs (lambda=1 correction + lambda=0 canary) and score compose-rate.
# Requires the prereq caches (gen_prereqs.sbatch) to exist first.
# Short training (400 epochs ~= 20k steps) for a cheap first-pass transfer read.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
POOL=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml
PROMPTS=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pair_prompts.yaml
SEEDS=$REPO/outputs/cross_pair_lora_pooling/seed_pool.yaml
OUTROOT=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora
RUNID=phase1_r8

echo "[$(date -u +%H:%M:%S)] node=$(hostname) === train pooled LoRA (11 pairs, ~20k steps) ==="
$PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
    --pair-pool "$POOL" --pair-prompts "$PROMPTS" --seed-pool-path "$SEEDS" \
    --cache-root "$POE_REPAIR_TRAINING_CACHE" \
    --lora-rank 8 --lora-alpha 8 --lr 1e-4 \
    --total-epochs 400 --epoch-size 50 --ckpt-every-epochs 100 \
    --output-root "$OUTROOT" --run-id "$RUNID"

CKPT=$(ls -t "$OUTROOT/$RUNID"/checkpoints/lora_step_*.pt | head -1)
echo "[$(date -u +%H:%M:%S)] latest checkpoint: $CKPT"

echo "[$(date -u +%H:%M:%S)] === sample held-out pairs (transfer test) lambda=1 + canary lambda=0 ==="
for LAM in 1.0 0.0; do
  $PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \
      --checkpoint "$CKPT" \
      --pair-pool "$POOL" --pair-prompts "$PROMPTS" --seed-pool-path "$SEEDS" \
      --cache-root "$POE_REPAIR_TRAINING_CACHE" \
      --quadrants out_out --lambda "$LAM" \
      --out-dir "$OUTROOT/$RUNID/samples_lambda${LAM}"
done

echo "[$(date -u +%H:%M:%S)] DONE train+sample. Score compose-rate next (compose_rate.py)."
