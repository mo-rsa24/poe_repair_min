#!/bin/bash
#SBATCH --job-name=animals_prereqs
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster108
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=results/animals_compose_transfer/slurm_prereqs_%j.log
# Phase-1 generation prerequisites for the animals-compose-transfer scope:
#   A) fail-rate: vanilla-PoE for all 19 pairs x 8 seeds, scored by the instance-count
#      scorer -> fail_rate.md (plan 01 task 4).
#   B) training caches for the 11 TRAIN pairs x 8 seeds (LoRA trains on these).
#   C) held-out-eval caches for the 5 eval pairs x 8 seeds (LoRA sampler pins init
#      latents from cache; no randn fallback).
# All idempotent (skip cells already on disk). Does NOT train.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
mkdir -p results/animals_compose_transfer

echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

echo "[$(date -u +%H:%M:%S)] === A) fail-rate (19 pairs x 8 seeds, vanilla PoE + score) ==="
$PY -m poe_repair.experiments.animals_compose_transfer.fail_rate

echo "[$(date -u +%H:%M:%S)] === B) training caches (11 train pairs x 8 seeds) ==="
$PY -m poe_repair.experiments.animals_compose_transfer.build_caches --which train

echo "[$(date -u +%H:%M:%S)] === C) held-out-eval caches (5 pairs x 8 seeds) ==="
$PY -m poe_repair.experiments.animals_compose_transfer.build_caches --which heldout_eval

echo "[$(date -u +%H:%M:%S)] DONE prereqs. fail_rate.md:"
cat artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md || true
