#!/bin/bash
#SBATCH --job-name=recap_g6
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=results/recap_landed/gen/slurm_%j.log
# Recap enactment generation: G6 survive-noise held-out-seed row.
# Generates the ONE missing qualitative enactment for the recall gallery —
# the pooled cat×dog LoRA sampled on held-out seeds {9,10,11,12} at λ=1 (the
# pooled correction) and λ=0 (canary = vanilla PoE). PoE/Mono reference columns
# come from the cache (already on disk); the beachhead λ-morph is reused from
# the existing 183 probes. Idempotent: skips if outputs already exist.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
CKPT=/datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/checkpoints/lora_step_100000.pt
POOL="$REPO/outputs/cross_seed_lora_pooling/seed_pool.yaml"
OUT="$REPO/results/recap_landed/gen/g6_survive_noise"

# --- disk guard (need a few GB for the decoded PNGs) ---
avail=$(df -BG --output=avail "$REPO/results" | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -lt 2 ]; then echo "ABORT: <2G free on results/"; exit 1; fi

test -f "$CKPT" || { echo "ABORT: checkpoint missing: $CKPT"; exit 1; }

echo "[$(date -u +%H:%M:%S)] pooled-LoRA λ=1 on held-out seeds 9-12"
$PY -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
    --checkpoint "$CKPT" \
    --seed-pool-path "$POOL" \
    --seeds 9,10,11,12 \
    --lambda 1.0 \
    --cache-root "$POE_REPAIR_TRAINING_CACHE" \
    --out-dir "$OUT/lambda1"

echo "[$(date -u +%H:%M:%S)] canary λ=0 (should equal vanilla PoE)"
$PY -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
    --checkpoint "$CKPT" \
    --seed-pool-path "$POOL" \
    --seeds 9,10,11,12 \
    --lambda 0.0 \
    --cache-root "$POE_REPAIR_TRAINING_CACHE" \
    --out-dir "$OUT/lambda0"

echo "[$(date -u +%H:%M:%S)] DONE recap_g6 → $OUT"
