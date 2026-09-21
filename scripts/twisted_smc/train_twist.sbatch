#!/bin/bash
#SBATCH --job-name=twist_smc
#SBATCH --partition=biggpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/twist_smc-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/twist_smc-%j.err
# Scope 06, baseline 09: train the twist head contrastively on the training cache and log
# Mono | PoE | twisted-SMC strips to W&B every --render-every steps.
#
# Usage:   sbatch --nodelist=mscluster110 scripts/twisted_smc/train_twist.sbatch [smoke|full] [extra args...]
# Blackwell nodes (110-112) need co3_bw; RTX 8000 / A6000 nodes (106, 108, 109) need co3.
set -euo pipefail

MODE="${1:-full}"; shift || true
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
cd "$REPO"
mkdir -p "$OUT" logs
export APPTAINER_BIND=/datasets
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/outputs/training_cache

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }
$PY -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'" \
  || { echo "torch.cuda not available" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) py=$PY disk_use=${USE_PCT}% mode=$MODE"

case "$MODE" in
  smoke)
    $PY -m poe_repair.experiments.twisted_smc.train --smoke --wandb-mode online "$@"
    ;;
  full)
    $PY -m poe_repair.experiments.twisted_smc.train \
      --total-steps 100000 --render-every 10000 --ckpt-every 10000 \
      --batch-size 16 --lr 1e-4 --particles 4 --eta 1.0 --render-steps 20 \
      --render-pairs a_wolf__x__a_husky a_lion__x__a_tiger a_cat__x__a_dog \
      --wandb-mode online "$@"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
