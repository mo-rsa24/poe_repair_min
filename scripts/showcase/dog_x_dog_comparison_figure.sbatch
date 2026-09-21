#!/bin/bash
#SBATCH --job-name=dogxdog_figure
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/figure_slurm_%j.log
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe
cd "$REPO"
mkdir -p "$OUT"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) disk_use=${USE_PCT}%"

$PY scripts/showcase/dog_x_dog_comparison_figure.py --seed 9

echo "[$(date -u +%H:%M:%S)] DONE"
