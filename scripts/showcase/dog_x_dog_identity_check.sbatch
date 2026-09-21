#!/bin/bash
#SBATCH --job-name=dogxdog_identity_check
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/identity_check_slurm_%j.log
# Plan 02 (01-showcase-the-trained-lora) task 1.2 preflight: which "Mono(A)" reference actually
# matches PoE("a dog","a dog") within fp16 drift. See dog_x_dog_identity_check.py's docstring.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe
cd "$REPO"
mkdir -p "$OUT"

# Disk guard: abort if the filesystem this script writes to is nearly full.
USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi

[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) disk_use=${USE_PCT}%"

$PY scripts/showcase/dog_x_dog_identity_check.py \
  --seed 9 \
  --out "$OUT/identity_check_seed9.json"

echo "[$(date -u +%H:%M:%S)] DONE"
