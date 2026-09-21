#!/bin/bash
#SBATCH --job-name=lora_dose_sweep
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster72
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/lora_dose_sweep-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/lora_dose_sweep-%j.err
# Plan 03 (01-showcase-the-trained-lora) tasks 1.1-1.5: capture per-seed LoRA deltas, print
# and record the manifest, render the sweep, score it, and compute the softness read.
#
# Stage is passed as $1: "capture", "manifest", "sweep", "score", or "softness".
set -euo pipefail

STAGE="${1:?usage: sbatch lora_dose_sweep.sbatch <capture|manifest|sweep|score|softness>}"

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/lora_dose
cd "$REPO"
mkdir -p "$OUT" logs
export APPTAINER_BIND=/datasets
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }
$PY -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'" \
  || { echo "torch.cuda not available" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) disk_use=${USE_PCT}% stage=$STAGE"

case "$STAGE" in
  capture)
    $PY scripts/showcase/lora_dose_sweep.py --capture-deltas
    ;;
  manifest)
    $PY scripts/showcase/lora_dose_sweep.py --print-manifest
    ;;
  sweep)
    $PY scripts/showcase/lora_dose_sweep.py --run-sweep
    ;;
  score)
    $PY scripts/showcase/lora_dose_sweep.py --score
    ;;
  softness)
    $PY scripts/showcase/lora_dose_sweep.py --softness
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 5
    ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
