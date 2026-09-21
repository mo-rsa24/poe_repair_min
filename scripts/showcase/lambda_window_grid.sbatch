#!/bin/bash
#SBATCH --job-name=lambda_window_grid
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster76
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/lambda_window_grid-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/lambda_window_grid-%j.err
# Plan 07 (01-showcase-the-trained-lora) task group 1: render the 5-cell x 5-lambda grid,
# measure compose/drift/sharpness per render, build the descriptive strip.
#
# Stage is passed as $1: "sweep", "measure", or "strip".

set -euo pipefail

STAGE="${1:?usage: sbatch lambda_window_grid.sbatch <sweep|measure|strip>}"

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/experiment_c
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

echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) " \
     "dirty=$(git status --porcelain | wc -l) disk_use=${USE_PCT}% stage=$STAGE"

case "$STAGE" in
  sweep)
    $PY scripts/showcase/lambda_window_grid.py --run-sweep
    ;;
  measure)
    $PY scripts/showcase/lambda_window_grid.py --measure
    ;;
  strip)
    $PY scripts/showcase/lambda_window_grid.py --strip
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 5
    ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
