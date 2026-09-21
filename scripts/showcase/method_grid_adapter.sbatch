#!/bin/bash
#SBATCH --job-name=mgrid_adapter
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/method_grid/slurm_%j.log
# One adapter's LoRA column for scripts/showcase/method_grid.py, so the four adapters can
# render on four nodes at once. Usage: sbatch this.sbatch <adapter name>
set -euo pipefail
A="${1:?usage: sbatch method_grid_adapter.sbatch <r8_240k|r16_50k|r32_60k|crossseed>}"
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/method_grid
cd "$REPO"; mkdir -p "$OUT"
USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "$USE_PCT" -lt 90 ] || { echo "disk guard: ${USE_PCT}%" >&2; exit 3; }
[ -x "$PY" ] || { echo "co3 python not found" >&2; exit 2; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || { echo "no GPU" >&2; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) adapter=$A"
$PY scripts/showcase/method_grid.py --adapter "$A"
echo "[$(date -u +%H:%M:%S)] DONE adapter=$A"
