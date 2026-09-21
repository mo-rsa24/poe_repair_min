#!/bin/bash
#SBATCH --job-name=energy_readout
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83,mscluster85
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/energy_penalty_readout_%j.log
# Plan 22 of scope 01: the readout, queued behind the three arms with
#   sbatch --dependency=afterany:<j0>:<j1>:<j2> scripts/showcase/energy_penalty_readout.sbatch
# Stage 1 (GPU): the baseline lineage's own 40,050 grid and the on-policy energy reads of the
# baseline at 30,050 and 40,050. Stage 2: figures, strips, verdict, W&B. Every read that finds an
# arm missing skips it and says so, so a dead arm does not block the others' readout.
set -uo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
SHOWCASE=/datasets/mmolefe/poe_repair_min/outputs/showcase
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO" || exit 2
USE_PCT=$(df --output=pcent "$SHOWCASE" | tail -1 | tr -dc '0-9')
[ "$USE_PCT" -ge 90 ] && { echo "ABORT: /datasets at ${USE_PCT}%" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: co3 python missing" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || { echo "ABORT: no GPU" >&2; exit 4; }
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)" || { echo "ABORT: cuda unavailable (poe-launch-002)" >&2; exit 7; }
echo "[$(date -u '+%F %T')] node=$(hostname -s) job=${SLURM_JOB_ID:-none} commit=$(git rev-parse --short HEAD) disk_use=${USE_PCT}%"
for b in 0 0.01 0.05; do
  ls "$SHOWCASE/figure_r32_energy${b}_040050/results.json" "$SHOWCASE/energy_penalty/on_policy/energy${b}_040050/on_policy_energy.json" 2>&1
done
$PY scripts/showcase/energy_penalty_readout.py --baseline
echo "[$(date -u '+%F %T')] baseline reads done exit=$?"
$PY scripts/showcase/energy_penalty_readout.py --all
echo "[$(date -u '+%F %T')] DONE readout exit=$?"
