#!/bin/bash
#SBATCH --job-name=stoch_sweep
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/stoch_sweep-%j.out
#SBATCH --error=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/logs/stoch_sweep-%j.err
# Plan 19 (01-showcase-the-trained-lora): the stochastic sampler sweep. Stage is $1:
# "render" (144 renders, resumable), "score", or "all" (render then score in one job).
# bigbatch nodes carry an RTX 3090 (24 GB); the excluded nodes have a dead or unreachable GPU
# (environment/hpc/nodes.md lists them).

set -euo pipefail

STAGE="${1:?usage: sbatch stochastic_sampler_sweep.sbatch <render|score|all>}"

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/stochastic_sampler_sweep
cd "$REPO"
mkdir -p "$OUT" logs
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
[ -x "$PY" ] || { echo "co3 python not found at $PY" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv \
  || { echo "no GPU visible" >&2; exit 4; }
$PY -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'" \
  || { echo "torch.cuda not available" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) commit=$(git rev-parse --short HEAD) " \
     "dirty=$(git status --porcelain | wc -l) disk_use=${USE_PCT}% stage=$STAGE"

case "$STAGE" in
  render) $PY scripts/showcase/stochastic_sampler_sweep.py --render ;;
  score)  $PY scripts/showcase/stochastic_sampler_sweep.py --score ;;
  all)    $PY scripts/showcase/stochastic_sampler_sweep.py --render --score ;;
  *) echo "unknown stage: $STAGE" >&2; exit 5 ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
