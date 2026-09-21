#!/bin/bash
#SBATCH --job-name=dogxdog_probe
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/dog_x_dog_probe/probe_slurm_%j.log
# Plan 02 (01-showcase-the-trained-lora) tasks 1.2-1.4: identity preflight, then (if it passes)
# the 8 held-out-seed dog x dog renders, then scoring. Stops after the identity check if it
# fails, per the plan's Fail criteria.
#
# Stage is passed as $1: "identity", "probe", or "score".
set -euo pipefail

STAGE="${1:?usage: sbatch dog_x_dog_probe.sbatch <identity|probe|score>}"

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

echo "[$(date -u +%H:%M:%S)] node=$(hostname) disk_use=${USE_PCT}% stage=$STAGE"

case "$STAGE" in
  identity)
    $PY scripts/showcase/dog_x_dog_probe.py --check-identity
    ;;
  probe)
    $PY scripts/showcase/dog_x_dog_probe.py --run-probe --seeds 9,10,11,12,13,14,15,16
    ;;
  score)
    $PY scripts/showcase/dog_x_dog_probe.py --score --seeds 9,10,11,12,13,14,15,16
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 5
    ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
