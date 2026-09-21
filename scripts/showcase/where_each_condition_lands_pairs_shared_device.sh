#!/bin/bash
# Shared-device launch for where_each_condition_lands_pairs_render.py (plan 06 task 4, scope 05),
# per environment/hpc/execution-protocol.md.
# Usage (from the session node, over SSH with nohup):
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/where_each_condition_lands_pairs_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/where_each_condition_lands_pairs.log 2>&1 &'
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"

# Blackwell nodes (110-112) need co3_bw; RTX 8000 / A6000 nodes (106, 108, 109) need co3.
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/pairs
cd "$REPO"
mkdir -p "$OUT"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then
  echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2
  exit 3
fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
if [ "${USED_MIB:-9999}" -gt 1024 ]; then
  echo "guard: device $GPU has ${USED_MIB}MiB in use (>1GiB), refusing to start on a foreign process" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export XFORMERS_DISABLED=1
# A card can be visible to nvidia-smi and still unusable ("GPU requires reset" in pstate;
# mscluster111, 2026-09-05): torch then falls back to CPU silently and the run crawls.
"$PY" -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available on this device'" \
  || { echo "guard: torch.cuda not available on device $GPU (nvidia-smi pstate: $(nvidia-smi --query-gpu=pstate --format=csv,noheader -i "$GPU"))" >&2; exit 4; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ disk_use=${USE_PCT}% commit=$(git rev-parse --short HEAD)"

"$PY" scripts/showcase/where_each_condition_lands_pairs_render.py "$@"

echo "[$(date -u +%H:%M:%S)] DONE"
