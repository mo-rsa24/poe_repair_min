#!/bin/bash
# Shared-device launch for where_each_condition_lands_render.py, per environment/hpc/execution-protocol.md.
# Usage (from the session node, over SSH with nohup):
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/where_each_condition_lands_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/where_each_condition_lands_render.log 2>&1 &'
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"
STAGE="${STAGE:-render}"   # render | frames

# Blackwell nodes (110-112) need co3_bw; RTX 8000 / A6000 nodes (106, 108, 109) need co3.
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands
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

# A faulted device passes the memory guard (known-failures poe-launch-002); torch must actually see it,
# or the sampler silently falls back to CPU and runs ~40x slower with no error.
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU")
case "$UTIL" in *N/A*|*reset*) echo "guard: device $GPU reports utilization '$UTIL' (faulted), refusing" >&2; exit 6 ;; esac
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing" >&2; exit 7; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ disk_use=${USE_PCT}% stage=$STAGE"

case "$STAGE" in
  render) "$PY" scripts/showcase/where_each_condition_lands_render.py "$@" ;;
  frames) "$PY" scripts/showcase/where_each_condition_lands_trajectories.py "$@" ;;
  *) echo "unknown stage: $STAGE" >&2; exit 5 ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE"
