#!/bin/bash
# Shared-device launch for dog_x_dog_probe.py, per environment/hpc/execution-protocol.md.
# Usage (from the session node): GPU=<idx> STAGE=<identity|probe|mono|poe|score|score_mono|score_poe> bash this_script.sh
# Intended to be invoked over SSH with nohup, per the protocol's shared-device path.
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"
STAGE="${STAGE:?set STAGE=<identity|probe|score>}"

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

# Re-check the device inside the launch script itself, not just at the caller's check minutes
# earlier — state can change between the check and the launch.
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
if [ "${USED_MIB:-9999}" -gt 1024 ]; then
  echo "guard: device $GPU has ${USED_MIB}MiB in use (>1GiB), refusing to start on a foreign process" >&2
  exit 4
fi

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ disk_use=${USE_PCT}% stage=$STAGE"

export CUDA_VISIBLE_DEVICES="$GPU"

case "$STAGE" in
  identity)
    "$PY" scripts/showcase/dog_x_dog_probe.py --check-identity
    ;;
  probe)
    "$PY" scripts/showcase/dog_x_dog_probe.py --run-probe --seeds 9,10,11,12,13,14,15,16
    ;;
  mono)
    "$PY" scripts/showcase/dog_x_dog_probe.py --run-mono --seeds 9,10,11,12,13,14,15,16
    ;;
  poe)
    "$PY" scripts/showcase/dog_x_dog_probe.py --run-poe --seeds 9,10,11,12,13,14,15,16
    ;;
  score)
    "$PY" scripts/showcase/dog_x_dog_probe.py --score --seeds 9,10,11,12,13,14,15,16
    ;;
  score_mono)
    "$PY" scripts/showcase/dog_x_dog_probe.py --score-mono --seeds 9,10,11,12,13,14,15,16
    ;;
  score_poe)
    "$PY" scripts/showcase/dog_x_dog_probe.py --score-poe --seeds 9,10,11,12,13,14,15,16
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 5
    ;;
esac

echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
