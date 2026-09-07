#!/bin/bash
# Shared-device launch for tangent_projection.py (plan 15, 01-showcase-the-trained-lora),
# per environment/hpc/execution-protocol.md.
# Usage (from the session node, over SSH with nohup; every path absolute):
#   ssh <node> 'GPU=<idx> STAGE=<smoke|read|render|score|figures|wandb|all> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/tangent_projection_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/tangent_projection_<stage>.log 2>&1 &'
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"
STAGE="${STAGE:?set STAGE=smoke|read|hsweep|render|score|figures|wandb|all|smoke_then_all}"

case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/tangent_projection
cd "$REPO"
mkdir -p "$OUT"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2; exit 3; fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }

# A device is free when nothing foreign is *running* on it (scripts/launch_sdlora.sh's rule):
# under 100 MiB nothing can be running, so memory alone decides; otherwise up to 4 GB of idle
# foreign residency at 0% utilisation is shared, anything more or anything active is refused.
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -dc '0-9')
UTIL_RAW=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU" | head -1)
case "$UTIL_RAW" in *N/A*|*reset*) echo "guard: device $GPU reports utilization '$UTIL_RAW' (faulted), refusing" >&2; exit 6 ;; esac
UTIL=$(echo "$UTIL_RAW" | tr -dc '0-9')
if [ "${MEM:-0}" -lt 100 ]; then :;
elif [ "${MEM:-0}" -lt 4096 ] && [ "${UTIL:-100}" -eq 0 ]; then :;
else echo "guard: device $GPU has ${MEM} MiB in use at ${UTIL}% utilisation, refusing to share" >&2; exit 4; fi
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # two UNets of different dtypes fragment the allocator otherwise
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing" >&2; exit 7; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ disk_use=${USE_PCT}% device_mem=${MEM}MiB util=${UTIL}% stage=$STAGE"
SCRIPT=scripts/showcase/tangent_projection.py
case "$STAGE" in
  smoke)   "$PY" $SCRIPT --cache-read --smoke; "$PY" $SCRIPT --render --smoke ;;
  read)    "$PY" $SCRIPT --cache-read ;;
  hsweep)  "$PY" $SCRIPT --h-sweep ;;
  render)  "$PY" $SCRIPT --render ;;
  score)   "$PY" $SCRIPT --score ;;
  figures) "$PY" $SCRIPT --figures ;;
  wandb)   "$PY" $SCRIPT --wandb ;;
  smoke_then_all)   # the smoke proves the memory fit on this card before the long stages start
    "$PY" $SCRIPT --cache-read --smoke
    "$PY" $SCRIPT --render --smoke
    "$PY" $SCRIPT --cache-read
    "$PY" $SCRIPT --render
    "$PY" $SCRIPT --score
    "$PY" $SCRIPT --figures
    "$PY" $SCRIPT --wandb
    ;;
  all)
    "$PY" $SCRIPT --cache-read
    "$PY" $SCRIPT --render
    "$PY" $SCRIPT --score
    "$PY" $SCRIPT --figures
    "$PY" $SCRIPT --wandb
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 5 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
