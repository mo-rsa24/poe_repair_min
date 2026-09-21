#!/bin/bash
# Scope 01, plan 16: noise trajectory search on the corrected sampler, shared-device launch per
# environment/hpc/execution-protocol.md.
# Usage (from the session node, over SSH with nohup; every path absolute):
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_trajectory_search/run_noise_trajectory_search.sh {smoke|full} [extra args] > /datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search/logs/<name>.log 2>&1 &'
set -euo pipefail
MODE="${1:-full}"; shift || true
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search
GPU="${GPU:?set GPU=<free device index>}"
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
cd "$REPO"
mkdir -p "$OUT/logs"
export APPTAINER_BIND=/datasets
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2; exit 3; fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }
# A device is free when nothing foreign is *running* on it (execution-protocol.md step 3): under
# 100 MiB nothing can be running; under 4 GiB at 0% utilisation is another user's idle residency
# (mscluster109 device 0, shared safely since 2026-09-03); anything else is refused.
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
raw=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader -i "$GPU")
case "$raw" in *N/A*|*reset*) echo "guard: device $GPU reports '$raw' (faulted, poe-launch-002), refusing" >&2; exit 6 ;; esac
if [ "${mem:-9999}" -lt 100 ]; then :;
elif [ "${mem:-9999}" -lt 4096 ] && [ "${util:-100}" -eq 0 ]; then :;
else echo "guard: device $GPU has ${mem} MiB in use at ${util}% utilisation, refusing to share" >&2; exit 4; fi
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing" >&2; exit 7; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname -s) device=$GPU pid=$$ py=$PY disk_use=${USE_PCT}% mode=$MODE commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"

case "$MODE" in
  smoke) exec "$PY" -m poe_repair.experiments.noise_trajectory_search.run --smoke "$@" ;;
  full)  exec "$PY" -m poe_repair.experiments.noise_trajectory_search.run --wandb-mode online "$@" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
