#!/bin/bash
# Shared-device launch for the zero-order noise search (scope 06, plan 11), per environment/hpc/execution-protocol.md.
# Usage, over SSH with nohup and absolute paths only:
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh {smoke|full|adapter-smoke|adapter-full} > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/<name>.log 2>&1 &'
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"
MODE="${1:-smoke}"; shift || true

case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search
cd "$REPO"
mkdir -p "$OUT/logs"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2; exit 3; fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
if [ "${USED_MIB:-9999}" -gt 1024 ]; then echo "guard: device $GPU has ${USED_MIB}MiB in use (>1GiB), refusing to start on a foreign process" >&2; exit 4; fi
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU")
case "$UTIL" in *N/A*|*reset*) echo "guard: device $GPU reports utilization '$UTIL' (faulted, poe-launch-002), refusing" >&2; exit 6 ;; esac
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing" >&2; exit 7; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ python=$PY disk_use=${USE_PCT}% mode=$MODE torch_sees=$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))')"

case "$MODE" in
  smoke) "$PY" -m poe_repair.experiments.noise_search.run --smoke "$@" ;;
  full)  "$PY" -m poe_repair.experiments.noise_search.run "$@" ;;
  adapter-smoke) "$PY" -m poe_repair.experiments.noise_search.run_adapter --smoke "$@" ;;
  adapter-full)  "$PY" -m poe_repair.experiments.noise_search.run_adapter "$@" ;;
  *) echo "unknown mode: $MODE (smoke|full|adapter-smoke|adapter-full)" >&2; exit 5 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE"
