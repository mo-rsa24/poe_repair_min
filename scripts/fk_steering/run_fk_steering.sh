#!/bin/bash
# Scope 06, plan 10: Feynman-Kac steering on the detector reward over the plain PoE sampler.
# Usage:  bash scripts/fk_steering/run_fk_steering.sh {smoke|full|adapter} [extra args...]
#         GPU=<index> pins the device (default 0). Blackwell nodes (110-112) take co3_bw.
set -euo pipefail
MODE="${1:-full}"; shift || true
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering
GPU="${GPU:-0}"
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
cd "$REPO"
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export APPTAINER_BIND=/datasets
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2; exit 3; fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
UTIL=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader -i "$GPU")
if [ "$USED" -gt 1024 ] || [[ "$UTIL" == *"N/A"* || "$UTIL" == *"reset"* ]]; then
  echo "ABORT: device $GPU used=${USED}MiB util/temp='$UTIL' (busy or faulted, poe-launch-002)" >&2; exit 4
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'; print('torch sees', torch.cuda.get_device_name(0))" \
  || { echo "torch.cuda not available on device $GPU (poe-launch-002)" >&2; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname -s) device=$GPU py=$PY disk_use=${USE_PCT}% mode=$MODE commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"

case "$MODE" in
  smoke) exec $PY -m poe_repair.experiments.fk_steering.run --smoke "$@" ;;
  full)  exec $PY -m poe_repair.experiments.fk_steering.run --wandb-mode online "$@" ;;
  adapter) exec $PY -m poe_repair.experiments.fk_steering.run --wandb-mode online \
      --adapter-checkpoint /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt \
      --adapter-rank 32 --adapter-lambdas 0.5 1.2 --control-pair-lambdas 1.2 --particles 16 \
      --fidelity imagereward "$@" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
