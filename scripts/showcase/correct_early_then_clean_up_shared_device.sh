#!/bin/bash
# Shared-device launch for correct_early_then_clean_up.py (plan 14, 01-showcase-the-trained-lora),
# per environment/hpc/execution-protocol.md.
# Usage (from the session node, over SSH with nohup; every path absolute):
#   ssh <node> 'GPU=<idx> STAGE=<detach|schedule|score|tail|renoise|sweep|strips|all> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/correct_early_then_clean_up_shared_device.sh > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/correct_early_<stage>.log 2>&1 &'
set -euo pipefail

GPU="${GPU:?set GPU=<free device index>}"
STAGE="${STAGE:?set STAGE=detach|schedule|score|tail|renoise|sweep|strips|rest|all}"

case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up
cd "$REPO"
mkdir -p "$OUT"

USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "disk guard: $OUT filesystem at ${USE_PCT}%, aborting" >&2; exit 3; fi
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
if [ "${USED_MIB:-9999}" -gt 1024 ]; then echo "guard: device $GPU has ${USED_MIB}MiB in use (>1GiB), refusing" >&2; exit 4; fi
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU")
case "$UTIL" in *N/A*|*reset*) echo "guard: device $GPU reports utilization '$UTIL' (faulted), refusing" >&2; exit 6 ;; esac
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing" >&2; exit 7; }

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$GPU pid=$$ disk_use=${USE_PCT}% stage=$STAGE"
SCRIPT=scripts/showcase/correct_early_then_clean_up.py
case "$STAGE" in
  detach)   "$PY" $SCRIPT --detach-check ;;
  schedule) "$PY" $SCRIPT --render --stage schedule ;;
  score)    "$PY" $SCRIPT --score ;;
  tail)     "$PY" $SCRIPT --render --stage tail ;;
  renoise)  "$PY" $SCRIPT --render --stage renoise ;;
  strips)   "$PY" $SCRIPT --strips ;;
  sweep)  # the re-noise level sweep, after the plan's own cells are scored: render, score, strips
    "$PY" $SCRIPT --render --stage renoise-sweep
    "$PY" $SCRIPT --score
    "$PY" $SCRIPT --strips
    ;;
  rest)   # resume after the schedule grid: score, tail, renoise, score
    "$PY" $SCRIPT --score
    "$PY" $SCRIPT --render --stage tail
    "$PY" $SCRIPT --render --stage renoise
    "$PY" $SCRIPT --score
    ;;
  all)
    "$PY" $SCRIPT --detach-check
    "$PY" - <<'PYEOF'
import json, sys
r = json.load(open("/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up/detach_check.json"))
print("detach check pass:", r["pass"]); sys.exit(0 if r["pass"] else 8)
PYEOF
    "$PY" $SCRIPT --render --stage schedule
    "$PY" $SCRIPT --score
    "$PY" $SCRIPT --render --stage tail
    "$PY" $SCRIPT --render --stage renoise
    "$PY" $SCRIPT --score
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 5 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE"
