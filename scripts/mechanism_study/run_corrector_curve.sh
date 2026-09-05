#!/usr/bin/env bash
# The Langevin-corrector runs of scope 06, on a shared biggpu device or in-session.
# Per environment/hpc/execution-protocol.md: every path absolute, the guards run inside
# this script, node/device/PID in the log header, and the python build picked by host.
#
#   STAGE=search  GPU=1 bash /abs/path/run_corrector_curve.sh   # plan 25: five c values at k=20
#   STAGE=smoke   GPU=1 bash ...                                # plan 26: k in {0,1}, wall time per level
#   STAGE=grid    GPU=1 bash ...                                # plan 26: the k grid, resumable
#   STAGE=window  GPU=1 bash ...                                # plan 27: the ten window columns
#   STAGE=sheet   GPU=1 bash ...                                # plan 27: the eight-seed triptych sheets
#   STAGE=tail    GPU=1 bash ...                                # plan 27: corrector on the last 15 steps of the adapter run
#
# Launch from the session node with every path absolute:
#   ssh mscluster108 'STAGE=grid GPU=1 nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/mechanism_study/run_corrector_curve.sh > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/grid.log 2>&1 &'
set -euo pipefail

STAGE="${STAGE:-grid}"
GPU="${GPU:-0}"
EXTRA="${EXTRA:-}"

case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector
cd "$REPO"
mkdir -p "$OUT"
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs

# The guard reads the filesystem this script writes to, derived from $OUT.
USE_PCT=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${USE_PCT:-0}" -ge 90 ] && { echo "disk guard: $(df --output=target "$OUT" | tail -1) at ${USE_PCT}%, aborting" >&2; exit 3; }
[ -x "$PY" ] || { echo "python not found at $PY" >&2; exit 2; }

USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc '0-9')
if [ "${USED_MIB:-9999}" -gt 1024 ]; then
  echo "guard: device $GPU has ${USED_MIB}MiB in use (>1GiB), refusing to start on a foreign process" >&2
  exit 4
fi
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU")
case "$UTIL" in *N/A*|*reset*) echo "guard: device $GPU reports utilization '$UTIL' (faulted, poe-launch-002), refusing" >&2; exit 6 ;; esac
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)' || { echo "guard: torch.cuda.is_available() is False on device $GPU, refusing (poe-launch-002)" >&2; exit 7; }

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] node=$(hostname -s) device=$GPU pid=$$ python=$PY disk_use=${USE_PCT}% stage=$STAGE extra='$EXTRA'"
"$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))"

case "$STAGE" in
  search) "$PY" scripts/corrector_residual_curve.py --step-size-search --k 20 --pair a_cat__x__a_dog --seed 9 $EXTRA ;;
  smoke)  "$PY" scripts/corrector_residual_curve.py --smoke --k 0,1 --pair a_cat__x__a_dog --seed 9 $EXTRA ;;
  grid)   "$PY" scripts/corrector_residual_curve.py --grid $EXTRA ;;
  window) "$PY" scripts/corrector_window_sweep.py --window-sweep $EXTRA ;;
  sheet)  "$PY" scripts/corrector_window_sweep.py --sheet $EXTRA ;;
  tail)   "$PY" scripts/corrector_window_sweep.py --tail $EXTRA ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 5 ;;
esac

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE stage=$STAGE"
