#!/bin/bash
# Separation steering on one shared biggpu device, launched over SSH outside Slurm:
#
#   ssh mscluster109 'GPU=1 MODE=calibrate nohup bash /datasets/.../separation_steering/_scripts/separation_steering.sh \
#       > /datasets/.../separation_steering/logs/calibrate.log 2>&1 &'
#
# MODE=calibrate  one seed, no steering applied, the loss terms and gradient norms at steps 0 to 9
# MODE=render     ETAS="small=.. medium=.. large=.." over SEEDS ("plain=0" renders the loop with no steering)
# Guards per environment/hpc/execution-protocol.md.
set -uo pipefail
GPU=${GPU:?set GPU to the device index}
MODE=${MODE:?calibrate or render}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8"}

echo "=== node $(hostname -s)  device $GPU  pid $$  mode $MODE  etas ${ETAS:-none}  seeds $SEEDS  $(date -Is)"
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "ABORT: no device $GPU" >&2; exit 4; }
[ "$used" -lt 1024 ] || { echo "ABORT: device $GPU has ${used} MiB in use" >&2; exit 4; }
[ -x "$PY" ] || { echo "ABORT: $PY missing" >&2; exit 5; }
pct=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${pct:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${pct}%" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU POE_REPO=$REPO CONCEPT_PAIR_OUT=$OUT PYTHONPATH=$REPO/scripts
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 5)" || { echo "ABORT: torch sees no CUDA device" >&2; exit 5; }

cd "$REPO" || exit 7
if [ "$MODE" = calibrate ]; then
  CALIBRATION_JSON=$OUT/separation_steering/calibration_seed${SEEDS%% *}.json \
    "$PY" "$HERE/separation_steering.py" --calibrate --seed ${SEEDS%% *} || exit 8
else
  "$PY" "$HERE/separation_steering.py" --render --eta $ETAS --seeds $SEEDS || exit 8
fi
echo "=== done $(date -Is)"
