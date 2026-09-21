#!/usr/bin/env bash
# Build fresh training-cache cells for new animal pairs, on one pinned GPU.
#
#   GPU=<idx> PAIRS="slug|prompt_a|prompt_b|joint ; ..." bash build_new_animal_cells.sh
#
# One cell per pair per seed, seeds 1 to 8, written under the train split.
# Guards first, then the loop. Idempotent: the builder skips a cell already complete.
set -uo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUTROOT=/datasets/mmolefe/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
GPU="${GPU:-0}"

echo "=== node   $(hostname)"
echo "=== gpu    $GPU"
echo "=== pid    $$"
echo "PID $$"
echo "=== start  $(date -Is)"

# --- guard 1: the Blackwell python exists
[ -x "$PY" ] || { echo "ABORT: co3_bw python not found at $PY" >&2; exit 2; }
echo "=== python $PY"

# --- guard 2: disk on the filesystem we actually write to
USEPCT=$(df --output=pcent "$OUTROOT" | tail -1 | tr -dc '0-9')
echo "=== disk   ${USEPCT}% used on $OUTROOT"
[ "$USEPCT" -ge 90 ] && { echo "ABORT: $OUTROOT is ${USEPCT}% full" >&2; exit 3; }

# --- guard 3: the device is idle and real
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu \
                      --format=csv,noheader,nounits | tr -d ',' )
echo "=== device ${USED} MiB used, ${UTIL}% util"
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu $GPU reports utilisation '$UTIL', it is faulted" >&2; exit 4;; esac
[ "$USED" -gt 1024 ] && { echo "ABORT: gpu $GPU has ${USED} MiB in use by someone else" >&2; exit 5; }

# --- guard 4: torch actually sees the card
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)" \
  || { echo "ABORT: torch.cuda.is_available() is False on gpu $GPU" >&2; exit 6; }
echo "=== torch  sees the device"

cd "$REPO" || exit 7
OK=0; FAIL=0
IFS=';' read -ra ENTRIES <<< "$PAIRS"
for e in "${ENTRIES[@]}"; do
  e="$(echo "$e" | sed 's/^ *//; s/ *$//')"
  [ -z "$e" ] && continue
  IFS='|' read -r SLUG PA PB PJ <<< "$e"
  for s in $SEEDS; do
    echo "--- $SLUG seed $s  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" -m scripts.build_training_cache \
      --prompt-a "$PA" --prompt-b "$PB" --joint-prompt "$PJ" \
      --seed "$s" --split train
    if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); echo "!!! failed: $SLUG seed $s"; fi
  done
done
echo "=== done   $(date -Is)   built $OK, failed $FAIL"
