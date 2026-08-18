#!/usr/bin/env bash
# Over-correction grid: does pushing lambda past 1.0 inside a window help,
# hurt, or do nothing, and does that depend on which window?
#
# 5 non-overlapping windows (0-10, 10-20, 20-30, 30-40, 40-50) x 4 lambdas
# (0.5, 1.0, 1.5, 2.0) x 1 seed = 20 images. Cat x dog, seed 9, matching the
# running example every other qualitative figure in this paper uses.
#
# Different window definition from the 9-window stride-5 grid F4a/F4b/F4c/F4e
# use (poe_repair/experiments/interaction_term/window_grid.py): this one is
# non-overlapping, by request, to read as "which tenth of the run" rather
# than "which sliding position".
#
# ~50s per cell, so about 17 minutes. Resumable: a cell whose image already
# exists is skipped.
#
#   bash scripts/mechanism_study/run_overcorrection_grid.sh
#   SEED=12 bash scripts/mechanism_study/run_overcorrection_grid.sh
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

# Large artifacts go to /datasets, never to /home-mscluster (filled to 100%
# once and silently killed checkpointing). The sampler reads this env var for
# its output root.
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs

EXP=interaction_term/overcorrection_grid
OUT=$POE_REPAIR_OUTPUT_ROOT/$EXP/pairs

PAIR=a_cat__x__a_dog
SEED=${SEED:-9}
WINDOWS=(0,10 10,20 20,30 30,40 40,50)
LAMBDAS=(0.5 1.0 1.5 2.0)

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
echo "output: $OUT"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#WINDOWS[@]} * ${#LAMBDAS[@]} ))
echo "cells: $TOTAL  (${#WINDOWS[@]} windows x ${#LAMBDAS[@]} lambdas x 1 seed)"

mkdir -p "$OUT"
FS=$(df --output=target "$OUT" | tail -1)
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
echo "disk: $FS at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: $FS over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for win in "${WINDOWS[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    i=$((i+1))
    start=${win%,*}; end=${win#*,}
    lam_tag=$(printf "%03d" "$(echo "$lam * 100" | bc | cut -d. -f1)")
    name="teacher_residual_const_lam${lam_tag}_w${start}-${end}"
    img="$OUT/$PAIR/seed_$SEED/$name/$name.png"
    if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
    echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) window $start-$end lambda $lam"
    if $PY scripts/interaction_term_window.py \
         --pair "$PAIR" --seed "$SEED" --window "$win" --lambda "$lam" \
         --exp-name "$EXP" >/dev/null 2>&1; then
      done_n=$((done_n+1))
    else
      echo "[$(date -u +%H:%M:%S)] FAILED: window $start-$end lambda $lam" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "generated $done_n, skipped $skip_n, failed $fail_n"
