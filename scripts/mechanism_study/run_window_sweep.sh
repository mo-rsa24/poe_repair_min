#!/usr/bin/env bash
# The timing sweep: when in the denoising run is the correction needed?
#
# A fixed-width window slides across the 50 steps. Inside it the correction is
# injected at full dose; outside it, nothing. Conditioning stays on at every
# step in every run, so the only thing changing across the sweep is WHEN the
# correction acts. That is what separates this from the conditioning-window
# ablation, which switches conditioning itself off outside the window and so
# cannot tell losing-the-correction apart from losing-the-prompt.
#
# 9 windows x 8 pairs x 4 seeds = 288 images. The grid is defined in
# poe_repair/experiments/interaction_term/window_grid.py and read from there, so
# this script cannot drift from what the scorer and the figure expect.
#
#   bash scripts/mechanism_study/run_window_sweep.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_window_sweep.sh
#   SMOKE=1 bash scripts/mechanism_study/run_window_sweep.sh   # 3 windows, 1 cell
#
# ~50s per cell, so about 4 hours for the full grid. Resumable: a cell whose
# image already exists is skipped, so Ctrl-C and re-run continues.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

# Large artifacts go to /datasets, never to /home-mscluster, which filled to
# 100% once and silently killed checkpointing. The sampler reads this env var
# for its output root, so setting it here is what actually moves the files.
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs

EXP=interaction_term/window
OUT=$POE_REPAIR_OUTPUT_ROOT/$EXP/pairs

readarray -t WINDOWS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(f'{a},{b}' for a, b in g.windows()))")
readarray -t PAIRS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(g.PAIRS))")
readarray -t SEEDS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(str(s) for s in g.SEEDS))")

# The smoke run is the same code path on one cell and three windows, one early,
# one over the fork step, one late. If these three do not differ visibly there
# is no point spending four hours on the other 285.
if [ "${SMOKE:-0}" = "1" ]; then
  WINDOWS=(0,10 15,25 40,50)
  PAIRS=(a_cat__x__a_dog)
  SEEDS=(9)
  echo "SMOKE: 3 windows, 1 pair, 1 seed"
fi

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
echo "output: $OUT"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#WINDOWS[@]} * ${#PAIRS[@]} * ${#SEEDS[@]} ))
echo "cells: $TOTAL  (${#WINDOWS[@]} windows x ${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds)"
echo "windows: ${WINDOWS[*]}"

# The guard has to look at the filesystem being written to, not at a filesystem
# named in a plan. Derive it from $OUT so the two cannot disagree.
mkdir -p "$OUT"
FS=$(df --output=target "$OUT" | tail -1)
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
echo "disk: $FS at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: $FS over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for win in "${WINDOWS[@]}"; do
      i=$((i+1))
      start=${win%,*}; end=${win#*,}
      name="teacher_residual_const_lam100_w${start}-${end}"
      img="$OUT/$pair/seed_$seed/$name/$name.png"
      if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
      echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $pair seed $seed window $start-$end"
      if $PY scripts/interaction_term_window.py \
           --pair "$pair" --seed "$seed" --window "$win" \
           --exp-name "$EXP" >/dev/null 2>&1; then
        done_n=$((done_n+1))
      else
        echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed window $start-$end" >&2
        fail_n=$((fail_n+1))
      fi
    done
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "generated $done_n, skipped $skip_n, failed $fail_n"
echo
echo "=== scoring ==="
$PY scripts/plot_window_curves.py --root "$OUT"
