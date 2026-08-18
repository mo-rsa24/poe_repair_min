#!/usr/bin/env bash
# The two growing-window sweeps: F4g (correct 0..c, then plain PoE) and F4h
# (plain PoE for 0..c, then correct c..50), c stepping through 10, 20, 30, 40,
# 50. F4a asks whether a narrow band suffices and where it has to sit; these
# ask how much of the trajectory, counted from one end, needs the correction
# on. Grid is poe_repair/experiments/interaction_term/window_grid.py's
# prefix_windows()/suffix_windows(), read from there so this script cannot
# drift from what the scorer and the figures expect.
#
# 10 windows (5 prefix + 5 suffix) x 8 pairs x 4 seeds = 320 cells, but the
# c=10 prefix window is (0,10), the same window F4a's own leftmost cell
# already ran, so those 32 images are skipped rather than resampled: 288 new
# cells. ~50s per cell, so about 4 hours.
#
#   bash scripts/mechanism_study/run_growing_window_sweep.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_growing_window_sweep.sh
#   SMOKE=1 bash scripts/mechanism_study/run_growing_window_sweep.sh   # 2 windows, 1 cell
#
# Resumable: a cell whose image already exists is skipped, so Ctrl-C and
# re-run continues. Writes to the same output root as run_window_sweep.sh, so
# the two sweeps' images live side by side and one manifest can see both.
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

readarray -t PREFIX < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(f'{a},{b}' for a, b in g.prefix_windows()))")
readarray -t SUFFIX < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(f'{a},{b}' for a, b in g.suffix_windows()))")
readarray -t PAIRS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(g.PAIRS))")
readarray -t SEEDS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(str(s) for s in g.SEEDS))")

# The smoke run is the same code path on one cell, one prefix and one suffix
# window. If these two don't differ visibly there is no point spending four
# hours on the other 318.
if [ "${SMOKE:-0}" = "1" ]; then
  PREFIX=(0,20)
  SUFFIX=(30,50)
  PAIRS=(a_cat__x__a_dog)
  SEEDS=(9)
  echo "SMOKE: 1 prefix window, 1 suffix window, 1 pair, 1 seed"
fi

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
echo "output: $OUT"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( (${#PREFIX[@]} + ${#SUFFIX[@]}) * ${#PAIRS[@]} * ${#SEEDS[@]} ))
echo "cells: $TOTAL  ((${#PREFIX[@]} prefix + ${#SUFFIX[@]} suffix) x ${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds)"
echo "prefix windows: ${PREFIX[*]}"
echo "suffix windows: ${SUFFIX[*]}"

# The guard has to look at the filesystem being written to, not at a
# filesystem named in a plan. Derive it from $OUT so the two cannot disagree.
mkdir -p "$OUT"
FS=$(df --output=target "$OUT" | tail -1)
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
echo "disk: $FS at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: $FS over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for direction in prefix suffix; do
  if [ "$direction" = "prefix" ]; then WINDOWS=("${PREFIX[@]}"); else WINDOWS=("${SUFFIX[@]}"); fi
  for pair in "${PAIRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for win in "${WINDOWS[@]}"; do
        i=$((i+1))
        start=${win%,*}; end=${win#*,}
        name="teacher_residual_const_lam100_w${start}-${end}"
        img="$OUT/$pair/seed_$seed/$name/$name.png"
        if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
        echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $direction $pair seed $seed window $start-$end"
        if $PY scripts/interaction_term_window.py \
             --pair "$pair" --seed "$seed" --window "$win" \
             --exp-name "$EXP" >/dev/null 2>&1; then
          done_n=$((done_n+1))
        else
          echo "[$(date -u +%H:%M:%S)] FAILED: $direction $pair seed $seed window $start-$end" >&2
          fail_n=$((fail_n+1))
        fi
      done
    done
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "generated $done_n, skipped $skip_n, failed $fail_n"
echo
echo "=== scoring ==="
$PY scripts/plot_growing_window_curves.py --root "$OUT"
