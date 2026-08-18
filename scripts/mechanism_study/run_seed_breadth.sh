#!/usr/bin/env bash
# Is "a good seed" a property of the noise, or of the noise AND the prompt?
#
# The timing grid runs 4 seeds, and at the earliest window the pairs disagree
# about which seeds work. With 4 binary outcomes per pair, the correlation
# between two pairs' seed patterns is computed from 4 points and is close to
# meaningless. This widens that one column to 12 seeds so the question can
# actually be asked.
#
# 8 pairs x 12 seeds x 1 window (0,10) = 96 cells, at full dose, the pair's own
# r_t, injected only inside the window. Seeds 9-12 already exist in the timing
# grid, but they are REGENERATED here rather than borrowed, so every cell in
# this table comes from one run of one script and no row mixes provenance.
#
# Why a separate experiment name. The timing grid's seeds are pinned in
# poe_repair/experiments/interaction_term/window_grid.py precisely so a figure
# cannot plot a grid nobody ran, and the dose sweep has a documented fault from
# exactly this: stray cells at other seeds under a shared root silently changed
# what the curve averaged over. So this writes to interaction_term/window_seeds
# and leaves the 288-cell timing grid untouched.
#
# Readings declared before this ran:
#   per-pair rates over 12 seeds separate by more than ~0.3
#       -> pair difficulty is real and measurable, and F4f's per-pair
#          differences are not an artefact of having only 4 seeds.
#   seed outcomes correlate across pairs, median |r| >= 0.3
#       -> some seeds are globally easier, so seed quality is a property of the
#          initial noise on its own and could in principle be read off x_T.
#   median |r| below 0.15
#       -> seed quality is an interaction between the noise and the prompt.
#          No per-seed predictor can exist that ignores the pair, and any noise
#          map has to be drawn one pair at a time.
#   anything between 0.15 and 0.3 is reported as inconclusive at this n, not
#          rounded to either story. With 12 binary points, |r| below about 0.58
#          is not individually significant, so the claim rests on the median
#          across the 28 pair-pairings rather than on any single one.
#
#   bash scripts/mechanism_study/run_seed_breadth.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_seed_breadth.sh
#
# ~50s per cell, so about 80 minutes. Resumable: a cell whose image exists is
# skipped, so Ctrl-C and re-run continues.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

# Large artifacts to /datasets, never /home-mscluster, which filled to 100%
# once and silently killed checkpointing.
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs

EXP=interaction_term/window_seeds
OUT=$POE_REPAIR_OUTPUT_ROOT/$EXP/pairs
WINDOW="0,10"

readarray -t PAIRS < <($PY -c "
from poe_repair.experiments.interaction_term import window_grid as g
print('\n'.join(g.PAIRS))")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12)

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#PAIRS[@]} * ${#SEEDS[@]} ))
echo "cells: $TOTAL  (${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds, window $WINDOW)"

USE=$(df --output=pcent /datasets/mmolefe | tail -1 | tr -dc '0-9')
echo "disk: /datasets at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: /datasets over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    if $PY scripts/interaction_term_window.py \
         --pair "$pair" --seed "$seed" --window "$WINDOW" \
         --exp-name "$EXP" >/dev/null 2>&1; then
      done_n=$((done_n+1))
      echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $pair seed $seed ok"
    else
      echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "generated $done_n, skipped $skip_n, failed $fail_n"
echo
echo "=== scoring ==="
$PY scripts/score_seed_breadth.py --root "$OUT"
