#!/usr/bin/env bash
# Both generation grids, in one detached run.
#
#   1. window   the 8-pair sliding-window sweep that feeds F4, 288 cells
#   2. cross    the conditioning x correction cross plus the dense stride-1
#               timing strip, 3 pairs, 702 cells
#
# The window grid is regenerated whole rather than resumed. The 77 cells left by
# the earlier single-cell runner were sampled one at a time, and the same UNet
# returns slightly different numbers at different batch shapes, so finishing the
# sweep at batch 8 would leave the curve made of two populations. Regenerating
# is also faster than resuming the old way: 18s a cell against 31s.
#
#   setsid nohup bash scripts/mechanism_study/run_all_grids.sh > log 2>&1 &
set -uo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) gpu $CUDA_VISIBLE_DEVICES"

echo
echo "############ 1/2  window grid (F4, 8 pairs, 288 cells) ############"
$PY scripts/run_cross_sweep.py --grid window --overwrite
w=$?
echo "window grid exit $w"

echo
echo "############ 2/2  cross grid (3 pairs, 702 cells) ############"
$PY scripts/run_cross_sweep.py --grid cross
c=$?
echo "cross grid exit $c"

echo
echo "############ scoring the window grid ############"
$PY scripts/plot_window_curves.py 2>&1 | grep -v "deformable attention"

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $(( w != 0 || c != 0 ))
