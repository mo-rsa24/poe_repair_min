#!/bin/bash
# Batch-size comparison: batch=1 then batch=4, same device, same rank, same
# step count (1000) — the only axis that differs is train-batch-size, so the
# step-time delta is attributable to batching, not to a hardware or rank
# change. Sequential on purpose: two configs sharing one device.
set -euo pipefail

DEVICE="${1:?usage: timing_batch_compare.sh <device> <rank> <alpha>}"
RANK="${2:?rank required}"
ALPHA="${3:?alpha required}"
DIR=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase

echo "[$(date -u +%H:%M:%S)] === batch=1 baseline, rank=$RANK ==="
bash "$DIR/timing_smoke.sh" "$DEVICE" "$RANK" "$ALPHA" 20 1

echo "[$(date -u +%H:%M:%S)] === batch=4, rank=$RANK ==="
bash "$DIR/timing_smoke.sh" "$DEVICE" "$RANK" "$ALPHA" 20 4

echo "[$(date -u +%H:%M:%S)] DONE batch_compare rank=$RANK."
