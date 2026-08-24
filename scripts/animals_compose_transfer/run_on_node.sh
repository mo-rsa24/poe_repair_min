#!/bin/bash
# Run the Phase-1 does-the-fix-reach-unseen-pairs prerequisites on whatever node you are on.
# Run this from an interactive shell ON the compute node (e.g. mscluster110), NOT the
# login node. It first proves the GPU actually runs, then does the work.
#
#   bash scripts/animals_compose_transfer/run_on_node.sh
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python  # Blackwell (cu128) clone of co3
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache

echo "node   : $(hostname)"
echo "gpu    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- make-or-break: can this torch actually run a kernel on this GPU? ---
echo "=== CUDA compute check (co3 env) ==="
if ! $PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); print('CUDA OK, checksum=', round((x@x).sum().item(),1))"; then
  echo ""
  echo "!! co3 (torch cu118, sm<=90) cannot run on this GPU."
  echo "!! If this node is a Blackwell (sm_120), build a cu128 env first (see notes at bottom of chat),"
  echo "!! or fall back to the local RTX 3090 / a compatible biggpu node (108)."
  exit 1
fi

echo "=== A) fail-rate: 19 pairs x 8 seeds vanilla PoE + instance-count scorer ==="
$PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.fail_rate

echo "=== B) training caches: 11 train pairs x 8 seeds ==="
$PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.build_caches --which train

echo "=== C) held-out-eval caches: 5 pairs x 8 seeds ==="
$PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.build_caches --which heldout_eval

echo "=== DONE. fail-rate table: ==="
cat outputs/animals_compose_transfer/fail_rate.md
