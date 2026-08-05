#!/usr/bin/env bash
# Generate the PoE and Mono trajectories the fork curve needs.
#
# Plan 05 task 2b. The training cache walks ONE path per cell (the PoE path),
# so d(t) = ||x_t_PoE - x_t_Mono|| cannot be computed from it: after step 1 the
# two paths visit different states and only one was recorded.
#
# This samples both, from the same pinned init, for every pool pair at one
# seed. ~38 runs at ~40s = about 25 minutes on a free A6000.
#
#   bash scripts/mechanism_study/generate_fork_paths.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/generate_fork_paths.sh
#
# Resumable: a run whose latent_trajectory.pt exists is skipped.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
mkdir -p results/mechanism_study

OUT=$REPO/outputs/interaction_term/dose/pairs
EXP=interaction_term/dose

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device visible." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

# The experiment's own pairs, and the first cached seed for each. Read from the
# pool file rather than scanned: the cache directory holds other experiments.
mapfile -t CELLS < <($PY - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from poe_repair.experiments.interaction_term.cache import CACHE_ROOT
from poe_repair.experiments.interaction_term.pool import load_pool
pool = load_pool()
for slug in pool.train + pool.heldout(roles=("transfer", "reference", "control")):
    for split in ("train", "heldout"):
        d = CACHE_ROOT / split / slug
        if not d.is_dir():
            continue
        seeds = sorted((x for x in d.glob("seed_*")
                        if len(list((x / "residuals").glob("step_*.pt"))) >= 2),
                       key=lambda p: int(p.name.split("_")[1]))
        if seeds:
            print(f"{slug} {int(seeds[0].name.split('_')[1])}")
            break
PYEOF
)

echo "cells: ${#CELLS[@]} pairs x 2 doses = $(( ${#CELLS[@]} * 2 )) runs"
echo

done_n=0; skip_n=0; fail_n=0
for cell in "${CELLS[@]}"; do
  read -r pair seed <<< "$cell"
  for lam in 0.0 1.0; do
    tag=$(printf "lam%03d" "$(python3 -c "print(int(round(float('$lam')*100)))")")
    traj="$OUT/$pair/seed_$seed/teacher_residual_const_$tag/latent_trajectory.pt"
    if [ -f "$traj" ]; then skip_n=$((skip_n+1)); continue; fi
    echo "[$(date -u +%H:%M:%S)] $pair seed $seed lambda=$lam"
    if $PY scripts/interaction_term_inject.py \
         --pair "$pair" --seed "$seed" --lambda "$lam" --exp-name "$EXP" >/dev/null; then
      done_n=$((done_n+1))
    else
      echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed lambda=$lam" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo
echo "generated $done_n, skipped $skip_n, failed $fail_n"
echo "=== fork curve ==="
$PY scripts/fork_curve.py --root "$OUT"
