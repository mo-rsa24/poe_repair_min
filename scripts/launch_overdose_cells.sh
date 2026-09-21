#!/usr/bin/env bash
# The four over-correction columns the dose strip is missing: lambda 1.25, 1.5,
# 1.75 and 2.0 on each of the four control arms, cat x dog seed 9.
#
#   bash scripts/launch_overdose_cells.sh <cuda index>
#
# The existing strip stops at lambda 1.0 (dose_curves.json lambdas), so nothing
# in it says what happens when the correction is pushed past the amount that
# reproduces the joint prompt. These 16 cells are the pictures for that; the
# compose RATE cannot follow without 8 pairs x 4 seeds, which is 512 samples,
# not 16.
#
# Writes into the same tree the strip already reads, interaction_term/dose, under
# tags teacher_residual_const_lam{125,150,175,200}[_<arm>]. Every path absolute;
# the script cd's into the repo itself.
set -euo pipefail
GPU=${1:?cuda index}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
# co3 for the RTX 8000 / A6000 nodes. co3_bw is the Blackwell env, but on
# mscluster111 (driver 595.71.05) it reports cuda available = False and torch
# silently runs on CPU: one cell took 39 minutes there on 2026-09-10. Check the
# torch guard below rather than trusting nvidia-smi.
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
# The sampler's output root is POE_REPAIR_OUTPUT_ROOT (poe_repair/config.py), and
# it defaults to <repo>/outputs. Unset, a run writes GBs to /home-mscluster,
# which is the failure CLAUDE.md records. Set it explicitly.
export POE_REPAIR_OUTPUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs
DOSE=$POE_REPAIR_OUTPUT_ROOT/interaction_term/dose
PAIR=a_cat__x__a_dog
SEED=9
# Overridable so a probe at higher strength needs no second copy of this file:
#   LAMBDAS="3 4 6 8" ARMS=oracle sbatch scripts/overdose_cells.sbatch
read -r -a LAMBDAS <<< "${LAMBDAS:-1.25 1.5 1.75 2.0}"
read -r -a ARMS <<< "${ARMS:-oracle wrong_pair wrong_seed wrong_step}"
cd "$REPO"

# Guards, per environment/hpc/execution-protocol.md step 4.
used_pct=$(df --output=pcent "$DOSE" | tail -1 | tr -dc '0-9'); echo "disk: $DOSE at ${used_pct}% used"
[ "$used_pct" -lt 90 ] || { echo "ERROR: /datasets over 90%"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: python missing at $PY"; exit 2; }
CUDA_VISIBLE_DEVICES=$GPU timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || { echo "ERROR: no GPU"; exit 2; }
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | head -1)
if [ "${mem:-0}" -lt 100 ]; then :;
elif [ "${mem:-0}" -lt 4096 ] && [ "${util:-100}" -eq 0 ]; then :;
else echo "ERROR: device $GPU has ${mem} MiB in use at ${util}% utilisation, refusing to share"; exit 2; fi
# nvidia-smi seeing a device does not mean torch can use it. Without this the
# run completes on CPU and looks fine in the log.
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('ERROR: torch reports no CUDA on this node; refusing to run on CPU')
print('torch sees', torch.cuda.get_device_name(0))" || exit 2
echo "node=$(hostname) device=$GPU pid=$$ pair=$PAIR seed=$SEED cells=$(( ${#LAMBDAS[@]} * ${#ARMS[@]} )) out=$POE_REPAIR_OUTPUT_ROOT"

export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=$GPU
n=0
for lam in "${LAMBDAS[@]}"; do
  for arm in "${ARMS[@]}"; do
    n=$((n+1))
    echo "=== cell $n/16  lambda=$lam  arm=$arm  $(date +%H:%M:%S) ==="
    "$PY" scripts/interaction_term_inject.py \
        --pair "$PAIR" --seed "$SEED" --lambda "$lam" --row "$arm"
  done
done
echo "=== done $n cells $(date +%H:%M:%S) ==="
