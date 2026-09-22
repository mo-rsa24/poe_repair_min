#!/bin/bash
#SBATCH --job-name=figwin
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster124,mscluster129
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates/logs/%x-%j.err
#
# One adapter, switched on for the first N of the fifty steps and off after, for several N in one
# process (the model loads once). Tiles land beside the joint-prompt and plain-product tiles of
# the same cell as ours_<TAG>_wNN.png.
#
#   sbatch --export=ALL,PAIRS="a tiger|a parrot;a goat|a cat",SEEDS="9 10 11 12",WINDOWS="10 20 50",CKPT=<pt>,TAG=run9 figure_windows.sbatch
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SCRIPT="$OUTROOT/figure_candidates.py"
: "${PAIRS:?set PAIRS}" "${SEEDS:?set SEEDS}" "${WINDOWS:?set WINDOWS}" "${CKPT:?set CKPT}" "${TAG:?set TAG}"

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
USE=$(df --output=pcent "$OUTROOT" | tail -1 | tr -dc '0-9')
[ "${USE:-100}" -ge 90 ] && { echo "ABORT: $OUTROOT is ${USE}% full" >&2; exit 3; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }

IFS=';' read -r -a P <<< "$PAIRS"
export POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$OUTROOT"
cd "$REPO"
"$PY" "$SCRIPT" --column ours --pairs "${P[@]}" --seeds $SEEDS --checkpoint "$CKPT" --window $WINDOWS --tag "$TAG"
echo "=== done $(date -Is)"
