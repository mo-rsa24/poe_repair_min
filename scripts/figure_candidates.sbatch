#!/bin/bash
#SBATCH --job-name=figcand
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates/logs/%x-%j.err
#
# Joint prompt, plain product and one adapter for a list of pairs and seeds, one column after the
# other so the joint-prompt column runs in a process that never attaches an adapter.
#
#   sbatch --export=ALL,PAIRS="a horse|an owl;a fox|a rabbit",SEEDS="9 10 11 12",CKPT=<pt> figure_candidates.sbatch
#
# PAIRS is ;-separated because each pair already contains a space and a |.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/figure_candidates
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SCRIPT="$OUTROOT/figure_candidates.py"
: "${PAIRS:?set PAIRS}" "${SEEDS:?set SEEDS}" "${CKPT:?set CKPT}"

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
USE=$(df --output=pcent "$OUTROOT" | tail -1 | tr -dc '0-9')
[ "${USE:-100}" -ge 90 ] && { echo "ABORT: $OUTROOT is ${USE}% full" >&2; exit 3; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }

IFS=';' read -r -a P <<< "$PAIRS"
export POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$OUTROOT"
cd "$REPO"
for col in mono poe ours; do
  echo "--- $col  $(date +%H:%M:%S)"
  if [ "$col" = mono ]; then
    "$PY" "$SCRIPT" --column mono --pairs "${P[@]}" --seeds $SEEDS || echo "COLUMN mono FAILED"
  else
    "$PY" "$SCRIPT" --column "$col" --pairs "${P[@]}" --seeds $SEEDS --checkpoint "$CKPT" || echo "COLUMN $col FAILED"
  fi
done
"$PY" "$SCRIPT" --sheet --pairs "${P[@]}" --seeds $SEEDS
echo "=== done $(date -Is)"
