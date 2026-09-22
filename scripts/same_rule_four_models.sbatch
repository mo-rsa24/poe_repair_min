#!/bin/bash
#SBATCH --job-name=samerule
#SBATCH --partition=bigbatch
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster72,mscluster73,mscluster74,mscluster76,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models/logs/%x-%j.err
#
# Figure 2 as a grid: three pairs by four models, joint prompt and PoE per cell. One model at a
# time, freed before the next, then the three sheets. The plan is
# plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md.
#
# Slurm does not create the log folder, so make it first:
#   mkdir -p /datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models/logs
#   sbatch --export=ALL,SEEDS=42 scripts/same_rule_four_models.sbatch
#   sbatch --export=ALL,RUN_MODELS=sd35 scripts/same_rule_four_models.sbatch   # one column only
#   sbatch --export=ALL,CHECK_LOAD=1 scripts/same_rule_four_models.sbatch    # load all four, render nothing
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SCRIPT="$REPO/scripts/same_rule_four_models.py"
SEEDS="${SEEDS:-42}"
RUN_MODELS="${RUN_MODELS:-sd14 sd21 sdxl sd35}"
# SD 3.5 medium needs diffusers >= 0.31 and co3 has 0.29.2, so its column puts a folder holding only a
# newer diffusers (and sentencepiece) ahead of co3 on PYTHONPATH. co3 itself is not changed.
SD35_PYLIB=/datasets/mmolefe/poe_repair_min/pylib/sd35

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
mkdir -p "$OUTROOT/logs"
USE=$(df --output=pcent "$OUTROOT" | tail -1 | tr -dc '0-9')
[ "${USE:-100}" -ge 90 ] && { echo "ABORT: $OUTROOT is ${USE}% full" >&2; exit 3; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
"$PY" -c "import torch; assert torch.cuda.is_available()" || { echo "ABORT: torch sees no GPU" >&2; exit 4; }

export SAME_RULE_OUT="$OUTROOT"
# Every model is already in ~/.cache/huggingface; online checks from a compute node stall for minutes.
export HF_HUB_OFFLINE=1
cd "$REPO"
if [ "${CHECK_LOAD:-0}" = 1 ]; then
  "$PY" "$SCRIPT" --check-load; rc=$?; echo "=== done $(date -Is)  rc $rc"; exit $rc
fi
for model in $RUN_MODELS; do
  echo "--- $model  $(date +%H:%M:%S)"
  if [ "$model" = sd35 ]; then
    PYTHONPATH="$SD35_PYLIB${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "import diffusers; print('diffusers', diffusers.__version__)"
    PYTHONPATH="$SD35_PYLIB${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$SCRIPT" --model sd35 --seeds $SEEDS || echo "MODEL sd35 FAILED"
  else
    "$PY" "$SCRIPT" --model "$model" --seeds $SEEDS || echo "MODEL $model FAILED"
  fi
done

# The sheets are small, so they are copied into the checkout's outputs/ where cluster.sh pull reaches them.
SHEETS="$REPO/outputs/same_rule_four_models"
mkdir -p "$SHEETS"
for s in $SEEDS; do
  "$PY" "$SCRIPT" --sheet --seed "$s" --out "$OUTROOT/sheets/seed-$s.pdf" && cp "$OUTROOT/sheets/seed-$s.pdf" "$SHEETS/"
done
find "$OUTROOT" -name '*.png' | wc -l | sed 's/^/pngs: /'
echo "=== done $(date -Is)"
