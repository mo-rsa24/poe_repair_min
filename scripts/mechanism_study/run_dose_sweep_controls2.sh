#!/usr/bin/env bash
# The two rows F2's redesign adds to the dose sweep: wrong_seed and wrong_step.
#
# 8 held-out pairs x 4 seeds x 4 doses x 2 rows = 256 images. Lambda 0 is not
# run: nothing is injected there, so the oracle's lambda-0 image already serves
# every row (plot_dose_curves.py shares it).
#
#   wrong_seed  the same pair's r_t, cached from the run at seed+4, injected
#               into this seed's run. Norm-matched like every control.
#   wrong_step  this cell's own cached r_t with the step order deranged by a
#               fixed permutation (no step keeps its own vector).
#
# Readings declared before this ran, so no outcome is negotiable after:
#   wrong_seed  composes -> r_t is a function of prompt + noise level, not of
#               the particular run, which is what makes it learnable;
#               stays fused -> the adapter story owes an explanation.
#   wrong_step  stays fused -> the correction's content is needed at the right
#               time, the injection-side twin of F4's window sweep;
#               composes -> only the total amount injected matters.
# Neither is a floor-expected control like random / wrong_pair; the causal
# verdict in plot_dose_curves.py still reads only those two.
#
#   bash scripts/mechanism_study/run_dose_sweep_controls2.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_dose_sweep_controls2.sh
#
# ~35s per cell, so about 2.5 hours. Resumable: a cell whose image exists is
# skipped, so Ctrl-C and re-run continues.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

EXP=interaction_term/dose
OUT=$REPO/outputs/$EXP/pairs

PAIRS=(
  a_leopard__x__a_jaguar a_frog__x__a_toad an_eagle__x__a_hawk
  a_seal__x__a_walrus a_goose__x__a_swan a_cow__x__a_buffalo
  a_cat__x__a_dog an_elephant__x__a_penguin
)
SEEDS=(9 10 11 12)
LAMBDAS=(0.25 0.5 0.75 1.0)
ROWS=(wrong_seed wrong_step)

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#PAIRS[@]} * ${#SEEDS[@]} * ${#LAMBDAS[@]} * ${#ROWS[@]} ))
echo "cells: $TOTAL  (${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds x ${#LAMBDAS[@]} doses x ${#ROWS[@]} rows)"

# Disk guard on the filesystem this actually writes to (outputs/ is on home).
USE=$(df --output=pcent /home-mscluster | tail -1 | tr -dc '0-9')
echo "disk: /home-mscluster at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: /home-mscluster over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      lamtag=$(printf "lam%03d" "$($PY -c "print(int(round(float('$lam')*100)))")")
      for row in "${ROWS[@]}"; do
        i=$((i+1))
        name="teacher_residual_const_${lamtag}_${row}"
        img="$OUT/$pair/seed_$seed/$name/$name.png"
        if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
        echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $pair seed $seed lam $lam $row"
        if $PY scripts/interaction_term_inject.py \
             --pair "$pair" --seed "$seed" --lambda "$lam" --row "$row" \
             --exp-name "$EXP" >/dev/null 2>&1; then
          done_n=$((done_n+1))
        else
          echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed lam $lam $row" >&2
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
$PY scripts/plot_dose_curves.py --root "$OUT"
