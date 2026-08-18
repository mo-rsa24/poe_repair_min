#!/usr/bin/env bash
# Does the SHARED part of the correction compose on its own?
#
# 8 pairs x 4 seeds x 4 doses x 1 row = 128 images, comparable to every other
# row of F2 (32 cells per dose).
#
# What is injected: at each step, the mean r_t over the OTHER cached seeds of
# the same pair, leaving the target seed out, norm-matched like every control.
# This is the part of the correction that every run of a pair agrees on,
# distilled and handed over on its own.
#
# Why it is worth 75 minutes. Measured on cat x dog's 17 cached runs, the mean
# of the other 16 runs explains 19.6% of a run's correction energy over steps
# 0 to 2, 0.57% over steps 3 to 9, and 0.01% after step 20 (random floor
# 0.00%). So a real shared component exists and it sits in the first few steps.
# F4 measured that the correction only works when it arrives in the first ten
# steps. The shared part and the load-bearing window are the same window, so
# the obvious question is whether the shared part alone is enough.
#
# Readings declared before this ran, so no outcome is negotiable after:
#   composes clearly above the wrong_seed row (which sits at 3-9%)
#       -> a pair-level correction exists that needs no per-run computation.
#          D2's claim narrows to "the whole correction is not reusable"; the
#          early shared part is. That is a shippable artefact and a new
#          baseline the adapter must beat.
#   sits at the wrong_seed floor
#       -> the shared component is real but not sufficient: composition needs
#          the state-specific 80%. D2's claim stands at full strength and the
#          adapter's job is confirmed to be per-run computation.
#   anything in between is a partial: report the rate, do not round it to
#          either story.
#
#   bash scripts/mechanism_study/run_dose_sweep_mean_others.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_dose_sweep_mean_others.sh
#
# ~35s per cell, so about 75 minutes. Resumable: a cell whose image exists is
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
ROW=mean_others

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#PAIRS[@]} * ${#SEEDS[@]} * ${#LAMBDAS[@]} ))
echo "cells: $TOTAL  (${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds x ${#LAMBDAS[@]} doses)"

USE=$(df --output=pcent /home-mscluster | tail -1 | tr -dc '0-9')
echo "disk: /home-mscluster at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: /home-mscluster over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      lamtag=$(printf "lam%03d" "$($PY -c "print(int(round(float('$lam')*100)))")")
      i=$((i+1))
      name="teacher_residual_const_${lamtag}_${ROW}"
      img="$OUT/$pair/seed_$seed/$name/$name.png"
      if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
      echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $pair seed $seed lam $lam $ROW"
      if $PY scripts/interaction_term_inject.py \
           --pair "$pair" --seed "$seed" --lambda "$lam" --row "$ROW" \
           --exp-name "$EXP" >/dev/null 2>&1; then
        done_n=$((done_n+1))
      else
        echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed lam $lam" >&2
        fail_n=$((fail_n+1))
      fi
    done
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "generated $done_n, skipped $skip_n, failed $fail_n"
echo
echo "=== scoring ==="
$PY scripts/plot_dose_curves.py --root "$OUT"
