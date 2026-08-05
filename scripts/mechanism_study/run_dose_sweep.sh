#!/usr/bin/env bash
# Plan 03's dose sweep: the causal headline.
#
# 8 held-out pairs x 4 seeds x 5 doses x 3 rows = 480 images.
#
# The three rows are the point. A rising oracle curve on its own proves
# nothing: it could mean "injecting anything of that magnitude helps".
#   oracle      the pair's own r_t
#   random      a norm-matched random vector
#   wrong_pair  another pair's r_t, norm-matched, token-disjoint donor
# Support means the oracle rises while both controls stay flat.
#
#   bash scripts/mechanism_study/run_dose_sweep.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/mechanism_study/run_dose_sweep.sh
#
# ~35s per cell, so about 4.5 hours. Resumable: a cell whose image exists is
# skipped, so Ctrl-C and re-run continues.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
mkdir -p results/mechanism_study

EXP=interaction_term/dose
OUT=$REPO/outputs/$EXP/pairs

PAIRS=(
  a_leopard__x__a_jaguar a_frog__x__a_toad an_eagle__x__a_hawk
  a_seal__x__a_walrus a_goose__x__a_swan a_cow__x__a_buffalo
  a_cat__x__a_dog an_elephant__x__a_penguin
)
SEEDS=(9 10 11 12)
LAMBDAS=(0.0 0.25 0.5 0.75 1.0)
ROWS=(oracle random wrong_pair)

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-all}"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device." >&2; exit 3; }
$PY -c "import torch; print(f'device: {torch.cuda.get_device_name(0)}')"

TOTAL=$(( ${#PAIRS[@]} * ${#SEEDS[@]} * ${#LAMBDAS[@]} * ${#ROWS[@]} ))
echo "cells: $TOTAL  (${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds x ${#LAMBDAS[@]} doses x ${#ROWS[@]} rows)"

# Disk guard: the plan says abort at 90% full. 480 images plus trajectories.
USE=$(df --output=pcent /datasets/mmolefe | tail -1 | tr -dc '0-9')
echo "disk: /datasets at ${USE}% used"
[ "${USE:-0}" -ge 90 ] && { echo "ERROR: /datasets over 90% full, aborting." >&2; exit 4; }
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      lamtag=$(printf "lam%03d" "$($PY -c "print(int(round(float('$lam')*100)))")")
      for row in "${ROWS[@]}"; do
        i=$((i+1))
        name="teacher_residual_const_${lamtag}"
        [ "$row" != "oracle" ] && name="${name}_${row}"
        img="$OUT/$pair/seed_$seed/$name/$name.png"
        if [ -f "$img" ]; then skip_n=$((skip_n+1)); continue; fi
        # lambda=0 is the same image for all three rows (nothing is injected),
        # so only the oracle row runs it. Saves 64 redundant samples.
        if [ "$lam" = "0.0" ] && [ "$row" != "oracle" ]; then
          skip_n=$((skip_n+1)); continue
        fi
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
