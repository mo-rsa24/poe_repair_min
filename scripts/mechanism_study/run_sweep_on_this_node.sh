#!/usr/bin/env bash
# Run the 64-cell mechanism re-probe directly on whatever node you are on.
#
# For when you have a VS Code session open on an idle compute node and want to
# use it, rather than queueing behind the biggpu one-job-per-user limit. No
# sbatch, no srun: this is the work itself, run here.
#
#   bash scripts/mechanism_study/run_sweep_on_this_node.sh
#
# Safe to interrupt and re-run: a cell whose manifest exists is skipped, so
# Ctrl-C and restart picks up where it left off. Safe to run twice by accident
# for the same reason.
#
# Runs in the foreground and prints one line per cell. To survive the terminal
# closing, run it under nohup:
#
#   nohup bash scripts/mechanism_study/run_sweep_on_this_node.sh \
#     > results/mechanism_study/sweep_$(hostname).log 2>&1 &
#   tail -f results/mechanism_study/sweep_*.log
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
mkdir -p results/mechanism_study

CKPT=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe
STEPS=10,25,40

PAIRS=(
  a_leopard__x__a_jaguar
  a_frog__x__a_toad
  an_eagle__x__a_hawk
  a_seal__x__a_walrus
  a_goose__x__a_swan
  a_cow__x__a_buffalo
  a_cat__x__a_dog
  an_elephant__x__a_penguin
)
SEEDS=(9 10 11 12 13 14 15 16)

echo "node   : $(hostname)"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Fail early and clearly rather than 64 times.
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 2; }
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ERROR: no CUDA device visible on $(hostname). This needs a GPU." >&2
  exit 3
}
$PY -c "import torch; print(f'gpu    : {torch.cuda.get_device_name(0)}')"

# Someone else's process on the same card is the one real hazard of running
# outside Slurm. Say so up front rather than dying on OOM at cell 40.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
echo "gpu mem: ${USED} MiB already in use"
if [ "${USED:-0}" -gt 4000 ]; then
  echo "WARNING: something else is using this GPU. Two SDXL processes on one" >&2
  echo "         card will OOM. Check 'nvidia-smi' before continuing." >&2
  echo "         Continuing in 15s; Ctrl-C to stop." >&2
  sleep 15
fi

TOTAL=$(( ${#PAIRS[@]} * ${#SEEDS[@]} ))
echo "cells  : $TOTAL  (${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds, steps $STEPS)"
echo

done_n=0; skip_n=0; fail_n=0; i=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    cell="$OUT/$pair/seed_$seed"
    if [ -f "$cell/value_probe_manifest.json" ]; then
      skip_n=$((skip_n+1)); continue
    fi
    echo "[$(date -u +%H:%M:%S)] ($i/$TOTAL) $pair seed $seed"
    if $PY -m poe_repair.experiments.mechanism_study.value_probe \
         --checkpoint "$CKPT" --pair-slug "$pair" --seed "$seed" \
         --steps "$STEPS" --out-root "$cell"; then
      done_n=$((done_n+1))
    else
      echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed (continuing)" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "captured $done_n, skipped $skip_n already-done, failed $fail_n"
echo "cells on disk: $(ls -d $OUT/*/seed_* 2>/dev/null | wc -l) / $TOTAL"
[ "$fail_n" -gt 0 ] && echo "re-run this script to retry the failures; finished cells are skipped."

echo
echo "=== scoring ==="
$PY scripts/mechanism_study/reprobe_table.py
