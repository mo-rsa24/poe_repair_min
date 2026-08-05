#!/bin/bash
#SBATCH --job-name=value_probe_sweep
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=results/mechanism_study/slurm_value_probe_%j.log
# Plan 02 task 3: the 64-cell mechanism re-probe.
#
# 8 held-out pairs x seeds 9-16, adapter OFF vs ON at matched x_t, three
# denoising steps each. Decides Goal 6: does the value-channel finding
# replicate beyond the single seed-9 cat/dog cell it was found on?
#
# Two things this depends on, both fixed 2026-08-05 before submission:
#   - the token map is derived per pair, not hardcoded to index 2. Three pool
#     pairs split (wal|rus, chim|pan|zee, porpoise), and a_seal__x__a_walrus
#     is in this sweep.
#   - the verdict compares the PATTERN change, not ||on-off||/||off||. The raw
#     norm is dominated by a ~25% uniform dimming of the weight maps and gives
#     the opposite answer. See docs/evidence/mechanism-reprobe/measure-fairness.md
#
# Resumable: a cell whose manifest already exists is skipped, so a timeout or
# preemption can be re-submitted without redoing finished work.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
mkdir -p results/mechanism_study

CKPT=$REPO/outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt
OUT=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe
STEPS=10,25,40

# The 8 held-out pairs from the experiment's own pool: 6 transfer tests, the
# known-failure reference, and the dissimilar control. Named explicitly rather
# than scanned, because the cache directory holds other experiments' pairs.
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

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 2; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) checkpoint=$CKPT"
echo "[$(date -u +%H:%M:%S)] ${#PAIRS[@]} pairs x ${#SEEDS[@]} seeds = $(( ${#PAIRS[@]} * ${#SEEDS[@]} )) cells, steps $STEPS"

done_n=0; skip_n=0; fail_n=0
for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cell="$OUT/$pair/seed_$seed"
    if [ -f "$cell/value_probe_manifest.json" ]; then
      skip_n=$((skip_n+1)); continue
    fi
    echo "[$(date -u +%H:%M:%S)] --- $pair seed $seed ---"
    if $PY -m poe_repair.experiments.mechanism_study.value_probe \
         --checkpoint "$CKPT" --pair-slug "$pair" --seed "$seed" \
         --steps "$STEPS" --out-root "$cell"; then
      done_n=$((done_n+1))
    else
      # One bad cell must not lose the other 63. Named, counted, continued.
      echo "[$(date -u +%H:%M:%S)] FAILED: $pair seed $seed" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo "[$(date -u +%H:%M:%S)] captured $done_n, skipped $skip_n already-done, failed $fail_n"
echo "[$(date -u +%H:%M:%S)] cells on disk: $(ls -d $OUT/*/seed_* 2>/dev/null | wc -l)"

if [ "$fail_n" -gt 0 ]; then
  echo "re-submit to retry the failed cells; finished ones are skipped." >&2
fi

echo "[$(date -u +%H:%M:%S)] DONE capture. Score next:"
echo "  $PY scripts/mechanism_study/reprobe_table.py"
