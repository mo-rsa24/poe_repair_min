#!/bin/bash
#SBATCH --job-name=experiment_b_rank
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster106
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_b_rank_slurm_%j.log
#
# Experiment B (01-showcase-the-trained-lora, plan 09): fresh rank 16 and
# rank 32 runs, 100k steps each, alpha = rank, batch size 1 — matching
# phase1_r8_100k's recipe exactly (rank is the only axis that differs, per
# throughput.md's finding that batch size neither speeds this up nor should
# change, to keep the length/capacity grid single-axis-clean).
#
# One sbatch job reserving the whole node (biggpu allocates whole nodes, not
# per-GPU), running both ranks as independent background processes pinned to
# their own device — not one rank split across both, which would need
# multi-GPU code this trainer doesn't have and would itself change the batch
# size axis.
set -uo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"

export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase

# --- Guards (execution-protocol.md, mandatory) -----------------------------
DISK_PCT=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9')
if [ "$DISK_PCT" -ge 90 ]; then
  echo "ABORT: /datasets at ${DISK_PCT}% (>=90% guard)" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ABORT: co3_bw python not found at $PY" >&2
  exit 1
fi
for DEV in 0 1; do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$DEV")
  if [[ "$UTIL" == *"N/A"* || "$UTIL" == *"reset"* ]]; then
    echo "ABORT: device $DEV reads '$UTIL' for utilization — hardware fault (poe-launch-002)" >&2
    exit 1
  fi
done
echo "[$(date -u +%H:%M:%S)] guards passed on $(hostname), /datasets=${DISK_PCT}%"

run_rank() {
  local DEVICE="$1"
  local RANK="$2"
  local ALPHA="$3"
  local RUNID="phase1_r${RANK}_100k"
  CUDA_VISIBLE_DEVICES="$DEVICE" $PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
      --pair-pool      "$SCOPE/pair_pool.yaml" \
      --pair-prompts   "$SCOPE/pair_prompts.yaml" \
      --seed-pool-path "$SCOPE/seed_pool.yaml" \
      --cache-root     "$POE_REPAIR_TRAINING_CACHE" \
      --lora-rank "$RANK" --lora-alpha "$ALPHA" --lr 1e-4 \
      --total-epochs 2000 --epoch-size 50 \
      --ckpt-every-epochs 100 --log-every-epochs 20 \
      --sample-every-epochs 200 \
      --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2 \
      --sample-train-pairs a_wolf__x__a_husky \
      --sample-heldout-pairs a_cat__x__a_dog \
      --sample-num-inference-steps 50 --sample-thumb 256 \
      --wandb-mode online --wandb-project poe-repair-animals-compose \
      --run-id "$RUNID" \
      --output-root "$OUTROOT" \
      > "$OUTROOT/logs/experiment_b_rank${RANK}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] rank $RANK (device $DEVICE) exited with code $?"
}

run_rank 0 16 16 &
PID16=$!
run_rank 1 32 32 &
PID32=$!

echo "[$(date -u +%H:%M:%S)] launched rank16 (pid $PID16, device 0) and rank32 (pid $PID32, device 1)"
wait "$PID16" "$PID32"
echo "[$(date -u +%H:%M:%S)] both ranks done."
