#!/bin/bash
# Scope 01 plan 16 (experiment E): the rank-32 pooled run again, trained on the clean-estimate
# residual instead of the noise residual, everything else copied from experiment_b_rank.sh at
# rank 32 (the run that made phase1_r32_100k). Three flags differ from that run, all named in the
# review file's fairness check:
#   --loss-space x0 --loss-weight-cap 22   the one axis under test: the noise-space MSE weighted
#                                          by (1 - abar_t)/abar_t clipped at 22 (its value at DDIM
#                                          step 10), so the loss is the MSE between clean estimates
#   --kill-halve-after-steps 1e9           same reason as experiment D: the kill rule aborted the
#                                          fresh baseline at 6,500 and could not fire on its resume
#   --total-epochs 800                     40k steps, not 100k: the baseline's best checkpoint is
#                                          30k and the question is asked there and at 40k
# After training, the same script probes the 30k and 40k checkpoints the way figure_r32_030050
# was made, renders the per-step frames of the new adapter, and runs the readout (figures, strips,
# verdict, W&B). STAGE=train|probe|readout|all picks the part; default all.
#
# Shared-device path (environment/hpc/execution-protocol.md): over SSH on a pinned free device of
# a biggpu node, invisible to squeue. Example (every path absolute):
#   ssh mscluster106 'nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase/experiment_e_x0_loss.sh 1 > /datasets/mmolefe/poe_repair_min/outputs/showcase/logs/experiment_e_x0loss.log 2>&1 &'
set -euo pipefail

DEVICE="${1:?usage: experiment_e_x0_loss.sh <cuda_device_index> [python]}"
case "$(hostname -s)" in
  mscluster110|mscluster111|mscluster112) DEFAULT_PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *) DEFAULT_PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
PY="${2:-$DEFAULT_PY}"
STAGE="${STAGE:-all}"
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$DEVICE"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
OUT_ROOT=/datasets/mmolefe/poe_repair_min/outputs/showcase
RUN_ID="phase1_r32_x0loss_40k"
LOSS_CAP=22

echo "[$(date -u +%H:%M:%S)] node=$(hostname) device=$DEVICE stage=$STAGE === experiment_e (${RUN_ID}) ==="

# --- Guards (execution-protocol.md), copied from experiment_d_weight_decay.sh ---------------
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE")
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$DEVICE")
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
  if [ "$USED" -gt 4096 ]; then
    echo "ABORT: device $DEVICE has ${USED}MiB in use (>4096MiB guard), not free" >&2; exit 1
  fi
  if [ "$UTIL" -gt 5 ]; then
    echo "ABORT: device $DEVICE at ${UTIL}% utilisation, someone is computing on it" >&2; exit 1
  fi
fi
DISK_PCT=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9')
if [ "$DISK_PCT" -ge 90 ]; then
  echo "ABORT: /datasets at ${DISK_PCT}% (>=90% guard)" >&2; exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ABORT: python not found at $PY" >&2; exit 1
fi
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ABORT: torch.cuda.is_available() is False on device $DEVICE (mscluster111-style dead GPU)" >&2; exit 1
fi
echo "[$(date -u +%H:%M:%S)] guards passed: device ${DEVICE} used=${USED}MiB util=${UTIL}%, /datasets=${DISK_PCT}%, cuda ok"

train_args=(
    --pair-pool      "$SCOPE/pair_pool.yaml"
    --pair-prompts   "$SCOPE/pair_prompts.yaml"
    --seed-pool-path "$SCOPE/seed_pool.yaml"
    --cache-root     "$POE_REPAIR_TRAINING_CACHE"
    --lora-rank 32 --lora-alpha 32 --lr 1e-4
    --loss-space x0 --loss-weight-cap "$LOSS_CAP"
    --kill-halve-after-steps 1000000000
    --epoch-size 50
    --ckpt-every-epochs 100 --log-every-epochs 10
    --sample-every-epochs 200
    --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2
    --sample-train-pairs a_wolf__x__a_husky --sample-heldout-pairs a_cat__x__a_dog
    --sample-num-inference-steps 50 --sample-thumb 256
    --wandb-project poe-repair-animals-compose
    --output-root "$OUT_ROOT"
)

if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
  if [ -e "$OUT_ROOT/$RUN_ID" ]; then
    echo "ABORT: $OUT_ROOT/$RUN_ID already exists; move it aside or run STAGE=probe / readout" >&2; exit 1
  fi
  # One epoch first with W&B off, so a broken loss shows up in three minutes and not after a
  # night on the device. The dry run's folder is removed once it passes.
  rm -rf "$OUT_ROOT/${RUN_ID}_dryrun"
  echo "[$(date -u +%H:%M:%S)] dry run: one epoch, wandb disabled"
  "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled "${train_args[@]}" \
      --total-epochs 1 --dry-run --wandb-mode disabled --run-id "${RUN_ID}_dryrun"
  rm -rf "$OUT_ROOT/${RUN_ID}_dryrun"
  echo "[$(date -u +%H:%M:%S)] dry run passed; launching the 40k run"
  "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled "${train_args[@]}" \
      --total-epochs 800 --wandb-mode online --run-id "$RUN_ID"
  echo "[$(date -u +%H:%M:%S)] training DONE ${RUN_ID}"
fi

if [ "$STAGE" = "probe" ] || [ "$STAGE" = "all" ]; then
  for STEP in 030000 040000; do
    CKPT="$OUT_ROOT/$RUN_ID/checkpoints/lora_step_${STEP}.pt"
    [ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 1; }
    # The 8-seed held-out grid, the same recipe as figure_r32_030050: lambda 0 (plain PoE) as
    # the reference column, then lambda 1.0 and 1.2 on all 50 steps; detector count and DINOv2 drift.
    "$PY" scripts/showcase/lambda_boundary_probe.py --rank 32 --checkpoint "$CKPT" \
        --out-root "$OUT_ROOT/figure_r32_x0loss_${STEP}" --windows full --lambdas 1.0,1.2
    echo "[$(date -u +%H:%M:%S)] probe DONE step ${STEP}"
  done
fi

if [ "$STAGE" = "readout" ] || [ "$STAGE" = "all" ]; then
  R=scripts/showcase/experiment_e_x0_loss.py
  "$PY" $R --frames      # per-step running estimates of the new adapter at lambda 1.0, 8 seeds
  "$PY" $R --figures     # training curves, contrast over steps, where it lands, both-ness over steps, checkpoint bars
  "$PY" $R --strips      # Mono | PoE | baseline adapter | new adapter, one strip per seed and one sheet
  "$PY" $R --verdict     # the bars in source against results.json; verdict.json and cell-table.md
  "$PY" $R --wandb       # everything above into one W&B run
  echo "[$(date -u +%H:%M:%S)] readout DONE"
fi

echo "[$(date -u +%H:%M:%S)] DONE ${RUN_ID} stage=${STAGE}."
