#!/bin/bash
# Wait for held-out caches, validate training with a 1-epoch dry-run, then launch the
# full 100k-step pooled-LoRA training with W&B. Run in background on node 110.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
cd "$REPO"
CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
TRAIN=scripts/animals_compose_transfer/train_phase1.sh

echo "[$(date +%H:%M:%S)] waiting for held-out cache workers to finish..."
until [ "$(pgrep -fc 'build_caches --which heldout')" -eq 0 ]; do sleep 15; done
NHELD=$(find "$CACHE/heldout" -name 'step_049.pt' 2>/dev/null | grep -cE 'leopard|frog|eagle|seal|elephant|giraffe|octopus|a_cat__x__a_dog')
echo "[$(date +%H:%M:%S)] held-out caches present (step_049 count incl cat/dog): $NHELD"

echo "[$(date +%H:%M:%S)] === DRY-RUN (1 epoch, wandb off) ==="
bash "$TRAIN" dry
DRC=$?
if [ $DRC -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] DRY-RUN FAILED rc=$DRC — NOT launching full training."
  exit $DRC
fi
echo "[$(date +%H:%M:%S)] dry-run OK. === LAUNCH FULL 100k TRAINING (wandb online) ==="
bash "$TRAIN" run
echo "[$(date +%H:%M:%S)] training exited rc=$?"
