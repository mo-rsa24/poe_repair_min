#!/usr/bin/env bash
# Reward fine-tune from v58-05 at step 15,000 against the plurality objective, two runs that differ
# in one thing: the KL-style anchor to the starting adapter. Both have the ImageReward fidelity term
# actually on, which refl-04-fidelity never did (BLIP_Pretrain has no forward()).
#
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/launch_reward_kl.sh <run> > <log> 2>&1 &'
#
#   K0   --kl-weight 0: refl-04-fidelity's arguments with the fidelity term working (the control)
#   K1   --kl-weight 10: the same, charged for drifting from the starting adapter (0.01 relative
#        squared drift, a 10% change, costs 0.1 of reward)
#   K2   --w-contrast 10, no anchor: K0 plus a charge on each region looking more like the other
#        concept than its own (a penguin turning into the elephant). Compared against K0 only.
#
# Bar, written before launch: K1 supports the anchor if, at its last checkpoint, reward/fidelity is
# no lower than K0's and train/drift stays under 0.05, and by eye on the tracking sheets it shows no
# more third animals or swapped species than K0. If K1's reward/total ends more than 0.05 below
# K0's with the pictures unchanged, the weight is too high and the run is inconclusive, not a null.
# K2 supports the contrast term if, at its last checkpoint, reward/contrast is above K0's (K0 does not
# optimise it but its value can be recomputed on K0's tracking renders), reward/fidelity is no more
# than 0.05 below K0's, and by eye the elephant-and-penguin and cat-and-dog sheets show fewer
# regions drawn as the other concept than K0's.
#
# On the Blackwell nodes (110, 112) pass PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python.
set -euo pipefail
RUN=${1:?run name: K0 K1 K2}; GPU=${GPU:?set GPU=<cuda index>}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
OUT=/datasets/mmolefe/poe_repair_min/outputs/reward_finetune
START=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-05-self-contrast/checkpoints/lora_step_015000.pt
cd "$REPO"

echo "=== $RUN on $(hostname -s) device $GPU pid $$ $(date -Is)"
[ -d /datasets/mmolefe/poe_repair_min ] || { echo "ERROR: /datasets not visible here"; exit 2; }
used_pct=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9'); echo "disk: /datasets at ${used_pct}% used"
[ -n "$used_pct" ] && [ "$used_pct" -lt 90 ] || { echo "ERROR: /datasets over 90% or unreadable"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: python missing at $PY"; exit 2; }
[ -f "$START" ] || { echo "ERROR: starting adapter missing at $START"; exit 2; }
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | head -1)
# ALLOW_SHARE=1 is for K0 and K1 side by side on one 96 GB Blackwell card, which is the point:
# the same hardware, so the anchor is the only thing that differs.
if [ "${ALLOW_SHARE:-0}" = 1 ]; then echo "sharing device $GPU (${mem} MiB already in use)";
elif [ "${mem:-0}" -lt 100 ]; then :;
elif [ "${mem:-0}" -lt 4096 ] && [ "${util:-100}" -eq 0 ]; then :;
else echo "ERROR: device $GPU has ${mem} MiB in use at ${util}% utilisation, refusing to share"; exit 2; fi
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no device"; print("torch device:", torch.cuda.get_device_name(0))'

BASE=(--checkpoint "$START" --pairs "a cat|a dog" "an elephant|a penguin" --seeds 1 2 3 4 5 6 7 8
      --steps 2000 --lr 1e-5 --grad-clip 1.0 --num-inference-steps 50 --guidance-scale 7.5
      --reward-step-lo 30 --reward-step-hi 45 --mask-at 8
      --w-identity 1.0 --w-distinct 1.0 --w-compact 1.0 --w-fidelity 1.0
      --gradient-checkpointing --sample-every 50 --ckpt-every 250
      --out-root "$OUT" --wandb-mode online --wandb-project poe-repair-animals-compose)
case "$RUN" in
  K0) EXTRA=(--kl-weight 0) ;;
  K1) EXTRA=(--kl-weight 10) ;;
  K2) EXTRA=(--kl-weight 0 --w-contrast 10) ;;
  *) echo "unknown run $RUN"; exit 2 ;;
esac
echo "=== argv: ${BASE[*]} ${EXTRA[*]} --run-id refl-kl-$RUN"
CUDA_VISIBLE_DEVICES=$GPU exec "$PY" -m poe_repair.experiments.reward_finetune.train \
  "${BASE[@]}" "${EXTRA[@]}" --run-id "refl-kl-$RUN"
