#!/usr/bin/env bash
# One recipe run on a shared card, started with nohup over SSH. The base is v58-05's exact
# argument list (recovered from its W&B metadata), and each run changes one thing against it.
#
#   ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/launch_recipe_run.sh <run> > <log> 2>&1 &'
#
#   R0   v58-05 again with torch seed 7 instead of 42: is a good step a property of the recipe or of one run
#   R1   --out-of-span-only 1: train only on the part re-weighting the two experts cannot supply
#   R2a  the on-policy cache (states a corrected run visits), on the 81 cells it holds
#   R2b  the plain cache on the same 81 cells: R2a's control
#
# On the Blackwell nodes (110, 112) pass PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python.
# Every path is absolute and the script cd's into the repo, so the SSH line needs no working directory.
set -euo pipefail
RUN=${1:?run name: R0 R1 R2a R2b}; GPU=${GPU:?set GPU=<cuda index>}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
OUT=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort5
POOL=$REPO/artifacts/_shared/cross_pair_pool_configs
ONPOLICY=/datasets/mmolefe/poe_repair_min/outputs/onpolicy_cache
cd "$REPO"

echo "=== $RUN on $(hostname -s) device $GPU pid $$ $(date -Is)"
used_pct=$(df --output=pcent /datasets | tail -1 | tr -dc '0-9'); echo "disk: /datasets at ${used_pct}% used"
[ "$used_pct" -lt 90 ] || { echo "ERROR: /datasets over 90%"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: python missing at $PY"; exit 2; }
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | head -1)
if [ "${mem:-0}" -lt 100 ]; then :;
elif [ "${mem:-0}" -lt 4096 ] && [ "${util:-100}" -eq 0 ]; then :;
else echo "ERROR: device $GPU has ${mem} MiB in use at ${util}% utilisation, refusing to share"; exit 2; fi
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no device"; print("torch device:", torch.cuda.get_device_name(0))'

BASE=(--pair-pool "$POOL/pair_pool_v58.yaml" --pair-prompts "$POOL/pair_prompts_v58.yaml"
      --seed-pool-path "$POOL/seed_pool_v58.yaml" --cells "$POOL/cells_v58.json"
      --train-step-range 0 10 --lora-rank 32 --lora-alpha 32 --lora-targets cross+self
      --lr 1e-4 --lr-decay cosine --weight-decay 0 --ema-decay 0.9999 --train-batch-size 1
      --kill-halve-after-steps 1000000000 --total-epochs 1000 --epoch-size 50
      --sample-every-epochs 50 --ckpt-every-epochs 50 --sample-num-inference-steps 50
      --sample-heldout-pairs a_cat__x__a_dog,an_elephant__x__a_penguin --sample-cells-per-heldout-pair 2
      --sample-train-pairs a_lion__x__a_meerkat,a_typewriter__x__a_cactus --sample-cells-per-train-pair 1
      --loss-space x0 --loss-weight-cap 1e9 --contrast-weight 0.01
      --output-root "$OUT" --wandb-mode online --wandb-project poe-repair-animals-compose)
case "$RUN" in
  R0)  EXTRA=(--torch-seed 7) ;;
  R1)  EXTRA=(--out-of-span-only 1) ;;
  R2a) EXTRA=(--cache-root "$ONPOLICY/cache" --cells "$ONPOLICY/cells_onpolicy81.json") ;;
  R2b) EXTRA=(--cells "$ONPOLICY/cells_onpolicy81.json") ;;
  *) echo "unknown run $RUN"; exit 2 ;;
esac
mkdir -p "$OUT"
echo "=== argv: ${BASE[*]} ${EXTRA[*]} --run-id $RUN"
CUDA_VISIBLE_DEVICES=$GPU exec "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
  "${BASE[@]}" "${EXTRA[@]}" --run-id "$RUN"
