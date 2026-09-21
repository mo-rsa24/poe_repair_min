#!/usr/bin/env bash
# Plan 08 (scope 06), task 2.1: train one SuperDiff-residual LoRA.
#
#   bash scripts/launch_sdlora.sh <rank> <cuda index>
#
# Mirrors the phase-1 PoE runs (phase1_r{8,16,32}_100k: 2000 epochs x 50 optimizer steps,
# batch 1, AdamW 1e-4, alpha = rank, attn2.to_q/k/v) with two differences: the cache is
# SuperDiff's trajectory at kappa 0.5 (scripts/build_superdiff_cache.py) and the step composes
# the three branches with SuperDiff's AND blend (--compose superdiff). Inline PoE sampling is
# off: it would render with the wrong sampler for this adapter. Every path is absolute; the
# script cd's into the repo itself, so the SSH launch line needs no working directory.
set -euo pipefail
RANK=${1:?rank}; GPU=${2:?cuda index}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
# co3 (torch 2.5.1, up to sm_90) for the RTX 8000 / A6000 nodes; on the Blackwell nodes
# (mscluster110-112, sm_120) pass PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python,
# the env the phase-1 rank-8 run used there. co3 produces no CUDA kernels on that card.
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
SD=/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff
POOL=$REPO/artifacts/results/does-the-fix-reach-unseen-pairs
RUN_ID=sdlora_r${RANK}_100k
cd "$REPO"

# Guards, per environment/hpc/execution-protocol.md step 4.
used_pct=$(df --output=pcent "$SD" | tail -1 | tr -dc '0-9'); echo "disk: $SD at ${used_pct}% used"
[ "$used_pct" -lt 90 ] || { echo "ERROR: /datasets over 90%"; exit 2; }
[ -x "$PY" ] || { echo "ERROR: co3 python missing"; exit 2; }
CUDA_VISIBLE_DEVICES=$GPU timeout 60 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || { echo "ERROR: no GPU"; exit 2; }
# A device is free when nothing foreign is *running* on it. mscluster109 device 0 carries an
# idle 2.9 GB residency from another user's allocation (0% utilisation for the whole of
# 2026-09-03, shared safely by three renders and a dry run), so the guard is memory under 4 GB
# and utilisation 0%, not memory under 1 GB.
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | head -1)
# nvidia-smi's utilisation is a windowed sample and can read 100% for a minute after a
# process is killed while memory already reads ~0; under 100 MiB nothing can be running, so
# memory alone decides there.
if [ "${mem:-0}" -lt 100 ]; then :;
elif [ "${mem:-0}" -lt 4096 ] && [ "${util:-100}" -eq 0 ]; then :;
else echo "ERROR: device $GPU has ${mem} MiB in use at ${util}% utilisation, refusing to share"; exit 2; fi
n_cells=$(find "$SD/cache/train" -name meta.json 2>/dev/null | wc -l)
[ "$n_cells" -ge 88 ] || { echo "ERROR: SuperDiff cache has $n_cells of 88 cells"; exit 2; }
echo "node=$(hostname) device=$GPU pid=$$ run_id=$RUN_ID rank=$RANK cache_cells=$n_cells"

# Optional resume: RESUME_FROM=<lora_step_*.pt> RESUME_WANDB_ID=<id> continues a run in place.
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  [ -f "$RESUME_FROM" ] || { echo "ERROR: RESUME_FROM not found: $RESUME_FROM"; exit 2; }
  RESUME_ARGS+=(--resume-from "$RESUME_FROM")
  [ -n "${RESUME_WANDB_ID:-}" ] && RESUME_ARGS+=(--resume-wandb-id "$RESUME_WANDB_ID")
  echo "resuming from $RESUME_FROM (wandb id ${RESUME_WANDB_ID:-new})"
fi

# The "commit loss must halve" stall guard was calibrated on the PoE residual at 50-step cells,
# where it halves well before step 5000. On SuperDiff's 200-step cache all three ranks fell
# 46% by step 5000 (0.0102 -> ~0.0055) and were killed just short; the budget makes ~4x fewer
# passes over the data, so the deadline is scaled by the same factor, to 20k steps. Recorded in
# plan 08's review and the scope CHANGELOG on 2026-09-03.
exec env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
  --kill-halve-after-steps 20000 ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --pair-pool "$POOL/pair_pool.yaml" \
  --seed-pool-path "$POOL/seed_pool.yaml" \
  --pair-prompts "$POOL/pair_prompts.yaml" \
  --cache-root "$SD/cache" \
  --compose superdiff --kappa 0.5 \
  --commit-window 20 100 \
  --lora-rank "$RANK" --lora-alpha "$RANK" \
  --total-epochs 2000 --epoch-size 50 --train-batch-size 1 --lr 1e-4 \
  --guidance-scale 7.5 --num-inference-steps 200 \
  --wandb-mode online --wandb-project poe-repair-animals-compose \
  --output-root "$SD" --run-id "$RUN_ID" \
  --ckpt-every-epochs 100 --log-every-epochs 20 --sample-every-epochs 0
