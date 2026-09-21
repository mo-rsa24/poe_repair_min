#!/usr/bin/env bash
# One pooled-adapter training run on a pinned card, with the guards that have caught real faults.
#
#   GPU=<idx> RUN=<name> CELLS=<json> [STEPRANGE="0 25"] [PY=<python>] [EXTRA="..."] \
#     bash train_pool_run.sh
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
# Where the run tree lands. Default keeps the rank-32 showcase runs where they already are;
# override it so a different rank or a different comparison does not mix into that tree.
OUTROOT="${OUTROOT:-/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32}"
CFG=$REPO/artifacts/_shared/cross_pair_pool_configs
# Which generation of pool/prompt/seed yamls to read. The prompts live in the yaml, not in each
# cell's meta.json, so a pool with new pairs needs its own generation or the trainer cannot resolve
# their prompts.
POOLGEN="${POOLGEN:-v54}"
# The reference run phase1_r32_100k, the one whose samples actually compose, used weight decay 0.
# This launcher defaulted to 1e-2, which is the largest unexplained difference between it and the
# four runs that underperformed it.
WD="${WD:-1e-2}"
# Samples per optimizer step. An epoch is EPOCHSIZE steps whatever this is, so raising it puts
# more samples in each update rather than shortening the run: it changes the optimisation, so it
# must be identical across every arm of a comparison.
BATCH="${BATCH:-1}"
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
GPU="${GPU:-0}"; RUN="${RUN:?set RUN}"; CELLS="${CELLS:?set CELLS}"
STEPRANGE="${STEPRANGE:-}"; EXTRA="${EXTRA:-}"
EPOCHS="${EPOCHS:-600}"; EPOCHSIZE="${EPOCHSIZE:-50}"
# Rank 32 needs about 25 GB. The 49 GB cards fit it outright, and checkpointing costs about a third
# of the speed for memory they do not need. Set CKPT=1 only on a 24 GB card.
CKPT="${CKPT:-}"
# Adapter rank, and the alpha that scales it. They have always moved together here, so alpha
# follows rank unless it is set on its own. Rank 32 is the showcase default; rank 16 has not
# been measured for memory, so leave CKPT on until one run reports its peak.
RANK="${RANK:-32}"; ALPHA="${ALPHA:-$RANK}"
# The commit-bucket kill rule snapshots the loss at step 200 and, from step 5000, kills the run on
# any single step whose running commit loss exceeds half of it. It is checked every step, not every
# epoch, so on a 54-cell pool it fires on one noisy step out of 1410: the first four runs all died
# at step 6409 having reached a tenth of their initial loss. Set KILLAFTER to a step past the run
# length to disable it; set it low again only for a pool smooth enough to trust it.
KILLAFTER="${KILLAFTER:-1000000000}"

# Render a strip while training, not only at the end. A loss curve cannot show the failure this
# scope exists to fix: an adapter can descend nicely and still draw two dogs. The held-out pairs are
# the two the paper is judged on, so they are what gets rendered.
SAMPLEEVERY="${SAMPLEEVERY:-25}"
SAMPLEHELD="${SAMPLEHELD:-a_cat__x__a_dog,an_elephant__x__a_penguin}"
SAMPLEHELDCELLS="${SAMPLEHELDCELLS:-2}"
SAMPLETRAIN="${SAMPLETRAIN:-a_giraffe__x__a_lion,a_lion__x__a_meerkat}"
SAMPLETRAINCELLS="${SAMPLETRAINCELLS:-1}"

echo "=== node $(hostname)  gpu $GPU  run $RUN"; echo "PID $$"; echo "=== start $(date -Is)"
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }

# Create the tree before the guard reads it. A df against a path that does not exist prints an
# error, leaves USE empty, and the >= 90 test then silently passes: the guard becomes a no-op
# exactly when a new output root is used for the first time. Creating it first also removes the
# race several arms launched together hit on NFS, where each tries to mkdir the shared parent.
# mkdir -p is not idempotent on this NFS: several arms launched together race on the shared
# parent and the losers exit non-zero with "Already exists". Test the directory, not mkdir's
# status, so an existing tree is success and only a genuinely absent one aborts.
# Retry rather than trust one test. Several arms launched together race on the shared parent,
# mkdir -p is not idempotent here (the losers exit "Already exists"), and a node whose NFS
# attribute cache is stale can fail -d on a directory another node created moments earlier.
# Five tries over eight seconds separates that from a genuinely unwritable path.
for _try in 1 2 3 4 5; do
  [ -d "$OUTROOT" ] && break
  mkdir -p "$OUTROOT" 2>/dev/null || true
  sleep 2
done
[ -d "$OUTROOT" ] || { echo "ABORT: $OUTROOT absent after 5 tries" >&2; exit 3; }
USE=$(df --output=pcent "$OUTROOT" | tail -1 | tr -dc '0-9')
[ -z "$USE" ] && { echo "ABORT: disk guard read no usage for $OUTROOT" >&2; exit 3; }
echo "=== disk ${USE}% on $OUTROOT"
[ "$USE" -ge 90 ] && { echo "ABORT: $OUTROOT is ${USE}% full" >&2; exit 3; }

read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
echo "=== device ${USED} MiB, ${UTIL}%"
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu $GPU is faulted" >&2; exit 4;; esac
# How much of someone else's memory this run will sit beside. 1024 MiB means "effectively
# empty" and is the right default: landing on an active card slows both jobs and can OOM either.
# Raise it deliberately, per launch, only for a card measured idle over hours with room to spare.
MAXUSED="${MAXUSED:-1024}"
[ "$USED" -gt "$MAXUSED" ] && { echo "ABORT: gpu $GPU has ${USED} MiB in use by someone else (limit $MAXUSED)" >&2; exit 5; }
[ "$USED" -gt 1024 ] && echo "=== NOTE sharing gpu $GPU: ${USED} MiB already in use by another user"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)" \
  || { echo "ABORT: torch cannot see gpu $GPU" >&2; exit 6; }

# 30,000 optimiser steps is EPOCHS x EPOCHSIZE; the trainer counts epochs, never steps.
echo "=== $((EPOCHS * EPOCHSIZE)) steps, rank $RANK/alpha $ALPHA, pool $POOLGEN, batch $BATCH, weight-decay $WD, cells $CELLS, step range ${STEPRANGE:-all}"
echo "=== outroot $OUTROOT"
echo "=== extra ${EXTRA:-none}"
cd "$REPO" || exit 7
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled \
  --pair-pool      "$CFG/pair_pool_$POOLGEN.yaml" \
  --pair-prompts   "$CFG/pair_prompts_$POOLGEN.yaml" \
  --seed-pool-path "$CFG/seed_pool_$POOLGEN.yaml" \
  --cells          "$CELLS" \
  ${STEPRANGE:+--train-step-range $STEPRANGE} \
  --lora-rank "$RANK" --lora-alpha "$ALPHA" --lr 1e-4 --weight-decay "$WD" --ema-decay 0.999 \
  --train-batch-size "$BATCH" ${COMPILE:+--compile} \
  --kill-halve-after-steps "$KILLAFTER" \
  --sample-every-epochs "$SAMPLEEVERY" \
  --ckpt-every-epochs   "$SAMPLEEVERY" \
  --sample-heldout-pairs "$SAMPLEHELD" \
  --sample-cells-per-heldout-pair "$SAMPLEHELDCELLS" \
  --sample-cells-per-train-pair "$SAMPLETRAINCELLS" \
  --sample-train-pairs "$SAMPLETRAIN" \
  --total-epochs "$EPOCHS" --epoch-size "$EPOCHSIZE" \
  ${CKPT:+--gradient-checkpointing} \
  --run-id "$RUN" --output-root "$OUTROOT" \
  --wandb-mode online --wandb-project poe-repair-animals-compose $EXTRA
RC=$?
# Capture the status BEFORE anything else runs. Writing `exit $?` inside an echo that also calls
# $(date) reports the date's status, not the training's, so a crashed run printed "exit 0".
if [ "$RC" -ne 0 ]; then
  echo "=== FAILED $(date -Is) exit $RC"
else
  echo "=== done $(date -Is) exit 0"
fi
exit "$RC"
