#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster124,mscluster129
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.err
#
# One run's checkpoints, at a stride, over the six probe cells. Pass 1 of the figure sweep: it says
# which checkpoints of a run are worth rendering the paper's full cell set on, and nothing else.
#
#   sbatch --export=ALL,RUN_DIR=<run folder>,STRIDE=10000 scripts/sweep_probe.sbatch
#
# Each tile lands at <pair>/seed<NN>/ours_<run>_s<step>_w50.png, so a tile always names the
# checkpoint that drew it. Tiles that exist are skipped, so a re-run costs nothing.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
: "${RUN_DIR:?set RUN_DIR}"
STRIDE="${STRIDE:-10000}"
PAIRS=('a cat|a dog' 'a chess board|an hourglass' 'a dog|a dog')
SEEDS="9 42"

RUN=$(basename "$RUN_DIR")
echo "=== node $(hostname -s)  run $RUN  stride $STRIDE  $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }

# Checkpoints on the stride, and always the last one, which is the state the run actually reached.
mapfile -t CKPTS < <(ls "$RUN_DIR"/checkpoints/lora_step_*.pt 2>/dev/null | sort)
[ "${#CKPTS[@]}" -eq 0 ] && { echo "ABORT: no checkpoints in $RUN_DIR" >&2; exit 2; }
PICK=()
for c in "${CKPTS[@]}"; do
  step=$(basename "$c" | sed 's/[^0-9]//g' | sed 's/^0*//')
  [ -z "$step" ] && continue
  if [ $((step % STRIDE)) -eq 0 ] || [ "$c" = "${CKPTS[-1]}" ]; then PICK+=("$c"); fi
done
echo "=== ${#PICK[@]} of ${#CKPTS[@]} checkpoints: $(for c in "${PICK[@]}"; do basename "$c"; done | tr '\n' ' ')"

export POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$OUT"
cd "$REPO" || exit 7
for c in "${PICK[@]}"; do
  step=$(basename "$c" | sed 's/lora_step_//; s/\.pt//')
  tag="${RUN}_s${step}"
  echo "--- $tag  $(date +%H:%M:%S)"
  "$PY" "$OUT/figure_candidates.py" --column ours --window 50 --tag "$tag" --checkpoint "$c" \
        --pairs "${PAIRS[@]}" --seeds $SEEDS || echo "FAILED $tag"
  # The renderer dropped writes silently once. Count what this checkpoint actually left behind.
  n=$(ls "$OUT"/*/seed*/ours_"${tag}"_w50.png 2>/dev/null | wc -l)
  echo "--- $tag wrote $n of 6 tiles"
done
echo "=== done $(date -Is)"
