#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --exclude=mscluster124,mscluster129
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.err
#
# One run's checkpoints, at a stride, over the probe cells. Pass 1 of the figure sweep: it says
# which checkpoints of a run compose across every kind of pair, and nothing else.
#
#   sbatch --export=ALL,RUN_DIR=<run folder>,STRIDE=10000,SEEDS="3 5" scripts/sweep_probe.sbatch
#
# Seeds are limited to the ones whose starting noise is cached (1 to 16, 42, and any minted
# since): every render borrows that seed's latent so a column is the same noise in every row.
#
# Each tile lands at <pair>/seed<NN>/ours_<run>_s<step>_w50.png, so a tile always names the
# checkpoint that drew it. Tiles that exist are skipped, so a re-run costs nothing.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
: "${RUN_DIR:?set RUN_DIR}"
STRIDE="${STRIDE:-10000}"
# Reward-finetuned runs name their checkpoints reward_step_*.pt and live under
# reward_finetune/ rather than showcase/; the adapter format inside is the same.
CKPT_GLOB="${CKPT_GLOB:-lora_step_*.pt}"
# Five kinds of pair, so "it composes consistently" is a claim about more than animals: two
# different animals, two that look alike, two objects, an object in a place, and one concept asked
# for twice. Held out of the v58 training pool.
PAIRS=('a cat|a dog' 'a tiger|a dog' 'a cat|a fox'
       'a wolf|a coyote' 'a leopard|a cheetah'
       'a chess board|an hourglass' 'a picnic basket|a watermelon' 'a saxophone|a music stand'
       'a lighthouse|a stormy sea' 'a tent|a snowy mountain' 'a canoe|a misty lake'
       'a dog|a dog' 'a cat|a cat')
SEEDS="${SEEDS:-9 42}"

RUN=$(basename "$RUN_DIR")
echo "=== node $(hostname -s)  run $RUN  stride $STRIDE  $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }

# Checkpoints on the stride, and always the last one, which is the state the run actually reached.
mapfile -t CKPTS < <(ls "$RUN_DIR"/checkpoints/$CKPT_GLOB 2>/dev/null | sort)
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
  step=$(basename "$c" | sed 's/[^0-9]//g')
  tag="${RUN}_s${step}"
  echo "--- $tag  $(date +%H:%M:%S)"
  "$PY" "$OUT/figure_candidates.py" --column ours --window 50 --tag "$tag" --checkpoint "$c" \
        --pairs "${PAIRS[@]}" --seeds $SEEDS || echo "FAILED $tag"
  # The renderer dropped writes silently once. Count what this checkpoint actually left behind.
  n=$(ls "$OUT"/*/seed*/ours_"${tag}"_w50.png 2>/dev/null | wc -l)
  echo "--- $tag wrote $n tiles for seeds $SEEDS"
done
echo "=== done $(date -Is)"
