#!/bin/bash
#SBATCH --job-name=repeat-ema
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --exclude=mscluster124,mscluster129
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%j.err
#
# Two free checks on the checkpoints the good pictures came from, before any new training:
#
#   1. Is the render repeatable? The same checkpoint rendered in two separate processes
#      (tags rawA and rawB). Bit-identical tiles mean every checkpoint-to-checkpoint swing seen
#      so far is in the weights. Different tiles mean the render itself flips, and no checkpoint
#      comparison to date can be read without a repeat.
#   2. Do the EMA weights draw a steadier picture? Every checkpoint carries `lora_state_ema`
#      beside `lora_state`, and no sheet has ever loaded it (tag ema).
#
# Checkpoints: v58-03-self at 25k, 27.5k and 30k (the run whose late renders swing while the
# weights barely move) and v58-05-self-contrast at 15k (the clearest elephant and penguin).
# Pairs are held out of the v58 pool. Seeds 9, 10 and 42 have cached starting noise.
#
#   sbatch scripts/render_repeat_and_ema.sbatch
#
# Tiles land at <pair>/seed<NN>/ours_<run>_s<step>_<tag>_w50.png in the ablation grid, and the
# job ends by printing the largest pixel difference for every rawA/rawB and rawA/ema pair.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
COHORT=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3
CKPTS=("$COHORT/v58-05-self-contrast/checkpoints/lora_step_015000.pt"
       "$COHORT/v58-03-self/checkpoints/lora_step_025000.pt"
       "$COHORT/v58-03-self/checkpoints/lora_step_027500.pt"
       "$COHORT/v58-03-self/checkpoints/lora_step_030000.pt")
PAIRS=('a cat|a dog' 'an elephant|a penguin' 'a tiger|a dog' 'a wolf|a coyote' 'a chess board|an hourglass')
SEEDS="9 10 42"

echo "=== node $(hostname -s)  $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
df -h /datasets | tail -1

export POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$OUT"
cd "$REPO" || exit 7
for c in "${CKPTS[@]}"; do
  [ -f "$c" ] || { echo "ABORT: missing $c" >&2; exit 2; }
  run=$(basename "$(dirname "$(dirname "$c")")")
  step=$(basename "$c" | sed 's/[^0-9]//g')
  for variant in rawA rawB ema; do
    tag="${run}_s${step}_${variant}"
    extra=""; [ "$variant" = ema ] && extra="--ema"
    echo "--- $tag  $(date +%H:%M:%S)"
    # One process per variant, so rawA and rawB never share a process.
    "$PY" scripts/figure_candidates.py --column ours --window 50 --tag "$tag" --checkpoint "$c" \
          --pairs "${PAIRS[@]}" --seeds $SEEDS $extra || echo "FAILED $tag"
    n=$(ls "$OUT"/*/seed*/ours_"${tag}"_w50.png 2>/dev/null | wc -l)
    echo "--- $tag wrote $n tiles (expected $(( ${#PAIRS[@]} * 3 )))"
  done
done

echo "=== pixel differences  $(date -Is)"
"$PY" - "$OUT" <<'EOF'
import sys, glob, os
from PIL import Image, ImageChops
out = sys.argv[1]
for a in sorted(glob.glob(f"{out}/*/seed*/ours_*_rawA_w50.png")):
    cell = os.path.relpath(a, out)
    for other in ("rawB", "ema"):
        b = a.replace("_rawA_", f"_{other}_")
        if not os.path.exists(b):
            print(f"{cell}  vs {other}: missing"); continue
        d = ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB"))
        mx = max(hi for _, hi in d.getextrema())
        mean = sum(d.convert("L").getdata()) / (d.width * d.height)
        print(f"{cell}  vs {other}: max {mx:3d}  mean {mean:.2f}  (0 to 255)")
EOF
echo "=== done $(date -Is)"
