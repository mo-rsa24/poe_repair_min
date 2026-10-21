#!/bin/bash
#SBATCH --job-name=seedgrid
#SBATCH --partition=batch
#SBATCH --time=10:00:00
#SBATCH --array=0-16
#SBATCH --exclude=mscluster124,mscluster129
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%A_%a.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid/logs/%x-%A_%a.err
#
# Seeds down the page, methods across: seven pairs, ten seeds, the joint prompt, the plain product,
# CO3, and every adapter trained from v58 on (freezenull excluded) plus the v57 self-decay run.
# One array task per column, so a column's 70 tiles run on one card.
#
#   sbatch scripts/seed_method_grid.sbatch
#
# Tiles land in the ablation_grid tree as <pair>/seed<NN>/<key>.png, the same names the figure
# sweep uses, so a tile that already exists is skipped. The grids are built afterwards on the
# laptop by scripts/seed_method_grid.py.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/ablation_grid
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
O=/datasets/mmolefe/poe_repair_min/outputs
SEEDS="1 2 4 9 10 11 12 13 14 15"
PAIRS=('an elephant|a penguin' 'a cat|a dog' 'a tiger|a dog'
       'a cat|a teapot' 'a horse|a bicycle'
       'a picnic basket|a watermelon'
       'a camel|a forest')
# A helper on a shared card works the same column from the far end, so it and the Slurm task meet
# in the middle instead of both rendering the next missing tile.
if [ "${REVERSE:-0}" = 1 ]; then
  mapfile -t PAIRS < <(printf '%s\n' "${PAIRS[@]}" | tac)
  SEEDS=$(echo $SEEDS | tr ' ' '\n' | tac | tr '\n' ' ')
fi

# name | checkpoint | extra renderer flags. v58-03 at 30k and v58-05 at 15k are the steps the
# training-recipe notes name as good; every other run is its last checkpoint. R1 learned only the
# part expert re-weighting cannot supply, so the re-weighting is applied at render, at the constant
# 3.0 and 3.0 the fidelity finding used.
ADAPTERS=(
  "v57-11b-x0-self-decay|$O/showcase/cohort2/v57-11b-x0-self-decay/checkpoints/lora_step_050000.pt|"
  "v58-01-plain|$O/showcase/cohort3/v58-01-plain/checkpoints/lora_step_047500.pt|"
  "v58-02-contrast|$O/showcase/cohort3/v58-02-contrast/checkpoints/lora_step_027500.pt|"
  "v58-03-self|$O/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt|"
  "v58-05-self-contrast|$O/showcase/cohort3/v58-05-self-contrast/checkpoints/lora_step_015000.pt|"
  "v58f-01-plain|$O/showcase/cohort4/v58f-01-plain/checkpoints/lora_step_020000.pt|"
  "v58f-05-self-contrast|$O/showcase/cohort4/v58f-05-self-contrast/checkpoints/lora_step_025000.pt|"
  "R1|$O/showcase/cohort5/R1/checkpoints/lora_step_050000.pt|--expert-weights 3.0 3.0"
  "R2a|$O/showcase/cohort5/R2a/checkpoints/lora_step_050000.pt|"
  "R2b|$O/showcase/cohort5/R2b/checkpoints/lora_step_050000.pt|"
  "refl-03-self-contrast-15k|$O/reward_finetune/refl-03-self-contrast-15k/checkpoints/reward_step_001250.pt|"
  "refl-kl-K0|$O/reward_finetune/refl-kl-K0/checkpoints/reward_step_002000.pt|"
  "refl-kl-K1|$O/reward_finetune/refl-kl-K1/checkpoints/reward_step_002000.pt|"
  "refl-kl-K2|$O/reward_finetune/refl-kl-K2/checkpoints/reward_step_002000.pt|"
)
# The plain-product column attaches an adapter with its window at zero steps; which one is moot.
ANY_CKPT=$O/showcase/cohort3/v58-05-self-contrast/checkpoints/lora_step_015000.pt

I=${SLURM_ARRAY_TASK_ID:?run as an array job}
echo "=== node $(hostname -s)  task $I  $(date -Is)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
[ -d "$OUT" ] || { echo "ABORT: $OUT not visible from this node" >&2; exit 6; }
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${used:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used}%" >&2; exit 3; }

export POE_REPO="$REPO" FIGURE_CANDIDATES_OUT="$OUT"
cd "$REPO" || exit 7

if [ "$I" -eq 0 ]; then
  "$PY" scripts/figure_candidates.py --column mono --pairs "${PAIRS[@]}" --seeds $SEEDS
  n=$(ls "$OUT"/*/seed*/mono.png | wc -l)
elif [ "$I" -eq 1 ]; then
  "$PY" scripts/figure_candidates.py --column poe --checkpoint "$ANY_CKPT" --pairs "${PAIRS[@]}" --seeds $SEEDS
  n=$(ls "$OUT"/*/seed*/poe.png | wc -l)
elif [ "$I" -eq 16 ]; then
  # CO3 draws its own noise: the seed is its own index, not the cached starting latent.
  # PAIRS is an array in this script and an array cannot be exported, so the runner gets its
  # pair string through env.
  CO3_OUT=$O/showcase/co3_seed_grid
  env GPU=${CO3_GPU:-0} OUT=$CO3_OUT SEEDS="$SEEDS" \
      PAIRS="an_elephant__x__a_penguin|an elephant|a penguin|an elephant and a penguin ; a_cat__x__a_dog|a cat|a dog|a cat and a dog ; a_tiger__x__a_dog|a tiger|a dog|a tiger and a dog ; a_cat__x__a_teapot|a cat|a teapot|a cat and a teapot ; a_horse__x__a_bicycle|a horse|a bicycle|a horse and a bicycle ; a_picnic_basket__x__a_watermelon|a picnic basket|a watermelon|a picnic basket and a watermelon ; a_camel__x__a_forest|a camel|a forest|a camel and a forest" \
      bash "$REPO/scripts/showcase/run_co3_targets.sh"
  OUT=$CO3_OUT
  n=$(find "$OUT" -name '*.png' | wc -l)
else
  IFS='|' read -r NAME CKPT EXTRA <<< "${ADAPTERS[$((I - 2))]}"
  step=$(basename "$CKPT" | tr -dc '0-9')
  tag="${NAME}_s${step}"
  [ -f "$CKPT" ] || { echo "ABORT: missing $CKPT" >&2; exit 2; }
  echo "--- $tag  $EXTRA"
  # shellcheck disable=SC2086
  "$PY" scripts/figure_candidates.py --column ours --window 50 --tag "$tag" --checkpoint "$CKPT" \
        $EXTRA --pairs "${PAIRS[@]}" --seeds $SEEDS
  # The renderer dropped writes silently once. Count what this column actually left behind.
  n=$(ls "$OUT"/*/seed*/ours_"${tag}"_w50.png 2>/dev/null | wc -l)
fi
echo "=== task $I wrote or found $n tiles  $(date -Is)"
