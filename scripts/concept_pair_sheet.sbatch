#!/bin/bash
#SBATCH --job-name=pairsheet
#SBATCH --partition=bigbatch
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster72,mscluster73,mscluster74,mscluster76,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/logs/%x-%j.out
#SBATCH --error=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/logs/%x-%j.err
#
# Concept A alone, concept B alone, joint prompt, plain PoE and SuperDiff for each pair and seed,
# one column per process so SuperDiff's own SDXL never shares a card with the repo's.
#
#   PAIRS="a dog|rain" SEEDS="1 2 4 9 10" sbatch scripts/concept_pair_sheet.sbatch
#   COLUMNS=pushes SEEDS=10 sbatch --partition=biggpu --nodelist=mscluster110 scripts/concept_pair_sheet.sbatch
#   COLUMNS=superdiff sbatch --partition=biggpu --nodelist=mscluster110 scripts/concept_pair_sheet.sbatch
#
# Tiles land under the concept_pair_sheet tree; the sheet is built on the laptop afterwards by
# scripts/concept_pair_sheet.py --sheet.
set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
OUT=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet
PAIRS=${PAIRS:-"a dog|rain"}
SEEDS=${SEEDS:-"1 2 4 9 10"}
COLUMNS=${COLUMNS:-"single mono poe superdiff"}

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-}  $(date -Is)  repo $REPO"
gpu=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader) || { echo "ABORT: no GPU" >&2; exit 4; }
echo "$gpu"
# Blackwell cards need the co3_bw build of torch; co3 dies with a CUDA error there.
case "$gpu" in *Blackwell*) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
               *) PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;; esac
[ -d "$OUT" ] || { echo "ABORT: $OUT not visible from this node" >&2; exit 6; }
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${used:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used}%" >&2; exit 3; }
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 5)" \
  || { echo "ABORT: torch sees no CUDA device" >&2; exit 5; }

export POE_REPO="$REPO" CONCEPT_PAIR_OUT="$OUT"
cd "$REPO" || exit 7
IFS=';' read -ra PAIR_LIST <<< "$PAIRS"
for column in $COLUMNS; do
  if [ "$column" = pushes ]; then
    # The per-step pushes on PoE's own path (scripts/concept_pair_pushes.py), one run per pair and seed.
    for pair in "${PAIR_LIST[@]}"; do for seed in $SEEDS; do
      "$PY" scripts/concept_pair_pushes.py --pair "$pair" --seed "$seed" \
        || { echo "pushes $pair seed $seed failed" >&2; exit 9; }
    done; done
    continue
  fi
  "$PY" scripts/concept_pair_sheet.py --column "$column" --pairs "${PAIR_LIST[@]}" --seeds $SEEDS \
    || { echo "column $column failed" >&2; exit 8; }
done
echo "=== done $(date -Is): $(find "$OUT" -name "*.png" -not -path "*_superdiff_raw*" | wc -l) tiles under $OUT"
