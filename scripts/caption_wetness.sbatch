#!/bin/bash
#SBATCH --job-name=wetness
#SBATCH --partition=bigbatch
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=mscluster72,mscluster73
#SBATCH --output=/datasets/mmolefe/relaion2b-en/logs/%x-%j.out
#
# Sample, fetch and score captioned camels and dogs from the Re-LAION-2B-en mirror.
#
#   N=100 OUT=/datasets/mmolefe/relaion2b-en/wetness/pilot sbatch scripts/caption_wetness.sbatch
#   STAGES=score OUT=... sbatch --gres=gpu:1 scripts/caption_wetness.sbatch
#
# CPU is enough for a pilot; the score stage uses a GPU when one is visible.
set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=${OUT:?set OUT to the output folder}
N=${N:-100}
STAGES=${STAGES:-"sample fetch score"}

echo "=== node $(hostname -s)  job ${SLURM_JOB_ID:-}  $(date -Is)  repo $REPO  out $OUT"
[ -d /datasets/mmolefe/relaion2b-en/parquet ] || { echo "ABORT: mirror not visible from this node" >&2; exit 6; }
mkdir -p "$OUT"
used=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${used:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${used:-unknown}%" >&2; exit 3; }
files=$(ls /datasets/mmolefe/relaion2b-en/parquet/*.parquet | wc -l)
[ "$files" -eq 128 ] || { echo "ABORT: mirror holds $files of 128 files" >&2; exit 8; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO" || exit 7
for stage in $STAGES; do
  echo "--- $stage  $(date -Is)"
  "$PY" scripts/caption_wetness.py "$stage" --out "$OUT" --n "$N" || { echo "ABORT: $stage failed" >&2; exit 9; }
done
echo "=== done $(date -Is)"
