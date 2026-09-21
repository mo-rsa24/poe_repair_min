#!/usr/bin/env bash
# Render CO3 targets for a pair, as a candidate replacement for the plain joint-prompt target.
#
#   GPU=<idx> PAIRS="slug|concept_a|concept_b|joint ; ..." SEEDS="1 2" bash run_co3_targets.sh
#
# CO3 is Dutta et al, "Steer Away From Mode Collisions", arXiv 2509.25940, vendored unmodified
# under composition/debottam_co3. It needs the joint prompt at inference, which is exactly what our
# adapter does not; here it is used only to produce a better teacher.
set -uo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
CO3=$REPO/composition/debottam_co3
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT="${OUT:-/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/co3_targets}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
GPU="${GPU:-0}"

echo "=== node $(hostname)"; echo "=== gpu $GPU"; echo "PID $$"; echo "=== start $(date -Is)"
[ -x "$PY" ] || { echo "ABORT: co3 python not found at $PY" >&2; exit 2; }

USEPCT=$(df --output=pcent "$OUT" 2>/dev/null || df --output=pcent /datasets | tail -1)
USEPCT=$(echo "$USEPCT" | tail -1 | tr -dc '0-9')
[ "$USEPCT" -ge 90 ] && { echo "ABORT: /datasets is ${USEPCT}% full" >&2; exit 3; }
echo "=== disk ${USEPCT}%"

read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
echo "=== device ${USED} MiB, ${UTIL}%"
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu $GPU faulted" >&2; exit 4;; esac
[ "$USED" -gt 1024 ] && { echo "ABORT: gpu $GPU has ${USED} MiB in use" >&2; exit 5; }

mkdir -p "$OUT"
cd "$CO3" || exit 7
OK=0; FAIL=0
IFS=';' read -ra ENTRIES <<< "$PAIRS"
for e in "${ENTRIES[@]}"; do
  e="$(echo "$e" | sed 's/^ *//; s/ *$//')"; [ -z "$e" ] && continue
  IFS='|' read -r SLUG CA CB PJ <<< "$e"

  # the barred-subject guard, same one the cache builder uses
  "$PY" - "$CA" "$CB" "$PJ" <<'PYG' || { echo "!!! refused: $SLUG"; FAIL=$((FAIL+1)); continue; }
import sys
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase")
from disallowed_subjects import check
for t in sys.argv[1:]:
    check(t)
PYG

  SEEDLIST="[$(echo $SEEDS | tr ' ' ',')]"
  echo "--- $SLUG  seeds $SEEDLIST  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" sample_co3.py \
    --guidance_scale 0.8 --n_timesteps 50 \
    --prompt "${CA}+${CB}+${PJ}" --prompt_orig "$PJ" \
    --output_path "$OUT/$SLUG/" --output_path_all "$OUT/$SLUG/" \
    --sd_version xl --resolution_h 1024 --resolution_w 1024 \
    --seeds "$SEEDLIST" --negative_prompt '' \
    --num_ts_to_correct 6 --num_latent_corrector_steps 5 --num_resampling_steps 3 \
    --corrector_algo co3-hybrid --modulate_comp_weights True --beta 0.9 --lmda 0.8
  if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); echo "!!! failed: $SLUG"; fi
done
echo "=== done $(date -Is)  pairs ok $OK, failed $FAIL"
