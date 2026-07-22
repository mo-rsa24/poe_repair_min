#!/usr/bin/env bash
# Plan 16 — build the within-group cross-pair training caches.
#
# Walks the five per-group pair_pool.yaml files, looks up each slug in
# pair_prompts.yaml, and calls scripts/build_training_cache.py for every
# (pair, seed) cell that isn't already on disk.
#
# Idempotent: build_training_cache.py is itself idempotent (per-cell
# sentinel + --overwrite flag), and this driver skips slug/seed combos
# whose seed_<N> directory already exists.
#
# Override via env vars:
#   PLAN16_GROUPS="g6 g1"                                # default: "g1 g2 g3 g4 g6"
#   SEEDS="1 2 3 4 5 6 7 8 9 10 11 12"            # default: seeds 1..12
#   PY=...
#   CUDA_VISIBLE_DEVICES=0
#   POE_REPAIR_TRAINING_CACHE=/datasets/.../training_cache
#   DRY_RUN=1                                     # echo commands only
#   OVERWRITE=1                                   # rebuild existing cells
set -euo pipefail

PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
PLAN16_GROUPS="${PLAN16_GROUPS:-g1 g2 g3 g4 g6}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

WITHIN_ROOT="outputs/cross_pair_lora_pooling/within_group"
PAIR_PROMPTS="outputs/cross_pair_lora_pooling/pair_prompts.yaml"

if [[ ! -f "$PAIR_PROMPTS" ]]; then
  echo "[err] missing $PAIR_PROMPTS" >&2; exit 2
fi

CACHE_ROOT="${POE_REPAIR_TRAINING_CACHE:-/datasets/mmolefe/poe_repair_min/outputs/training_cache}"
echo "[info] cache root: $CACHE_ROOT"

# Helper: emit "<pair_slug> <prompt_a> <prompt_b> <joint_prompt>" lines
# for every slug in $GROUP_DIR/pair_pool.yaml (train + heldout combined),
# read from pair_prompts.yaml.
emit_pair_specs() {
  local pair_pool_path="$1"
  "$PY" - "$pair_pool_path" "$PAIR_PROMPTS" <<'PY'
import sys, yaml
pair_pool_path, prompts_path = sys.argv[1], sys.argv[2]
with open(pair_pool_path) as fh:
    pool = yaml.safe_load(fh)
with open(prompts_path) as fh:
    prompts = yaml.safe_load(fh)
slugs = list(pool.get("train") or []) + list(pool.get("heldout") or [])
missing = [s for s in slugs if s not in prompts]
if missing:
    raise SystemExit(f"[err] missing pair_prompts entries: {missing}")
for s in slugs:
    p = prompts[s]
    # Tab-separated, one slug per line.
    print(f"{s}\t{p['prompt_a']}\t{p['prompt_b']}\t{p['joint_prompt']}")
PY
}

total_planned=0
total_skipped=0
total_built=0

for g in $PLAN16_GROUPS; do
  PAIR_POOL="$WITHIN_ROOT/$g/pair_pool.yaml"
  if [[ ! -f "$PAIR_POOL" ]]; then
    echo "[err] $g: missing $PAIR_POOL" >&2; exit 2
  fi
  echo "===== $g ====="

  while IFS=$'\t' read -r slug A B J; do
    [[ -z "$slug" ]] && continue
    for s in $SEEDS; do
      total_planned=$((total_planned+1))
      cell_dir="$CACHE_ROOT/heldout/$slug/seed_$s"
      if [[ "${OVERWRITE:-0}" != "1" && -d "$cell_dir" ]]; then
        total_skipped=$((total_skipped+1))
        printf '[skip] %-50s seed=%d (exists)\n' "$slug" "$s"
        continue
      fi
      cmd=( "$PY" -m scripts.build_training_cache \
            --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J" \
            --seed "$s" --split heldout )
      [[ "${OVERWRITE:-0}" == "1" ]] && cmd+=( --overwrite )
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[dry ] %s\n' "${cmd[*]}"
      else
        printf '[run ] %-50s seed=%d\n' "$slug" "$s"
        "${cmd[@]}"
        total_built=$((total_built+1))
      fi
    done
  done < <(emit_pair_specs "$PAIR_POOL")
done

echo "------"
echo "planned=$total_planned  skipped=$total_skipped  built=$total_built"
