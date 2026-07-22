#!/usr/bin/env bash
# Build the five sibling-pair eval-only caches at seeds {9..12} for
# plan-10's held-out-pair driver. Idempotent — re-running skips cells
# that already have meta.json + the requested refs.
#
# Pairs (matching the sibling column in plan 10's "Representative
# pairs" table):
#   G1  a_polar_bear__x__an_iceberg
#   G2  a_cat__x__charcoal_drawing_style
#   G3  a_fire_hydrant__x__a_snowfield
#   G4  a_drum_set__x__a_snowman
#   G6  a_wolf__x__a_husky
#
# Env:
#   CUDA_VISIBLE_DEVICES   default 1
#   SEEDS                  default "9 10 11 12"
#   LORA_GROUPS            default "G1 G2 G3 G4 G6"
#   SKIP_REFS=1            do not render PoE/Mono PNGs (~1 s/cell instead of ~75 s)
#   DRY_RUN=1              print invocations only
#   POE_REPAIR_TRAINING_CACHE  required (cache root)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
CUDA=${CUDA_VISIBLE_DEVICES:-1}
SEEDS=${SEEDS:-"9 10 11 12"}
LORA_GROUPS=${LORA_GROUPS:-"G1 G2 G3 G4 G6"}
SKIP_REFS=${SKIP_REFS:-0}
DRY_RUN=${DRY_RUN:-0}

cd "$REPO_ROOT"

if [ -z "${POE_REPAIR_TRAINING_CACHE:-}" ]; then
    echo "error: POE_REPAIR_TRAINING_CACHE must be set" >&2
    exit 2
fi

# Per-group sibling-pair spec: A | B | J. Slug is computed by slugify
# inside build_eval_cache; we mirror the plan-10 table here for sanity.
sibling_spec () {
    case "$1" in
        G1) echo "a polar bear|an iceberg|a polar bear and an iceberg" ;;
        G2) echo "a cat|charcoal drawing style|a cat in charcoal drawing style" ;;
        G3) echo "a fire hydrant|a snowfield|a fire hydrant in a snowfield" ;;
        G4) echo "a drum set|a snowman|a drum set and a snowman" ;;
        G6) echo "a wolf|a husky|a wolf and a husky" ;;
        *)  echo "" ;;
    esac
}

skip_flag=""
[ "$SKIP_REFS" = "1" ] && skip_flag="--skip-refs"

for g in $LORA_GROUPS; do
    spec="$(sibling_spec "$g")"
    if [ -z "$spec" ]; then
        echo "skip: unknown group $g" >&2
        continue
    fi
    IFS='|' read -r A B J <<< "$spec"
    echo "==> $g: $A × $B  →  $J"
    for s in $SEEDS; do
        cmd=(
            "$PY" scripts/build_eval_cache.py
            --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J"
            --seed "$s"
        )
        [ -n "$skip_flag" ] && cmd+=("$skip_flag")
        if [ "$DRY_RUN" = "1" ]; then
            printf '    DRY_RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
        else
            CUDA_VISIBLE_DEVICES=$CUDA "${cmd[@]}"
        fi
    done
done

echo "==> sibling caches built. Verify:"
for g in $LORA_GROUPS; do
    spec="$(sibling_spec "$g")"
    [ -z "$spec" ] && continue
    IFS='|' read -r A B J <<< "$spec"
    slug="$("$PY" -c "from poe_repair.experiments._eval_common import slugify; import sys; print(slugify(sys.argv[1], sys.argv[2]))" "$A" "$B")"
    echo "  $g  $POE_REPAIR_TRAINING_CACHE/heldout/$slug"
done
