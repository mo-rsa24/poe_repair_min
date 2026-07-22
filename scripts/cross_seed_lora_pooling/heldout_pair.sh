#!/usr/bin/env bash
# Plan-10 held-out-pair driver — within-group, cross-pair LoRA transfer.
#
# For each studied group (G1..G4, G6), load that group's pooled LoRA
# (trained on the representative pair) and sample it on the four
# held-out seeds of a sibling pair from the same group. Vanilla PoE on
# the same sibling cells is the comparison panel (rendered by
# build_eval_cache.py at cache-build time).
#
# Sibling pairs come from plan 10's "Representative pairs" table:
#   G1  a_dolphin__x__an_ocean_wave        → a_polar_bear__x__an_iceberg
#   G2  a_dog__x__oil_painting_style       → a_cat__x__charcoal_drawing_style
#   G3  a_mailbox__x__a_snowfield          → a_fire_hydrant__x__a_snowfield
#   G4  a_typewriter__x__a_cactus          → a_drum_set__x__a_snowman
#   G6  a_cat__x__a_dog                    → a_wolf__x__a_husky
#
# Per group we resolve the pooled checkpoint via the per-pair run dir's
# checkpoints/latest.json (written by train_pooled). For G6 the legacy
# (non-per-pair) run dir at task_b_learning_curve/k04__ep2000_resumed is
# used by default — override with G6_CHECKPOINT if a per-pair k08__ep1600
# run exists.
#
# Outputs land at:
#   outputs/cross_seed_lora_pooling/<train-pair>/heldout_pair/<sibling>/
#       sample_seed_{09,10,11,12}.png
#       manifest.json
#
# Env vars:
#   CUDA_VISIBLE_DEVICES   default 1
#   SEEDS                  default "9,10,11,12"
#   LORA_GROUPS            default "G1 G2 G3 G4 G6"
#   OUT_ROOT               default $REPO_ROOT/outputs/cross_seed_lora_pooling
#   POE_REPAIR_TRAINING_CACHE  required for sample_heldout (CellPath lookup)
#   G{N}_CHECKPOINT        override the resolved checkpoint for group N
#   DRY_RUN=1              print commands instead of running them
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
CUDA=${CUDA_VISIBLE_DEVICES:-1}
SEEDS=${SEEDS:-"9,10,11,12"}
LORA_GROUPS=${LORA_GROUPS:-"G1 G2 G3 G4 G6"}
OUT_ROOT=${OUT_ROOT:-"$REPO_ROOT/outputs/cross_seed_lora_pooling"}
DRY_RUN=${DRY_RUN:-0}

cd "$REPO_ROOT"

# Per-group config: TRAIN_PAIR | SIBLING | A | B | J.
group_spec () {
    case "$1" in
        G1) echo "a_dolphin__x__an_ocean_wave|a_polar_bear__x__an_iceberg|a polar bear|an iceberg|a polar bear and an iceberg" ;;
        G2) echo "a_dog__x__oil_painting_style|a_cat__x__charcoal_drawing_style|a cat|charcoal drawing style|a cat in charcoal drawing style" ;;
        G3) echo "a_mailbox__x__a_snowfield|a_fire_hydrant__x__a_snowfield|a fire hydrant|a snowfield|a fire hydrant in a snowfield" ;;
        G4) echo "a_typewriter__x__a_cactus|a_drum_set__x__a_snowman|a drum set|a snowman|a drum set and a snowman" ;;
        G6) echo "a_cat__x__a_dog|a_wolf__x__a_husky|a wolf|a husky|a wolf and a husky" ;;
        *)  echo "" ;;
    esac
}

read_latest () {
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$1"
}

# Default checkpoint resolver. G1-G4 use the per-pair run dir; G6 uses
# the legacy non-per-pair k04__ep2000_resumed run unless overridden.
default_ckpt () {
    local group="$1" train_pair="$2"
    case "$group" in
        G6)
            local legacy="/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed/checkpoints/lora_step_100000.pt"
            local per_pair_dir="$REPO_ROOT/outputs/cross_seed_lora_pooling/$train_pair/task_b_learning_curve/k08__ep1600/checkpoints/latest.json"
            if [ -f "$per_pair_dir" ]; then
                read_latest "$per_pair_dir"
            else
                echo "$legacy"
            fi ;;
        *)
            local k08="$REPO_ROOT/outputs/cross_seed_lora_pooling/$train_pair/task_b_learning_curve/k08__ep1600/checkpoints/latest.json"
            local k04="$REPO_ROOT/outputs/cross_seed_lora_pooling/$train_pair/task_b_learning_curve/k04__ep2000/checkpoints/latest.json"
            local k04_ds="/datasets/mmolefe/poe_repair_min/outputs/cross_seed_lora_pooling/$train_pair/task_b_learning_curve/k04__ep2000/checkpoints/latest.json"
            if   [ -f "$k08" ];    then read_latest "$k08"
            elif [ -f "$k04" ];    then read_latest "$k04"
            elif [ -f "$k04_ds" ]; then read_latest "$k04_ds"
            else echo ""
            fi ;;
    esac
}

if [ -z "${POE_REPAIR_TRAINING_CACHE:-}" ]; then
    echo "warn: POE_REPAIR_TRAINING_CACHE is unset; sample_heldout will fall back to DEFAULT_CACHE_ROOT" >&2
fi

run_group () {
    local group="$1"
    local spec; spec="$(group_spec "$group")"
    if [ -z "$spec" ]; then
        echo "skip: unknown group $group" >&2
        return 0
    fi

    local train_pair sibling A B J
    IFS='|' read -r train_pair sibling A B J <<< "$spec"

    local override_var="${group}_CHECKPOINT"
    local ckpt="${!override_var:-}"
    if [ -z "$ckpt" ]; then
        ckpt="$(default_ckpt "$group" "$train_pair")"
    fi
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "skip $group: no checkpoint found for train_pair=$train_pair (resolved=\"$ckpt\")" >&2
        echo "      set $override_var=/path/to/lora_step_*.pt to force" >&2
        return 0
    fi

    local out_dir="$OUT_ROOT/$train_pair/heldout_pair/$sibling"
    mkdir -p "$out_dir"

    echo "==> $group: $train_pair --[lora]--> $sibling"
    echo "    ckpt   : $ckpt"
    echo "    out    : $out_dir"
    echo "    seeds  : $SEEDS"

    local cmd=(
        "$PY" -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout
        --checkpoint "$ckpt"
        --heldout-pair "$sibling"
        --prompt-a "$A" --prompt-b "$B" --joint-prompt "$J"
        --seeds "$SEEDS"
        --out-dir "$out_dir"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '    DRY_RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
    else
        CUDA_VISIBLE_DEVICES=$CUDA "${cmd[@]}"
    fi
}

for g in $LORA_GROUPS; do
    run_group "$g"
done

echo "==> heldout_pair sweep complete."
echo "    next: render per-group contact sheets, e.g."
echo "    \$PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet \\"
echo "        --pair <train-pair> --task heldout_pair"
