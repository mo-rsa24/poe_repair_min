#!/usr/bin/env bash
# Phase C — kick off all GPU sweeps required for the Phase 0 Veracity
# figure set: anti-corroboration controls (3 seeds × 2 controls), CFG
# sweep (5 guidance scales), and the cross-attention capture run.
#
# Sequential by design — each sweep allocates ~5GB GPU and we don't want
# OOM contention with the butterfly × meadow run already in flight.
#
# Logs land under outputs/_phase_c_logs/<run_name>.log; tail any of them
# to follow progress. Re-running this script is idempotent — the
# teacher_residual composer skips runs whose PNG already exists.

set -euo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
LOG_DIR="${REPO}/outputs/_phase_c_logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

run_one() {
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label}.log"
    echo "===== [$(date +%H:%M:%S)] starting ${label} =====" | tee -a "${logfile}"
    if "${PY}" -m poe_repair.experiments.veracity "$@" >> "${logfile}" 2>&1; then
        echo "===== [$(date +%H:%M:%S)] DONE ${label} =====" | tee -a "${logfile}"
    else
        echo "===== [$(date +%H:%M:%S)] FAILED ${label} (exit $?) =====" | tee -a "${logfile}"
        return 1
    fi
}

CONTROL_SEEDS=(42 4 123)
CFG_VALUES=(1.0 3.0 5.0 10.0)  # 7.5 already covered by main veracity exp

# 1. Self-pair control: cat × cat, 3 seeds.
for seed in "${CONTROL_SEEDS[@]}"; do
    run_one "self_pair_seed${seed}" \
        --pair "a cat|a cat" --seed "${seed}" \
        --exp-name veracity_self_pair --skip-figures
done

# 2. Disjoint control: cat × car, 3 seeds.
for seed in "${CONTROL_SEEDS[@]}"; do
    run_one "disjoint_seed${seed}" \
        --pair "a cat|a car" --seed "${seed}" \
        --exp-name veracity_disjoint --skip-figures
done

# 3. CFG sweep on cat × dog seed 42 (excluding 7.5, which lives in
#    outputs/veracity already).
for cfg in "${CFG_VALUES[@]}"; do
    suffix=${cfg/./p}
    run_one "cfg_${suffix}" \
        --pair "a cat|a dog" --seed 42 \
        --guidance-scale "${cfg}" \
        --exp-name "veracity_cfg_${suffix}" --skip-figures
done

# 4. (Cross-attention capture cut in v3 — replaced by App-B' detection-based
#    failure-mode classification, which works directly off cached PoE PNGs
#    and doesn't need a separate sweep.)

echo "===== [$(date +%H:%M:%S)] PHASE C COMPLETE =====" | tee -a "${LOG_DIR}/_summary.log"
