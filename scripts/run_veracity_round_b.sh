#!/usr/bin/env bash
# Round B — GPU sweeps for the v2 corroboration round.
#
# 1. App-A variance bands need 3 new sweeps:
#    - cat × dog seed 123  (also used by Round A3 anti-corroboration check)
#    - butterfly × meadow seed 4
#    - butterfly × meadow seed 123
# 2. App-E window-localised injection: 10 single-trajectory runs at λ=1
#    with correction_window={[0,5], [5,10], …, [45,50]} on cat × dog seed 42.
#
# Sequential. Logs at outputs/_round_b_logs/<run_name>.log.
# Idempotent — teacher_residual composer skips runs whose PNG already exists.

set -euo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
LOG_DIR="${REPO}/outputs/_round_b_logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

run_one() {
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label}.log"
    echo "===== [$(date +%H:%M:%S)] starting ${label} =====" | tee -a "${logfile}"
    if "${PY}" -m poe_repair.experiments.veracity "$@" >> "${logfile}" 2>&1; then
        echo "===== [$(date +%H:%M:%S)] DONE ${label} =====" | tee -a "${logfile}"
    else
        echo "===== [$(date +%H:%M:%S)] FAILED ${label} (continuing) =====" | tee -a "${logfile}"
    fi
}

# 1a. cat × dog seed 123 — full 11-λ sweep so it's usable for both Round A3
#     and App-A variance bands. ~15 min.
run_one "cat_dog_seed123" \
    --pair "a cat|a dog" --seed 123 \
    --exp-name veracity --skip-figures

# 1b/c. butterfly × meadow seeds 4 and 123 — full 11-λ sweeps for App-A.
for seed in 4 123; do
    run_one "butterfly_seed${seed}" \
        --pair "a butterfly|a flower meadow" --seed "${seed}" \
        --exp-name veracity --skip-figures
done

# 2. App-E window-localised injection sweep. λ=1 only is implicit because
#    we use --only-lambdas 1.0; correction_window is set by a small inline
#    Python helper instead of a CLI flag (the existing veracity main does
#    not expose --correction-window). We invoke teacher_residual.run
#    directly via the experiments.veracity.sweep module.
#
# Window starts: 0, 5, 10, …, 45 (each width 5). 10 windows.
echo "===== [$(date +%H:%M:%S)] starting App-E window-injection sweep =====" \
    | tee -a "${LOG_DIR}/window_injection.log"
"${PY}" - <<'PYEOF' >> "${LOG_DIR}/window_injection.log" 2>&1
from poe_repair.run import make_ctx
from poe_repair.composers import teacher_residual as cmp_tr
from poe_repair.experiments.veracity.sweep import make_cell

cell = make_cell("a cat", "a dog", 42)
ctx = make_ctx()
EXP_NAME = "veracity_window_injection"
WINDOW_WIDTH = 5
STARTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

for start in STARTS:
    end = start + WINDOW_WIDTH
    method_name = cmp_tr.method_name_for(
        lambda_schedule="constant",
        lambda_max=1.0,
        correction_window=(start, end),
    )
    print(f"[window_injection] window=[{start},{end}]  ->  {method_name}", flush=True)
    cmp_tr.run(
        cell, ctx,
        lambda_schedule="constant",
        lambda_max=1.0,
        correction_window=(start, end),
        save_residuals=False,
        save_x0_estimates=False,
        save_trajectory=False,
        exp_name=EXP_NAME,
        overwrite=False,
    )
print("[window_injection] done", flush=True)
PYEOF
echo "===== [$(date +%H:%M:%S)] DONE window_injection =====" \
    | tee -a "${LOG_DIR}/window_injection.log"

echo "===== [$(date +%H:%M:%S)] ROUND B COMPLETE =====" | tee -a "${LOG_DIR}/_summary.log"
