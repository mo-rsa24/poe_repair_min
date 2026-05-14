#!/usr/bin/env bash
# App-E only — 10 window-localised injection runs at λ=1 on cat × dog seed 42.
# Each window is width 5; starts at {0, 5, 10, ..., 45}.
#
# Outputs land at outputs/veracity_window_injection/pairs/a_cat__x__a_dog/seed_42/
#   teacher_residual_const_lam100_w{start}-{end}/teacher_residual_const_lam100_w{start}-{end}.png
#
# Wallclock: ~12 min per window on a single GPU; ~2h total.
# Idempotent: the teacher_residual composer skips runs whose PNG already exists.

set -euo pipefail

REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
LOG_DIR="${REPO}/outputs/_app_e_logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

LOGFILE="${LOG_DIR}/window_injection.log"
echo "===== [$(date +%H:%M:%S)] starting App-E window-injection sweep =====" \
    | tee -a "${LOGFILE}"

"${PY}" - <<'PYEOF' >> "${LOGFILE}" 2>&1
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

echo "===== [$(date +%H:%M:%S)] DONE App-E window-injection =====" \
    | tee -a "${LOGFILE}"
