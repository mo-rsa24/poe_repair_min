#!/usr/bin/env bash
# Run the three caveat scripts in the order: tempered -> cheap ULA sweep
# -> verify (on the cheapest-default config; rerun manually with the best
# config from the sweep if you want a tighter cross-seed check).
#
# Usage:
#   bash scripts/run_mcmc_caveats.sh                 # tempered + cheap sweep + verify
#   bash scripts/run_mcmc_caveats.sh --full-sweep    # also do the 60-config grid
#   bash scripts/run_mcmc_caveats.sh --skip-sweep    # tempered + verify only
#
# Honours $CUDA_VISIBLE_DEVICES; defaults to GPU 1 (the idle one) if unset.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$HOME/miniforge3/envs/co3/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

FULL_SWEEP=0
SKIP_SWEEP=0
for arg in "$@"; do
    case "$arg" in
        --full-sweep) FULL_SWEEP=1 ;;
        --skip-sweep) SKIP_SWEEP=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "=== [1/3] tempered β schedules ==="
$PY scripts/tempered_schedules_seed42.py --include-baseline

if [ "$SKIP_SWEEP" -eq 0 ]; then
    if [ "$FULL_SWEEP" -eq 1 ]; then
        echo "=== [2/3] full ULA sweep (5 × 3 × 4 = 60 configs) ==="
        $PY scripts/sweep_mcmc_corrector.py --include-baseline
    else
        echo "=== [2/3] cheap ULA sweep (3 × 2 × 2 = 12 configs) ==="
        $PY scripts/sweep_mcmc_corrector.py \
            --ss-base 1e-4 1e-3 1e-2 \
            --corrector-steps 5 25 \
            --windows 5 25 0 50 \
            --include-baseline
    fi
fi

echo "=== [3/3] cross-seed verify (seeds 4, 42, 123, default config) ==="
$PY scripts/verify_mcmc_seeds.py --seeds 4 42 123

echo
echo "All done. Outputs:"
echo "  outputs/tempered_schedules/a_cat__x__a_dog/seed_42/"
echo "  outputs/mcmc_sweep/a_cat__x__a_dog/seed_42/"
echo "  outputs/mcmc_verify/a_cat__x__a_dog/seed_{4,42,123}/"
