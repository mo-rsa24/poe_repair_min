#!/usr/bin/env bash
# Grow the working-Mono pool by generating new seeds for all 5 pairs.
# Only PoE + Mono(literal e_J) are run — CO3 and sched-M2 are skipped
# because the helper-network training only needs PoE and a clean Mono
# image per cell. Roughly 60s per cell (vs 210s for the full e1 grid).
#
# Two workers, one per CUDA device:
#   GPU 0 : a cat × a dog, a cow × a horse, a lion × a horse  (3 pairs)
#   GPU 1 : a lion × a dog, a tiger × a dog                   (2 pairs)
#
# After both workers finish, build_dataset.py is rerun with the extended
# SEEDS list, and build_label_strips.py regenerates the per-pair strips
# so you can label the new cells.
#
# Usage:
#   scripts/gen_extra_seeds.sh                  # default: seeds 5 6 7
#   NEW_SEEDS="5 6 7 8 9" scripts/gen_extra_seeds.sh
#   scripts/gen_extra_seeds.sh --overwrite      # force regenerate
#
# Extra args after the env-var are passed through to the inner driver.

set -euo pipefail

PYTHON="${PYTHON:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
NEW_SEEDS="${NEW_SEEDS:-5 6 7}"

cd "$(dirname "$0")/.."

LOG_DIR="logs/gen_extra_seeds"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_GPU0="${LOG_DIR}/${TS}_gpu0.log"
LOG_GPU1="${LOG_DIR}/${TS}_gpu1.log"

echo "[gen] new seeds: $NEW_SEEDS"

echo "[gen] launching GPU 0 (cat-dog, cow-horse, lion-horse) -> $LOG_GPU0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/_gen_poe_mono.py \
    --pairs "a cat|a dog" "a cow|a horse" "a lion|a horse" \
    --seeds $NEW_SEEDS \
    "$@" >"$LOG_GPU0" 2>&1 &
PID_GPU0=$!

echo "[gen] launching GPU 1 (lion-dog, tiger-dog) -> $LOG_GPU1"
CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/_gen_poe_mono.py \
    --pairs "a lion|a dog" "a tiger|a dog" \
    --seeds $NEW_SEEDS \
    "$@" >"$LOG_GPU1" 2>&1 &
PID_GPU1=$!

echo "[gen] waiting on pids $PID_GPU0, $PID_GPU1"
EXIT=0
if ! wait "$PID_GPU0"; then
    echo "[gen] GPU 0 worker FAILED — see $LOG_GPU0"
    EXIT=1
fi
if ! wait "$PID_GPU1"; then
    echo "[gen] GPU 1 worker FAILED — see $LOG_GPU1"
    EXIT=1
fi

if [[ $EXIT -ne 0 ]]; then
    echo "[gen] one or more workers failed; logs in $LOG_DIR"
    exit 1
fi

echo "[gen] both workers done; updating SEEDS in build_dataset.py"
"$PYTHON" - <<EOF
from pathlib import Path
import re
existing = [1, 2, 3, 4, 42]
new = [int(s) for s in "$NEW_SEEDS".split()]
seeds = sorted(set(existing) | set(new))
p = Path("scripts/build_dataset.py")
text = p.read_text()
text = re.sub(
    r"^SEEDS: list\[int\] = .*$",
    f"SEEDS: list[int] = {seeds}",
    text, count=1, flags=re.M,
)
p.write_text(text)
print(f"  SEEDS now: {seeds}")
EOF

echo "[gen] rebuilding dataset/ symlinks"
"$PYTHON" scripts/build_dataset.py

echo "[gen] rebuilding label strips and template"
"$PYTHON" scripts/build_label_strips.py

echo "[gen] done. Inspect dataset/strips/<pair>.png and add new-seed labels"
echo "      to dataset/labels.json (keep existing labels intact)."
