#!/usr/bin/env bash
# Launch the authors'-settings generation on one free device. Usage: GPU=<idx> bash launch.sh
set -euo pipefail
ROOT=/datasets/mmolefe/poe_repair_min/pair_test
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=$ROOT/authors_settings
[ -x "$PY" ] || { echo "co3 python missing: $PY"; exit 2; }
used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$used" -lt 1024 ] || { echo "device $GPU has $used MiB in use, aborting"; exit 3; }
pct=$(df --output=pcent /datasets | tail -1 | tr -dc 0-9)
[ "$pct" -lt 90 ] || { echo "/datasets is ${pct}% full, aborting"; exit 4; }
echo "node=$(hostname) device=$GPU pid=$$ start=$(date -Is)"
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_CACHE=$ROOT/hf_cache
cd "$ROOT/code"
exec "$PY" medical_pair_generate_authors.py --out "$OUT" \
  --shard $ROOT/rg2_synth/data/train-00016-of-00033.parquet
