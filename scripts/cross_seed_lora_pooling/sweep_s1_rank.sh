#!/usr/bin/env bash
# Sweep S1 — Rank sweep at k=8 (pooled only).
#
# Run only if Task C reads "pooled < per-seed" and you suspect capacity.
# Trains pooled-k8 at ranks ∈ {16, 32} (rank=8 is already in Task B);
# samples held-out seeds after each. Same train pool, only rank changes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}
CUDA=${CUDA_VISIBLE_DEVICES:-1}
EPOCHS=${EPOCHS:-1600}
RANKS=${RANKS:-"16 32"}
OUT_ROOT="$REPO_ROOT/outputs/cross_seed_lora_pooling/sweep_s1_rank"
mkdir -p "$OUT_ROOT"

cd "$REPO_ROOT"

for r in $RANKS; do
    run_id="k08_r${r}__ep${EPOCHS}"
    CUDA_VISIBLE_DEVICES=$CUDA "$PY" -m \
        poe_repair.experiments.cross_seed_lora_pooling.train_pooled \
        --k 8 --total-epochs "$EPOCHS" \
        --lora-rank "$r" --lora-alpha "$r" \
        --output-root "$OUT_ROOT" \
        --run-id "$run_id"

    run_dir="$OUT_ROOT/$run_id"
    ckpt=$(jq -r .path "$run_dir/checkpoints/latest.json")

    CUDA_VISIBLE_DEVICES=$CUDA "$PY" -m \
        poe_repair.experiments.cross_seed_lora_pooling.sample_heldout \
        --checkpoint "$ckpt" \
        --out-dir "$run_dir/samples/heldout"
done

echo "==> Sweep S1 complete."
