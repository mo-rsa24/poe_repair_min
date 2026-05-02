#!/usr/bin/env bash
# Train the embedding synthesizer (text-only, MSE+cosine on (e_A, e_B, e_empty) -> e_J).
#
# The training data (text-encoder outputs over the pair pool) is precomputed
# once and cached under outputs/synthesizer/cache/. The text encoders are then
# released and the loop becomes a pure MLP train, so VRAM is mostly used for
# the cached tensors plus optimizer state.
#
# Wall-clock breakdown after the cache is built:
#   - precompute (one-time): ~10-30 min, dominated by 3 SDXL text-encoder
#     forward passes per pair.
#   - training: a few minutes for 100k steps at batch_size=512.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ "${GPU_RESIDENT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--gpu-resident)
fi

python -m poe_repair.embeddings.train \
    --arch residual_mlp \
    --num-steps "${NUM_STEPS:-100000}" \
    --batch-size "${BATCH_SIZE:-512}" \
    --output-name residual_mlp \
    "${EXTRA_ARGS[@]}" \
    "$@"
