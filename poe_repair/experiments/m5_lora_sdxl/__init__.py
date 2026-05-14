"""M5 — LoRA on SDXL UNet cross-attention for guided-residual prediction.

Phase 2 of phase0-consolidated-plan §3. Trains a rank-8 LoRA on the
``attn2`` cross-attention projections (``to_q``, ``to_k``, ``to_v``) of the
SDXL UNet to predict the guided residual ``Δ_t = ε̃_J − ε̃_PoE`` on a
single (pair, seed) cell. Periodic inference probes over a λ-grid score
the result against §4's detection + VQA protocol; figures and artifacts
are logged to Weights & Biases.
"""
