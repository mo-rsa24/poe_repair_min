"""Cross-seed LoRA pooling experiment.

Tests whether a LoRA trained on a pool of cat×dog seeds encodes a
seed-invariant correction direction vs. a collection of seed-idiosyncratic
corrections that pooled training merely averages. Plan:
``.claude/plans/cross-seed-lora-pooling.md``.

This package is a *sibling* of ``poe_repair.experiments.one_pair_one_seed``. The
existing single-seed LoRA training path (and its reference artifact at
``artifacts/results/can-lora-learn-a-residual-that-corrects-poe/results-by-pair/a_cat__x__a_dog/seed_42/results/``) is untouched.
"""
