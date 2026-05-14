"""Diagnostic 0 — single-cell overfit + replay.

Trains M2a and M2b on (cat × dog, seed 42) only, then replays at
inference on the same cell. Tests whether the residual can be memorised
and deployed at all. See ``main.py`` for orchestration.
"""
