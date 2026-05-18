"""Residual-existence diagnostic — verifies r_t exists and is structured.

Eleven-point λ-sweep on a single (pair, seed) cell, with per-step residual
artifacts, distance-to-anchor curves, residual-norm trajectories, spatial
and directional diagnostics, and a self-consistency check that the
teacher residual ``Δ_t = ε̃_J − ε̃_PoE`` equals the algebraic PMI
rearrangement ``w · (ε_J + ε_∅ − ε_A − ε_B)`` to numerical precision.

Output: ``outputs/residual_diagnostics/existence/...`` — used by the
sibling ``clip_window`` sub-experiment.
"""

from __future__ import annotations

from poe_repair.experiments.residual_diagnostics.existence.main import main

__all__ = ["main"]
