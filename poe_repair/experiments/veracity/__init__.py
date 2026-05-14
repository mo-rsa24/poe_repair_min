"""Veracity — scientific demonstration of the residual mechanism.

Eleven-point λ-sweep on a single (pair, seed) cell, with per-step residual
artefacts, distance-to-anchor curves, residual-norm trajectories, spatial
and directional diagnostics, and a self-consistency check that the
deployed teacher residual ``Δ_t = ε̃_J − ε̃_PoE`` equals the algebraic
PMI rearrangement ``w · (ε_J + ε_∅ − ε_A − ε_B)`` to numerical precision.

Stages:
  - ``sweep``    : run the eleven λ values + a Mono "reference" panel.
  - ``metrics``  : compute distance tables, residual stats, identity curve.
  - ``figures``  : render Figures 1–9 from the on-disk artefacts.
  - ``main``     : CLI orchestrator wiring the three stages.
"""

from __future__ import annotations

from poe_repair.experiments.veracity.main import main

__all__ = ["main"]
