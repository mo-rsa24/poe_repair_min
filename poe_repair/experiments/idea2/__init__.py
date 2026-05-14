"""Idea 2 — adaptive schedule with basin-proximity monitoring.

A schedule wrapper that decides *when* to fire an underlying corrective
force, based on a per-step basin-axis projection. Three trigger rules
shipped together (threshold / persistence / velocity), four force
sources supported (residual / force_a / force_b / clip).

The headline claim: same-or-better images at lower total injected
correction, by spending budget only on steps where the trajectory is
actively drifting toward PoE.
"""

from __future__ import annotations

from poe_repair.experiments.idea2.main import main

__all__ = ["main"]
