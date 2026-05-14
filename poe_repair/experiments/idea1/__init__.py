"""Idea 1 — PoE-internal corrective forces.

A Mono-free repair experiment: at every denoising step, construct a
corrective force from PoE's own UNet outputs (per-concept eps and
cross-attention maps), scale it to clear the basin barrier measured by
veracity, and add it to ``ε_PoE``. No 4th UNet branch, no ``e_J``
encoding, no synthesiser at inference.

Two force variants are shipped together:
  - ``overlap``   — attention-overlap repulsion (Force A).
  - ``alignment`` — score-alignment damping (Force B).

Stages:
  - ``sweep``    : capacity check + 11-point strength sweep per force.
  - ``metrics``  : distance tables, force stats, method comparison.
  - ``figures``  : Figs 1–9 (mirror veracity) + N1–N4 (method-specific).
  - ``main``     : CLI orchestrator.
"""

from __future__ import annotations

from poe_repair.experiments.idea1.main import main

__all__ = ["main"]
