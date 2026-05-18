"""Internal-force failure case — Mono-free PoE-internal corrective forces.

A repair attempt that does not invoke Mono at inference: at every
denoising step it constructs a corrective force from PoE's own UNet
outputs (per-concept eps + cross-attention maps), scales it to clear
the basin barrier measured by the residual-existence diagnostic, and
adds it to ``ε_PoE``. No 4th UNet branch, no ``e_J`` encoding, no
synthesiser at inference.

Two force variants are reported as failure cases alongside the group-A
architectural correctors:
  - ``overlap``   — attention-overlap repulsion (Force A).
  - ``alignment`` — score-alignment damping (Force B).

Output: ``outputs/internal_force_failure/...``.
"""

from __future__ import annotations

from poe_repair.experiments.internal_force_failure.main import main

__all__ = ["main"]
