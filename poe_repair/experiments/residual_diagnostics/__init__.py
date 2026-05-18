"""Residual diagnostics — characterise the guided Mono–PoE residual r_t.

Two sub-experiments sharing the same λ-sweep substrate:

  - ``existence``    : verifies the residual exists and is structured;
                       runs the algebraic PMI identity check; renders
                       Fig 1 / Fig 4 / App-A / App-B'.
  - ``clip_window``  : CLIP-as-diagnostic on commitment window; when does
                       CLIP first separate PoE from Mono, and is there a
                       still-correctable interval before commitment?

Shared modules:
  - ``sweep``   : 11-point λ-sweep on the headline cell, caching per-step
                  residuals + decoded images.
  - ``metrics`` : distance tables, residual stats, CLIP / detection / VQA
                  caches reused by LoRA and group_a_failure as well.

Entry points::

    python -m poe_repair.experiments.residual_diagnostics                 # both
    python -m poe_repair.experiments.residual_diagnostics.existence       # one
    python -m poe_repair.experiments.residual_diagnostics.clip_window     # one
"""

from __future__ import annotations
