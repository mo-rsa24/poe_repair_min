"""CLIP commitment-window diagnostic — when does CLIP separate PoE from Mono?

Passive-observer experiment. Reads cached artifacts from the sibling
``existence`` sub-experiment, reconstructs Tweedie x̂_0 at chosen step
indices along three reference trajectories (PoE λ=0, mid-λ basin
transition, Mono λ=1), decodes through SDXL's VAE, and scores against
CLIP text targets.

The questions it answers:
  - At which denoising step does CLIP first separate PoE from Mono?
  - Does CLIP cleanly distinguish "cat and dog" from "cat" / "dog" alone
    on the Mono trajectory?
  - Is there a step window where CLIP is informative *and* the
    trajectory is still correctable (before commitment)?

No new sampler. Reads from ``outputs/residual_diagnostics/existence/...``.
"""

from __future__ import annotations

from poe_repair.experiments.residual_diagnostics.clip_window.main import main

__all__ = ["main"]
