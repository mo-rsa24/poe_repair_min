"""Idea 5a — CLIP-as-diagnostic.

A passive observer experiment: at sparse intervals along three reference
trajectories on cat × dog seed 42 (PoE, basin-transition λ=0.6, Mono),
compute Tweedie x̂_0 estimates, decode them through SDXL's VAE, and
score them against several CLIP text targets.

The questions we answer:
  - At which denoising step does CLIP first separate PoE from Mono?
  - Does CLIP cleanly distinguish "cat and a dog" from "cat" / "dog" on
    the Mono trajectory?
  - Is there a step window where CLIP is informative *and* the trajectory
    is still correctable (i.e. before commitment)?

If yes-yes-yes, Idea 5b (CLIP-guided corrective sampler) is viable.
If any answer is no, we pivot.

No new sampler. Pure post-hoc analysis on artefacts already produced by
the veracity sweep (`outputs/veracity/...`).
"""

from __future__ import annotations

from poe_repair.experiments.idea5a.main import main

__all__ = ["main"]
