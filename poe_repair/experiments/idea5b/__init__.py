"""Idea 5b — CLIP-guided PoE repair.

A Mono-free repair: at every step in a configured correction window,
backpropagate the cosine similarity of the Tweedie x̂_0 against a CLIP
text embedding (e.g. "a cat and a dog") to obtain a gradient w.r.t. the
latent, then add that gradient as a corrective term to ε_PoE. The
diffusion model's joint-prompt UNet branch is never invoked at inference;
the corrective signal comes from a separate vision-language model.

Stages:
  - sweep    : capacity check + 11-point α-multiplier sweep.
  - metrics  : distance tables, gradient stats, four-method comparison.
  - figures  : Figs 1–9 (mirror veracity / idea1) + N1, N3, N4.
  - main     : CLI orchestrator.
"""

from __future__ import annotations

from poe_repair.experiments.idea5b.main import main

__all__ = ["main"]
