"""Group A — external score-space correctors (A1 / A2 / A3).

Stage 3, Group A of [poe-correction-mvp.md]. A small standalone network
(latent-CNN / latent-UNet / frozen-feature-MLP) reads the current latent
plus joint conditioning and outputs an additive ε-correction. SDXL stays
frozen end-to-end.

The package mirrors [lora]'s artifact layout — probes every
``probe.every_epochs`` epochs across a λ-grid, decoded thumbnails per λ,
cumulative grid of (epochs × λ), where-applied overlay for the top λ.
"""
