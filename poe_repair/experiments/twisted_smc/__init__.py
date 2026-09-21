"""Twisted sequential Monte Carlo on product-of-experts SDXL sampling.

A learned twist ``log psi_t(x_t) ~ log p_J,t(x_t) - log p_PoE,t(x_t)`` trained
contrastively (joint-prompt latents against PoE latents, both re-noised to the
same timestep), then used to reweight and resample particles of the plain PoE
sampler. No correction is ever added to the score: the sampler only chooses
among states the product already proposes.
"""
