"""Interaction-term scope: reading r_t out of the cache.

The sampler that *injects* r_t already exists as
``poe_repair.methods._sampling.run_teacher_residual`` (λ dose, correction
window, and the PMI identity check). This package does not duplicate it.
What lives here is the read side: loading cached residuals and computing

    r_t = ε̃_J − ε̃_PoE

the same way the sampler does, so cache-derived analyses and sampler runs
cannot drift apart.
"""

from poe_repair.experiments.interaction_term.cache import (  # noqa: F401
    load_cell,
    r_t,
)
