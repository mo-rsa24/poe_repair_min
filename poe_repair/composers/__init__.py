"""Thin composers — one entry per method.

Active composers:

  - ``mono``       : single CFG branch on the literal joint embedding e_J
                     (diagnostic ceiling).
  - ``poe``        : vanilla PoE ε̃_A + ε̃_B − ε_∅ (the CI baseline that fails).
  - ``direct_eps`` : direct-ε student wrapper used by group-A failure cases.

Each composer accepts a (PairSeedCell, MethodCtx). They do not own a
scheduler, model loading, or seed logic — those come from MethodCtx.
"""
