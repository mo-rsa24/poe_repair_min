"""Thin composers — one entry per method.

Each composer accepts a (PairSeedCell, MethodCtx) and an optional
``anchor_source`` (``"literal"`` | ``"synth"``) selecting whether the joint
embedding is the literal e_J or the synthesised ê_J. Each composer runs
the published algorithm (faithful SDXL port, or unmodified call into the
``composition/`` reference) and writes one PNG + summary JSON.

Composers do NOT define their own scheduler, model loading, or seed logic
— those come from MethodCtx. They are deliberately tiny: their job is to
glue our infrastructure to a published method without embellishment.
"""
