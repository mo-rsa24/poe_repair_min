"""Held-out validation pairs for synthesizer training.

These are *not* used at inference time — only for validation cosine on text-
encoder embeddings during training. Two regimes:

- ``COLLISION_PAIRS``  — same-supercategory G6-style pairs. The synthesizer's
  primary deployment target.
- ``COOPERATIVE_PAIRS`` — different-supercategory cooperative pairs, drawn
  from G1-style co-occurrence. Sanity that validation cosine isn't only high
  on collisions.

Used by ``embeddings/dataset.py`` for training oversampling and by
``embeddings/train.py`` for held-out validation cosine.
"""

COLLISION_PAIRS: list[tuple[str, str]] = [
    ("a cat", "a dog"),
    ("a lion", "a horse"),
    ("a cow", "a horse"),
    ("a lion", "a dog"),
    ("a tiger", "a dog"),
    ("a cat", "a horse"),
    ("a wolf", "a horse"),
    ("a cat", "a lion"),
    ("a dog", "a horse"),
    ("a mug", "a wine glass"),
]

COOPERATIVE_PAIRS: list[tuple[str, str]] = [
    ("a butterfly", "a flower meadow"),
    ("a camel", "a desert landscape"),
    ("a deer", "a forest clearing"),
    ("a dolphin", "an ocean wave"),
    ("a duck", "a pond"),
    ("a flamingo", "a lagoon"),
    ("a horse", "a grassy field"),
    ("a polar bear", "an iceberg"),
    ("a sailboat", "a harbor"),
]

ALL_HOLDOUT_PAIRS: list[tuple[str, str]] = COLLISION_PAIRS + COOPERATIVE_PAIRS
