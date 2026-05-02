"""Pair × seed configuration for the minimal proof-of-mechanism.

Two pairs at one seed:
    - cat × dog                  — collision regime, PoE fails
    - butterfly × flower meadow  — cooperative regime, PoE works

Pilot data layout: data/pilot/seed_<n>/<slug>/
"""

PAIRS = [
    {
        "slug": "a_cat__x__a_dog",
        "prompt_a": "a cat",
        "prompt_b": "a dog",
        "regime": "collision",
    },
    {
        "slug": "a_butterfly__x__a_flower_meadow",
        "prompt_a": "a butterfly",
        "prompt_b": "a flower meadow",
        "regime": "cooperative",
    },
]

SEEDS = [42]
