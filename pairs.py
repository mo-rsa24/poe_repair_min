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
    {
        "slug": "a_cat__x__a_cat",
        "prompt_a": "a cat",
        "prompt_b": "a cat",
        "regime": "self_pair_control",
    },
    {
        "slug": "a_cat__x__a_car",
        "prompt_a": "a cat",
        "prompt_b": "a car",
        "regime": "disjoint_control",
    },
]

SEEDS = [42]
CONTROL_SEEDS = [42, 4, 123]
