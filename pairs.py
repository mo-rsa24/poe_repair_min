"""Pair × seed configuration for the LoRA-checkpoint state.

One pair, one seed:
    - cat × dog (collision regime, PoE fails)  @ seed 42

Pilot data layout: data/pilot/seed_<n>/<slug>/
"""

PAIRS = [
    {
        "slug": "a_cat__x__a_dog",
        "prompt_a": "a cat",
        "prompt_b": "a dog",
        "regime": "collision",
    },
]

SEEDS = [42]
CONTROL_SEEDS = [42]
