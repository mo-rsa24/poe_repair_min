"""Which pair donates its r_t for the wrong-pair control.

Plan 03's second control asks: if you inject *another* pair's correction,
norm-matched, does the compose rate still rise? If it does, the effect is not
about this pair's interaction term and the causal claim is in trouble.

The donor assignment has to be fixed and stated, not chosen per run. Two
reasons:

- **Reproducibility.** A random donor makes the control unrepeatable, and a
  donor chosen after seeing the result is not a control at all.
- **Similarity is a confound.** Donating cheetah's r_t to cougar tests almost
  nothing: they are the same kind of animal and the corrections may genuinely
  be interchangeable. The donor should be a pair with no shared concept, so
  that "it still worked" cannot be explained by the two pairs being alike.

So the map below pairs each slug with a deliberately dissimilar donor, fixed
here in source. Cross-checked to be token-disjoint from its recipient: no
animal word is shared between a pair and its donor.
"""

from __future__ import annotations

from poe_repair.experiments.interaction_term.cache import CACHE_ROOT

# Recipient -> donor. Deliberately across animal families in every case
# (canid <- cetacean, big cat <- bird, and so on), and token-disjoint.
DONOR = {
    "a_wolf__x__a_husky":           "a_dolphin__x__a_porpoise",
    "a_lion__x__a_tiger":           "a_crow__x__a_raven",
    "a_cheetah__x__a_cougar":       "a_turtle__x__a_tortoise",
    "a_horse__x__a_zebra":          "a_frog__x__a_toad",
    "a_donkey__x__a_pony":          "a_crocodile__x__an_alligator",
    "a_crocodile__x__an_alligator": "a_rabbit__x__a_hare",
    "a_rabbit__x__a_hare":          "a_lion__x__a_tiger",
    "a_dolphin__x__a_porpoise":     "a_horse__x__a_zebra",
    "a_crow__x__a_raven":           "a_wolf__x__a_husky",
    "a_gorilla__x__a_chimpanzee":   "a_goose__x__a_swan",
    "a_turtle__x__a_tortoise":      "a_cheetah__x__a_cougar",
    # held-out
    "a_leopard__x__a_jaguar":       "a_goose__x__a_swan",
    "a_frog__x__a_toad":            "a_horse__x__a_zebra",
    "an_eagle__x__a_hawk":          "a_donkey__x__a_pony",
    "a_seal__x__a_walrus":          "a_crow__x__a_raven",
    "a_goose__x__a_swan":           "a_gorilla__x__a_chimpanzee",
    "a_cow__x__a_buffalo":          "a_dolphin__x__a_porpoise",
    "a_cat__x__a_dog":              "a_turtle__x__a_tortoise",
    "an_elephant__x__a_penguin":    "a_cheetah__x__a_cougar",
}


def partner_for(pair_slug: str) -> str:
    """The fixed donor for this pair. Raises rather than guessing."""
    try:
        donor = DONOR[pair_slug]
    except KeyError:
        raise KeyError(
            f"no wrong-pair donor assigned for {pair_slug!r}. Add one to "
            "poe_repair/experiments/interaction_term/wrong_pair.py rather than "
            "picking at run time: an unrecorded donor makes the control "
            "unreproducible."
        ) from None
    if _shares_a_word(pair_slug, donor):
        raise ValueError(
            f"donor {donor!r} shares a concept with {pair_slug!r}; that is a "
            "confound, pick a token-disjoint donor")
    return donor


def _words(slug: str) -> set[str]:
    out: set[str] = set()
    for part in slug.split("__x__"):
        out |= {w for w in part.split("_") if w not in ("a", "an")}
    return out


def _shares_a_word(a: str, b: str) -> bool:
    return bool(_words(a) & _words(b))


def donor_seed_for(pair_slug: str, seed: int, *, root=CACHE_ROOT,
                   min_steps: int = 50) -> int:
    """The fixed donor seed for the wrong-seed control: seed + 4.

    The dose sweep runs seeds 9 to 12 and the cache holds full 50-step
    trajectories at seeds 9 to 16 for every sweep pair, so seed + 4 always
    lands on a fully cached run outside the sweep itself. Fixed here in source
    for the same reason as DONOR above: a donor chosen at run time is not a
    control. Raises rather than falling back, because a donor with fewer steps
    than the run would silently repeat its last vector.
    """
    donor = seed + 4
    for split in ("train", "heldout"):
        d = root / split / pair_slug / f"seed_{donor}" / "residuals"
        if len(list(d.glob("step_*.pt"))) >= min_steps:
            return donor
    raise FileNotFoundError(
        f"no full cached trajectory for {pair_slug!r} seed {donor} "
        f"(wanted >= {min_steps} residual steps). The wrong-seed control "
        "needs a complete donor run; cache it first."
    )


def first_cached_seed(pair_slug: str, *, root=CACHE_ROOT, min_steps: int = 2) -> int:
    """First seed of this pair with a real trajectory, not an eval stub."""
    for split in ("train", "heldout"):
        d = root / split / pair_slug
        if not d.is_dir():
            continue
        for sd in sorted(d.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
            if len(list((sd / "residuals").glob("step_*.pt"))) >= min_steps:
                return int(sd.name.split("_")[1])
    raise FileNotFoundError(f"no cached trajectory for {pair_slug!r}")
