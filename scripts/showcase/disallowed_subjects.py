"""Subjects that must never be rendered, scored, or shown in any sheet.

This is a hard constraint, not a preference, and it is enforced in code rather than kept as a note
because a note is something a later session forgets. Every script that renders a cell or builds a
sheet imports ``check`` and refuses on a match.

Two lists. ``BLOCKED`` is absolute: no rendering, no scoring, no appearance in any figure, ever.
``ASK_FIRST`` is not blocked outright but may not be used without an explicit decision recorded in
the pair ledger, so a borderline subject cannot enter a sheet by default.

Matching is on whole words after the slug or prompt is split on non-letters, so ``a_bathtub`` does
not match ``bat`` and ``a_chessboard`` does not match ``boa``.
"""
from __future__ import annotations

import re

BLOCKED = {
    # arachnids
    "spider", "spiders", "tarantula", "tarantulas", "arachnid", "arachnids",
    "scorpion", "scorpions", "mite", "mites", "tick", "ticks",
    # snakes
    "snake", "snakes", "serpent", "serpents", "cobra", "cobras", "viper", "vipers",
    "python", "pythons", "anaconda", "anacondas", "rattlesnake", "rattlesnakes",
    "adder", "adders", "mamba", "mambas", "boa",
    # crawling and burrowing
    "centipede", "centipedes", "millipede", "millipedes", "cockroach", "cockroaches",
    "roach", "roaches", "maggot", "maggots", "larva", "larvae", "grub", "grubs",
    "leech", "leeches", "worm", "worms", "slug", "slugs", "silverfish", "earwig",
    "earwigs", "weevil", "weevils", "louse", "lice", "flea", "fleas",
    # stinging and swarming
    "wasp", "wasps", "hornet", "hornets", "mosquito", "mosquitoes", "mosquitos",
    "locust", "locusts", "termite", "termites",
    # other commonly feared small creatures
    "moth", "moths", "beetle", "beetles", "mantis", "cricket", "crickets",
    "bat", "bats", "vermin", "parasite", "parasites",
}

ASK_FIRST = {
    "crocodile", "crocodiles", "alligator", "alligators", "gator", "lizard", "lizards",
    "iguana", "iguanas", "gecko", "geckos", "reptile", "reptiles",
    "toad", "toads", "frog", "frogs", "eel", "eels", "shark", "sharks",
    "rat", "rats", "mouse", "mice", "spiderweb", "web",
}


def words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", text.lower()) if w}


def check(text: str, *, allow_ask_first: bool = False) -> None:
    """Raise on a disallowed subject. ``text`` is a pair slug or a prompt.

    ``allow_ask_first`` is passed only by a caller that has an explicit recorded decision for that
    subject in the pair ledger. It never lifts ``BLOCKED``.
    """
    w = words(text)
    hit = sorted(w & BLOCKED)
    if hit:
        raise ValueError(
            "refusing %r: contains a permanently disallowed subject (%s). "
            "This subject may never be rendered, scored, or shown." % (text, ", ".join(hit))
        )
    if not allow_ask_first:
        ask = sorted(w & ASK_FIRST)
        if ask:
            raise ValueError(
                "refusing %r: contains a subject needing an explicit decision first (%s). "
                "Record the decision in the pair ledger, then pass allow_ask_first=True."
                % (text, ", ".join(ask))
            )


def is_allowed(text: str, *, allow_ask_first: bool = False) -> bool:
    try:
        check(text, allow_ask_first=allow_ask_first)
        return True
    except ValueError:
        return False
