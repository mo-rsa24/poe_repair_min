"""Shared utilities for experiments.

- ``cell_for``: minimal PairSeedCell factory using fallback init latents.
- ``slugify``: filesystem-friendly pair slug.
- ``HEADLINE_PAIR``: the pair the active experiments target.
"""

from __future__ import annotations

import re

from poe_repair.config import RunConfig
from poe_repair.runtime import PairSeedCell


HEADLINE_PAIR = ("a cat", "a dog")


def slugify(prompt_a: str, prompt_b: str) -> str:
    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")
    return f"{_clean(prompt_a)}__x__{_clean(prompt_b)}"


def cell_for(
    prompt_a: str, prompt_b: str, seed: int,
    *, height: int = 1024, width: int = 1024,
    regime: str = "collision",
) -> PairSeedCell:
    cfg = RunConfig()
    slug = slugify(prompt_a, prompt_b)
    pair_dir = cfg.paths.pilot_dir / f"seed_{seed}" / slug
    return PairSeedCell(
        pair_dir=pair_dir,
        pair_slug=slug,
        prompt_a=prompt_a, prompt_b=prompt_b,
        seed=seed, regime=regime,
        height=height, width=width,
        grid_assets={},
    )
