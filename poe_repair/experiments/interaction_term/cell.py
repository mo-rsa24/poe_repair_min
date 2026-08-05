"""Build a PairSeedCell from a cached pair slug.

The rest of the codebase constructs cells from an explicit (prompt_a, prompt_b)
pair. The interaction-term command lines are driven by a slug instead, because
that is how the cache and the plan files name things. Every cached cell carries
its own ``meta.json`` with the original prompts, so the slug is enough: no
separate slug-to-prompt table to keep in sync.
"""

from __future__ import annotations

import json

from poe_repair.experiments._eval_common import cell_for
from poe_repair.experiments.interaction_term.cache import cell_dir
from poe_repair.runtime import PairSeedCell


def prompts_for_slug(pair_slug: str, seed: int) -> tuple[str, str]:
    """Recover (prompt_a, prompt_b) from a cached cell's meta.json."""
    meta = json.loads((cell_dir(pair_slug, seed) / "meta.json").read_text())
    a, b = meta["pair"]
    return str(a), str(b)


def cell_from_slug(pair_slug: str, seed: int, **kw) -> PairSeedCell:
    """The cell for a cached (pair, seed), with prompts read from the cache.

    Raises FileNotFoundError if the pair is not cached, which is the honest
    failure: without the cache there is no r_t to inject.
    """
    prompt_a, prompt_b = prompts_for_slug(pair_slug, seed)
    cell = cell_for(prompt_a, prompt_b, seed, **kw)
    if cell.pair_slug != pair_slug:
        raise ValueError(
            f"slug mismatch: cache says {pair_slug!r} but slugify gives "
            f"{cell.pair_slug!r} for ({prompt_a!r}, {prompt_b!r})"
        )
    return cell
