"""Which pairs belong to the experiment, read from its pool file.

The training cache directory holds more than one experiment's pairs. As of
2026-08-05 `training_cache/train/` has 18 pair folders of which 11 belong to
animals-compose-transfer, and `heldout/` has 58 of which 8 do. Nothing on disk
marks which is which, so any script that scans the directory silently mixes
experiments together and reports a number about neither.

This module makes the pair set explicit. Scripts take `--pool <file>` and get
exactly the pairs that experiment declared, in the roles it declared them.

The held-out list is NOT uniform, and that matters for how results are read:

    transfer   unseen blend pairs. The actual transfer test.
    reference  a_cat__x__a_dog, the known-failure case, blends by default.
    control    dissimilar pairs that compose fine anyway (do-no-harm check).

Averaging all three together answers no question cleanly, so `heldout_roles()`
keeps them separate and `heldout()` defaults to transfer pairs only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_POOL = Path("outputs/animals_compose_transfer/pair_pool.yaml")

# The pool file records roles in trailing comments rather than structured
# fields, so they are matched here. A pair whose comment says nothing
# recognisable is treated as a transfer pair and named in `unlabelled`, rather
# than silently dropped or silently counted.
_ROLE_MARKERS = (
    ("transfer", "transfer test"),
    ("reference", "known-failure"),
    ("control", "control"),
)


@dataclass(frozen=True)
class Pool:
    """One experiment's declared pairs, with held-out roles kept apart."""

    path: Path
    train: list[str]
    transfer: list[str]
    reference: list[str]
    control: list[str]
    unlabelled: list[str] = field(default_factory=list)

    def heldout(self, *, roles: tuple[str, ...] = ("transfer",)) -> list[str]:
        """Held-out pairs for the given roles. Transfer only by default."""
        out: list[str] = []
        for r in roles:
            out.extend(getattr(self, r))
        return out

    def summary(self) -> str:
        s = (f"pool {self.path}: {len(self.train)} train, "
             f"{len(self.transfer)} transfer, {len(self.reference)} reference, "
             f"{len(self.control)} control")
        if self.unlabelled:
            s += f", {len(self.unlabelled)} unlabelled (counted as transfer)"
        return s


def _roles_from_comments(path: Path) -> dict[str, str]:
    """Map pair slug -> role, read from the trailing comment on each line."""
    roles: dict[str, str] = {}
    in_heldout = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("heldout:"):
            in_heldout = True
            continue
        if stripped and not stripped.startswith(("-", "#")) and ":" in stripped:
            in_heldout = False
        if not in_heldout or not stripped.startswith("- "):
            continue
        body = stripped[2:]
        slug, _, comment = body.partition("#")
        slug = slug.strip()
        low = comment.lower()
        for role, marker in _ROLE_MARKERS:
            if marker in low:
                roles[slug] = role
                break
    return roles


def load_pool(path: Path | str = DEFAULT_POOL) -> Pool:
    """Read an experiment's pair pool. Raises if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no pair pool at {path}. Scripts need an explicit pair list: the "
            "cache directory holds more than one experiment's pairs, so "
            "scanning it mixes them together."
        )
    data = yaml.safe_load(path.read_text())
    train = [str(p) for p in data.get("train", [])]
    heldout = [str(p) for p in data.get("heldout", [])]
    roles = _roles_from_comments(path)

    buckets: dict[str, list[str]] = {"transfer": [], "reference": [], "control": []}
    unlabelled: list[str] = []
    for slug in heldout:
        role = roles.get(slug)
        if role is None:
            unlabelled.append(slug)
            buckets["transfer"].append(slug)
        else:
            buckets[role].append(slug)

    return Pool(path=path, train=train, unlabelled=unlabelled, **buckets)
