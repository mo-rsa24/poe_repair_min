"""Schedule grammar for the CFG conditioning-window ablation.

A schedule is a length-``n`` boolean mask. ``mask[i] = True`` means the
conditional branch is applied at DDIM step ``i``; ``False`` means the
sampler collapses to ε_uncond at that step.

``step_index`` runs 0 → n-1 (noisy → clean), so ``prefix(k)`` enables CFG
for the k earliest/noisiest steps and ``suffix(k)`` enables it for the k
latest/cleanest steps.

Schedules are grouped by ``family``:

  - ``prefix``    — ``[0, k)``
  - ``suffix``    — ``[n-k, n)``
  - ``window``    — ``[a, b)``
  - ``punctate``  — short pulse(s) at single positions
  - ``sanity``    — all-on / all-off equivalence references

``STANDARD_SUITE`` enumerates ~50 schedules covering the four production
families. ``SANITY_SUITE`` is the two sanity schedules. ``SMOKE`` selects
a 2-schedule subset for fast wiring checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Schedule:
    """A named on/off mask for the CFG conditional branch."""
    id: str
    family: str
    mask: tuple[bool, ...]


def all_on(n: int) -> tuple[bool, ...]:
    return tuple(True for _ in range(n))


def all_off(n: int) -> tuple[bool, ...]:
    return tuple(False for _ in range(n))


def prefix(k: int, n: int) -> tuple[bool, ...]:
    """``mask[i] = True`` iff ``i < k``."""
    return tuple(i < k for i in range(n))


def suffix(k: int, n: int) -> tuple[bool, ...]:
    """``mask[i] = True`` iff ``i >= n - k``."""
    return tuple(i >= n - k for i in range(n))


def window(a: int, b: int, n: int) -> tuple[bool, ...]:
    """``mask[i] = True`` iff ``a <= i < b``. Half-open."""
    return tuple(a <= i < b for i in range(n))


def punctate(positions: Sequence[int], width: int, n: int) -> tuple[bool, ...]:
    """1–3-step pulses at each position p (covers ``[p, p+width)``)."""
    pulse_steps: set[int] = set()
    for p in positions:
        for off in range(width):
            j = p + off
            if 0 <= j < n:
                pulse_steps.add(j)
    return tuple(i in pulse_steps for i in range(n))


def mask_to_str(mask: Sequence[bool]) -> str:
    """Render a mask as a fixed-width binary string ("10101...")."""
    return "".join("1" if x else "0" for x in mask)


def num_on(mask: Sequence[bool]) -> int:
    return int(sum(bool(x) for x in mask))


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


N_DEFAULT = 50


def _make_standard_suite(n: int = N_DEFAULT) -> list[Schedule]:
    schedules: list[Schedule] = []

    # Prefix-only: on for [0, k).
    for k in [5, 10, 15, 20, 25, 30, 40, 50]:
        schedules.append(Schedule(
            id=f"prefix_k{k:02d}",
            family="prefix",
            mask=prefix(k, n),
        ))

    # Suffix-only: on for [n-k, n).
    for k in [5, 10, 15, 20, 25, 30, 40, 50]:
        schedules.append(Schedule(
            id=f"suffix_k{k:02d}",
            family="suffix",
            mask=suffix(k, n),
        ))

    # Single contiguous window [a, b).
    window_pairs = [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
        (25, 30), (30, 40), (40, 50),
        (0, 10), (0, 20), (10, 30), (20, 40), (5, 15),
    ]
    for a, b in window_pairs:
        schedules.append(Schedule(
            id=f"window_{a:02d}_{b:02d}",
            family="window",
            mask=window(a, b, n),
        ))

    # Punctate pulses: minimum-dose probes at a few timesteps.
    for p in [3, 7, 12, 17, 22, 27, 32, 37, 42, 47]:
        for w in [1, 2, 3]:
            schedules.append(Schedule(
                id=f"pulse_t{p:02d}_w{w}",
                family="punctate",
                mask=punctate([p], w, n),
            ))

    return schedules


def _make_sanity_suite(n: int = N_DEFAULT) -> list[Schedule]:
    return [
        Schedule(id="sanity_all_on",  family="sanity", mask=all_on(n)),
        Schedule(id="sanity_all_off", family="sanity", mask=all_off(n)),
    ]


STANDARD_SUITE: list[Schedule] = _make_standard_suite()
SANITY_SUITE: list[Schedule] = _make_sanity_suite()

SMOKE_IDS = ["prefix_k10", "window_10_20"]


def select(names: Sequence[str], suite: Sequence[Schedule] | None = None) -> list[Schedule]:
    """Filter ``suite`` (default STANDARD_SUITE) to the given ids, preserving order."""
    pool = list(suite or STANDARD_SUITE)
    by_id = {s.id: s for s in pool}
    missing = [n for n in names if n not in by_id]
    if missing:
        known = ", ".join(by_id.keys())
        raise KeyError(f"unknown schedule id(s): {missing}; known: {known}")
    return [by_id[n] for n in names]


FAMILY_ORDER = ("prefix", "suffix", "window", "punctate", "sanity")
