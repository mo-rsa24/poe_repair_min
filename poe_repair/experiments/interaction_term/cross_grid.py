"""The crossed grid: when the prompt acts, crossed with when the correction acts.

Two axes, deliberately separated, because collapsing them is what makes a
timing result unreadable:

  **Conditioning window** decides which steps the prompt acts on at all.
  Outside it the sampler runs unconditional.
  **Correction window** decides which steps the interaction term is injected
  on, within the steps that have a prompt.

The existing conditioning-window ablation moves only the first, and the timing
sweep moves only the second with the prompt on throughout. Crossing them is what
lets the correction's effect be read at a fixed conditioning schedule, instead of
confounding "the correction stopped acting" with "the prompt stopped acting".

Read the grid along a row or a column, never along a diagonal: two cells that
differ in both axes cannot say which axis caused the difference.

Three pairs, chosen to bracket how hard composition is: a_cat__x__a_dog in the
middle, a_frog__x__a_toad and a_leopard__x__a_jaguar as near-duplicate pairs
that compose badly.
"""

from __future__ import annotations

from dataclasses import dataclass

from poe_repair.experiments.interaction_term.window_grid import (  # noqa: F401
    FORK_STEP,
    NUM_STEPS,
    WIDTH,
)

PAIRS = (
    "a_cat__x__a_dog",
    "a_frog__x__a_toad",
    "a_leopard__x__a_jaguar",
)
SEEDS = (9, 10, 11)

# Every cell is sampled at this batch size, padded if the last group is short.
# Not a throughput choice: an A6000 is compute-bound from about four cells up,
# so this buys a few percent. It is fixed because the UNet returns slightly
# different numbers at different batch shapes, so a short final batch would put
# a systematic difference into whichever cells landed in it.
BATCH = 8


@dataclass(frozen=True)
class CondSchedule:
    """When the prompt acts. ``window=None`` means every step."""

    tag: str
    window: tuple[int, int] | None
    outside: bool = False

    def describe(self) -> str:
        if self.window is None:
            return "prompt on at every step"
        a, b = self.window
        if self.outside:
            return f"prompt off in steps {a}-{b}, on elsewhere"
        return f"prompt on in steps {a}-{b}, unconditional elsewhere"


# The baseline first: every other schedule is read against it.
COND_SCHEDULES = (
    CondSchedule("all",   None),
    CondSchedule("pre5",  (0, 5)),
    CondSchedule("pre10", (0, 10)),
    CondSchedule("pre20", (0, 20)),
    CondSchedule("mid",   (10, 20)),
    CondSchedule("post10", (10, NUM_STEPS)),   # unconditional first, then guided
)

# Correction windows crossed against every conditioning schedule. "off" injects
# nothing anywhere and "all" injects at every step; both are anchors rather than
# positions, so they are named rather than given step bounds.
CROSS_CORRECTIONS = (
    "off", "0-10", "10-20", "15-25", "20-30", "30-40", "all",
)


def dense_windows(width: int = WIDTH, stride: int = 1, n: int = NUM_STEPS):
    """Every placement of the fixed-width window, one step apart.

    The sparse stride-5 grid resolves position well enough for a rate; this is
    for looking at the pictures, where a one-step shift is the thing worth
    seeing.
    """
    return [(s, s + width) for s in range(0, n - width + 1, stride)]


def corr_tag(spec: str | tuple[int, int]) -> str:
    return spec if isinstance(spec, str) else f"{spec[0]}-{spec[1]}"


def cell_id(cond_tag: str, corr: str | tuple[int, int]) -> str:
    """The directory name for one cell. Both axes readable without a lookup."""
    return f"c{cond_tag}__r{corr_tag(corr)}"


def corr_window(spec: str | tuple[int, int]) -> tuple[tuple[int, int] | None, float]:
    """Turn a correction spec into (window, lambda_max).

    ``correction_window=None`` in the sampler means "λ applies at every step",
    which is the opposite of off, so "off" has to be expressed as λ=0 instead.
    """
    if spec == "off":
        return None, 0.0
    if spec == "all":
        return None, 1.0
    return (tuple(spec) if not isinstance(spec, str)
            else tuple(int(x) for x in spec.split("-"))), 1.0


def jobs(
    *, pairs=PAIRS, seeds=SEEDS, dense_stride: int = 1,
) -> list[dict]:
    """Every cell in blocks A and B, deduplicated, in a stable order.

    Block A is the cross: every conditioning schedule against every correction
    window. Block B is the dense timing strip at the baseline conditioning
    schedule. They overlap where block A's correction windows coincide with a
    dense placement, and the overlap is emitted once.
    """
    out: list[dict] = []
    seen: set[tuple[str, int, str]] = set()

    def add(pair, seed, cond: CondSchedule, corr, block):
        cid = cell_id(cond.tag, corr)
        key = (pair, seed, cid)
        if key in seen:
            return
        seen.add(key)
        window, lam = corr_window(corr)
        out.append({
            "pair": pair, "seed": int(seed), "cell_id": cid, "block": block,
            "cond_tag": cond.tag, "cond_window": cond.window,
            "cond_outside": cond.outside,
            "corr": corr_tag(corr), "corr_window": window, "lambda_max": lam,
        })

    base = COND_SCHEDULES[0]
    for pair in pairs:
        for seed in seeds:
            # Block B first, so the dense strip exists early if the run is cut
            # short: a complete strip at one conditioning schedule is readable,
            # a scattered fragment of the cross is not.
            for w in dense_windows(stride=dense_stride):
                add(pair, seed, base, w, "dense")
            for cond in COND_SCHEDULES:
                for corr in CROSS_CORRECTIONS:
                    add(pair, seed, cond, corr, "cross")
    return out
