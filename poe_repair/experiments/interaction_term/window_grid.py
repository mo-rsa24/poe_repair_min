"""The timing sweep's grid: which windows, which pairs, which seeds.

One place, in source, so the settings show up in a diff if they move after the
answer is known. The runner, the scorer, and the inspector all read this rather
than each hard-coding their own list, which is how a figure ends up plotting a
grid nobody actually ran.

The window is a half-open range of denoising step indices. Step 0 is the
noisiest, step 49 the cleanest. Inside the window the correction is injected at
full dose; outside it, nothing is injected.

Why width 10 at stride 5, and not something picked from the correction's size:
``scripts/window_width.py`` shows ‖r_t‖ is nearly flat over the run (each tenth
carries 15-22% of the total), so the narrowest band holding half the correction
is 25 steps. At width 25 every placement contains the fork step and the curve
cannot help but come out flat. Width 10 gives nine placements, each a fifth of
the run, with the fork step inside only two of them (10-20 and 15-25), so the
experiment can come out either way.
"""

from __future__ import annotations

NUM_STEPS = 50

# Chosen for resolution, not for energy. See the module docstring and
# scripts/window_width.py.
WIDTH = 10
STRIDE = 5

# The step where the broken and working paths pull apart, median over 19 cached
# 50-step cells, from outputs/interaction_term/cache_analyses/fork_curve.json.
# An independent estimate of the same moment this sweep measures: if the peak
# lands here the two agree, and if it does not that disagreement is the finding.
FORK_STEP = 16

# The same eight held-out pairs and four seeds the dose sweep used, so the
# timing curve and the dose curve describe one population of cells.
PAIRS = (
    "a_leopard__x__a_jaguar",
    "a_frog__x__a_toad",
    "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus",
    "a_goose__x__a_swan",
    "a_cow__x__a_buffalo",
    "a_cat__x__a_dog",
    "an_elephant__x__a_penguin",
)
SEEDS = (9, 10, 11, 12)


def windows(
    *, width: int = WIDTH, stride: int = STRIDE, n: int = NUM_STEPS,
) -> list[tuple[int, int]]:
    """Every [start, end) placement of the fixed-width window, in step order.

    The last placement ends exactly at ``n``, so no window runs off the end and
    every window has the same width. With the defaults: (0,10), (5,15) ...
    (40,50), nine of them.
    """
    return [(s, s + width) for s in range(0, n - width + 1, stride)]


def centre(window: tuple[int, int]) -> float:
    """The step a window is plotted at."""
    return (window[0] + window[1]) / 2.0


def contains_fork(window: tuple[int, int], *, fork: int = FORK_STEP) -> bool:
    """Does this window cover the step where the two paths pull apart?"""
    return window[0] <= fork < window[1]


# F4g/F4h: a growing window anchored at one end, instead of a fixed width
# slid along the run. F4a asks whether a narrow band suffices and where it has
# to sit; these ask how much of the trajectory, counted from one end, needs
# the correction on. Same PAIRS and SEEDS as the fixed-width grid, so all
# three curves describe one population of cells.
CUTOFFS = (10, 20, 30, 40, 50)


def prefix_windows(cutoffs: tuple[int, ...] = CUTOFFS) -> list[tuple[int, int]]:
    """Correct 0..c, plain PoE from c to the end. c=NUM_STEPS corrects every step.

    c=10 is (0, 10), the same window F4a's own leftmost cell already used, so a
    sweep over these windows reuses those images instead of resampling them.
    """
    return [(0, c) for c in cutoffs]


def suffix_windows(
    cutoffs: tuple[int, ...] = CUTOFFS, *, n: int = NUM_STEPS,
) -> list[tuple[int, int]]:
    """Plain PoE for 0..c, correct c..end. c=NUM_STEPS corrects no step.

    ``correction_window=(c, n)`` requires ``c < n``; a plain-PoE-everywhere
    point can't be written that way. c=NUM_STEPS is written as ``(n, n+10)``
    instead: a window entirely past the last step index, so lambda is forced
    to 0 at every step. This is the same "window past the end" trick
    ``interaction_term_window.py``'s own identity check already uses to
    represent off without leaving the windowed code path.
    """
    return [(c, n) if c < n else (n, n + 10) for c in cutoffs]
