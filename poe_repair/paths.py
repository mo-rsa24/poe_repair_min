"""Where every run family's bytes live, named once.

The problem this solves: 103 places in this codebase spell a path that exists on
*both* filesystems, and a relative path has no defined answer there. Thirty of them
name ``outputs/interaction_term/dose``, which is an empty directory in the repository
and 6.3G on ``/datasets``. Which one a process reads depends on where it is standing.

So a path is never spelled by hand again. Each run family has one constant here, the
constant carries the name the project uses for that family, and resolution against the
two filesystems happens in one place where it can be tested.

**A constant is a name, not a location.** Its value is where the bytes sit today. When
the retrofit moves a family, this file changes and nothing else does. That is the whole
point of the indirection, and it is why the names below are the new vocabulary even
though the directories still carry the old one.

Two roots:

``REPO_ROOT``    this checkout. Small things: indexes, manifests, symlink trees, the
                early runs.
``MOUNT_ROOT``   ``/datasets/mmolefe/poe_repair_min`` by default, overridable with
                ``POE_REPAIR_MOUNT``. The bulk: checkpoints, sample sweeps, caches.

Neither is authoritative in general, and assuming one is was wrong. Measured on
2026-08-23: under ``held-out-seeds`` the repository holds the one-seed pool, the
per-seed ceiling run, the prescreen and the held-out probes, while the mount holds the
four-seed pooled runs. The two file sets are disjoint. They are different experiments
that share a directory name, so the run tag decides the filesystem, not a rule.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "MOUNT_ROOT",
    "resolve",
    "roots_holding",
    "NOT_YET_PRODUCED",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mount_root() -> Path:
    env = os.environ.get("POE_REPAIR_MOUNT")
    if env:
        return Path(env).expanduser()
    return Path("/datasets/mmolefe/poe_repair_min")


REPO_ROOT = _repo_root()
MOUNT_ROOT = _mount_root()


# ---------------------------------------------------------------------------
# The run families.
#
# Left of the arrow is what the project calls the family. Right of it is where the
# bytes are today. When the retrofit moves a family, only the right-hand side moves.
#
# Names follow the walk of 2026-08-23: what was measured against what was varied.
# A family whose name was already honest keeps it.
# ---------------------------------------------------------------------------

# How much correction, and when it arrives
HOW_MUCH_CORRECTION_IS_NEEDED = "outputs/interaction_term/dose"
SAME_TOTAL_CORRECTION_DIFFERENT_WINDOW = "outputs/interaction_term/dose_matched"
WINDOW = "outputs/interaction_term/window"
SAMPLES_AS_THE_WINDOW_MOVES_ONE_STEP_AT_A_TIME = "outputs/interaction_term/cross"
COMPOSE_RATE_IN_THE_FIRST_WINDOW_ACROSS_TWELVE_SEEDS = "outputs/interaction_term/window_seeds"
SAMPLES_AS_THE_WINDOW_MOVES_AND_STRENGTH_GOES_PAST_ONE = "outputs/interaction_term/overcorrection_grid"

# What the correction is, and what it does inside the model
DECODED_PREDICTIONS_PER_STEP_FOR_EACH_EXPERT = "outputs/interaction_term/experts"
CONTENT_CHANGE_RELATIVE_TO_ATTENTION_CHANGE = "outputs/interaction_term/reprobe"
PREDICTED_CLEAN_IMAGE_PER_STEP = "outputs/interaction_term/xhat0_readback"
DIRECTION_WALL = "outputs/interaction_term/direction_wall"
NOISE_SLICE = "outputs/interaction_term/noise_slice"

# Analyses over cached trajectories. Dissolving into the question folders its sixteen
# figures answer; the constant stays until the move so nothing dangles meanwhile.
CACHE_ANALYSES = "outputs/interaction_term/cache_analyses"

# Training
TRAINING_RUN_SCORED_WHILE_IT_TRAINS = "outputs/interaction_term/live_curves_smoke_run"
DOES_THE_FIX_REACH_UNSEEN_PAIRS = "outputs/animals_compose_transfer"

# The earlier eras. Renamed off private labels and off stated claims.
CORRECTION_OUTSIDE_THE_UNET = "outputs/group_a_failure"
RESIDUAL_BETWEEN_MONO_AND_POE = "outputs/residual_diagnostics"
CFG_WINDOW_WITHOUT_LORA = "outputs/conditioning_window"
CFG_WINDOW_WITH_LORA = "outputs/conditioning_window_lora"
INTERNAL_FORCE_FAILURE = "outputs/internal_force_failure"  # declared, never produced
ATTENTION_MECHANISM = "outputs/attn_mechanism"

# Baselines and instruments
POE_BASELINE_SAMPLES = "outputs/poe"
COMPOSE_SCORER_VALIDATION = "outputs/compose_scorer"

# The pooled-adapter runs. The `rung` level is dropped: the level below it already
# carried the axis, and the number carried nothing.
ONE_PAIR_ONE_SEED = "artifacts/rung1-overfit/lora"
HELD_OUT_SEEDS = "artifacts/rung2-survive-noise/cross_seed"
HELD_OUT_SEEDS_INDEX = "outputs/cross_seed_lora_pooling"

# Cut by group, so out of scope for the animal-pair work and left where they are.
WITHIN_GROUP = "artifacts/rung3-group-wise/cross_pair/within_group"
ALL_GROUPS = "artifacts/rung4-scale/cross_pair/all_groups"
GROUP_POOL_CONFIGS = "outputs/cross_pair_lora_pooling"

# Caches
TRAINING_CACHE = "artifacts/caches/training_cache"
MANIFOLD_CACHE = "artifacts/caches/manifold_cache"


# Families whose code names an output directory that has never been written.
# `internal_force_failure` has a full package under `poe_repair/experiments/` and a
# writeup in `docs/results-archive/`, and `metrics.py:45` documents the layout it
# writes; no run has produced it on either filesystem. That is a fact worth keeping
# rather than a constant worth deleting, so it is stated here and the resolve check
# permits it. Anything not in this set that stops resolving is drift, and fails.
NOT_YET_PRODUCED = frozenset({"INTERNAL_FORCE_FAILURE"})


def roots_holding(rel: str | os.PathLike[str]) -> list[Path]:
    """Every root whose copy of ``rel`` exists, repository first.

    An empty list means nothing holds it. Two entries mean the bare relative path is
    ambiguous, which is the condition this module exists to make visible rather than
    silently pick a side of.
    """
    return [root for root in (REPO_ROOT, MOUNT_ROOT) if (root / rel).exists()]


def resolve(rel: str | os.PathLike[str], *, prefer: str = "mount") -> Path:
    """The path to ``rel`` on whichever filesystem holds it.

    ``prefer`` decides only when both hold it, and defaults to the mount because that
    is where the bulk sits. Where the two copies are different experiments rather than
    one copied twice, name the run tag in ``rel`` and this becomes unambiguous without
    anyone having to choose.

    Raises ``FileNotFoundError`` naming both roots tried, because a path that resolves
    nowhere should say so at the call site rather than fail later as an empty result.
    """
    if prefer not in ("mount", "repo"):
        raise ValueError(f"prefer must be 'mount' or 'repo', not {prefer!r}")

    found = roots_holding(rel)
    if not found:
        raise FileNotFoundError(
            f"{rel} exists under neither {REPO_ROOT} nor {MOUNT_ROOT}"
        )
    if len(found) == 1:
        return found[0] / rel
    return (MOUNT_ROOT if prefer == "mount" else REPO_ROOT) / rel
