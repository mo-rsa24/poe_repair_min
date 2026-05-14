"""C2 pre-committed pass/fail thresholds for Thread C diagnostics.

These are encoded *before* plotting so the verdict is mechanical, not
post-hoc. See phase0-consolidated-plan.md §7b additions C2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    d1a_window: tuple[int, int] = (5, 25)
    d1a_min_mean_cos: float = 0.85

    d1b_min_top3_share: float = 0.80

    d1c_window: tuple[int, int] = (5, 25)
    d1c_min_best_cos: float = 0.50

    d3_cooperative_window: tuple[int, int] = (5, 25)
    d3_cooperative_min_mean_cos: float = 0.50

    # D4-A: shared-mean ≥ zero + 0.7·(oracle − zero) on VQA min on ≥ 2/3 seeds
    # AND shared-mean produces `both_distinct` on ≥ 2/3 seeds. The fraction
    # is encoded as ``min_pass_count_out_of_total`` evaluated against the
    # number of seeds actually run.
    d4a_oracle_zero_fraction: float = 0.7
    d4a_min_seed_pass_fraction: float = 2.0 / 3.0

    # D4-A-t: same fraction, evaluated inside the commit-window panel.
    d4a_t_window_label: str = "commit"

    # D4-B: per-seed cos(seed, loo-mean) averaged in window ≥ this threshold
    # on the cooperative pair.
    d4b_window: tuple[int, int] = (5, 25)
    d4b_min_cooperative_mean_cos: float = 0.5


DEFAULT_THRESHOLDS = Thresholds()


def verdict_for_d1a(mean_cos: float, t: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    passed = mean_cos >= t.d1a_min_mean_cos
    return {
        "metric": "D1-A direction stability",
        "measured": {"mean_consecutive_cos_in_window": float(mean_cos)},
        "window_step_indices": list(t.d1a_window),
        "threshold": t.d1a_min_mean_cos,
        "passed": bool(passed),
    }


def verdict_for_d1b(top3_share: float, t: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    passed = top3_share >= t.d1b_min_top3_share
    return {
        "metric": "D1-B low-rank energy",
        "measured": {"top3_variance_share": float(top3_share)},
        "threshold": t.d1b_min_top3_share,
        "passed": bool(passed),
    }


def verdict_for_d1c(
    best_candidate_mean_cos: float,
    candidate_means: dict[str, float],
    t: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    passed = best_candidate_mean_cos >= t.d1c_min_best_cos
    return {
        "metric": "D1-C Mono-free basis alignment",
        "measured": {
            "best_candidate_mean_cos_in_window": float(best_candidate_mean_cos),
            "per_candidate_mean_cos_in_window": {
                k: float(v) for k, v in candidate_means.items()
            },
        },
        "window_step_indices": list(t.d1c_window),
        "threshold": t.d1c_min_best_cos,
        "passed": bool(passed),
    }


def verdict_for_d3_cooperative(
    mean_cos: float, t: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    passed = mean_cos >= t.d3_cooperative_min_mean_cos
    return {
        "metric": "D3 cross-seed cosine (cooperative pair)",
        "measured": {"mean_cross_pair_mean_cos_in_window": float(mean_cos)},
        "window_step_indices": list(t.d3_cooperative_window),
        "threshold": t.d3_cooperative_min_mean_cos,
        "passed": bool(passed),
        "note": (
            "C1: collision-pair D3 is reported for context but does NOT gate "
            "the structured-vs-noise verdict. Three cat × dog seeds fail in "
            "qualitatively different ways, so cross-seed cosine is "
            "over-determined to look low."
        ),
    }


def verdict_for_d4b(
    cooperative_mean_cos: float | None,
    t: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    if cooperative_mean_cos is None:
        return {
            "metric": "D4-B direction-vs-magnitude (cooperative pair)",
            "measured": {"mean_cos_in_window": None},
            "window_step_indices": list(t.d4b_window),
            "threshold": t.d4b_min_cooperative_mean_cos,
            "passed": None,
            "note": (
                "Cooperative-pair cache missing — D4-B verdict deferred "
                "until butterfly × meadow seeds are materialised."
            ),
        }
    passed = cooperative_mean_cos >= t.d4b_min_cooperative_mean_cos
    return {
        "metric": "D4-B direction-vs-magnitude (cooperative pair)",
        "measured": {"mean_cos_in_window": float(cooperative_mean_cos)},
        "window_step_indices": list(t.d4b_window),
        "threshold": t.d4b_min_cooperative_mean_cos,
        "passed": bool(passed),
    }


def verdict_for_d4a(
    *,
    seeds: list[int],
    shared_mean_vqa_pass_count: int,
    shared_mean_detection_pass_count: int,
    t: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    n = max(1, len(seeds))
    threshold_n = max(1, int(round(t.d4a_min_seed_pass_fraction * n)))
    vqa_pass = shared_mean_vqa_pass_count >= threshold_n
    det_pass = shared_mean_detection_pass_count >= threshold_n
    return {
        "metric": "D4-A shared-mean substitution",
        "measured": {
            "vqa_pass_count": int(shared_mean_vqa_pass_count),
            "detection_both_distinct_count": int(shared_mean_detection_pass_count),
            "seeds_total": int(n),
            "seed_pass_threshold": int(threshold_n),
        },
        "oracle_zero_fraction": t.d4a_oracle_zero_fraction,
        "passed": bool(vqa_pass and det_pass),
    }


def verdict_for_d4a_t(
    *,
    seeds: list[int],
    commit_window_shared_mean_pass_count: int,
    t: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    n = max(1, len(seeds))
    threshold_n = max(1, int(round(t.d4a_min_seed_pass_fraction * n)))
    passed = commit_window_shared_mean_pass_count >= threshold_n
    return {
        "metric": "D4-A-t windowed substitution (commit window)",
        "measured": {
            "shared_mean_pass_count_in_commit_window": int(commit_window_shared_mean_pass_count),
            "seeds_total": int(n),
            "seed_pass_threshold": int(threshold_n),
        },
        "window_label": t.d4a_t_window_label,
        "passed": bool(passed),
    }


def overall_verdict(verdicts: list[dict]) -> dict:
    pass_count = sum(1 for v in verdicts if v.get("passed"))
    fail_count = sum(1 for v in verdicts if v.get("passed") is False)
    if fail_count == 0 and pass_count == len(verdicts):
        label = "STRUCTURED"
    elif pass_count == 0:
        label = "NOISE"
    else:
        label = "MIXED"
    return {
        "label": label,
        "passed_count": pass_count,
        "failed_count": fail_count,
        "total": len(verdicts),
    }
