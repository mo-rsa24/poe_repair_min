"""Idea 2 — basin-proximity monitor + trigger rules.

A *schedule* utility that decides, at every denoising step, whether the
underlying sampler should fire a corrective force. It does *not*
compute the force itself; the inner sampler (idea1, idea5b, veracity)
provides the force, this module provides the timing.

Components:

  - ``BasinMonitor`` — wraps two saved trajectories (PoE and Mono).
    ``project(x_t, step_index)`` returns a scalar in roughly ``[0, 1]``:
    0 = "I am where PoE was at this step", 1 = "I am where Mono was".

  - Three ``Trigger`` classes (threshold / persistence / velocity).
    All take a history of recent projections and return a bool.

  - ``AdaptiveSchedule`` — bundles a monitor + a trigger + a small
    history buffer. ``alpha(step_index, x_t, base_alpha)`` returns the
    α(t) the inner sampler should use this step.

The history is owned by the schedule, not by the sampler, so no
modification to the sampler's per-step state is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Basin monitor
# ---------------------------------------------------------------------------


@dataclass
class BasinMonitor:
    """Project the current latent onto the line between PoE and Mono templates.

    ``x_t_poe`` and ``x_t_mono`` are ``(T+1, B, C, H, W)`` tensors saved
    by veracity at λ=0 and λ=1 respectively. We use them as
    step-indexed templates: at step ``s``, the basin axis is
    ``x_t_mono[s] − x_t_poe[s]``.
    """

    x_t_poe: torch.Tensor      # (T+1, ...)
    x_t_mono: torch.Tensor     # (T+1, ...)

    def project(self, *, x_t: torch.Tensor, step_index: int) -> float:
        if step_index >= self.x_t_poe.shape[0]:
            # Off the end of the saved templates — assume neutral.
            return 0.5
        x_poe = self.x_t_poe[step_index].float()
        x_mono = self.x_t_mono[step_index].float()
        axis = (x_mono - x_poe).flatten()
        offset = (x_t.detach().cpu() - x_poe).flatten().float()
        denom = float((axis * axis).sum().item())
        if denom <= 1e-12:
            return 0.5
        return float((offset * axis).sum().item() / denom)


def load_basin_monitor(
    *, pair_slug: str, seed: int, output_root: Path,
) -> BasinMonitor | None:
    """Load templates from veracity's λ=0 and λ=1 saved trajectories.

    Returns ``None`` if either trajectory is missing — callers should
    raise a friendly error in that case (run veracity first).
    """
    veracity_root = output_root / "veracity" / "pairs" / pair_slug / f"seed_{seed}"
    poe_path = (
        veracity_root / "teacher_residual_const_lam000" / "latent_trajectory.pt"
    )
    mono_path = (
        veracity_root / "teacher_residual_const_lam100" / "latent_trajectory.pt"
    )
    if not poe_path.exists() or not mono_path.exists():
        return None
    poe = torch.load(poe_path, map_location="cpu", weights_only=False)
    mono = torch.load(mono_path, map_location="cpu", weights_only=False)
    return BasinMonitor(
        x_t_poe=poe["trajectories"],
        x_t_mono=mono["trajectories"],
    )


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


class Trigger:
    """Common interface for trigger rules.

    Subclasses implement ``fire(history)`` where ``history`` is the list
    of all projection readings observed so far (most recent at the end).
    """

    name: str = "base"

    def fire(self, history: Sequence[float]) -> bool:        # pragma: no cover
        raise NotImplementedError


@dataclass
class ThresholdTrigger(Trigger):
    """Fire iff the latest projection is below ``theta``."""

    theta: float
    name: str = "threshold"

    def fire(self, history: Sequence[float]) -> bool:
        if not history:
            return False
        return float(history[-1]) < float(self.theta)


@dataclass
class PersistenceTrigger(Trigger):
    """Fire iff the last ``K`` readings are *all* below ``theta``.

    With ``K=1`` reduces to ``ThresholdTrigger``.
    """

    theta: float
    K: int = 3
    name: str = "persistence"

    def fire(self, history: Sequence[float]) -> bool:
        K = max(1, int(self.K))
        if len(history) < K:
            return False
        recent = history[-K:]
        return all(float(r) < float(self.theta) for r in recent)


@dataclass
class VelocityTrigger(Trigger):
    """Fire iff the latest is below ``theta`` AND below ``history[-lookback-1]``.

    "Sliding further toward PoE" — only fires while drifting wrong.
    """

    theta: float
    lookback: int = 2
    name: str = "velocity"

    def fire(self, history: Sequence[float]) -> bool:
        L = max(1, int(self.lookback))
        if len(history) <= L:
            # need at least one earlier reading to compare against
            if not history:
                return False
            return float(history[-1]) < float(self.theta)
        latest = float(history[-1])
        earlier = float(history[-L - 1])
        return latest < float(self.theta) and latest < earlier


def make_trigger(
    *,
    rule: str,
    theta: float,
    persistence_K: int = 3,
    velocity_lookback: int = 2,
) -> Trigger:
    if rule == "threshold":
        return ThresholdTrigger(theta=theta)
    if rule == "persistence":
        return PersistenceTrigger(theta=theta, K=persistence_K)
    if rule == "velocity":
        return VelocityTrigger(theta=theta, lookback=velocity_lookback)
    raise ValueError(
        f"unknown trigger rule {rule!r}; expected one of "
        "{'threshold','persistence','velocity'}"
    )


# ---------------------------------------------------------------------------
# Adaptive schedule (bundle)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSchedule:
    """Per-step decision-maker. Owns its own history buffer."""

    monitor: BasinMonitor
    trigger: Trigger
    history: list[float] = field(default_factory=list)
    fired_steps: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.history = []
        self.fired_steps = []

    def alpha(
        self, *, step_index: int, x_t: torch.Tensor, base_alpha: float,
    ) -> tuple[float, float]:
        """Return ``(alpha_t, projection_t)``.

        ``alpha_t`` is ``base_alpha`` if the trigger fires, else 0.0.
        ``projection_t`` is the raw monitor reading (kept for figures).
        """
        proj = self.monitor.project(x_t=x_t, step_index=step_index)
        self.history.append(float(proj))
        fire = self.trigger.fire(self.history)
        if fire:
            self.fired_steps.append(int(step_index))
        return (float(base_alpha) if fire else 0.0, float(proj))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_adaptive_schedule(
    *,
    pair_slug: str,
    seed: int,
    output_root: Path,
    rule: str,
    theta: float,
    persistence_K: int = 3,
    velocity_lookback: int = 2,
) -> AdaptiveSchedule:
    monitor = load_basin_monitor(
        pair_slug=pair_slug, seed=seed, output_root=output_root,
    )
    if monitor is None:
        raise FileNotFoundError(
            f"Basin templates missing for {pair_slug}/seed_{seed}. "
            "Run veracity at λ=0,1 first."
        )
    trigger = make_trigger(
        rule=rule, theta=theta,
        persistence_K=persistence_K, velocity_lookback=velocity_lookback,
    )
    return AdaptiveSchedule(monitor=monitor, trigger=trigger)
