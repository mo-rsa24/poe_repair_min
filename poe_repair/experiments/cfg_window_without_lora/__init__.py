"""CFG conditioning-window ablation — no-LoRA baseline.

Research objective: identify, for "a cat and a dog" at seed 42, which
contiguous segments of the 50-step DDIM trajectory genuinely require the
conditional branch of CFG. At each step the CFG mask gates the conditional
contribution; "off" steps collapse to ε_uncond while keeping the σ-schedule
and DDIM noise direction identical to the normal CFG step.

The interactive inspector slider (``/conditioning_window`` in
``scripts/lora_inspector.py``) is the primary readout — scrub across
schedules to identify the minimum conditioning window that still produces
a recognisable cat+dog. This is the no-residual baseline against which
the LoRA experiment measures its marginal effect.

Schedule grammar lives in ``schedules.py`` (prefix, suffix, single-window,
punctate). The standard suite renders ~50 schedules at seed 42 and writes
PNGs + an inspector manifest under ``outputs/conditioning_window/a_cat__x__a_dog/
seed_42/``.
"""

from __future__ import annotations

from poe_repair.experiments.cfg_window_without_lora.main import main

__all__ = ["main"]
