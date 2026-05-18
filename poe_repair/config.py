"""Configuration for the minimal PoE-repair project.

All paths are project-relative by default. Override via environment vars:
    POE_REPAIR_PILOT_DIR, POE_REPAIR_OUTPUT_ROOT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_DTYPE = "fp16"
DEFAULT_GUIDANCE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_pilot_dir() -> Path:
    env = os.environ.get("POE_REPAIR_PILOT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "data" / "pilot"


def _default_output_root() -> Path:
    env = os.environ.get("POE_REPAIR_OUTPUT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "outputs"


@dataclass
class Paths:
    pilot_dir: Path = field(default_factory=_default_pilot_dir)
    output_root: Path = field(default_factory=_default_output_root)


@dataclass
class RunConfig:
    model_id: str = DEFAULT_MODEL_ID
    dtype: str = DEFAULT_DTYPE
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    guidance: float = DEFAULT_GUIDANCE
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    joint_template: str = "{a} and {b}"
    paths: Paths = field(default_factory=Paths)


def joint_prompt(prompt_a: str, prompt_b: str, template: str = "{a} and {b}") -> str:
    return template.format(a=prompt_a, b=prompt_b)
