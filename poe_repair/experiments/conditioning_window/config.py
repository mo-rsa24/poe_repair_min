"""Run configuration for the CFG conditioning-window ablation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    import os
    env = os.environ.get("POE_REPAIR_OUTPUT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "outputs"


@dataclass
class RunConfig:
    prompt: str = "a cat and a dog"
    prompt_a: str = "a cat"
    prompt_b: str = "a dog"
    pair_slug: str = "a_cat__x__a_dog"
    seed: int = 42
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_root: Path = field(default_factory=_default_output_root)

    def run_dir(self) -> Path:
        return (
            self.output_root / "conditioning_window"
            / self.pair_slug / f"seed_{self.seed}"
        )

    def schedules_dir(self) -> Path:
        return self.run_dir() / "schedules"

    def results_dir(self) -> Path:
        return self.run_dir() / "results"
