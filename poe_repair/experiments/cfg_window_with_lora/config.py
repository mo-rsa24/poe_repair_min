"""Run configuration for the LoRA-on-CFG-mask ablation.

Each ``(checkpoint, lambda)`` pair is one "cell". Output layout::

    outputs/conditioning_window_lora/<pair>/seed_<n>/<mode>/
        epoch_<step>/lambda_<lam>/
            schedules/<schedule_id>/{image.png, summary.json}
            results/cell_manifest.json
        results/inspector_manifest.json   ← rollup across all cells

The "epoch" label here is the LoRA optimizer-step count parsed from the
checkpoint filename ``lora_step_XXXXXX.pt`` — same naming the existing
``/`` inspector uses for its epoch slider.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_root() -> Path:
    env = os.environ.get("POE_REPAIR_OUTPUT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "outputs"


def _default_lora_ckpt() -> Path:
    return (
        _repo_root() / "outputs/lora/a_cat__x__a_dog/seed_42/results/"
        "checkpoints/lora_step_062500.pt"
    )


def _default_lora_run_dir() -> Path:
    return _repo_root() / "outputs/lora/a_cat__x__a_dog/seed_42/results"


COMPOSITION_MODES = ("with_prompt", "always")

_STEP_RE = re.compile(r"lora_step_(\d+)\.pt$")


def epoch_for_ckpt(ckpt_path: Path) -> int:
    """Extract the optimizer-step count from a ``lora_step_*.pt`` filename."""
    m = _STEP_RE.search(str(ckpt_path))
    if not m:
        raise ValueError(
            f"cannot parse step number from checkpoint name {ckpt_path!r}; "
            f"expected ``lora_step_<step>.pt``"
        )
    return int(m.group(1))


def lambda_tag(lam: float) -> str:
    """Canonical zero-padded tag for a lambda value — ``0.00``, ``0.50``, ``1.00``."""
    return f"{float(lam):.2f}"


def epoch_tag(epoch: int) -> str:
    """Canonical zero-padded tag for an optimizer-step count."""
    return f"{int(epoch):06d}"


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

    # LoRA bits — now multi-cell. Each (ckpt, lambda) pair is a rendered cell.
    lora_ckpt_paths: tuple[Path, ...] = field(
        default_factory=lambda: (_default_lora_ckpt(),),
    )
    lora_run_dir: Path = field(default_factory=_default_lora_run_dir)
    lambda_values: tuple[float, ...] = (1.0,)
    composition_modes: tuple[str, ...] = COMPOSITION_MODES

    def run_dir(self) -> Path:
        return (
            self.output_root / "conditioning_window_lora"
            / self.pair_slug / f"seed_{self.seed}"
        )

    def mode_dir(self, mode: str) -> Path:
        if mode not in COMPOSITION_MODES:
            raise ValueError(
                f"composition_mode must be one of {COMPOSITION_MODES}, got {mode!r}"
            )
        return self.run_dir() / mode

    def cell_dir(self, mode: str, epoch: int, lam: float) -> Path:
        return (
            self.mode_dir(mode) / f"epoch_{epoch_tag(epoch)}"
            / f"lambda_{lambda_tag(lam)}"
        )

    def schedules_dir(self, mode: str, epoch: int, lam: float) -> Path:
        return self.cell_dir(mode, epoch, lam) / "schedules"

    def results_dir_for_cell(self, mode: str, epoch: int, lam: float) -> Path:
        return self.cell_dir(mode, epoch, lam) / "results"

    def mode_results_dir(self, mode: str) -> Path:
        """Mode-level rollup directory (holds the inspector_manifest.json)."""
        return self.mode_dir(mode) / "results"
