"""Configuration for the minimal PoE-repair project.

All paths are project-relative by default. Override via environment vars:
    POE_REPAIR_PILOT_DIR, POE_REPAIR_OUTPUT_ROOT, POE_REPAIR_SYNTH_CKPT.
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


def _default_synthesizer_checkpoint() -> Path:
    env = os.environ.get("POE_REPAIR_SYNTH_CKPT")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "checkpoints" / "synthesizer" / "residual_mlp" / "best.pt"


@dataclass
class Paths:
    pilot_dir: Path = field(default_factory=_default_pilot_dir)
    output_root: Path = field(default_factory=_default_output_root)
    synthesizer_checkpoint: Path = field(default_factory=_default_synthesizer_checkpoint)


@dataclass
class SynthConfig:
    """Defaults for the embedding synthesizer pipeline (text-only training)."""

    arch: str = "residual_mlp"
    seq_dim: int = 2048
    pooled_dim: int = 1280
    seq_len: int = 77
    hidden_dim: int = 1024
    num_layers: int = 3
    dropout: float = 0.0

    n_train_pairs: int = 60_000
    n_val_pairs: int = 5_000
    n_holdout_oversample: int = 5_000
    n_supercategory_oversample: int = 5_000
    batch_size: int = 512
    num_steps: int = 100_000
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    cosine_loss_weight: float = 1.0
    mse_loss_weight: float = 1.0
    pooled_loss_weight: float = 1.0
    seq_loss_weight: float = 1.0
    log_interval: int = 250
    val_interval: int = 1_000
    val_batches: int = 32
    seed: int = 0

    joint_template: str = "{a} and {b}"


@dataclass
class RunConfig:
    model_id: str = DEFAULT_MODEL_ID
    dtype: str = DEFAULT_DTYPE
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    guidance: float = DEFAULT_GUIDANCE
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    paths: Paths = field(default_factory=Paths)
    synth: SynthConfig = field(default_factory=SynthConfig)


def joint_prompt(prompt_a: str, prompt_b: str, template: str = "{a} and {b}") -> str:
    return template.format(a=prompt_a, b=prompt_b)
