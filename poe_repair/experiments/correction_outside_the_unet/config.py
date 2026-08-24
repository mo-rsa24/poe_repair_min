"""Run configuration for Group A correctors.

One ``RunConfig`` dataclass tree shared across A1 / A2 / A3. The
``TechniqueConfig`` slot selects which corrector is built; the rest of
the config (cell, optim, schedule, probe, sampler, wandb) is identical
across techniques.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CellConfig:
    pair_slug: str = "a_cat__x__a_dog"
    seed: int = 42
    split: str = "heldout"
    prompt_a: str = "a cat"
    prompt_b: str = "a dog"
    joint_prompt: str = "a cat and a dog"


@dataclass
class TechniqueConfig:
    """Selects which corrector is built and exposes its width knobs."""

    name: str = "latent_cnn"           # "latent_cnn" | "latent_unet" | "frozen_feature_mlp"
    # A1 / shared: latent-CNN body channels and number of blocks
    body_channels: int = 128
    n_blocks: int = 5
    cond_dim: int = 256
    # A2: latent-UNet base channels + blocks per level
    base_channels: int = 96
    blocks_per_level: int = 2
    # A3: frozen-feature MLP head
    feature_channels: int = 1280
    head_channels: int = 256
    head_blocks: int = 3
    hook_module: str = "mid_block"


@dataclass
class OptimConfig:
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    total_epochs: int = 600
    epoch_size: int = 50               # optimizer steps per epoch
    train_batch_size: int = 4
    log_every: int = 20
    # σ-windowed loss is the default per the rewritten MVP plan.
    t_sampler: str = "sigma_weighted"   # "uniform" | "sigma_weighted" | "commit_window"
    t_sampler_floor: float = 0.01       # minimum relative weight per step (avoid tail starvation)


@dataclass
class ProbeConfig:
    every_epochs: int = 50
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    commit_window: tuple[int, int] = (5, 25)
    where_applied_steps: tuple[int, ...] = (7, 15, 22)


@dataclass
class KillConfig:
    # Group A kill: if r̂_t norm at the top-λ probe collapses to < r_collapse_threshold
    # × cached ‖r_t‖ for ``r_collapse_n_probes`` consecutive probes, abort.
    r_collapse_threshold: float = 0.05
    r_collapse_n_probes: int = 3


@dataclass
class SamplerConfig:
    scheduler: str = "ddim"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    euler_init_noise_sigma: float = 1.0
    height: int = 1024
    width: int = 1024


@dataclass
class WandBConfig:
    project: str = "poe-repair-group-a"
    entity: str | None = None
    tags: tuple[str, ...] = ("group_a_failure",)
    mode: str = "online"               # "online" | "offline" | "disabled"


@dataclass
class RunConfig:
    cell: CellConfig = field(default_factory=CellConfig)
    technique: TechniqueConfig = field(default_factory=TechniqueConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    kill: KillConfig = field(default_factory=KillConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    seed: int = 42
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "cuda"
    dtype: str = "float16"

    run_id: str = ""
    run_dir: str = ""
    dry_run: bool = False
    skip_scoring: bool = True           # Group A's headline is qualitative.
    resume_from: str | None = None
    git_commit: str = ""

    correction_max_rel_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_run_id(cfg: RunConfig, *, timestamp: str) -> str:
    short = {
        "latent_cnn": "lcnn",
        "latent_unet": "lunet",
        "frozen_feature_mlp": "ffmlp",
    }.get(cfg.technique.name, cfg.technique.name)
    return (
        f"groupA__{short}__{cfg.cell.pair_slug}__seed{cfg.cell.seed}"
        f"__lr{cfg.optim.lr:.0e}__{timestamp}"
    )


def run_dir_for(cfg: RunConfig, *, output_root: Path) -> Path:
    """``output_root`` is the resolved family root (``paths.CORRECTION_OUTSIDE_THE_UNET``),
    not a generic outputs directory — the caller resolves that before calling in.
    """
    return (
        output_root / cfg.technique.name
        / cfg.cell.pair_slug / f"seed_{cfg.cell.seed}"
        / cfg.run_id
    )
