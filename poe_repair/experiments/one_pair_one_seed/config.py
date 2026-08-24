"""Run configuration for LoRA on SDXL.

A single ``RunConfig`` dataclass capturing everything that goes into the
W&B ``config`` block. Constructed from CLI flags in ``main.py``.
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
class LoRAConfig:
    rank: int = 8
    alpha: int = 8
    dropout: float = 0.0
    # peft list-form: each entry is suffix-matched as f".{entry}". Using
    # plain suffix strings (not regex) so we hit attn2 (cross-attn) and
    # never attn1 (self-attn). Verified: key.endswith(".attn2.to_q") is
    # True for cross-attn projections, False for self-attn.
    target_modules: tuple[str, ...] = (
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
    )
    init: str = "gaussian"
    adapter_name: str = "lora"


@dataclass
class OptimConfig:
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    total_epochs: int = 200
    epoch_size: int = 50               # number of optimizer steps per epoch
    train_batch_size: int = 1          # cached steps processed per grad step
    log_every: int = 20                # bucket-loss aggregation cadence


@dataclass
class ProbeConfig:
    every_epochs: int = 50
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    commit_window: tuple[int, int] = (5, 25)
    where_applied_steps: tuple[int, ...] = (7, 15, 22)


@dataclass
class KillConfig:
    loss_threshold: float = 0.1
    after_steps: int = 3_000
    commit_bucket_halve_after_steps: int = 5_000


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
    project: str = "poe-repair-lora"
    entity: str | None = None
    tags: tuple[str, ...] = ("lora",)
    mode: str = "online"               # "online" | "offline" | "disabled"


@dataclass
class RunConfig:
    cell: CellConfig = field(default_factory=CellConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
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
    skip_scoring: bool = False
    resume_from: str | None = None
    git_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_run_id(cfg: RunConfig, *, timestamp: str) -> str:
    return (
        f"lora_sdxl__{cfg.cell.pair_slug}__seed{cfg.cell.seed}"
        f"__r{cfg.lora.rank}__lr{cfg.optim.lr:.0e}__{timestamp}"
    )


def run_dir_for(cfg: RunConfig, *, output_root: Path) -> Path:
    return (
        output_root / "lora"
        / cfg.cell.pair_slug / f"seed_{cfg.cell.seed}"
        / cfg.run_id
    )
