"""Load cached residuals and compute r_t the way the sampler computes it.

The cache stores the four *raw* UNet outputs per step (``eps_a_raw``,
``eps_b_raw``, ``eps_j_raw``, ``eps_uncond``) in fp16. Everything downstream
wants the guided quantities, so the arithmetic that turns raw into guided, and
guided into r_t, has to match ``run_teacher_residual`` exactly. It does, because
this module imports the same two helpers the sampler uses rather than restating
the formulas:

    eps_a  = guided_eps(eps_a_raw, eps_uncond, w)
    eps_b  = guided_eps(eps_b_raw, eps_uncond, w)
    eps_j  = guided_eps(eps_j_raw, eps_uncond, w)
    eps_poe = poe_eps(eps_a, eps_b, eps_uncond)
    r_t     = eps_j - eps_poe

Everything is upcast to fp32 before arithmetic: the cache is fp16, and norms
accumulated in fp16 lose precision that matters at 50 steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps

CACHE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")


@dataclass(frozen=True)
class Cell:
    """One cached (pair, seed) trajectory, all tensors fp32, step-ordered."""

    pair_slug: str
    seed: int
    split: str
    meta: dict
    timesteps: torch.Tensor      # [T] int
    x_t: torch.Tensor            # [T,1,4,128,128]
    eps_a_raw: torch.Tensor
    eps_b_raw: torch.Tensor
    eps_j_raw: torch.Tensor
    eps_uncond: torch.Tensor

    @property
    def guidance_scale(self) -> float:
        return float(self.meta.get("guidance_scale", 7.5))

    @property
    def n_steps(self) -> int:
        return int(self.x_t.shape[0])

    def eps_poe(self) -> torch.Tensor:
        """Guided PoE prediction per step: ε̃_A + ε̃_B − ε_∅."""
        w = self.guidance_scale
        a = guided_eps(self.eps_a_raw, self.eps_uncond, w)
        b = guided_eps(self.eps_b_raw, self.eps_uncond, w)
        return poe_eps(a, b, self.eps_uncond)

    def eps_mono(self) -> torch.Tensor:
        """Guided Mono prediction per step: ε̃_J."""
        return guided_eps(self.eps_j_raw, self.eps_uncond, self.guidance_scale)

    def r_t(self) -> torch.Tensor:
        """The interaction term: ε̃_J − ε̃_PoE, per step. [T,1,4,128,128]."""
        return self.eps_mono() - self.eps_poe()

    def r_t_norm(self) -> torch.Tensor:
        """‖r_t‖ per step, flattened over channels and space. [T]."""
        return self.r_t().flatten(1).norm(dim=1)

    def log_snr(self) -> torch.Tensor:
        """log-SNR per step from the cached timesteps, for collapse plots.

        Uses the scheduler's alphas_cumprod, so this matches what the sampler
        saw rather than an approximation from the timestep index.
        """
        ab = _alphas_cumprod()[self.timesteps.long()]
        return torch.log(ab / (1.0 - ab))


@lru_cache(maxsize=1)
def _alphas_cumprod() -> torch.Tensor:
    """SDXL's alphas_cumprod, cached. Falls back to the standard schedule."""
    try:
        from poe_repair.config import RunConfig
        from diffusers import DDIMScheduler

        model_id = RunConfig().models.base_model_id
        sched = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        return sched.alphas_cumprod.float()
    except Exception:
        # scaled_linear betas, the SDXL default: reproduces the schedule
        # without needing the model files present.
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000) ** 2
        return torch.cumprod(1.0 - betas, dim=0).float()


def cell_dir(pair_slug: str, seed: int, *, root: Path = CACHE_ROOT) -> Path:
    """Locate a cell without the caller needing to know its split."""
    for split in ("train", "heldout"):
        d = root / split / pair_slug / f"seed_{seed}"
        if d.is_dir():
            return d
    raise FileNotFoundError(
        f"no cached cell for pair={pair_slug!r} seed={seed} under {root}"
    )


def load_cell(
    pair_slug: str, seed: int, *, root: Path = CACHE_ROOT, max_steps: int | None = None,
) -> Cell:
    """Load one cached cell, upcast to fp32, ordered by step index."""
    d = cell_dir(pair_slug, seed, root=root)
    meta = json.loads((d / "meta.json").read_text())
    step_files = sorted((d / "residuals").glob("step_*.pt"))
    if not step_files:
        raise FileNotFoundError(f"no step files under {d / 'residuals'}")
    if max_steps is not None:
        step_files = step_files[:max_steps]

    stacks: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("x_t", "eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond")
    }
    timesteps: list[int] = []
    for f in step_files:
        sd = torch.load(f, map_location="cpu", weights_only=True)
        for k in stacks:
            stacks[k].append(sd[k].float())      # fp16 -> fp32 before any math
        timesteps.append(int(sd["timestep"]))

    return Cell(
        pair_slug=pair_slug,
        seed=int(seed),
        split=d.parent.parent.name,
        meta=meta,
        timesteps=torch.tensor(timesteps),
        **{k: torch.stack(v) for k, v in stacks.items()},
    )


def r_t(pair_slug: str, seed: int, **kw) -> torch.Tensor:
    """Convenience: the interaction term for one cell. [T,1,4,128,128]."""
    return load_cell(pair_slug, seed, **kw).r_t()
