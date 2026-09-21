#!/usr/bin/env python
"""Shared pieces for "what the correction is made of" (plan 05/tests/07).

The three bar constants live here and nowhere else, so the verdict on rung 2 is read from
source and a change to it shows up in a diff:

    ORTHO_SHARE_REWEIGHT_IMPOSSIBLE   above this, PoE could not have supplied the correction by
                                      re-weighting its own three predictions (support)
    ORTHO_SHARE_REWEIGHT_CANDIDATE    below this, a per-step guidance re-weighting is a
                                      candidate fix (null)
    EARLY_STEPS                       the window the statistic is averaged over

The statistic: mean over seeds of the per-seed mean, over EARLY_STEPS, of the orthogonal share
of ||r_t||^2, where r_t = eps~_j - eps_PoE is projected onto span{eps_a - eps_u, eps_b - eps_u,
eps_u} at the same cached state.

The rule the cache used (scripts/build_training_cache.py, poe_repair/_sdxl/metrics.py):
    eps~_k  = eps_u + g (eps_k - eps_u),  g = 7.5
    eps_PoE = eps~_a + eps~_b - eps_u
    r_t     = eps~_j - eps_PoE = g (eps_j - eps_a - eps_b + eps_u)
Every cached tensor is float16 on disk and is upcast to float32 here before any arithmetic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---- the bar (written before any number existed, 2026-09-05) ---------------------------------
ORTHO_SHARE_REWEIGHT_IMPOSSIBLE = 0.5
ORTHO_SHARE_REWEIGHT_CANDIDATE = 0.25
EARLY_STEPS = range(0, 11)          # steps 0 to 10 inclusive

# ---- the data ----------------------------------------------------------------------------------
PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
GUIDANCE = 7.5
CACHE_ROOT = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout")
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/what_the_correction_is_made_of")
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "artifacts/results/what-the-correction-is-made-of"
STRIP_SEED = 15
STRIP_STEPS = (0, 2, 5, 10, 20, 30, 49)   # 49 is the last cached step (timestep 1)


def cell_dir(seed: int, pair: str = PAIR) -> Path:
    return CACHE_ROOT / pair / f"seed_{seed}"


@dataclass
class Step:
    seed: int
    step_index: int
    timestep: int
    x_t: torch.Tensor        # (1, 4, 128, 128) float32
    eps_a: torch.Tensor
    eps_b: torch.Tensor
    eps_j: torch.Tensor
    eps_u: torch.Tensor

    @property
    def eps_poe(self) -> torch.Tensor:
        """The prediction the cached run stepped with: guided a + guided b - unconditional."""
        return guided(self.eps_a, self.eps_u) + guided(self.eps_b, self.eps_u) - self.eps_u

    @property
    def eps_joint(self) -> torch.Tensor:
        return guided(self.eps_j, self.eps_u)

    @property
    def r_t(self) -> torch.Tensor:
        """The correction: guided joint minus PoE, = g (eps_j - eps_a - eps_b + eps_u)."""
        return self.eps_joint - self.eps_poe


def guided(eps_cond: torch.Tensor, eps_u: torch.Tensor, g: float = GUIDANCE) -> torch.Tensor:
    return eps_u + g * (eps_cond - eps_u)


def load_step(seed: int, step_index: int, pair: str = PAIR) -> Step:
    p = cell_dir(seed, pair) / "residuals" / f"step_{step_index:03d}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return Step(seed=seed, step_index=int(d["step_index"]), timestep=int(d["timestep"]),
                x_t=d["x_t"].float(), eps_a=d["eps_a_raw"].float(), eps_b=d["eps_b_raw"].float(),
                eps_j=d["eps_j_raw"].float(), eps_u=d["eps_uncond"].float())


def num_steps(seed: int, pair: str = PAIR) -> int:
    return len(list((cell_dir(seed, pair) / "residuals").glob("step_*.pt")))


# ---- the span and the projection -----------------------------------------------------------------
def span_basis(s: Step) -> torch.Tensor:
    """(65536, 3) matrix whose columns are eps_a - eps_u, eps_b - eps_u, eps_u."""
    return torch.stack([(s.eps_a - s.eps_u).reshape(-1),
                        (s.eps_b - s.eps_u).reshape(-1),
                        s.eps_u.reshape(-1)], dim=1).double()


def project(basis: torch.Tensor, v: torch.Tensor) -> dict:
    """Least-squares projection of v onto the column span of basis.

    Returns the coefficients (alpha on eps_u last, matching the basis column order
    [a - u, b - u, u]), the in-span and orthogonal parts, and the two shares of ||v||^2."""
    v = v.reshape(-1).double()
    coef = torch.linalg.lstsq(basis, v.unsqueeze(1)).solution.squeeze(1)
    v_in = basis @ coef
    v_out = v - v_in
    vv = float(v @ v)
    in_share = float(v_in @ v_in) / vv if vv > 0 else float("nan")
    return {"coef": coef, "in": v_in, "out": v_out, "norm": vv ** 0.5,
            "in_share": in_share, "ortho_share": 1.0 - in_share}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def verdict(early_ortho_mean: float) -> str:
    if early_ortho_mean > ORTHO_SHARE_REWEIGHT_IMPOSSIBLE:
        return "support: PoE could not have supplied the correction by re-weighting"
    if early_ortho_mean < ORTHO_SHARE_REWEIGHT_CANDIDATE:
        return "null: a per-step guidance re-weighting is a candidate fix"
    return "inconclusive: between the two constants"


def bar_block() -> dict:
    return {"ORTHO_SHARE_REWEIGHT_IMPOSSIBLE": ORTHO_SHARE_REWEIGHT_IMPOSSIBLE,
            "ORTHO_SHARE_REWEIGHT_CANDIDATE": ORTHO_SHARE_REWEIGHT_CANDIDATE,
            "EARLY_STEPS": list(EARLY_STEPS), "source": "scripts/showcase/correction_span_common.py"}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def per_position_norm(t: torch.Tensor) -> np.ndarray:
    """(128, 128) map of the L2 norm over the 4 latent channels at each position."""
    return t.reshape(4, 128, 128).float().norm(dim=0).cpu().numpy()
