"""Batched sampler that gates the prompt and the correction independently.

Two things this adds over ``run_teacher_residual``, which the crossed timing
experiment needs and which that sampler cannot do:

1. **A per-step conditioning gate.** At a step where the prompt is off, the
   prediction is the raw unconditional ε_∅, exactly as ``run_cfg_masked``
   defines it for the plain-CFG path. So a sample can be guided early and
   unconditional late, or the reverse.
2. **Many cells in one call.** ``run_teacher_residual`` samples one cell per
   process, which means reloading SDXL for every image. Here N cells share one
   loaded model and one UNet call per step, which sees 4N branches.

Each sample carries its own two masks, so a batch can mix pairs, seeds,
conditioning schedules and correction windows freely:

    cond_on[i, t]  the prompt acts on sample i at step t
    lam[i, t]      how much correction goes into sample i at step t

and the prediction stepped with is

    ε_i,t = ε_∅            where the prompt is off
          = ε̃_PoE + λ·r_t  where it is on

Batching does not make the card faster: throughput is flat from batch 4 upward
on an A6000, because the UNet is compute-bound well before it is memory-bound.
The win is loading the model once, which is roughly 2x on wall clock.

Two identities pin this to the samplers it replaces, both checked in
tests/test_window_batch_identity.py:

  - prompt on everywhere, λ=0 everywhere  ≡  plain PoE
  - prompt on everywhere, any λ schedule  ≡  run_teacher_residual with the same
    correction window, cell for cell
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from poe_repair._sdxl.metrics import (
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)
from poe_repair._sdxl.runtime import decode_latents
from poe_repair.methods._sampling import add_time_ids


@dataclass
class BatchOutputs:
    """One entry per input sample, in input order."""

    latents: torch.Tensor                  # [N,4,H/8,W/8] final
    images: list[torch.Tensor]             # N decoded images, CPU
    trajectories: torch.Tensor             # [T+1,N,4,H/8,W/8] fp16, CPU
    x0_estimates: torch.Tensor             # [T,N,4,H/8,W/8] fp16, CPU
    timesteps: list[int]
    extras: list[dict] = field(default_factory=list)


def schedule_masks(
    *,
    num_steps: int,
    cond_window: tuple[int, int] | None,
    corr_window: tuple[int, int] | None,
    lambda_max: float = 1.0,
    cond_outside: bool = False,
) -> tuple[list[bool], list[float]]:
    """Turn one cell's two windows into its two per-step masks.

    ``cond_window=None`` means the prompt is on at every step, which is the
    ordinary guided run and the baseline every other schedule is read against.
    ``corr_window=None`` means λ applies at every step; to inject nothing
    anywhere, pass ``lambda_max=0``.

    ``cond_outside=True`` inverts the conditioning window: the prompt is off
    inside it and on outside. That is what gives the "unconditional early, then
    guided" direction without a second window grammar.

    Where the prompt is off the correction is forced to zero as well. Injecting
    a correction into a step that has no prompt is not a weaker version of the
    experiment, it is a different and meaningless quantity: r_t is defined as
    the gap between two conditional predictions, and neither is being made.
    """
    cond_on: list[bool] = []
    lam: list[float] = []
    for t in range(num_steps):
        if cond_window is None:
            on = True
        else:
            inside = cond_window[0] <= t < cond_window[1]
            on = (not inside) if cond_outside else inside
        l = float(lambda_max)
        if corr_window is not None and not (corr_window[0] <= t < corr_window[1]):
            l = 0.0
        if not on:
            l = 0.0
        cond_on.append(bool(on))
        lam.append(l)
    return cond_on, lam


@torch.no_grad()
def run_window_batch(
    *,
    init_latents: torch.Tensor,          # [N,4,h,w], already noise-scaled input
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,   # [N,77,2048] / [N,1280]
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cond_on: torch.Tensor,               # [N,T] bool
    lam: torch.Tensor,                   # [N,T] float
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    save_trajectory: bool = True,
) -> BatchOutputs:
    """Sample N cells together, each with its own prompt and correction masks."""
    n = init_latents.shape[0]
    if cond_on.shape != (n, num_inference_steps):
        raise ValueError(
            f"cond_on must be [N,T] = {(n, num_inference_steps)}, got "
            f"{tuple(cond_on.shape)}"
        )
    if lam.shape != (n, num_inference_steps):
        raise ValueError(
            f"lam must be [N,T] = {(n, num_inference_steps)}, got {tuple(lam.shape)}"
        )

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    cond_on = cond_on.to(device=device)
    lam = lam.to(device=device, dtype=torch.float32)

    # Branch order [A(N), B(N), J(N), ∅(N)]: chunk(4) then recovers each branch
    # for all N samples at once, which is what the arithmetic below assumes.
    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0).to(device=device, dtype=dtype)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0).to(device=device, dtype=dtype)
    cond_kwargs = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=4 * n, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    traj = (
        torch.empty(num_inference_steps + 1, n, *latents.shape[1:], dtype=torch.float16)
        if save_trajectory else None
    )
    # The Tweedie x̂_0 estimate per step: what the model currently believes the
    # finished picture is. Decoding x_t instead shows mostly noise until the
    # last few steps, which cannot show where two runs diverge. It is computed
    # anyway to take the step, so keeping it costs nothing but the memory.
    x0s = (
        torch.empty(num_inference_steps, n, *latents.shape[1:], dtype=torch.float16)
        if save_trajectory else None
    )
    delta_norm = torch.zeros(n, num_inference_steps)
    pmi_resid = torch.zeros(n, num_inference_steps)
    timesteps: list[int] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        if traj is not None:
            traj[step_index] = latents.detach().to(torch.float16).cpu()
        timesteps.append(int(timestep.item()))

        latent_input = scheduler.scale_model_input(
            latents.repeat(4, 1, 1, 1), timestep,
        )
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond_kwargs, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)

        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_j = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
        eps_poe = poe_eps(eps_a, eps_b, eps_uncond)
        delta = eps_j - eps_poe

        delta_norm[:, step_index] = delta.flatten(1).float().norm(dim=1).cpu()
        rhs = float(guidance_scale) * (eps_j_raw + eps_uncond - eps_a_raw - eps_b_raw)
        num = (delta - rhs).flatten(1).float().norm(dim=1)
        den = delta.flatten(1).float().norm(dim=1).clamp_min(1e-12)
        pmi_resid[:, step_index] = (num / den).cpu()

        l = lam[:, step_index].to(dtype).view(n, 1, 1, 1)
        eps_cond = eps_poe + l * delta
        # Reproduce the single-cell sampler exactly at the two endpoints rather
        # than relying on floating-point luck: at λ=0 it steps with eps_poe, and
        # at λ=1 with eps_j itself.
        is0 = (lam[:, step_index] == 0.0).view(n, 1, 1, 1)
        is1 = (lam[:, step_index] == 1.0).view(n, 1, 1, 1)
        eps_cond = torch.where(is0, eps_poe, eps_cond)
        eps_cond = torch.where(is1, eps_j, eps_cond)

        # The prompt gate. Off means the raw unconditional prediction, the same
        # thing run_cfg_masked collapses to.
        on = cond_on[:, step_index].view(n, 1, 1, 1)
        eps_t = torch.where(on, eps_cond, eps_uncond)

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=dtype,
        )
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        if x0s is not None:
            x0s[step_index] = x0.detach().to(torch.float16).cpu()
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_t,
        )

    if traj is not None:
        traj[num_inference_steps] = latents.detach().to(torch.float16).cpu()

    # Decoded one at a time on purpose: the VAE peaks at 36 GiB decoding eight
    # 1024x1024 frames at once, and gains nothing for it.
    images = [decode_latents(models, latents[i:i + 1]).cpu() for i in range(n)]

    extras = [
        {
            "lambda_per_step": [float(x) for x in lam[i].tolist()],
            "cond_on_per_step": [bool(x) for x in cond_on[i].tolist()],
            "delta_norm_per_step": [float(x) for x in delta_norm[i].tolist()],
            "pmi_identity_residual_per_step": [float(x) for x in pmi_resid[i].tolist()],
        }
        for i in range(n)
    ]
    return BatchOutputs(
        latents=latents, images=images,
        trajectories=traj if traj is not None else torch.empty(0),
        x0_estimates=x0s if x0s is not None else torch.empty(0),
        timesteps=timesteps, extras=extras,
    )


def stack_embeddings(
    per_cell: Sequence[dict], key: str, device, dtype,
) -> torch.Tensor:
    """Stack one embedding field across cells into a [N,...] batch tensor."""
    return torch.cat([c[key].to(device=device, dtype=dtype) for c in per_cell], dim=0)
