"""Shared sampling primitives.

Four samplers, one per "column" of the qualitative grid:

- ``run_vanilla_poe``  — eps_PoE = eps_A + eps_B - eps_uncond. Reference baseline.
- ``run_m2_replace``   — single CFG branch on synthesised joint embedding ê_J.
- ``run_c_poe``        — PoE pull modulated by max(0, cos<u_A, u_B>)^gamma.
- ``run_m2_c_poe``     — combined: alpha·(u_A + u_B) + lambda_j·u_J.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from poe_repair.runtime import (
    LatentTrajectoryCollector,
    PairSeedCell,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    load_shared_latents,
    poe_eps,
    tweedie_mean,
)


def add_time_ids(*, height, width, batch_size, device, dtype):
    base = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=device)
    return base.repeat(batch_size, 1)


@dataclass
class SamplerOutputs:
    latents: torch.Tensor
    image: torch.Tensor
    tracker: LatentTrajectoryCollector
    extras: dict


def write_decoded_image(image_tensor: torch.Tensor, path: Path) -> Path:
    """Save a VAE-decoded tensor (already in [0, 1] from decode_latents) as PNG.

    The earlier vendored convention applied a redundant [-1, 1] -> [0, 1]
    remap on top of input that was already in [0, 1], compressing every
    pixel into [0.5, 1] — visibly washed-out edges / faded background.
    Removed: input is clamped to [0, 1] and quantised directly.
    """
    arr = image_tensor.detach().float().clamp(0.0, 1.0)
    arr = (arr * 255.0).round().to(torch.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    Image.fromarray(arr.cpu().numpy()).save(str(path))
    return path


def initial_latents_for_pair(
    *,
    cell: PairSeedCell,
    models: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """Recover the same x_T used in the pilot for a fair comparison."""
    return load_shared_latents(
        pair_dir=cell.pair_dir,
        seed=cell.seed,
        device=device,
        dtype=dtype,
        models=models,
        height=cell.height,
        width=cell.width,
    )


def _conflict_modulation(
    u_a: torch.Tensor, u_b: torch.Tensor, gamma: float, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos_theta = <u_a, u_b> / (||u_a|| ||u_b||)``, ``alpha = max(0, cos_theta)^gamma``."""
    flat_a = u_a.reshape(-1)
    flat_b = u_b.reshape(-1)
    num = torch.dot(flat_a, flat_b)
    den = torch.linalg.vector_norm(flat_a) * torch.linalg.vector_norm(flat_b) + eps
    cos_theta = num / den
    alpha = torch.clamp(cos_theta, min=0.0).pow(gamma)
    return cos_theta, alpha


@torch.no_grad()
def run_vanilla_poe(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor,
    pool_a: torch.Tensor,
    seq_b: torch.Tensor,
    pool_b: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> SamplerOutputs:
    """eps_PoE = eps_A + eps_B - eps_uncond."""
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=3, device=device, dtype=dtype)}
    unet = models["unet"]
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        tracker.store_step(step_index, latents, eps_p, float(step_index) / float(num_inference_steps), int(timestep.item()))
        # Standard DDIM: same eps for Tweedie x0 and the noise direction.
        # The earlier vendored convention passed eps_uncond here, which
        # partially undoes the conditioning each step.
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_p)
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras={})


@torch.no_grad()
def run_m2_replace(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_j: torch.Tensor,
    pool_j: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> SamplerOutputs:
    """Single guided UNet branch using the synthesised joint embedding ê_J."""
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
    pe_in = torch.cat([seq_j, seq_e], dim=0)
    pool_in = torch.cat([pool_j, pool_e], dim=0)
    cond = {"text_embeds": pool_in, "time_ids": add_time_ids(height=height, width=width, batch_size=2, device=device, dtype=dtype)}
    unet = models["unet"]

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe_in, added_cond_kwargs=cond, timestep_cond=None).sample
        eps_j_raw, eps_uncond = noise.chunk(2)
        eps_j = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_j)
        tracker.store_step(step_index, latents, eps_j, float(step_index) / float(num_inference_steps), int(timestep.item()))
        # Standard DDIM: pass the same composed/guided eps to ddim_prev as was
        # used for Tweedie x0 (see notes on the eps_uncond convention bug).
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_j)
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras={})


@torch.no_grad()
def run_c_poe(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor,
    pool_a: torch.Tensor,
    seq_b: torch.Tensor,
    pool_b: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    gamma: float = 2.0,
) -> SamplerOutputs:
    """C-PoE — conflict-aware composition.

    Per step:
        u_A = eps_A - eps_uncond,  u_B = eps_B - eps_uncond
        cos_theta = <u_A, u_B> / (||u_A|| ||u_B||)
        alpha     = max(0, cos_theta)^gamma
        eps_t     = eps_uncond + alpha * (u_A + u_B)
    """
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=3, device=device, dtype=dtype)}
    unet = models["unet"]

    cos_theta_per_step: list[float] = []
    alpha_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        u_a = eps_a - eps_uncond
        u_b = eps_b - eps_uncond
        cos_theta, alpha = _conflict_modulation(u_a, u_b, gamma)
        eps_t = eps_uncond + alpha * (u_a + u_b)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        tracker.store_step(step_index, latents, eps_t, float(step_index) / float(num_inference_steps), int(timestep.item()))
        # Standard DDIM: same eps_t for x0 and noise direction.
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_t)
        cos_theta_per_step.append(float(cos_theta.detach().cpu()))
        alpha_per_step.append(float(alpha.detach().cpu()))

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    extras = {
        "gamma": float(gamma),
        "cos_theta_per_step": cos_theta_per_step,
        "alpha_per_step": alpha_per_step,
        "mean_cos_theta": float(np.mean(cos_theta_per_step)) if cos_theta_per_step else float("nan"),
        "mean_alpha": float(np.mean(alpha_per_step)) if alpha_per_step else float("nan"),
        "frac_steps_in_conflict": float(np.mean([1.0 if c < 0.0 else 0.0 for c in cos_theta_per_step])) if cos_theta_per_step else float("nan"),
    }
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras=extras)


@torch.no_grad()
def run_beta_inject_online(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor,
    pool_a: torch.Tensor,
    seq_b: torch.Tensor,
    pool_b: torch.Tensor,
    seq_j: torch.Tensor,
    pool_j: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    beta: float = 0.0,
) -> SamplerOutputs:
    """β-injected PoE sampler with an online residual.

    Per step the UNet is evaluated on four branches (A, B, J, empty) and the
    composed update is

        ε̃_β = (1-β)·ε̃_PoE + β·ε̃_J = ε̃_PoE + β·r_t,

    with r_t = ε̃_J - ε̃_PoE recomputed at the current x_t^β each step
    (online). Endpoints by construction:

        β = 0  →  ε̃_β = ε̃_PoE   (reduces to vanilla PoE).
        β = 1  →  ε̃_β = ε̃_J     (reduces to mono).

    Returns SamplerOutputs with extras:
        beta:         float
        eps_poe_traj: Tensor[T, C, H, W]  — composed ε̃_PoE per step.
        eps_j_traj:   Tensor[T, C, H, W]  — guided ε̃_J at the same x_t.
        timesteps:    Tensor[T]
    """
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(height=height, width=width, batch_size=4, device=device, dtype=dtype),
    }
    unet = models["unet"]

    latent_shape = tuple(latents.shape[1:])
    eps_poe_traj = torch.zeros(num_inference_steps, *latent_shape, dtype=torch.float32)
    eps_j_traj = torch.zeros(num_inference_steps, *latent_shape, dtype=torch.float32)
    timesteps_arr = torch.zeros(num_inference_steps)

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
        noise = unet(
            latent_input,
            timestep,
            encoder_hidden_states=pe,
            added_cond_kwargs=cond,
            timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_j = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        eps_beta = (1.0 - beta) * eps_p + beta * eps_j

        eps_poe_traj[step_index] = eps_p.detach().to(torch.float32).cpu().squeeze(0)
        eps_j_traj[step_index] = eps_j.detach().to(torch.float32).cpu().squeeze(0)
        timesteps_arr[step_index] = float(timestep.item())

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_beta)
        tracker.store_step(
            step_index, latents, eps_beta,
            float(step_index) / float(num_inference_steps),
            int(timestep.item()),
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_beta,
        )

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents,
        image=image,
        tracker=tracker,
        extras={
            "beta": float(beta),
            "eps_poe_traj": eps_poe_traj,
            "eps_j_traj": eps_j_traj,
            "timesteps": timesteps_arr,
        },
    )


@torch.no_grad()
def run_beta_inject_frozen(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor,
    pool_a: torch.Tensor,
    seq_b: torch.Tensor,
    pool_b: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    r_traj: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    beta: float = 0.0,
) -> SamplerOutputs:
    """β-injected PoE sampler with a *frozen* (precomputed) residual.

    Per step:
        ε̃_β = ε̃_PoE(x_t^β) + β · r_traj[t],

    where r_traj is precomputed along a separate clean PoE rollout — it does
    NOT depend on the current modified latent. Compare with
    `run_beta_inject_online`, which recomputes r_t at the modified latent.

    Frozen ≈ Online ⇒ residual is approximately a fixed vector field along
    the trajectory.
    Frozen ≠ Online ⇒ residual is state-dependent.
    """
    if r_traj.shape[0] != num_inference_steps:
        raise ValueError(
            f"r_traj has {r_traj.shape[0]} steps, expected {num_inference_steps}"
        )

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(height=height, width=width, batch_size=3, device=device, dtype=dtype),
    }
    unet = models["unet"]
    r_traj_dev = r_traj.to(device=device, dtype=dtype)

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input,
            timestep,
            encoder_hidden_states=pe,
            added_cond_kwargs=cond,
            timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        eps_beta = eps_p + beta * r_traj_dev[step_index].unsqueeze(0)

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_beta)
        tracker.store_step(
            step_index, latents, eps_beta,
            float(step_index) / float(num_inference_steps),
            int(timestep.item()),
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_beta,
        )

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents,
        image=image,
        tracker=tracker,
        extras={"beta": float(beta), "frozen": True},
    )


@torch.no_grad()
def run_m2_c_poe(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor,
    pool_a: torch.Tensor,
    seq_b: torch.Tensor,
    pool_b: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    seq_j: torch.Tensor,
    pool_j: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    gamma: float = 2.0,
    lambda_j: float = 1.0,
) -> SamplerOutputs:
    """M2 + C-PoE — combined channel-C1 (synthesizer) and channel-C2 (conflict gate).

    Per step:
        u_J     = eps_synth - eps_uncond
        eps_t   = eps_uncond + alpha*(u_A + u_B) + lambda_j * u_J
    """
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])

    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=height, width=width, batch_size=4, device=device, dtype=dtype)}
    unet = models["unet"]

    cos_theta_per_step: list[float] = []
    alpha_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
        noise = unet(latent_input, timestep, encoder_hidden_states=pe, added_cond_kwargs=cond, timestep_cond=None).sample
        eps_a_raw, eps_b_raw, eps_synth_raw, eps_uncond = noise.chunk(4)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_synth = guided_eps(eps_synth_raw, eps_uncond, guidance_scale)
        u_a = eps_a - eps_uncond
        u_b = eps_b - eps_uncond
        u_j = eps_synth - eps_uncond
        cos_theta, alpha = _conflict_modulation(u_a, u_b, gamma)
        eps_t = eps_uncond + alpha * (u_a + u_b) + lambda_j * u_j
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        tracker.store_step(step_index, latents, eps_t, float(step_index) / float(num_inference_steps), int(timestep.item()))
        # Standard DDIM: same eps_t for x0 and noise direction.
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index, x0=x0, eps=eps_t)
        cos_theta_per_step.append(float(cos_theta.detach().cpu()))
        alpha_per_step.append(float(alpha.detach().cpu()))

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    extras = {
        "gamma": float(gamma),
        "lambda_j": float(lambda_j),
        "cos_theta_per_step": cos_theta_per_step,
        "alpha_per_step": alpha_per_step,
        "mean_cos_theta": float(np.mean(cos_theta_per_step)) if cos_theta_per_step else float("nan"),
        "mean_alpha": float(np.mean(alpha_per_step)) if alpha_per_step else float("nan"),
        "frac_steps_in_conflict": float(np.mean([1.0 if c < 0.0 else 0.0 for c in cos_theta_per_step])) if cos_theta_per_step else float("nan"),
    }
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras=extras)
