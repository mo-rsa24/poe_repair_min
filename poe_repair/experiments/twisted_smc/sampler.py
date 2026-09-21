"""Particle sampler for Mono and PoE with an optional learned twist.

``run_particles`` runs K particles through the stochastic DDIM chain (``eta``
sets the fresh noise per step; ``eta=0`` is the repo's deterministic DDIM).
With ``twist=None`` every particle keeps weight 1 and no resampling happens, so
the output is K independent draws of the base sampler. With a twist, each step
adds ``log psi_{t'}(x_{t'}) - log psi_t(x_t)`` to a particle's log-weight and
particles are systematically resampled whenever the effective sample size drops
below ``ess_threshold * K``. That is twisted SMC with the base as proposal
(CDM, arXiv 2605.23346, algorithm 1, continuous case).

Comparing ``twist=None`` against a twist at the same ``eta``, K and noise draws
is the one-axis comparison: only the reweighting differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean
from poe_repair._sdxl.runtime import decode_latents
from poe_repair.methods._sampling import add_time_ids


@dataclass
class ParticleOutputs:
    latents: torch.Tensor                     # (K, 4, h, w) final
    log_weights: torch.Tensor                 # (K,) at the end (0 if just resampled)
    ess: list[float] = field(default_factory=list)
    resampled_at: list[int] = field(default_factory=list)
    ancestry: list[list[int]] = field(default_factory=list)
    log_psi_final: torch.Tensor | None = None  # (K,) twist at the clean end


def _alpha_prev(scheduler, step_index: int, ref: torch.Tensor) -> torch.Tensor:
    if step_index + 1 < len(scheduler.timesteps):
        prev_t = int(scheduler.timesteps[step_index + 1].item())
        return scheduler.alphas_cumprod[prev_t].to(device=ref.device, dtype=ref.dtype), prev_t
    return torch.as_tensor(scheduler.final_alpha_cumprod, device=ref.device, dtype=ref.dtype), 0


def ddim_step_eta(
    *, scheduler, timestep, step_index: int, x_t: torch.Tensor, x0: torch.Tensor,
    eps: torch.Tensor, eta: float, generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    """One DDIM step with stochasticity ``eta`` (Song et al. 2021, eq. 12)."""
    ab_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=x0.device, dtype=x0.dtype)
    ab_prev, prev_t = _alpha_prev(scheduler, step_index, x0)
    sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / ab_prev)
    dir_coef = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0))
    noise = torch.randn(x0.shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=x0.device, dtype=x0.dtype)
    x_prev = torch.sqrt(ab_prev) * x0 + dir_coef * eps + sigma * noise
    return x_prev, prev_t


def systematic_resample(log_w: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    K = log_w.shape[0]
    w = torch.softmax(log_w.float().cpu(), dim=0)
    cum = torch.cumsum(w, dim=0)
    u0 = torch.rand((), generator=generator).item() / K
    u = u0 + torch.arange(K, dtype=torch.float32) / K
    idx = torch.searchsorted(cum, u).clamp(max=K - 1)
    return idx


def effective_sample_size(log_w: torch.Tensor) -> float:
    w = torch.softmax(log_w.float(), dim=0)
    return float(1.0 / (w * w).sum().item())


@torch.no_grad()
def run_particles(
    *,
    init_latents: torch.Tensor,             # (K, 4, h, w), already divided by euler sigma
    models: dict,
    scheduler,
    composition: str,                       # "poe" or "mono"
    seq_a, pool_a, seq_b, pool_b, seq_j, pool_j, seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
    eta: float = 1.0,
    generator: torch.Generator | None = None,
    twist=None,
    twist_temperature: float = 1.0,
    ess_threshold: float = 0.5,
    unet_chunk: int = 2,
) -> ParticleOutputs:
    scheduler.set_timesteps(num_inference_steps)
    K = init_latents.shape[0]
    latents = init_latents.to(device=device, dtype=dtype)
    unet = models["unet"]

    if composition == "poe":
        branches = [(seq_a, pool_a), (seq_b, pool_b), (seq_e, pool_e)]
    elif composition == "mono":
        branches = [(seq_j, pool_j), (seq_e, pool_e)]
    else:
        raise ValueError(composition)
    nb = len(branches)
    pe_one = torch.cat([s for s, _ in branches], dim=0)
    pool_one = torch.cat([p for _, p in branches], dim=0)
    cond_pool_j = pool_j.reshape(1, -1).float().to(device).repeat(K, 1)

    def eps_for(lat: torch.Tensor, timestep) -> torch.Tensor:
        out = []
        for s in range(0, lat.shape[0], unet_chunk):
            chunk = lat[s: s + unet_chunk]
            k = chunk.shape[0]
            latent_input = scheduler.scale_model_input(chunk.repeat_interleave(nb, dim=0), timestep)
            cond = {
                "text_embeds": pool_one.repeat(k, 1),
                "time_ids": add_time_ids(height=height, width=width, batch_size=nb * k,
                                         device=device, dtype=dtype),
            }
            noise = unet(latent_input, timestep, encoder_hidden_states=pe_one.repeat(k, 1, 1),
                         added_cond_kwargs=cond, timestep_cond=None).sample
            noise = noise.view(k, nb, *noise.shape[1:])
            if composition == "poe":
                ea = guided_eps(noise[:, 0], noise[:, 2], guidance_scale)
                eb = guided_eps(noise[:, 1], noise[:, 2], guidance_scale)
                out.append(poe_eps(ea, eb, noise[:, 2]))
            else:
                out.append(guided_eps(noise[:, 0], noise[:, 1], guidance_scale))
        return torch.cat(out)

    def log_psi(lat: torch.Tensor, t_int: int) -> torch.Tensor:
        t = torch.full((lat.shape[0],), float(t_int), device=device)
        return twist(lat.float(), t, cond_pool_j).float() * float(twist_temperature)

    log_w = torch.zeros(K, device=device)
    res = ParticleOutputs(latents=latents, log_weights=log_w)
    if twist is not None:
        log_w = log_psi(latents, int(scheduler.timesteps[0].item()))
        prev_log_psi = log_w.clone()

    for step_index, timestep in enumerate(scheduler.timesteps):
        eps_t = eps_for(latents, timestep)
        ab_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, ab_t, eps_t)
        latents, prev_t = ddim_step_eta(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x_t=latents, x0=x0, eps=eps_t, eta=eta, generator=generator,
        )
        if twist is None:
            continue
        cur_log_psi = log_psi(latents, prev_t)
        log_w = log_w + cur_log_psi - prev_log_psi
        prev_log_psi = cur_log_psi
        ess = effective_sample_size(log_w)
        res.ess.append(ess)
        if ess < ess_threshold * K and step_index + 1 < len(scheduler.timesteps):
            idx = systematic_resample(log_w, generator)
            latents = latents[idx.to(device)]
            prev_log_psi = prev_log_psi[idx.to(device)]
            log_w = torch.zeros(K, device=device)
            res.resampled_at.append(step_index)
            res.ancestry.append(idx.tolist())
    res.latents = latents
    res.log_weights = log_w
    if twist is not None:
        res.log_psi_final = prev_log_psi
    return res


@torch.no_grad()
def decode_one(models: dict, latent: torch.Tensor) -> torch.Tensor:
    return decode_latents(models, latent[None]).cpu()
