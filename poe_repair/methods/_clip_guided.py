"""CLIP-guided PoE repair (Idea 5b).

A Mono-free repair that imports the corrective signal from a *separate*
vision-language model. At each step in a configured correction window:

  1. Run PoE's standard 3-branch UNet (A, B, ∅). No 4th branch on ``e_J``.
  2. Form ``ε_PoE = ε̃_A + ε̃_B − ε_∅`` (detached — we don't backprop
     through the UNet).
  3. Build Tweedie ``x̂_0 = (x_t − √(1−ᾱ_t)·ε_PoE) / √ᾱ_t`` with
     ``requires_grad=True`` on ``x_t``.
  4. Decode ``x̂_0`` through SDXL's VAE *with grad enabled*.
  5. Resize + normalise to CLIP's input convention (224×224 + the
     standard CLIP ImageNet-style mean/std).
  6. Run CLIP's image encoder, take cosine against a cached text
     embedding (e.g. ``"a cat and a dog"``).
  7. ``sim.backward()`` to obtain ``g_t = ∂sim / ∂x_t``.
  8. Score-space update: ``ε_t = ε_PoE − α(t) · σ_t · g_t``.
  9. Standard DDIM step.

The diffusion model never runs on the joint embedding ``e_J``; the
synthesiser is never invoked. Mono enters the pipeline only via the
basin barrier number from veracity (used for calibration of α₀).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
)


# ---------------------------------------------------------------------------
# Schedule resolution (mirror of _poe_internal._alpha_static)
# ---------------------------------------------------------------------------


def _alpha_static(
    schedule: str, step_index: int, num_steps: int, schedule_max: float,
) -> float:
    if schedule == "constant":
        return float(schedule_max)
    if schedule == "linear_decay":
        frac = float(step_index) / float(max(1, num_steps))
        return float(schedule_max) * max(0.0, 1.0 - frac)
    if schedule == "early_only":
        return float(schedule_max) if step_index < (num_steps // 5) else 0.0
    if schedule == "closed_loop":
        return float(schedule_max)   # placeholder; resolved later
    raise ValueError(
        f"unknown schedule {schedule!r}; expected one of "
        "{'constant','linear_decay','early_only','closed_loop'}"
    )


def _basin_projection(
    *, x_t: torch.Tensor, x_poe: torch.Tensor, x_mono: torch.Tensor,
) -> float:
    axis = (x_mono - x_poe).flatten().float()
    offset = (x_t.detach().cpu() - x_poe).flatten().float()
    denom = (axis * axis).sum().item()
    if denom <= 1e-12:
        return 0.0
    return float((offset * axis).sum().item() / denom)


# ---------------------------------------------------------------------------
# CLIP image-pipeline normalisation (kept differentiable)
# ---------------------------------------------------------------------------


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _clip_preprocess(
    image_01: torch.Tensor,    # (1, 3, H, W) in [0, 1]
    *,
    target_size: int = 224,
) -> torch.Tensor:
    if image_01.shape[-1] != target_size or image_01.shape[-2] != target_size:
        image_01 = F.interpolate(
            image_01, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
    mean = torch.tensor(
        _CLIP_MEAN, device=image_01.device, dtype=image_01.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        _CLIP_STD, device=image_01.device, dtype=image_01.dtype,
    ).view(1, 3, 1, 1)
    return (image_01 - mean) / std


def _vae_decode_grad(
    vae: Any, latents: torch.Tensor, *, output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Differentiable mirror of ``decode_latents_to_tensor``.

    Identical scaling convention to ``poe_repair/_sdxl/sdipc_utils.py:133``
    but without the ``@torch.no_grad`` wrapper. Returns a ``[1, 3, H, W]``
    tensor in ``[0, 1]`` of dtype ``output_dtype``.
    """
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    z = latents.to(dtype=vae.dtype)
    images = vae.decode(
        z / vae.config.scaling_factor + shift_factor,
        return_dict=False,
    )[0]
    return (images / 2 + 0.5).clamp(0.0, 1.0).to(dtype=output_dtype)


def _half_res_decode(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode at half latent resolution (cheaper backward graph)."""
    half = F.interpolate(
        latents, size=(latents.shape[-2] // 2, latents.shape[-1] // 2),
        mode="bilinear", align_corners=False,
    )
    return _vae_decode_grad(vae, half)


def _decode_for_clip(
    *,
    decode_strategy: str,
    vae: Any,
    latents: torch.Tensor,
    taesd: Any | None = None,
) -> torch.Tensor:
    if decode_strategy == "full_vae":
        return _vae_decode_grad(vae, latents)
    if decode_strategy == "half_res":
        return _half_res_decode(vae, latents)
    if decode_strategy == "taesd":
        if taesd is None:
            raise ValueError("decode_strategy='taesd' requires a loaded taesd model")
        return taesd.decode(latents).clamp(0.0, 1.0)
    raise ValueError(
        f"unknown decode_strategy {decode_strategy!r}; expected one of "
        "{'full_vae','half_res','taesd'}"
    )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def run_clip_guided_repair(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    # CLIP guidance specifics
    clip_model: Any,
    clip_text_embed: torch.Tensor,           # (1, D), L2-normalised
    correction_window: tuple[int, int] = (10, 25),
    force_scaler: float = 1.0,
    schedule: str = "constant",
    schedule_max: float = 1.0,
    basin_templates: dict | None = None,
    closed_loop_threshold: float = 0.5,
    adaptive_schedule: Any | None = None,    # idea 2: BasinMonitor + Trigger
    decode_strategy: str = "full_vae",       # "full_vae" | "half_res" | "taesd"
    taesd_model: Any | None = None,
    grad_norm_clip: float | None = None,
    # Persistence
    save_residuals_dir: Path | None = None,
    save_dtype: torch.dtype = torch.float16,
) -> SamplerOutputs:
    """3-branch PoE sampler with a CLIP-gradient corrective force.

    The UNet is invoked under ``no_grad``; only the VAE + CLIP path is
    differentiable. ``clip_text_embed`` is passed in pre-normalised to
    avoid re-encoding the text every step.
    """
    if schedule == "closed_loop" and basin_templates is None and adaptive_schedule is None:
        raise ValueError(
            "schedule='closed_loop' requires basin_templates or adaptive_schedule"
        )
    if adaptive_schedule is not None:
        adaptive_schedule.reset()
    if save_residuals_dir is not None:
        save_residuals_dir = Path(save_residuals_dir)
        save_residuals_dir.mkdir(parents=True, exist_ok=True)

    cw_start, cw_end = int(correction_window[0]), int(correction_window[1])

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]
    vae = models["vae"]

    clip_dtype = next(clip_model.parameters()).dtype
    clip_device = next(clip_model.parameters()).device
    clip_text_embed = clip_text_embed.to(device=clip_device, dtype=clip_dtype)
    if clip_text_embed.dim() == 1:
        clip_text_embed = clip_text_embed.unsqueeze(0)

    x_poe_traj = x_mono_traj = None
    if schedule == "closed_loop" and adaptive_schedule is None:
        x_poe_traj = basin_templates["x_t_poe"].float().cpu()
        x_mono_traj = basin_templates["x_t_mono"].float().cpu()

    alpha_per_step: list[float] = []
    in_window_per_step: list[bool] = []
    clip_similarity_per_step: list[float] = []
    grad_norm_per_step: list[float] = []
    force_norm_per_step: list[float] = []
    basin_projection_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        in_window = (cw_start <= step_index < cw_end)
        in_window_per_step.append(bool(in_window))

        # ----- schedule resolution -----
        proj_value: float = -1.0
        if adaptive_schedule is not None:
            alpha_t, proj_value = adaptive_schedule.alpha(
                step_index=step_index, x_t=latents, base_alpha=schedule_max,
            )
        elif schedule == "closed_loop":
            proj_value = _basin_projection(
                x_t=latents,
                x_poe=x_poe_traj[step_index],
                x_mono=x_mono_traj[step_index],
            )
            alpha_t = float(schedule_max) if proj_value < closed_loop_threshold else 0.0
        else:
            alpha_t = _alpha_static(
                schedule, step_index, num_inference_steps, schedule_max,
            )
        if not in_window:
            alpha_t = 0.0
        alpha_per_step.append(float(alpha_t))
        basin_projection_per_step.append(float(proj_value))

        # ----- 3-branch UNet (no_grad) -----
        with torch.no_grad():
            latent_input = scheduler.scale_model_input(
                latents.repeat(3, 1, 1, 1), timestep,
            )
            noise = unet(
                latent_input, timestep, encoder_hidden_states=pe,
                added_cond_kwargs=cond, timestep_cond=None,
            ).sample
            eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
            eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
            eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
            eps_p = poe_eps(eps_a, eps_b, eps_uncond)

        # ----- CLIP gradient (only when firing) -----
        sim_value: float = 0.0
        grad_norm: float = 0.0
        force_norm: float = 0.0
        decoded_for_save: torch.Tensor | None = None
        correction: torch.Tensor | None = None
        g_t: torch.Tensor | None = None

        if alpha_t != 0.0:
            x_t_grad = latents.detach().clone().to(dtype=torch.float32).requires_grad_(True)
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=torch.float32,
            )
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
            sigma_t = (sqrt_one_minus / sqrt_alpha).detach()

            x_hat_0 = (x_t_grad - sqrt_one_minus * eps_p.detach().to(torch.float32)) / sqrt_alpha

            # VAE decode (grad enabled).
            decoded = _decode_for_clip(
                decode_strategy=decode_strategy, vae=vae,
                latents=x_hat_0.to(vae.dtype), taesd=taesd_model,
            )
            decoded_for_save = decoded.detach().to(save_dtype).cpu()

            # CLIP forward (grad enabled).
            clip_input = _clip_preprocess(decoded.to(clip_device).to(clip_dtype))
            img_feat = clip_model.get_image_features(pixel_values=clip_input)
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (img_feat * clip_text_embed).sum()
            sim_value = float(sim.detach().item())

            sim.backward()
            assert x_t_grad.grad is not None
            g_t = x_t_grad.grad.detach().to(device=device, dtype=dtype)

            if grad_norm_clip is not None:
                gn = g_t.float().norm()
                if float(gn.item()) > float(grad_norm_clip):
                    g_t = g_t * (float(grad_norm_clip) / float(gn.item()))

            grad_norm = float(g_t.float().norm().item())

            correction = (
                float(force_scaler) * float(alpha_t) * sigma_t.to(dtype) * g_t
            )
            force_norm = float(correction.float().norm().item())

            # Score-space update: ascend on similarity ⇒ subtract grad in eps space.
            eps_t = eps_p - correction
        else:
            eps_t = eps_p

        clip_similarity_per_step.append(float(sim_value))
        grad_norm_per_step.append(float(grad_norm))
        force_norm_per_step.append(float(force_norm))

        # ----- save per-step artefact -----
        if save_residuals_dir is not None:
            payload = {
                "x_t": latents.detach().to(save_dtype).cpu(),
                "timestep": int(timestep.item()),
                "step_index": int(step_index),
                "eps_poe": eps_p.detach().to(save_dtype).cpu(),
                "alpha_t": float(alpha_t),
                "in_correction_window": bool(in_window),
                "force_scaler": float(force_scaler),
                "guidance_scale": float(guidance_scale),
                "clip_similarity": float(sim_value),
                "grad_norm": float(grad_norm),
                "force_norm": float(force_norm),
            }
            if decoded_for_save is not None:
                payload["decoded_x_hat_0"] = decoded_for_save
            if correction is not None:
                payload["force"] = correction.detach().to(save_dtype).cpu()
            if g_t is not None:
                payload["clip_grad"] = g_t.detach().to(save_dtype).cpu()
            torch.save(
                payload,
                save_residuals_dir / f"step_{step_index:03d}.pt",
            )

        # ----- DDIM step -----
        with torch.no_grad():
            alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype,
            )
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
            x0 = (latents - sqrt_one_minus * eps_t) / sqrt_alpha
            tracker.store_step(
                step_index, latents, eps_t,
                float(step_index) / float(num_inference_steps),
                int(timestep.item()),
            )
            latents = ddim_prev_from_x0_eps(
                scheduler=scheduler, timestep=timestep, step_index=step_index,
                x0=x0, eps=eps_t,
            )

    tracker.store_final(latents)
    with torch.no_grad():
        image = decode_latents(models, latents).cpu()

    K_clip = float(sum(force_norm_per_step))
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "force_scaler": float(force_scaler),
            "schedule": schedule,
            "schedule_max": float(schedule_max),
            "correction_window": [cw_start, cw_end],
            "decode_strategy": decode_strategy,
            "alpha_per_step": alpha_per_step,
            "in_correction_window_per_step": in_window_per_step,
            "clip_similarity_per_step": clip_similarity_per_step,
            "grad_norm_per_step": grad_norm_per_step,
            "force_norm_per_step": force_norm_per_step,
            "basin_projection_per_step": basin_projection_per_step,
            "fired_steps": (
                list(adaptive_schedule.fired_steps)
                if adaptive_schedule is not None else None
            ),
            "K_clip": K_clip,
            "saved_residuals_dir": (
                None if save_residuals_dir is None else str(save_residuals_dir)
            ),
            "closed_loop_threshold": (
                float(closed_loop_threshold) if schedule == "closed_loop" else None
            ),
        },
    )
