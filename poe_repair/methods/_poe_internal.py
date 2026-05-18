"""PoE-internal corrective forces (Idea 1).

A Mono-free repair: at every denoising step, construct a corrective force
from PoE's own UNet outputs (per-concept eps + cross-attention maps),
scale it to clear the empirically measured basin barrier, and add it to
``ε_PoE``. The sampler runs a 3-branch UNet (A, B, ∅) — the same shape as
``run_cfg_poe`` — with no 4th branch on the joint embedding ``e_J``.

Two force variants:

  - **overlap** — uses the cat-token and dog-token cross-attention maps
    extracted from the A and B branches respectively to detect spatial
    collision, then resolves it by pushing each pixel toward the dominant
    concept.
  - **alignment** — uses the pixel-wise dot product of the per-concept
    score deltas to detect score-space collision, then drags the prediction
    back toward unconditional at collision pixels.

Both forces are self-gating: zero where there is no collision.

Calibration (`force_scaler`) is set so that the integrated force
``α₀ · Σ_t ‖F_t‖`` matches the basin barrier height measured by the
veracity diagnostic on the same cell.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.methods._sampling import (
    SamplerOutputs,
    _CrossAttnRecorder,
    add_time_ids,
)
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)


# ---------------------------------------------------------------------------
# Schedule resolution
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
        # Resolved later from basin templates; placeholder value.
        return float(schedule_max)
    raise ValueError(
        f"unknown schedule {schedule!r}; expected one of "
        "{'constant','linear_decay','early_only','closed_loop'}"
    )


def _basin_projection(
    *, x_t: torch.Tensor, x_poe: torch.Tensor, x_mono: torch.Tensor,
) -> float:
    """Coordinate of ``x_t`` along the line from ``x_poe`` to ``x_mono``.

    Returns 0 when ``x_t == x_poe``, 1 when ``x_t == x_mono``. Values in
    between mean the trajectory is partway across the barrier.
    """
    axis = (x_mono - x_poe).flatten().float()
    offset = (x_t.detach().cpu() - x_poe).flatten().float()
    denom = (axis * axis).sum().item()
    if denom <= 1e-12:
        return 0.0
    return float((offset * axis).sum().item() / denom)


# ---------------------------------------------------------------------------
# Force constructions
# ---------------------------------------------------------------------------


def _force_overlap(
    *,
    recorder: _CrossAttnRecorder,
    eps_a: torch.Tensor, eps_b: torch.Tensor,
    token_index_a_solo: int, token_index_b_solo: int,
    latent_h: int, latent_w: int,
    softness: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Force A — attention-overlap repulsion.

    Returns ``(F_t, M_cat_up, M_dog_up)`` where ``F_t`` has shape
    ``[B, C, H, W]`` and the upsampled token maps are ``[H, W]`` each
    (or ``None`` if extraction failed).
    """
    # Aggregate at AAE-canon 16×16 (stable across SDXL resolutions), then
    # bilinear-upsample to the latent grid for broadcasting against eps.
    M_cat = recorder.aggregate_token_map(
        token_index=token_index_a_solo,
        target_hw=(16, 16),
        branch_index=0,
        agg_resolution=16,
    )
    M_dog = recorder.aggregate_token_map(
        token_index=token_index_b_solo,
        target_hw=(16, 16),
        branch_index=1,
        agg_resolution=16,
    )
    if M_cat is None or M_dog is None:
        zero = torch.zeros_like(eps_a)
        return zero, M_cat, M_dog

    def _to_latent(m: torch.Tensor) -> torch.Tensor:
        m = m.to(device=eps_a.device, dtype=eps_a.dtype)
        if m.shape[-2:] == (latent_h, latent_w):
            return m
        return F.interpolate(
            m.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    M_cat_l = _to_latent(M_cat)
    M_dog_l = _to_latent(M_dog)

    def _norm01(x: torch.Tensor) -> torch.Tensor:
        m = x.max()
        if float(m.item()) <= 1e-12:
            return torch.zeros_like(x)
        return x / m

    M_cat_n = _norm01(M_cat_l)
    M_dog_n = _norm01(M_dog_l)

    overlap = M_cat_n * M_dog_n                                  # [H, W]
    sign_field = torch.tanh(softness * (M_cat_n - M_dog_n))      # [H, W]
    f_pixel = (overlap * sign_field).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]

    # Broadcast over batch + channel against (eps_a − eps_b).
    F_t = f_pixel * (eps_a - eps_b)                              # [B, C, H, W]
    # Return the latent-resolution maps so saved artefacts are usable for figs.
    return F_t, M_cat_l, M_dog_l


def _force_alignment(
    *,
    eps_a: torch.Tensor, eps_b: torch.Tensor,
    eps_uncond: torch.Tensor, eps_poe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Force B — score-alignment damping.

    Returns ``(F_t, alignment_field)`` where ``alignment_field`` is the
    per-pixel positive alignment ``a_pos`` of shape ``[B, 1, H, W]``.
    """
    da = eps_a - eps_uncond                                # [B, C, H, W]
    db = eps_b - eps_uncond
    a_field = (da * db).sum(dim=1, keepdim=True)           # [B, 1, H, W]
    a_pos = torch.relu(a_field)
    F_t = a_pos * (eps_uncond - eps_poe)                   # broadcast over channels
    return F_t, a_pos


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_poe_internal_repair(
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
    force_kind: str,
    force_scaler: float = 1.0,
    schedule: str = "constant",
    schedule_max: float = 1.0,
    correction_window: tuple[int, int] | None = None,
    token_index_a_solo: int = 2,
    token_index_b_solo: int = 2,
    basin_templates: dict | None = None,
    closed_loop_threshold: float = 0.5,
    adaptive_schedule: object | None = None,    # idea 2: BasinMonitor + Trigger
    save_residuals_dir: Path | None = None,
    save_dtype: torch.dtype = torch.float16,
) -> SamplerOutputs:
    """3-branch PoE sampler with a Mono-free corrective force.

    ``ε_t = ε̃_PoE + α(t) · force_scaler · F_t``

    where ``F_t`` is computed from PoE-internal signals only (see
    ``_force_overlap`` / ``_force_alignment``). No 4th UNet branch, no
    ``e_J`` encoding.

    Schedule semantics:
      - ``constant`` — α(t) = schedule_max.
      - ``linear_decay`` — α(t) = schedule_max · (1 − t/T).
      - ``early_only`` — α(t) = schedule_max if t < T/5 else 0.
      - ``closed_loop`` — α(t) = schedule_max if the basin projection of
        ``x_t`` is below ``closed_loop_threshold``, else 0. Requires
        ``basin_templates = {'x_t_poe': T+1 traj, 'x_t_mono': T+1 traj}``
        loaded from veracity's λ=0 / λ=1 trajectories on the same cell.
    """
    if force_kind not in {"overlap", "alignment"}:
        raise ValueError(f"force_kind must be 'overlap' or 'alignment', got {force_kind!r}")
    if schedule == "closed_loop" and basin_templates is None and adaptive_schedule is None:
        raise ValueError(
            "schedule='closed_loop' requires basin_templates or adaptive_schedule"
        )
    if adaptive_schedule is not None:
        adaptive_schedule.reset()
    if save_residuals_dir is not None:
        save_residuals_dir = Path(save_residuals_dir)
        save_residuals_dir.mkdir(parents=True, exist_ok=True)

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
    latent_h, latent_w = int(latents.shape[2]), int(latents.shape[3])

    # Pre-load basin templates onto CPU for projection (cheap; small).
    x_poe_traj = x_mono_traj = None
    if schedule == "closed_loop" and adaptive_schedule is None:
        x_poe_traj = basin_templates["x_t_poe"].float().cpu()
        x_mono_traj = basin_templates["x_t_mono"].float().cpu()

    alpha_per_step: list[float] = []
    force_norm_per_step: list[float] = []
    scaled_force_norm_per_step: list[float] = []
    basin_projection_per_step: list[float] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        # --- schedule ---
        if adaptive_schedule is not None:
            alpha_t, proj = adaptive_schedule.alpha(
                step_index=step_index, x_t=latents, base_alpha=schedule_max,
            )
            basin_projection_per_step.append(proj)
        elif schedule == "closed_loop":
            proj = _basin_projection(
                x_t=latents,
                x_poe=x_poe_traj[step_index],
                x_mono=x_mono_traj[step_index],
            )
            basin_projection_per_step.append(proj)
            alpha_t = float(schedule_max) if proj < closed_loop_threshold else 0.0
        else:
            alpha_t = _alpha_static(
                schedule, step_index, num_inference_steps, schedule_max,
            )
        if correction_window is not None:
            t_start, t_end = correction_window
            if step_index < int(t_start) or step_index >= int(t_end):
                alpha_t = 0.0
        alpha_per_step.append(float(alpha_t))

        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)

        # --- UNet forward (with optional cross-attn recorder) ---
        if force_kind == "overlap":
            with _CrossAttnRecorder(unet) as recorder:
                noise = unet(
                    latent_input, timestep, encoder_hidden_states=pe,
                    added_cond_kwargs=cond, timestep_cond=None,
                ).sample
                eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
                eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
                eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
                eps_p = poe_eps(eps_a, eps_b, eps_uncond)
                F_t, M_cat, M_dog = _force_overlap(
                    recorder=recorder,
                    eps_a=eps_a, eps_b=eps_b,
                    token_index_a_solo=token_index_a_solo,
                    token_index_b_solo=token_index_b_solo,
                    latent_h=latent_h, latent_w=latent_w,
                )
            alignment_field = None
        else:
            noise = unet(
                latent_input, timestep, encoder_hidden_states=pe,
                added_cond_kwargs=cond, timestep_cond=None,
            ).sample
            eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
            eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
            eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
            eps_p = poe_eps(eps_a, eps_b, eps_uncond)
            F_t, alignment_field = _force_alignment(
                eps_a=eps_a, eps_b=eps_b, eps_uncond=eps_uncond, eps_poe=eps_p,
            )
            M_cat = M_dog = None

        force_norm_per_step.append(float(F_t.float().norm().item()))
        scaled = float(force_scaler) * float(alpha_t) * F_t
        scaled_force_norm_per_step.append(float(scaled.float().norm().item()))

        # --- save per-step artefact ---
        if save_residuals_dir is not None:
            payload = {
                "x_t": latents.detach().to(save_dtype).cpu(),
                "timestep": int(timestep.item()),
                "step_index": int(step_index),
                "force": F_t.detach().to(save_dtype).cpu(),
                "force_scaled": scaled.detach().to(save_dtype).cpu(),
                "force_kind": force_kind,
                "guidance_scale": float(guidance_scale),
                "alpha_t": float(alpha_t),
                "force_scaler": float(force_scaler),
                "eps_poe": eps_p.detach().to(save_dtype).cpu(),
            }
            if M_cat is not None:
                payload["M_cat"] = M_cat.detach().to(save_dtype).cpu()
            if M_dog is not None:
                payload["M_dog"] = M_dog.detach().to(save_dtype).cpu()
            if alignment_field is not None:
                payload["alignment_field"] = (
                    alignment_field.detach().to(save_dtype).cpu()
                )
            torch.save(
                payload,
                save_residuals_dir / f"step_{step_index:03d}.pt",
            )

        # --- final eps + DDIM step ---
        eps_t = eps_p + scaled
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=dtype,
        )
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
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
    image = decode_latents(models, latents).cpu()

    K_force = float(sum(force_norm_per_step))
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "force_kind": force_kind,
            "force_scaler": float(force_scaler),
            "schedule": schedule,
            "schedule_max": float(schedule_max),
            "correction_window": (
                None if correction_window is None
                else [int(correction_window[0]), int(correction_window[1])]
            ),
            "token_index_a_solo": int(token_index_a_solo),
            "token_index_b_solo": int(token_index_b_solo),
            "closed_loop_threshold": (
                float(closed_loop_threshold) if schedule == "closed_loop" else None
            ),
            "alpha_per_step": alpha_per_step,
            "force_norm_per_step": force_norm_per_step,
            "scaled_force_norm_per_step": scaled_force_norm_per_step,
            "basin_projection_per_step": (
                basin_projection_per_step
                if (schedule == "closed_loop" or adaptive_schedule is not None)
                else None
            ),
            "fired_steps": (
                list(adaptive_schedule.fired_steps)
                if adaptive_schedule is not None else None
            ),
            "K_force": K_force,
            "saved_residuals_dir": (
                None if save_residuals_dir is None else str(save_residuals_dir)
            ),
        },
    )
