"""Sampling primitives.

Reference samplers used by the experiments:

- ``run_cfg_poe``: ε = ε̃_A + ε̃_B − ε_∅ (CI baseline; the failure case).
- ``run_cfg``: plain single-prompt CFG (1 conditional + 1 unconditional);
  used for solo subjects and the Mono diagnostic ceiling.
- ``run_teacher_residual``: λ-interpolated PoE↔Mono sampler for the
  mono-residual diagnostic (idea1, idea5a, veracity).
- ``run_lora_residual_inject``: per-arm LoRA composition (success thread).
- ``run_external_corrector_inject``: group-A external residual corrector
  (failure thread).
- ``run_direct_eps_inject``: direct-ε student wrapper used by group-A.

``_CrossAttnRecorder`` is exposed for cross-attention diagnostics
(used by the veracity diagnostic experiment).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def add_time_ids(*, height, width, batch_size, device, dtype):
    base = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=dtype, device=device,
    )
    return base.repeat(batch_size, 1)


@dataclass
class SamplerOutputs:
    latents: torch.Tensor
    image: torch.Tensor
    tracker: LatentTrajectoryCollector
    extras: dict


def write_decoded_image(image_tensor: torch.Tensor, path: Path) -> Path:
    """Write a [0, 1] decode_latents tensor as PNG.

    The earlier vendored convention applied a redundant ``[-1, 1] -> [0, 1]``
    remap on top of an input that was already in ``[0, 1]``, washing out
    contrast. We clamp to ``[0, 1]`` and quantise directly.
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
    """Recover x_T from disk if present; else from-seed fallback."""
    return load_shared_latents(
        pair_dir=cell.pair_dir,
        seed=cell.seed,
        device=device,
        dtype=dtype,
        models=models,
        height=cell.height,
        width=cell.width,
    )


def _maybe_cap_correction(
    correction: torch.Tensor,
    eps_ref: torch.Tensor,
    max_rel_norm: float | None,
) -> tuple[torch.Tensor, float]:
    """Cap a learned correction's norm to ``max_rel_norm × ||eps_ref||``.

    Returns ``(possibly_scaled_correction, applied_scale)``. If
    ``max_rel_norm`` is None / non-positive the correction is returned
    unchanged with scale=1.0. The cap is on the un-scaled correction
    (independent of any λ_t), so callers should still multiply by their
    schedule afterwards.
    """
    if max_rel_norm is None or float(max_rel_norm) <= 0.0:
        return correction, 1.0
    ref_norm = float(eps_ref.float().norm().item())
    cor_norm = float(correction.float().norm().item())
    cap = float(max_rel_norm) * ref_norm
    if cor_norm > cap and cor_norm > 0.0:
        scale = cap / cor_norm
        return correction * scale, float(scale)
    return correction, 1.0


# ---------------------------------------------------------------------------
# A. Reference samplers
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_cfg_poe(
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
) -> SamplerOutputs:
    """ε̃_PoE = ε̃_A + ε̃_B − ε_∅ — the CI baseline."""
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
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_p = poe_eps(eps_a, eps_b, eps_uncond)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        tracker.store_step(
            step_index, latents, eps_p,
            float(step_index) / float(num_inference_steps),
            int(timestep.item()),
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_p,
        )
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras={})


@torch.no_grad()
def run_cfg(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_cond: torch.Tensor, pool_cond: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
) -> SamplerOutputs:
    """Plain single-prompt CFG: ε̃ = ε_uncond + w·(ε_cond − ε_uncond).

    Used for solo subjects and for the Mono baseline (where the
    conditional is the joint embedding e_J — literal or synthesised).
    """
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_cond, seq_e], dim=0)
    pool = torch.cat([pool_cond, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=2, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]
    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(2, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_cond_raw, eps_uncond = noise.chunk(2)
        eps_t = guided_eps(eps_cond_raw, eps_uncond, guidance_scale)
        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
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
    return SamplerOutputs(latents=latents, image=image, tracker=tracker, extras={})



def _lambda_value(
    schedule: str, step_index: int, num_steps: int, lambda_max: float,
) -> float:
    if schedule == "constant":
        return float(lambda_max)
    if schedule == "linear_decay":
        frac = float(step_index) / float(max(1, num_steps))
        return float(lambda_max) * max(0.0, 1.0 - frac)
    if schedule == "early_only":
        return float(lambda_max) if step_index < (num_steps // 5) else 0.0
    raise ValueError(
        f"unknown lambda_schedule {schedule!r}; expected one of "
        "{'constant','linear_decay','early_only'}"
    )


@torch.no_grad()
def run_teacher_residual(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    lambda_schedule: str = "constant",
    lambda_max: float = 1.0,
    correction_window: tuple[int, int] | None = None,
    save_residuals_dir: Path | None = None,
    save_dtype: torch.dtype = torch.float16,
    save_x0_estimates: bool = False,
    adaptive_schedule: object | None = None,
    attn_capture_dir: Path | None = None,
    attn_token_indices: dict | None = None,
    attn_resolution: int = 32,
) -> SamplerOutputs:
    """Teacher-residual sampler.

    At each step, runs a single 4-branch UNet call (A, B, J, ∅), builds the
    guided PoE prediction ``ε̃_PoE = ε̃_A + ε̃_B − ε_∅`` and the guided Mono
    prediction ``ε̃_Mono = ε̃_J``, and steps with::

        ε_final = ε̃_PoE + λ_t · (ε̃_Mono − ε̃_PoE)

    With ``lambda_max=0`` this reduces to vanilla PoE; with ``lambda_max=1``
    and ``schedule='constant'`` it reduces to literal-e_J Mono. Any value
    in between blends the teacher residual ``Δ_t = ε̃_Mono − ε̃_PoE`` into
    the PoE trajectory.

    Args specific to this sampler:
      lambda_schedule: 'constant' | 'linear_decay' | 'early_only'.
      lambda_max:      peak λ.
      correction_window: optional ``(start, end)`` step-index range outside
        which λ is forced to 0. Inclusive start, exclusive end.
      save_residuals_dir: if set, writes one ``.pt`` per step containing
        ``{x_t, t, step_index, seq_a, seq_b, delta}`` for use as idea-2
        training data. Stored in ``save_dtype`` to halve disk usage.
      save_x0_estimates: if True (and ``save_residuals_dir`` set), also
        persist the guided ``eps_poe`` and ``eps_j`` tensors per step so
        downstream code can compute Tweedie ``x̂_0`` panels at any step.

    Always populates ``extras['pmi_identity_residual_per_step']`` with the
    relative residual of the algebraic identity
    ``Δ_t == w · (ε_J + ε_∅ − ε_A − ε_B)`` (raw conditionals, raw uncond).
    A flat-near-zero curve is the empirical proof that the deployed
    teacher residual is the PMI gradient up to the known constant.
    """
    if save_residuals_dir is not None:
        save_residuals_dir = Path(save_residuals_dir)
        save_residuals_dir.mkdir(parents=True, exist_ok=True)
    if adaptive_schedule is not None:
        adaptive_schedule.reset()
    if attn_capture_dir is not None:
        attn_capture_dir = Path(attn_capture_dir)
        attn_capture_dir.mkdir(parents=True, exist_ok=True)

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe = torch.cat([seq_a, seq_b, seq_j, seq_e], dim=0)
    pool = torch.cat([pool_a, pool_b, pool_j, pool_e], dim=0)
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=4, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    lambda_per_step: list[float] = []
    delta_norm_per_step: list[float] = []
    pmi_identity_residual_per_step: list[float] = []
    basin_projection_per_step: list[float] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        if adaptive_schedule is not None:
            lam, proj_value = adaptive_schedule.alpha(
                step_index=step_index, x_t=latents, base_alpha=lambda_max,
            )
            basin_projection_per_step.append(float(proj_value))
        else:
            lam = _lambda_value(
                lambda_schedule, step_index, num_inference_steps, lambda_max,
            )
        if correction_window is not None:
            t_start, t_end = correction_window
            if step_index < int(t_start) or step_index >= int(t_end):
                lam = 0.0
        lambda_per_step.append(float(lam))

        latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
        if attn_capture_dir is not None and attn_token_indices is not None:
            with _CrossAttnRecorder(unet, keep_grad=False) as _attn_rec:
                noise = unet(
                    latent_input, timestep, encoder_hidden_states=pe,
                    added_cond_kwargs=cond, timestep_cond=None,
                ).sample
                # attn_token_indices schema:
                #   {"<token_key>_<branch_role>": {
                #       "branch_index": int, "token_index": int}}
                # File names: step_XXX_token_<tok>_branch_<role>.pt
                # branch_index: 0=A, 1=B, 2=J, 3=∅ (matches `pe` cat order).
                for save_key, spec in attn_token_indices.items():
                    amap = _attn_rec.aggregate_token_map(
                        int(spec["token_index"]),
                        target_hw=(int(attn_resolution), int(attn_resolution)),
                        branch_index=int(spec["branch_index"]),
                        agg_resolution=int(attn_resolution),
                    )
                    if amap is None:
                        continue
                    fname = f"step_{step_index:03d}_token_{save_key}.pt"
                    torch.save(
                        {
                            "map": amap.float().cpu(),
                            "spec": dict(spec),
                            "step_index": int(step_index),
                            "timestep": int(timestep.item()),
                        },
                        attn_capture_dir / fname,
                    )
        else:
            noise = unet(
                latent_input, timestep, encoder_hidden_states=pe,
                added_cond_kwargs=cond, timestep_cond=None,
            ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_j = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
        eps_poe = poe_eps(eps_a, eps_b, eps_uncond)
        delta = eps_j - eps_poe
        delta_norm_per_step.append(float(delta.float().norm().item()))

        # PMI identity check: Δ should equal w·(ε_J + ε_∅ − ε_A − ε_B) using
        # the raw four UNet outputs. Algebraic derivation:
        #   ε̃ = (1−w)ε_∅ + w·ε  ⇒  ε̃_J − (ε̃_A + ε̃_B − ε_∅)
        #                          = w·(ε_J + ε_∅ − ε_A − ε_B).
        # In score space this is −w·σ_t · ∇_{x_t} PMI(c_1; c_2 | x_t).
        rhs = float(guidance_scale) * (
            eps_j_raw + eps_uncond - eps_a_raw - eps_b_raw
        )
        delta_norm = float(delta.float().norm().item())
        identity_err = float((delta - rhs).float().norm().item())
        pmi_identity_residual_per_step.append(
            identity_err / max(delta_norm, 1e-12)
        )

        if save_residuals_dir is not None:
            payload = {
                "x_t": latents.detach().to(save_dtype).cpu(),
                "timestep": int(timestep.item()),
                "step_index": int(step_index),
                "seq_a": seq_a.detach().to(save_dtype).cpu(),
                "pool_a": pool_a.detach().to(save_dtype).cpu(),
                "seq_b": seq_b.detach().to(save_dtype).cpu(),
                "pool_b": pool_b.detach().to(save_dtype).cpu(),
                "delta": delta.detach().to(save_dtype).cpu(),
                "guidance_scale": float(guidance_scale),
                "eps_a_raw": eps_a_raw.detach().to(save_dtype).cpu(),
                "eps_b_raw": eps_b_raw.detach().to(save_dtype).cpu(),
                "eps_j_raw": eps_j_raw.detach().to(save_dtype).cpu(),
                "eps_uncond": eps_uncond.detach().to(save_dtype).cpu(),
            }
            if save_x0_estimates:
                payload["eps_poe"] = eps_poe.detach().to(save_dtype).cpu()
                payload["eps_j"] = eps_j.detach().to(save_dtype).cpu()
            torch.save(
                payload,
                save_residuals_dir / f"step_{step_index:03d}.pt",
            )

        if lam == 0.0:
            eps_t = eps_poe
        elif lam == 1.0:
            eps_t = eps_j
        else:
            eps_t = eps_poe + float(lam) * delta

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
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "lambda_schedule": lambda_schedule,
            "lambda_max": float(lambda_max),
            "correction_window": (
                None if correction_window is None
                else [int(correction_window[0]), int(correction_window[1])]
            ),
            "lambda_per_step": lambda_per_step,
            "delta_norm_per_step": delta_norm_per_step,
            "pmi_identity_residual_per_step": pmi_identity_residual_per_step,
            "saved_residuals_dir": (
                None if save_residuals_dir is None else str(save_residuals_dir)
            ),
            "saved_x0_estimates": bool(save_x0_estimates),
            "basin_projection_per_step": (
                basin_projection_per_step
                if adaptive_schedule is not None else None
            ),
            "fired_steps": (
                list(adaptive_schedule.fired_steps)
                if adaptive_schedule is not None else None
            ),
        },
    )


@torch.no_grad()
def run_lora_residual_inject(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    lambda_value: float,
    lora_adapter_name: str = "lora",
    record_delta_at_steps: list[int] | None = None,
    correction_max_rel_norm: float | None = None,
) -> SamplerOutputs:
    """LoRA per-arm sampler: PoE with a LoRA-corrected per-arm composition.

    Per step, two 3-branch forwards on (A, B, ∅):
      adapter OFF → ε̃_PoE_frozen   (the failing baseline)
      adapter ON  → ε̃_PoE_lora     (corrected by the trained LoRA)
      Δ̂ = ε̃_PoE_lora − ε̃_PoE_frozen      ← the learned residual, explicit
      ε_final = ε̃_PoE_frozen + lambda_value · Δ̂

    With ``lambda_value = 0`` we use plain frozen PoE (canary).
    With ``lambda_value = 1`` we use the LoRA-corrected composition.

    The joint prompt ``(seq_j, pool_j)`` is no longer consumed at
    inference — the LoRA is the only thing carrying the correction. It is
    kept in the signature for backwards-compat with the probe wiring;
    pass anything (e.g. the unconditional embedding) without effect.

    ``record_delta_at_steps``: cache ``{Δ̂, ε̃_PoE_frozen, x_t,
    tweedie_x0, timestep}`` for the where-applied overlay.

    Adapter management uses the diffusers PeftAdapterMixin API
    (``disable_adapters`` / ``enable_adapters`` / ``set_adapter``).
    """
    del seq_j, pool_j  # unused — kept for signature parity with prior wiring
    record_set = (
        {int(s) for s in record_delta_at_steps}
        if record_delta_at_steps is not None else set()
    )
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )

    pe_3 = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond_3 = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    # Adapter management. Use the PEFT-bridge methods that diffusers exposes
    # on the UNet (PeftAdapterMixin). Falls back to set_adapter / enable.
    def _adapter_disable():
        if hasattr(unet, "disable_adapters"):
            unet.disable_adapters()
        elif hasattr(unet, "disable_adapter_layers"):
            unet.disable_adapter_layers()

    def _adapter_enable():
        if hasattr(unet, "enable_adapters"):
            unet.enable_adapters()
        elif hasattr(unet, "enable_adapter_layers"):
            unet.enable_adapter_layers()
        if hasattr(unet, "set_adapter"):
            try:
                unet.set_adapter(lora_adapter_name)
            except Exception:
                pass

    def _three_branch_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_input_3 = scheduler.scale_model_input(
            latents.repeat(3, 1, 1, 1), timestep,
        )
        noise = unet(
            latent_input_3, timestep, encoder_hidden_states=pe_3,
            added_cond_kwargs=cond_3, timestep_cond=None,
        ).sample
        return noise.chunk(3)

    delta_norm_per_step: list[float] = []
    cap_scale_per_step: list[float] = []
    where_applied: dict[int, dict[str, torch.Tensor]] = {}

    for step_index, timestep in enumerate(scheduler.timesteps):
        # --- frozen 3-branch (adapter OFF) -------------------------------
        _adapter_disable()
        eps_a_raw_f, eps_b_raw_f, eps_uncond_f = _three_branch_forward()
        eps_a_f = guided_eps(eps_a_raw_f, eps_uncond_f, guidance_scale)
        eps_b_f = guided_eps(eps_b_raw_f, eps_uncond_f, guidance_scale)
        eps_poe_frozen = poe_eps(eps_a_f, eps_b_f, eps_uncond_f)

        if float(lambda_value) == 0.0:
            # λ=0 canary — never invokes the adapter. Byte-identical to plain PoE.
            delta_hat = torch.zeros_like(eps_poe_frozen)
            delta_norm_per_step.append(0.0)
            cap_scale_per_step.append(1.0)
            eps_t = eps_poe_frozen
        else:
            # --- LoRA-corrected 3-branch (adapter ON) ---------------------
            _adapter_enable()
            eps_a_raw_l, eps_b_raw_l, eps_uncond_l = _three_branch_forward()
            eps_a_l = guided_eps(eps_a_raw_l, eps_uncond_l, guidance_scale)
            eps_b_l = guided_eps(eps_b_raw_l, eps_uncond_l, guidance_scale)
            eps_poe_lora = poe_eps(eps_a_l, eps_b_l, eps_uncond_l)
            delta_hat = eps_poe_lora - eps_poe_frozen
            delta_capped, applied = _maybe_cap_correction(
                delta_hat, eps_poe_frozen, correction_max_rel_norm,
            )
            delta_norm_per_step.append(float(delta_hat.float().norm().item()))
            cap_scale_per_step.append(float(applied))
            eps_t = eps_poe_frozen + float(lambda_value) * delta_capped
        eps_poe = eps_poe_frozen  # alias for the where-applied cache

        # --- where-applied cache --------------------------------------------
        if step_index in record_set:
            alpha_bar_cache = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype,
            )
            x0_cache = tweedie_mean(latents, alpha_bar_cache, eps_t)
            where_applied[int(step_index)] = {
                "delta_hat": delta_hat.detach().float().cpu(),
                "eps_poe": eps_poe.detach().float().cpu(),
                "x_t": latents.detach().float().cpu(),
                "tweedie_x0": x0_cache.detach().float().cpu(),
                "timestep": int(timestep.item()),
            }

        # --- DDIM step ------------------------------------------------------
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

    # Restore adapter-enabled state on exit so the caller sees a normal UNet.
    _adapter_enable()

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "lambda_value": float(lambda_value),
            "lora_adapter_name": str(lora_adapter_name),
            "delta_norm_per_step": delta_norm_per_step,
            "cap_scale_per_step": cap_scale_per_step,
            "correction_max_rel_norm": (
                None if correction_max_rel_norm is None
                else float(correction_max_rel_norm)
            ),
            "where_applied_cache": where_applied,
            "record_delta_at_steps": sorted(record_set),
        },
    )


@torch.no_grad()
def run_external_corrector_inject(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_j: torch.Tensor, pool_j: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    lambda_value: float,
    corrector,
    record_delta_at_steps: list[int] | None = None,
    correction_max_rel_norm: float | None = None,
) -> SamplerOutputs:
    """Group A sampler: PoE with an external corrector adding an additive ε-residual.

    Per step:
      1. 3-branch (A, B, ∅) frozen UNet forward → ε̃_PoE.
      2. corrector(z_t, t, seq_j, pool_j) → r̂_t (the learned guided residual).
      3. ε_final = ε̃_PoE + λ · r̂_t  (with optional norm cap on r̂_t).
      4. Standard DDIM update.

    With ``lambda_value = 0`` we never call the corrector — the rollout is
    byte-identical to ``run_cfg_poe`` (canary). The joint embedding
    ``(seq_j, pool_j)`` is consumed only by the corrector.

    The corrector is any ``nn.Module`` whose forward signature is
    ``corrector(z_t, t, seq_j, pool_j) -> r̂_t`` with shapes:
        z_t:      (1, 4, H, W)
        t:        (1,) long
        seq_j:    (1, 77, 2048)
        pool_j:   (1, 1280)
        r̂_t:      (1, 4, H, W)
    """
    record_set = (
        {int(s) for s in record_delta_at_steps}
        if record_delta_at_steps is not None else set()
    )
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )

    pe_3 = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond_3 = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    # The corrector lives next to SDXL; keep it on the same device/dtype.
    if corrector is not None:
        corrector_was_training = corrector.training
        corrector.eval()

    delta_norm_per_step: list[float] = []
    cap_scale_per_step: list[float] = []
    where_applied: dict[int, dict[str, torch.Tensor]] = {}

    for step_index, timestep in enumerate(scheduler.timesteps):
        # --- frozen 3-branch PoE forward ----------------------------------
        latent_input_3 = scheduler.scale_model_input(
            latents.repeat(3, 1, 1, 1), timestep,
        )
        noise = unet(
            latent_input_3, timestep, encoder_hidden_states=pe_3,
            added_cond_kwargs=cond_3, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_poe = poe_eps(eps_a, eps_b, eps_uncond)

        if float(lambda_value) == 0.0 or corrector is None:
            # λ=0 canary — corrector is never invoked. Vanilla PoE.
            r_hat = torch.zeros_like(eps_poe)
            delta_norm_per_step.append(0.0)
            cap_scale_per_step.append(1.0)
            eps_t = eps_poe
        else:
            # --- external corrector --------------------------------------
            t_scalar = torch.tensor(
                [int(timestep.item())], device=device, dtype=torch.long,
            )
            r_hat_raw = corrector(latents, t_scalar, seq_j, pool_j)
            r_hat = r_hat_raw.to(dtype=eps_poe.dtype)
            r_hat_capped, applied = _maybe_cap_correction(
                r_hat, eps_poe, correction_max_rel_norm,
            )
            delta_norm_per_step.append(float(r_hat.float().norm().item()))
            cap_scale_per_step.append(float(applied))
            eps_t = eps_poe + float(lambda_value) * r_hat_capped
            r_hat = r_hat_capped  # what gets stored for the overlay

        # --- where-applied cache --------------------------------------------
        if step_index in record_set:
            alpha_bar_cache = scheduler.alphas_cumprod[int(timestep.item())].to(
                device=device, dtype=dtype,
            )
            x0_cache = tweedie_mean(latents, alpha_bar_cache, eps_t)
            where_applied[int(step_index)] = {
                "delta_hat": r_hat.detach().float().cpu(),
                "eps_poe": eps_poe.detach().float().cpu(),
                "x_t": latents.detach().float().cpu(),
                "tweedie_x0": x0_cache.detach().float().cpu(),
                "timestep": int(timestep.item()),
            }

        # --- DDIM step ------------------------------------------------------
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

    if corrector is not None and corrector_was_training:
        corrector.train()

    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "lambda_value": float(lambda_value),
            "delta_norm_per_step": delta_norm_per_step,
            "cap_scale_per_step": cap_scale_per_step,
            "correction_max_rel_norm": (
                None if correction_max_rel_norm is None
                else float(correction_max_rel_norm)
            ),
            "where_applied_cache": where_applied,
            "record_delta_at_steps": sorted(record_set),
        },
    )


@torch.no_grad()
def run_direct_eps_inject(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    student,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device: torch.device, dtype: torch.dtype,
    lambda_schedule: torch.Tensor,
    correction_max_rel_norm: float | None = None,
) -> SamplerOutputs:
    """Method 2b sampler: PoE plus a learned eps-space residual student.

        ε_t = ε̃_PoE + λ_t · δ_θ(x_t, t, pool_a, pool_b, pool_uncond)

    The student is a small CNN trained to match the guided PMI residual
    r_t = ε̃_J − ε̃_PoE = w·(ε_J + ε_∅ − ε_A − ε_B). It conditions on the
    pre-scale latent ``x_t`` (matching the cache convention) and the SDXL
    pooled embeddings of A, B, ∅. No extra UNet branch is needed for the
    student — only the standard 3-branch (A, B, ∅) PoE forward.

    ``correction_max_rel_norm`` (optional safety belt): if set, cap the
    student's correction at ``max_rel_norm × ||ε̃_PoE||`` per step.
    """
    if int(lambda_schedule.shape[0]) != int(num_inference_steps):
        raise ValueError(
            f"lambda_schedule has {int(lambda_schedule.shape[0])} steps, "
            f"expected {num_inference_steps}"
        )
    schedule_list = [float(v) for v in lambda_schedule.tolist()]
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
    student_param = next(student.parameters())
    student_dtype = student_param.dtype
    student_device = student_param.device
    student.eval()
    pool_a_s = pool_a.to(device=student_device, dtype=student_dtype)
    pool_b_s = pool_b.to(device=student_device, dtype=student_dtype)
    pool_e_s = pool_e.to(device=student_device, dtype=student_dtype)

    lambda_per_step: list[float] = []
    delta_norm_per_step: list[float] = []
    cap_scale_per_step: list[float] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        lam = float(schedule_list[step_index])
        lambda_per_step.append(lam)
        latent_input = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_poe = poe_eps(eps_a, eps_b, eps_uncond)

        if lam == 0.0:
            delta_norm_per_step.append(0.0)
            cap_scale_per_step.append(1.0)
            eps_t = eps_poe
        else:
            t_in = torch.tensor(
                [int(timestep.item())], device=student_device, dtype=torch.long,
            )
            x_t_s = latents.to(device=student_device, dtype=student_dtype)
            delta = student(
                x_t=x_t_s, t=t_in,
                pool_a=pool_a_s, pool_b=pool_b_s, pool_uncond=pool_e_s,
            )
            delta = delta.to(device=device, dtype=dtype)
            delta_norm_per_step.append(float(delta.float().norm().item()))
            delta_capped, applied = _maybe_cap_correction(
                delta, eps_poe, correction_max_rel_norm,
            )
            cap_scale_per_step.append(float(applied))
            eps_t = eps_poe + lam * delta_capped

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
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "lambda_per_step": lambda_per_step,
            "delta_norm_per_step": delta_norm_per_step,
            "cap_scale_per_step": cap_scale_per_step,
            "correction_max_rel_norm": (
                None if correction_max_rel_norm is None
                else float(correction_max_rel_norm)
            ),
        },
    )



# ---------------------------------------------------------------------------
# B. Cross-attention recorder (used by diagnostic experiments)
# ---------------------------------------------------------------------------


class _CrossAttnRecorder:
    """Forward-hook based cross-attention recorder, AAE-canon faithful.

    Hooks every cross-attention module (those with ``is_cross_attention=True``)
    and recomputes ``softmax(QK^T/√d)`` from ``to_q(hidden)`` and
    ``to_k(encoder)``. With ``keep_grad=True`` the stored maps preserve
    grad through the latent (used by FOCUS's velocity correction). With
    ``track_self_attn=True`` self-attention modules are also hooked.

    Aggregation matches ``composition/aae/utils/ptp_utils.py:aggregate_attention``:
    filter to ``query_len ≤ 32**2``, optional AAE softmax-renorm over the
    real text tokens (``[1:text_token_count]``), bilinear-resize each layer's
    per-token map to a fixed ``agg_resolution`` (16 for AAE/FOCUS canon),
    average across layers.
    """

    def __init__(self, unet, *, keep_grad: bool = False, track_self_attn: bool = False):
        self.unet = unet
        self.keep_grad = bool(keep_grad)
        self.track_self_attn = bool(track_self_attn)
        self.attn_maps: list[torch.Tensor] = []
        self.self_attn_maps: list[torch.Tensor] = []
        self._hook_handles: list = []

    def __enter__(self):
        self.attn_maps = []
        self.self_attn_maps = []
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ in {"Attention", "CrossAttention"}:
                is_cross = getattr(module, "is_cross_attention", None)
                if is_cross is None:
                    is_cross = getattr(module, "cross_attention_dim", None) is not None
                if not is_cross and not self.track_self_attn:
                    continue
                handle = module.register_forward_hook(
                    self._make_hook(module, is_cross=bool(is_cross)),
                    with_kwargs=True,
                )
                self._hook_handles.append(handle)
        return self

    def _make_hook(self, module, *, is_cross: bool):
        def hook(_mod, args, kwargs, _output):
            hidden = args[0] if len(args) >= 1 else kwargs.get("hidden_states")
            encoder = (
                args[1] if len(args) >= 2 else kwargs.get("encoder_hidden_states")
            )
            if hidden is None:
                return
            if encoder is None:
                if is_cross:
                    return
                encoder = hidden
            try:
                q = module.to_q(hidden)
                k = module.to_k(encoder)
                heads = getattr(module, "heads", 1)
                head_dim = q.shape[-1] // heads
                q = q.view(q.shape[0], q.shape[1], heads, head_dim).transpose(1, 2)
                k = k.view(k.shape[0], k.shape[1], heads, head_dim).transpose(1, 2)
                scale = float(head_dim) ** -0.5
                attn = torch.softmax((q * scale) @ k.transpose(-1, -2), dim=-1)
                stored = attn if self.keep_grad else attn.detach()
                if is_cross:
                    self.attn_maps.append(stored)
                else:
                    self.self_attn_maps.append(stored)
            except Exception:
                pass
        return hook

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def aggregate_token_map(
        self,
        token_index: int,
        target_hw: tuple[int, int],
        *,
        branch_index: int = 0,
        max_query_len: int = 32 * 32,
        text_token_count: int | None = None,
        drop_bos: bool = False,
        agg_resolution: int | None = None,
        keep_grad: bool = False,
    ) -> torch.Tensor | None:
        """Aggregate cross-attn → per-token spatial map.

        - ``max_query_len``: layers above this query-length (i.e., higher
          resolution than ``sqrt(max_query_len)``) are dropped. AAE canon = 1024 (32²).
        - ``drop_bos`` + ``text_token_count``: AAE-style softmax renorm over
          ``[1:text_token_count]`` with ``token_index`` shifted by -1.
        - ``agg_resolution``: bilinear-resize each layer's map to
          ``(agg_resolution, agg_resolution)`` before accumulation. AAE/FOCUS = 16.
        - ``keep_grad``: when False, returns a detached CPU tensor (legacy
          diagnostic path); when True, stays on GPU with grad preserved
          (FOCUS gradient path).
        """
        if not self.attn_maps:
            return None
        if agg_resolution is None:
            H, W = target_hw
        else:
            H = W = int(agg_resolution)
        accum = None
        n = 0
        for attn in self.attn_maps:
            if attn.shape[2] > max_query_len:
                continue
            if attn.shape[-1] <= token_index:
                continue
            head_avg = attn.mean(dim=1)
            if text_token_count is not None and text_token_count > 0:
                tcount = min(int(text_token_count), int(head_avg.shape[-1]))
                if drop_bos:
                    if tcount <= 1 or token_index < 1:
                        continue
                    text_attn = head_avg[..., 1:tcount].float() * 100.0
                    text_attn = torch.softmax(text_attn, dim=-1)
                    shifted = token_index - 1
                    if shifted >= text_attn.shape[-1]:
                        continue
                    tok = text_attn[..., shifted]
                else:
                    if tcount <= token_index:
                        continue
                    text_attn = head_avg[..., :tcount].float() * 100.0
                    text_attn = torch.softmax(text_attn, dim=-1)
                    tok = text_attn[..., token_index]
            else:
                tok = head_avg[..., token_index]
            B, ql = tok.shape
            side = int(round(ql ** 0.5))
            if side * side != ql:
                continue
            if branch_index < 0 or branch_index >= B:
                continue
            spatial = tok[branch_index].reshape(side, side)
            spatial = F.interpolate(
                spatial.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)
            accum = spatial if accum is None else accum + spatial
            n += 1
        if accum is None or n == 0:
            return None
        out = accum / float(n)
        return out if keep_grad else out.detach().cpu()

    def aggregate_self_attention(
        self,
        target_hw: tuple[int, int],
        *,
        branch_index: int = 0,
        max_query_len: int = 32 * 32,
    ) -> torch.Tensor | None:
        """Average self-attn matrices across hooked layers at the target resolution.

        Returns ``[HW, HW]`` row-stochastic. Used by Self-Cross-style losses
        — currently unused in the kept pipeline but preserved as a hook
        point for future experiments.
        """
        if not self.self_attn_maps:
            return None
        H, W = target_hw
        target_qlen = H * W
        accum = None
        n = 0
        for attn in self.self_attn_maps:
            if attn.shape[2] != attn.shape[3]:
                continue
            if attn.shape[2] != target_qlen:
                continue
            if branch_index < 0 or branch_index >= attn.shape[0]:
                continue
            head_avg = attn[branch_index].mean(dim=0)
            accum = head_avg if accum is None else accum + head_avg
            n += 1
        if accum is None or n == 0:
            return None
        return accum / float(n)

