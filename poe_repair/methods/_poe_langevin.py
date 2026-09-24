"""Langevin corrector on the product-of-experts score.

At each of the 50 DDIM noise levels the latent is first settled by ``k`` unadjusted
Langevin steps driven by the guided product-of-experts prediction, and then the
ordinary DDIM step is taken from the settled point. ``k = 0`` is plain
product-of-experts and runs the exact ops of ``run_cfg_poe``, so it is byte-identical
to it.

The update at noise level ``t`` with ``ᾱ_t`` the scheduler's cumulative alpha and
``β_t`` its per-timestep beta::

    s_t = -eps_PoE(x) / sqrt(1 - ᾱ_t)          the PoE score in latent units
    δ_t = c · β_t                               the step size, c the one free parameter
    x  <- x + δ_t · s_t + sqrt(2 δ_t) · z       z ~ N(0, I), repeated k times

This is the vendored ``AnnealedULASampler.sample_step`` in
``composition/reduce_reuse_recycle/anneal_samplers.py`` (``x = x + grad*ss + noise``),
with the gradient function being the PoE score and ``ss = c·β_t`` following that
code's ``la_step_sizes = betas * 0.035``.

Two precision rules. The chain latent is kept in float32 while the corrector runs,
because ``δ_t · s_t`` is around 1e-4 of the latent and float16 cannot represent that
increment near 1.0; the UNet still sees float16 inputs. The DDIM step is taken on the
float16 latent with the reference sampler's own ops, so the ``k = 0`` path is the
reference sampler exactly. Every norm this module records is taken in float32.

The residual measurement is a pure observer. When ``measure_residual`` is on, one
extra two-branch UNet call (joint prompt, unconditional) is made at the settled point
so ``r_t^(k) = eps_J(x_t^(k)) - eps_PoE(x_t^(k))`` can be recorded, and its output is
never fed back into the trajectory. So a run with measurement on and a run with it
off produce the same latents.

Written for plans 25 to 27 of ``plans/06-is-the-gap-the-samplers-or-the-models``.
"""

from __future__ import annotations

import math

import torch

from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
from poe_repair.runtime import (
    LatentTrajectoryCollector,
    decode_latents,
    ddim_prev_from_x0_eps,
    guided_eps,
    poe_eps,
    tweedie_mean,
)


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().item())


def corrector_active(
    step_index: int, corrector_window: tuple[int, int] | None,
) -> bool:
    """Half-open window on step indices, matching the injected-correction runs:
    ``None`` means every step, ``(start, end)`` means ``start <= step < end``."""
    if corrector_window is None:
        return True
    start, end = corrector_window
    return int(start) <= int(step_index) < int(end)


@torch.no_grad()
def run_poe_langevin(
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
    k: int,
    c: float,
    corrector_window: tuple[int, int] | None = None,
    noise_seed: int = 0,
    seq_j: torch.Tensor | None = None, pool_j: torch.Tensor | None = None,
    measure_residual: bool = False,
    probe_inner_counts: tuple[int, ...] = (),
    decode: bool = True,
) -> SamplerOutputs:
    """PoE with ``k`` Langevin corrector steps at every noise level inside the window.

    Returns ``SamplerOutputs`` whose ``extras['per_step']`` holds one dict per noise
    level with, in float32: the corrector's step size, the chain's relative
    displacement ``‖x_end - x_start‖/‖x_start‖`` and norm ratio ``‖x_end‖/‖x_start‖``
    within that level, the largest norm ratio seen along the chain, and, when
    ``measure_residual`` is on, the numerator ``‖eps_J - eps_PoE‖``, the denominator
    ``‖eps_PoE‖`` and their ratio at the settled point, plus the same three at each
    inner count named in ``probe_inner_counts`` (``0`` is the level's starting point).
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if measure_residual and (seq_j is None or pool_j is None):
        raise ValueError("measure_residual=True needs seq_j and pool_j")

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe3 = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool3 = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond3 = {
        "text_embeds": pool3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    if measure_residual:
        pe2 = torch.cat([seq_j, seq_e], dim=0)
        pool2 = torch.cat([pool_j, pool_e], dim=0)
        cond2 = {
            "text_embeds": pool2,
            "time_ids": add_time_ids(
                height=height, width=width, batch_size=2, device=device, dtype=dtype,
            ),
        }
    unet = models["unet"]
    gen = torch.Generator(device=device).manual_seed(int(noise_seed))
    probes = set(int(p) for p in probe_inner_counts)

    def poe_forward(x16: torch.Tensor, timestep) -> torch.Tensor:
        """The reference sampler's three-branch call, op for op."""
        latent_input = scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe3,
            added_cond_kwargs=cond3, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        eps_a = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        return poe_eps(eps_a, eps_b, eps_uncond)

    def joint_forward(x16: torch.Tensor, timestep) -> torch.Tensor:
        latent_input = scheduler.scale_model_input(x16.repeat(2, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe2,
            added_cond_kwargs=cond2, timestep_cond=None,
        ).sample
        eps_j_raw, eps_uncond = noise.chunk(2)
        return guided_eps(eps_j_raw, eps_uncond, guidance_scale)

    def residual_row(eps_p: torch.Tensor, x16: torch.Tensor, timestep) -> dict:
        eps_j = joint_forward(x16, timestep)
        num = _norm(eps_j.float() - eps_p.float())
        den = _norm(eps_p)
        return {"numerator": num, "denominator": den,
                "ratio": num / max(den, 1e-12)}

    per_step: list[dict] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        t_int = int(timestep.item())
        alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=dtype)
        beta_t = float(scheduler.betas[t_int].item())
        sigma_t = math.sqrt(max(1.0 - float(alpha_bar_t.item()), 1e-12))
        n_inner = int(k) if corrector_active(step_index, corrector_window) else 0
        delta = float(c) * beta_t

        row = {
            "step_index": int(step_index), "timestep": t_int,
            "beta_t": beta_t, "alpha_bar_t": float(alpha_bar_t.item()),
            "k_applied": n_inner, "delta_t": delta if n_inner > 0 else 0.0,
        }
        x_start32 = latents.float()
        start_norm = _norm(x_start32)
        max_norm_ratio = 1.0
        probe_rows: dict[str, dict] = {}

        if n_inner > 0:
            x32 = x_start32.clone()
            noise_std = math.sqrt(2.0 * delta)
            for j in range(n_inner):
                x16 = x32.to(dtype)
                eps_p = poe_forward(x16, timestep)
                if measure_residual and j in probes:
                    probe_rows[str(j)] = residual_row(eps_p, x16, timestep)
                score = -eps_p.float() / sigma_t
                z = torch.randn(x32.shape, generator=gen, device=device, dtype=torch.float32)
                x32 = x32 + delta * score + noise_std * z
                max_norm_ratio = max(max_norm_ratio, _norm(x32) / max(start_norm, 1e-12))
            latents = x32.to(dtype)
            row["chain_disp_rel"] = _norm(x32 - x_start32) / max(start_norm, 1e-12)
            row["latent_norm_rel"] = _norm(x32) / max(start_norm, 1e-12)
        else:
            row["chain_disp_rel"] = 0.0
            row["latent_norm_rel"] = 1.0
        row["max_latent_norm_rel"] = max_norm_ratio

        # The settled point. This three-branch call is the reference sampler's own
        # per-step call, and at k = 0 it is the only call made.
        eps_p = poe_forward(latents, timestep)
        if measure_residual:
            row.update(residual_row(eps_p, latents, timestep))
            if n_inner > 0 and n_inner in probes:
                probe_rows[str(n_inner)] = {kk: row[kk] for kk in ("numerator", "denominator", "ratio")}
            if probe_rows:
                row["probes"] = probe_rows
        per_step.append(row)

        x0 = tweedie_mean(latents, alpha_bar_t, eps_p)
        tracker.store_step(
            step_index, latents, eps_p,
            float(step_index) / float(num_inference_steps),
            t_int,
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_p,
        )
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu() if decode else None
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "k": int(k), "c": float(c),
            "corrector_window": (
                None if corrector_window is None
                else [int(corrector_window[0]), int(corrector_window[1])]
            ),
            "noise_seed": int(noise_seed),
            "measure_residual": bool(measure_residual),
            "step_size_rule": "delta_t = c * beta_t; x <- x + delta_t * (-eps_PoE / sqrt(1 - alpha_bar_t)) + sqrt(2 delta_t) z",
            "per_step": per_step,
        },
    )


# ---------------------------------------------------------------------------
# The corrector on top of the trained adapter (the fidelity read, plan 27's addition)
# ---------------------------------------------------------------------------


def _adapter_disable(unet) -> None:
    if hasattr(unet, "disable_adapters"):
        unet.disable_adapters()
    elif hasattr(unet, "disable_adapter_layers"):
        unet.disable_adapter_layers()


def _adapter_enable(unet, name: str) -> None:
    if hasattr(unet, "enable_adapters"):
        unet.enable_adapters()
    elif hasattr(unet, "enable_adapter_layers"):
        unet.enable_adapter_layers()
    if hasattr(unet, "set_adapter"):
        try:
            unet.set_adapter(name)
        except Exception:
            pass


@torch.no_grad()
def run_lora_langevin_windowed_poe(
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
    lambda_value: float,
    k: int,
    c: float,
    corrector_window: tuple[int, int] | None,
    noise_seed: int = 0,
    lora_adapter_name: str = "lora",
    lambda_window: tuple[int, int] | None = None,
    corrector_score: str = "corrected",
    lora_adapter_name_late: str | None = None,
    adapter_switch_at: int | None = None,
    spread_match: float = 0.0,
    reward_guidance: dict | None = None,
    contrast_steps: int = 0,
    contrast_iters: int = 5,
    contrast_beta: float = 0.9,
    contrast_w_multi: float = 2.0,
) -> SamplerOutputs:
    """The rank-32 corrected run with ``k`` Langevin steps inside ``corrector_window``.

    Every step's prediction is the one ``run_lora_residual_inject_windowed_poe`` in
    ``scripts/showcase/lambda_window_grid.py`` makes with the adapter on at all 50
    steps: ``eps_t = eps_PoE_frozen + λ · (eps_PoE_adapter - eps_PoE_frozen)``. Inside
    the corrector window the Langevin drift is that same corrected prediction, so the
    chain settles into the distribution the corrected model describes, not plain
    product-of-experts. With ``k = 0`` this is the adapter-alone run, op for op.

    Two switches for the clean-tail conditions. ``lambda_window`` restricts the adapter
    to a half-open step range: outside it λ is 0 and the adapter forward is skipped, so
    the step is the frozen model's plain PoE step. ``corrector_score`` picks the Langevin
    drift: ``"corrected"`` (the default, the λ-corrected prediction) or ``"frozen"`` (the
    frozen model's PoE prediction, so the chain settles into the frozen model's own
    low-noise distribution whatever the adapter did earlier).

    A third switch hands the run from one adapter to another part-way through. Attach both
    under different names, pass the second as ``lora_adapter_name_late`` and the step it takes
    over as ``adapter_switch_at``, and the first draws steps ``[0, switch)`` while the second
    draws ``[switch, end)``. Composition is decided in the early steps and the picture is
    finished in the late ones, so this lets one adapter do each job. Leave both at ``None`` and
    a single adapter runs the whole path, unchanged.

    ``spread_match`` is Lin et al.'s guidance rescale (2305.08891, section 3.4) moved onto the
    product: each step's combined prediction is rescaled so its standard deviation matches the
    mean of the two unguided concept predictions, then blended back in at this weight (their
    choice is 0.7). 0 leaves the prediction as it was, which is every run before it existed.

    ``contrast_steps`` switches on CO3's contrast corrector (Dutta et al., 2509.25940, its
    ``co3_corrector``) over the first that many steps, ``contrast_iters`` times each. CO3 subtracts
    the single concepts from a joint-prompt prediction; here the corrected product stands in for the
    joint prompt, so no joint prompt is needed. Each pass takes the corrected product's predicted
    clean image at weight ``contrast_w_multi``, subtracts each concept's own predicted clean image
    with weights ``-exp(-beta d_k)`` normalised to sum to -1 (``d_k`` is how far that concept's
    prediction sits from the product's, so whichever concept is taking over is pushed away hardest),
    rescales the result to the size the product alone would have had, and rebuilds the latent with
    the empty branch. 0 steps leaves the path untouched, which is every run before it existed.

    One Langevin step costs two three-branch UNet calls (frozen and adapter) when the
    drift is corrected and the adapter is on, one call otherwise.
    """
    if corrector_score not in ("corrected", "frozen"):
        raise ValueError(f"corrector_score must be 'corrected' or 'frozen', got {corrector_score!r}")
    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)
    tracker = LatentTrajectoryCollector(
        num_inference_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3]
    )
    pe3 = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool3 = torch.cat([pool_a, pool_b, pool_e], dim=0)
    cond3 = {
        "text_embeds": pool3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]
    gen = torch.Generator(device=device).manual_seed(int(noise_seed))

    def branch_parts(x16: torch.Tensor, timestep):
        """The three guided pieces of one step: each concept on its own, and the empty branch."""
        latent_input = scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe3,
            added_cond_kwargs=cond3, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_uncond = noise.chunk(3)
        return (guided_eps(eps_a_raw, eps_uncond, guidance_scale),
                guided_eps(eps_b_raw, eps_uncond, guidance_scale), eps_uncond)

    def compose(eps_a: torch.Tensor, eps_b: torch.Tensor, eps_uncond: torch.Tensor) -> torch.Tensor:
        out = poe_eps(eps_a, eps_b, eps_uncond)
        if spread_match > 0.0:
            out32 = out.float()
            ref = 0.5 * (eps_a.float().std() + eps_b.float().std())
            rescaled = out32 * (ref / out32.std().clamp_min(1e-8))
            out = (float(spread_match) * rescaled + (1.0 - float(spread_match)) * out32).to(out.dtype)
        return out

    # The last step's two concept predictions, kept so reward guidance can ask where each concept
    # wants mass without a second forward pass.
    last_branches: dict[str, torch.Tensor] = {}

    def three_branch(x16: torch.Tensor, timestep) -> torch.Tensor:
        eps_a, eps_b, eps_uncond = branch_parts(x16, timestep)
        last_branches["a"], last_branches["b"] = eps_a.detach(), eps_b.detach()
        return compose(eps_a, eps_b, eps_uncond)

    def adapter_for(step_index: int) -> str:
        """Which attached adapter draws this step."""
        if lora_adapter_name_late is None or adapter_switch_at is None:
            return lora_adapter_name
        return lora_adapter_name if step_index < int(adapter_switch_at) else lora_adapter_name_late

    def corrected_forward(x16: torch.Tensor, timestep, adapter_on: bool,
                          step_index: int = 0) -> tuple[torch.Tensor, float]:
        _adapter_disable(unet)
        eps_frozen = three_branch(x16, timestep)
        if not adapter_on:
            return eps_frozen, 0.0
        _adapter_enable(unet, adapter_for(step_index))
        eps_lora = three_branch(x16, timestep)
        delta_hat = eps_lora - eps_frozen
        return eps_frozen + float(lambda_value) * delta_hat, _norm(delta_hat)

    def frozen_forward(x16: torch.Tensor, timestep) -> torch.Tensor:
        _adapter_disable(unet)
        return three_branch(x16, timestep)

    per_step: list[dict] = []
    delta_norm_per_step: list[float] = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        t_int = int(timestep.item())
        alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=dtype)
        beta_t = float(scheduler.betas[t_int].item())
        sigma_t = math.sqrt(max(1.0 - float(alpha_bar_t.item()), 1e-12))
        n_inner = int(k) if corrector_active(step_index, corrector_window) else 0
        adapter_on = corrector_active(step_index, lambda_window) and float(lambda_value) != 0.0
        delta = float(c) * beta_t
        x_start32 = latents.float()
        start_norm = _norm(x_start32)
        row = {"step_index": int(step_index), "timestep": t_int, "k_applied": n_inner,
               "adapter_on": bool(adapter_on), "delta_t": delta if n_inner > 0 else 0.0,
               "adapter_name": adapter_for(step_index) if adapter_on else None}
        if n_inner > 0:
            x32 = x_start32.clone()
            noise_std = math.sqrt(2.0 * delta)
            for _ in range(n_inner):
                if corrector_score == "frozen":
                    eps_t = frozen_forward(x32.to(dtype), timestep)
                else:
                    eps_t, _dn = corrected_forward(x32.to(dtype), timestep, adapter_on, step_index)
                score = -eps_t.float() / sigma_t
                z = torch.randn(x32.shape, generator=gen, device=device, dtype=torch.float32)
                x32 = x32 + delta * score + noise_std * z
            latents = x32.to(dtype)
            row["chain_disp_rel"] = _norm(x32 - x_start32) / max(start_norm, 1e-12)
            row["latent_norm_rel"] = _norm(x32) / max(start_norm, 1e-12)
        else:
            row["chain_disp_rel"] = 0.0
            row["latent_norm_rel"] = 1.0
        if contrast_steps > 0 and step_index < int(contrast_steps):
            sqrt_1mab = (1.0 - alpha_bar_t).sqrt()
            for _ in range(int(contrast_iters)):
                _adapter_disable(unet)
                eps_a_g, eps_b_g, eps_uncond = branch_parts(latents, timestep)
                eps_multi = compose(eps_a_g, eps_b_g, eps_uncond)
                if adapter_on:
                    _adapter_enable(unet, adapter_for(step_index))
                    eps_multi = eps_multi + float(lambda_value) * (three_branch(latents, timestep) - eps_multi)
                x32 = latents.float()
                tweedie_multi = x32 - sqrt_1mab.float() * eps_multi.float()
                singles = [eps_a_g.float(), eps_b_g.float()]
                dists = torch.stack([(e - eps_multi.float()).norm() for e in singles])
                w = -torch.exp(-float(contrast_beta) * dists)
                w = w / w.sum().abs().clamp_min(1e-8)
                comp = float(contrast_w_multi) * tweedie_multi
                for w_k, e_k in zip(w, singles):
                    comp = comp + w_k * (x32 - sqrt_1mab.float() * e_k)
                comp = comp / comp.max().clamp_min(1e-8) * tweedie_multi.max()
                x2 = comp + sqrt_1mab.float() * eps_uncond.float()
                x2 = x2 * (x32.norm() / x2.norm().clamp_min(1e-8))
                latents = x2.to(dtype)
            row["contrast_disp_rel"] = _norm(latents.float() - x_start32) / max(start_norm, 1e-12)
        eps_t, dn = corrected_forward(latents, timestep, adapter_on, step_index)
        delta_norm_per_step.append(dn)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        # Reward guidance: decode the clean picture this step is heading for, score it for two
        # concepts that are each themselves and unlike each other, and push the latent up that
        # gradient. The same objective is what the reward fine-tune trains against, so a setting
        # that moves pictures here is a setting worth training with.
        if reward_guidance and reward_guidance["lo"] <= step_index < reward_guidance["hi"]:
            rg = reward_guidance
            with torch.enable_grad():
                z = latents.detach().float().requires_grad_(True)
                x0_g = tweedie_mean(z.to(dtype), alpha_bar_t, eps_t.detach())
                # Both ordinary decoders are wrapped in no_grad, which would cut the chain
                # before the reward ever sees the picture.
                from poe_repair._sdxl.sdipc_utils import decode_latents_with_grad
                img = decode_latents_with_grad(models["vae"], x0_g.to(dtype))
                mask_a = mask_b = None
                if "a" in last_branches:
                    from poe_repair.rewards.plurality import soft_masks
                    x0_a = tweedie_mean(latents, alpha_bar_t, last_branches["a"]).float()
                    x0_b = tweedie_mean(latents, alpha_bar_t, last_branches["b"]).float()
                    mask_a, mask_b = soft_masks(x0_a, x0_b)
                total, terms = rg["reward"].score(
                    img, prompt_a=rg["prompt_a"], prompt_b=rg["prompt_b"],
                    mask_a=mask_a, mask_b=mask_b)
                grad, = torch.autograd.grad(total, z)
            step = rg["weight"] * grad / grad.norm().clamp_min(1e-8) * z.detach().norm()
            latents = (z.detach() + step).to(dtype)
            row["reward"] = terms.as_dict()
            row["reward_step_rel"] = float(step.norm() / z.detach().norm().clamp_min(1e-8))
            eps_t, dn = corrected_forward(latents, timestep, adapter_on, step_index)
            x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        per_step.append(row)
        tracker.store_step(
            step_index, latents, eps_t,
            float(step_index) / float(num_inference_steps), t_int,
        )
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_t,
        )
    _adapter_enable(unet, lora_adapter_name)
    tracker.store_final(latents)
    image = decode_latents(models, latents).cpu()
    return SamplerOutputs(
        latents=latents, image=image, tracker=tracker,
        extras={
            "lambda_value": float(lambda_value), "k": int(k), "c": float(c),
            "lambda_window": (
                None if lambda_window is None
                else [int(lambda_window[0]), int(lambda_window[1])]
            ),
            "corrector_score": corrector_score,
            "spread_match": float(spread_match),
            "reward_guidance": (None if not reward_guidance else
                                {k: v for k, v in reward_guidance.items() if k != "reward"}),
            "contrast": {"steps": int(contrast_steps), "iters": int(contrast_iters),
                         "beta": float(contrast_beta), "w_multi": float(contrast_w_multi)},
            "corrector_window": (
                None if corrector_window is None
                else [int(corrector_window[0]), int(corrector_window[1])]
            ),
            "noise_seed": int(noise_seed),
            "delta_norm_per_step": delta_norm_per_step,
            "per_step": per_step,
        },
    )
