"""LoRA trainer: LoRA injection + training loop on the cached (x_t, t, Δ_t) set.

Single-cell training. The cache holds (50 steps) raw eps tensors and x_t;
we compute Δ_t in closed form per step. LoRA is attached to SDXL UNet
cross-attention projections (``attn2``) at rank 8. The loss is MSE on the
guided residual:

    Δ̂ = guided(ε_LoRA(joint)) − guided_PoE(ε_a, ε_b, ε_uncond)
    loss = ‖Δ̂ − Δ_t‖²

Algebraically this reduces to ``MSE(eps_lora_raw, eps_j_raw_cached)``
(see consolidated plan §1.3 equivalence note), but we keep the explicit
residual form so wiring bugs in either branch surface clearly.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from poe_repair.experiments.one_pair_one_seed.config import RunConfig
from poe_repair.training_cache import (
    CellPath,
    delta_t_from_raw,
    load_step_raw,
)
from poe_repair.runtime import guided_eps, poe_eps


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CachedStep:
    step_index: int
    timestep: int
    x_t: torch.Tensor          # (1, 4, H, W), float32 CPU
    eps_a_raw: torch.Tensor
    eps_b_raw: torch.Tensor
    eps_j_raw: torch.Tensor
    eps_uncond: torch.Tensor
    delta_t: torch.Tensor      # guided Δ_t target
    render_latent: torch.Tensor | None = None   # V6: the picked mono render, VAE-encoded
    source_seed: int = -1      # -1 = legacy single-seed (unset)
    source_pair: str = ""      # empty = legacy single-pair (unset)


def load_cached_steps(
    cell: CellPath,
    *,
    guidance_scale: float,
) -> list[CachedStep]:
    """Load all step files for a cell. ~250 MB for 50 SDXL steps."""
    entries: list[CachedStep] = []
    for path in cell.step_files():
        raw = load_step_raw(path)
        delta = delta_t_from_raw(
            raw["eps_a_raw"], raw["eps_b_raw"],
            raw["eps_j_raw"], raw["eps_uncond"],
            guidance_scale,
        )
        entries.append(
            CachedStep(
                step_index=raw["step_index"],
                timestep=raw["timestep"],
                x_t=raw["x_t"],
                eps_a_raw=raw["eps_a_raw"],
                eps_b_raw=raw["eps_b_raw"],
                eps_j_raw=raw["eps_j_raw"],
                eps_uncond=raw["eps_uncond"],
                delta_t=delta,
                source_seed=int(cell.seed),
            )
        )
    if not entries:
        raise FileNotFoundError(
            f"no cached steps under {cell.residuals_dir}"
        )
    return entries


def load_cached_steps_pooled(
    cells: list[CellPath],
    *,
    guidance_scale: float,
) -> list[CachedStep]:
    """Concatenate cached steps across multiple cells. Order: cells[0]'s
    steps first, then cells[1], etc. Each entry tagged with ``source_seed``.
    Uniform random sampling over the returned list gives per-seed mixing
    for free (no special scheduler needed)."""
    out: list[CachedStep] = []
    for c in cells:
        out.extend(load_cached_steps(c, guidance_scale=guidance_scale))
    if not out:
        raise FileNotFoundError(
            f"no cached steps across {len(cells)} cells"
        )
    return out


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def _bucket(step_index: int, *, commit_window: tuple[int, int]) -> str:
    lo, hi = int(commit_window[0]), int(commit_window[1])
    if step_index < lo:
        return "early"
    if step_index < hi:
        return "commit"
    return "late"


def _loss_weights(
    timesteps: list[int],
    alphas_cumprod: torch.Tensor,
    *,
    loss_space: str,
    cap: float,
    num_inference_steps: int,
) -> torch.Tensor:
    """Per-sample multiplier on the noise-space squared error (scope 01 plan 16, experiment E).

    ``"eps"``: 1 for every sample, the original trainer.

    ``"x0"``: ``(1 - abar_t) / abar_t`` clipped at ``cap``. Two Tweedie clean estimates formed
    from the same ``x_t`` differ by ``sqrt(1 - abar) / sqrt(abar)`` times the difference of the
    two noise predictions, so this factor turns the noise-space MSE into the clean-estimate MSE
    (the loss JiT, Li and He 2025, argue for). At the first DDIM timestep the factor is 172 and at
    the last 0.002, so the clip keeps the first few steps from owning every gradient. The weights
    are divided by the mean of the clipped factor over the DDIM grid the caches use, so the loss
    scale, and with it the learning rate, stays comparable with the unweighted baseline.
    """
    if loss_space == "eps":
        return torch.ones(len(timesteps), dtype=torch.float32)
    if loss_space != "x0":
        raise ValueError(f"unknown loss_space {loss_space!r}; expected 'eps' or 'x0'")
    ab = alphas_cumprod.detach().float().cpu()
    factor = (1.0 - ab) / ab
    grid = torch.arange(num_inference_steps) * (len(ab) // num_inference_steps) + 1  # 1, 21, ..., 981
    norm = factor[grid].clamp(max=cap).mean()
    w = factor[torch.tensor(timesteps, dtype=torch.long)].clamp(max=cap) / norm
    return w.float()


# ---------------------------------------------------------------------------
# LoRA injection
def control_energy_weights(
    timesteps: list[int] | torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    num_inference_steps: int,
) -> torch.Tensor:
    """Per-step weight that turns a squared noise-prediction difference into control energy.

    Read the reverse run as an SDE with the same marginals (the Schrödinger-bridge / stochastic
    optimal control view, plan 19 of scope 01). A change ``r`` to the noise prediction at
    timestep ``t`` is a change of ``-r / sigma_t`` to the score, ``sigma_t^2 = 1 - abar_t``, and
    the reverse SDE's drift moves by ``g_t^2`` times the score change. Over one DDIM step from
    ``t`` down to ``t_prev`` the integrated ``g_t^2`` is ``beta_step = 1 - abar_t / abar_prev``.
    Girsanov then prices the path-space KL between the corrected and the plain run at
    ``sum_k 0.5 * beta_step_k / (1 - abar_k) * ||r_k||^2``, so the weight returned here is
    ``beta_step_k / (1 - abar_k)`` per sample; multiply by ``0.5 * ||r||^2`` for nats.
    ``t_prev`` is ``t`` minus the DDIM stride (1000 // num_inference_steps), floored at 0.
    """
    ab = alphas_cumprod.detach().float().cpu()
    stride = len(ab) // int(num_inference_steps)
    t = torch.as_tensor(list(timesteps) if not torch.is_tensor(timesteps) else timesteps,
                        dtype=torch.long)
    t_prev = (t - stride).clamp(min=0)
    beta_step = 1.0 - ab[t] / ab[t_prev]
    return (beta_step / (1.0 - ab[t])).float()


def control_energy_weights_normalised(
    timesteps: list[int] | torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    num_inference_steps: int,
) -> torch.Tensor:
    """``control_energy_weights`` divided by its mean over the DDIM grid the caches use, so an
    energy penalty of ``beta`` on the per-element mean square of the correction reads as
    ``beta`` times a plain mean-square penalty on average, and the late steps (where the weight
    is largest) carry proportionally more."""
    ab = alphas_cumprod.detach().float().cpu()
    stride = len(ab) // int(num_inference_steps)
    grid = torch.arange(int(num_inference_steps)) * stride + 1                   # 1, 21, ..., 981
    norm = control_energy_weights(grid, ab, num_inference_steps=num_inference_steps).mean()
    return control_energy_weights(timesteps, ab, num_inference_steps=num_inference_steps) / norm


# ---------------------------------------------------------------------------


def attach_lora(unet: torch.nn.Module, cfg: RunConfig) -> dict[str, Any]:
    """Attach a LoRA adapter to the UNet's cross-attention projections.

    Returns a dict with ``{"adapter_name", "matched_modules",
    "trainable_params"}`` for sanity-printing at run start.
    """
    from peft import LoraConfig

    lora_cfg = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        bias="none",
        target_modules=list(cfg.lora.target_modules),
        init_lora_weights=(
            True if cfg.lora.init == "gaussian"
            else cfg.lora.init  # passthrough for "loftq" / etc.
        ),
    )

    # Diffusers UNet PeftAdapterMixin: ``add_adapter`` attaches the LoRA
    # and registers it under ``adapter_name``.
    if not hasattr(unet, "add_adapter"):
        raise RuntimeError(
            "UNet does not expose add_adapter — diffusers >=0.21 is required."
        )
    unet.add_adapter(lora_cfg, adapter_name=cfg.lora.adapter_name)

    # A module is "matched" if it has injected lora_A / lora_B sub-modules.
    matched: list[str] = []
    for name, module in unet.named_modules():
        sub_names = {n for n, _ in module.named_children()}
        if "lora_A" in sub_names and "lora_B" in sub_names:
            matched.append(name)

    # Enable grad on adapter params only. Cast LoRA params to fp32 — fp16
    # LoRA training underflows immediately (loss NaN at step 1). Frozen
    # base UNet stays in its original dtype; peft's LoraLayer.forward casts
    # the LoRA delta back to the base layer's input dtype before adding,
    # so the mixed-precision path is safe.
    n_trainable = 0
    n_total = 0
    with torch.no_grad():
        for name, p in unet.named_parameters():
            n_total += p.numel()
            if "lora_" in name:
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
                n_trainable += p.numel()
            else:
                p.requires_grad_(False)

    if n_trainable == 0:
        raise RuntimeError(
            "LoRA attach produced 0 trainable parameters. "
            "Check target_modules regex — SDXL cross-attn lives at attn2."
        )

    return {
        "adapter_name": cfg.lora.adapter_name,
        "matched_modules": matched,
        "n_matched": len(matched),
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
    }


def lora_state_dict(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA-only state for checkpointing (~20 MB at rank 8)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in unet.named_parameters():
        if "lora_" in name:
            out[name] = p.detach().to("cpu", torch.float32).clone()
    return out


def load_lora_state(unet: torch.nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Inverse of ``lora_state_dict``. Tensors are placed on the matching
    module's device/dtype.

    Returns how many tensors were actually copied. A key that names no parameter is skipped,
    which is what happens when the checkpoint was saved under one adapter name and the model
    carries another: every key misses, nothing is written, and the adapter stays at its zero
    initialisation. The count is the only thing that tells those two cases apart, so callers
    report it rather than the size of the file."""
    own_params = dict(unet.named_parameters())
    copied = 0
    with torch.no_grad():
        for name, t in state.items():
            if name not in own_params:
                continue
            p = own_params[name]
            p.copy_(t.to(device=p.device, dtype=p.dtype))
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _build_added_cond_kwargs_3branch(
    pool_a: torch.Tensor, pool_b: torch.Tensor, pool_e: torch.Tensor,
    *,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Concat 3-branch (A, B, ∅) embeddings + the time-id batch."""
    from poe_repair.methods._sampling import add_time_ids
    pool_3 = torch.cat([pool_a, pool_b, pool_e], dim=0).to(device=device, dtype=dtype)
    cond = {
        "text_embeds": pool_3,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=3, device=device, dtype=dtype,
        ),
    }
    return pool_3, cond


def _compose(
    eps_a_raw: torch.Tensor, eps_b_raw: torch.Tensor, eps_uncond: torch.Tensor,
    gs: float, compose: str, kappa: float,
) -> torch.Tensor:
    """Combine the three raw branch predictions under one composition rule.

    "poe":       w·(ε̃_a + ε̃_b − ε_∅), the original trainer, each branch guided first.
    "superdiff": ε_∅ + w·((ε_b − ε_∅) + κ·(ε_a − ε_b)), the AND blend of
                 `superdiff/superdiff-sdxl-v1-0` at a fixed κ, a = prompt 1, b = prompt 2,
                 matching poe_repair/composers/superdiff.py with kappa_override=κ.
    """
    if compose == "poe":
        return poe_eps(guided_eps(eps_a_raw, eps_uncond, gs),
                       guided_eps(eps_b_raw, eps_uncond, gs), eps_uncond)
    if compose == "superdiff":
        return eps_uncond + gs * ((eps_b_raw - eps_uncond) + kappa * (eps_a_raw - eps_b_raw))
    raise ValueError(f"unknown compose rule {compose!r}; expected 'poe' or 'superdiff'")


def _train_one_step(
    *,
    unet: torch.nn.Module,
    scheduler,
    step_entries: list[CachedStep],
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cfg: RunConfig,
    device: torch.device,
    train_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One forward+backward step over a batch of K cached entries.

    Per cached ``(x_t, t)`` in the batch:
      1. Run LoRA-enabled UNet on (A, B, ∅) → ε_a_lora, ε_b_lora, ε_∅_lora.
      2. Compose ε̃_PoE_lora = w·(ε̃_a + ε̃_b − ε_∅).
      3. Target ε̃_J_cached (guided from cached eps_j_raw + cached eps_∅).
      4. Loss_k = ‖ε̃_PoE_lora − ε̃_J_cached‖² for cached step k.
      5. Total loss = mean over batch.

    All K samples run through a single 3K-wide UNet forward; each sample's
    A/B/∅ branches sit at contiguous indices and use their own timestep.
    Wall-clock per optimizer step grows sub-linearly with K because the
    UNet underutilises the GPU at batch=1.
    """
    assert len(step_entries) >= 1
    K = len(step_entries)
    # V6 (--render-teacher): evaluate the branches at a noised picked render and train against
    # the noise that was added, instead of at the cached trajectory against the joint-prompt
    # prediction. Read here because the cache loop below needs it.
    _render_teacher = bool(getattr(cfg, "render_teacher", False))      # V6
    _freeze_null = bool(getattr(cfg, "freeze_null", False))            # V1, and V6A with V6
    _adapt_null_only = bool(getattr(cfg, "adapt_null_only", False))    # V3, and V6B with V6
    _no_dial = bool(getattr(cfg, "no_dial", False))                    # V4
    # With the empty branch read from the cache there is nothing for the adapter to move in it,
    # so its row is dropped from the batch and the forward pass is a third smaller. V6 keeps all
    # three: its frozen branch comes from a second adapter-off forward over this same batch.
    _drop_null_row = _freeze_null and not _render_teacher
    n_branches = 2 if _drop_null_row else 3
    alphas_cumprod = scheduler.alphas_cumprod
    gs = float(cfg.sampler.guidance_scale)
    H, W = cfg.sampler.height, cfg.sampler.width
    # Which composition rule the three branches are combined with. "poe" is the original
    # trainer; "superdiff" learns SuperDiff's residual at a fixed kappa (plan 08 of scope 06).
    compose = str(getattr(cfg, "compose", "poe"))
    kappa = float(getattr(cfg, "kappa", 0.5))

    # Build the batched 3K input. Per-sample tensors are concatenated in
    # blocks of 3 (A, B, ∅), so the output can be reshaped (K, 3, ...).
    with torch.no_grad():
        x_t_list, eps_j_target_list, eps_poe_frozen_list = [], [], []
        eps_j_raw_list = []
        eps_a_frozen_list, eps_b_frozen_list = [], []
        drawn_noise_list = []
        eps_uncond_frozen_list = []
        span_basis_list = []
        timestep_per_sample = []
        delta_target_norms = []
        for e in step_entries:
            if _render_teacher:
                # V6: the branches are evaluated at a NOISED PICKED RENDER, not at the cached
                # PoE trajectory latent, so every cached eps_* below is at the wrong x_t and
                # goes unused. x_t = sqrt(abar) * x + sqrt(1-abar) * z is SDXL's own forward
                # process; the tex writes the variance-exploding form x + sigma*z, which is the
                # same thing up to the scaling this scheduler folds in.
                if e.render_latent is None:
                    raise RuntimeError(
                        f"--render-teacher but no cached render latent for pair={e.source_pair} "
                        f"seed={e.source_seed}; run scripts/showcase/build_render_teacher_cache.py")
                x0 = e.render_latent.to(device=device, dtype=train_dtype)
                abar = alphas_cumprod[int(e.timestep)].to(device=device, dtype=train_dtype)
                z = torch.randn_like(x0)
                x_t = abar.sqrt() * x0 + (1.0 - abar).sqrt() * z
                drawn_noise_list.append(z)
            else:
                x_t   = e.x_t.to(device=device, dtype=train_dtype)
            ej    = e.eps_j_raw.to(device=device, dtype=train_dtype)
            eu    = e.eps_uncond.to(device=device, dtype=train_dtype)
            ea    = e.eps_a_raw.to(device=device, dtype=train_dtype)
            eb    = e.eps_b_raw.to(device=device, dtype=train_dtype)
            ej_g  = guided_eps(ej, eu, gs)
            ea_g  = guided_eps(ea, eu, gs)
            eb_g  = guided_eps(eb, eu, gs)
            x_t_list.append(x_t.repeat(n_branches, 1, 1, 1))  # (n_branches, 4, H, W)
            eps_j_target_list.append(ej_g)                    # (1, 4, H, W)
            eps_j_raw_list.append(ej)                         # (1, 4, H, W), un-guided
            eps_a_frozen_list.append(ea)                      # (1, 4, H, W), V3's a-bar
            eps_b_frozen_list.append(eb)                      # (1, 4, H, W), V3's b-bar
            eps_poe_frozen_list.append(_compose(ea, eb, eu, gs, compose, kappa))
            eps_uncond_frozen_list.append(eu)                 # (1, 4, H, W)
            # The plane the two experts can already reach on their own: the directions each
            # single-concept prediction points relative to the unconditional one. Anything the
            # error has inside this plane, re-weighting the two experts could have supplied;
            # anything outside it is the part product-of-experts structurally cannot produce.
            # Orthonormalised here, under no_grad, so it is a fixed frame the loss projects onto.
            d1 = (ea - eu).reshape(1, -1).float()
            d2 = (eb - eu).reshape(1, -1).float()
            n1 = d1.norm().clamp_min(1e-12)
            u1 = d1 / n1
            d2p = d2 - (d2 * u1).sum() * u1
            n2 = d2p.norm().clamp_min(1e-12)
            u2 = d2p / n2
            span_basis_list.append(torch.cat([u1, u2], dim=0))    # (2, D)
            timestep_per_sample.append(int(e.timestep))
            delta_target_norms.append(float(e.delta_t.float().norm().item()))

        latent_input_3K = torch.cat(x_t_list, dim=0)          # (n_branches*K, 4, H, W)
        eps_j_target = torch.cat(eps_j_target_list, dim=0)    # (K, 4, H, W)
        eps_j_raw = torch.cat(eps_j_raw_list, dim=0)          # (K, 4, H, W)
        eps_a_frozen = torch.cat(eps_a_frozen_list, dim=0)    # (K, 4, H, W)
        eps_b_frozen = torch.cat(eps_b_frozen_list, dim=0)    # (K, 4, H, W)
        drawn_noise = (torch.cat(drawn_noise_list, dim=0)      # (K, 4, H, W), V6's target z
                       if drawn_noise_list else None)
        eps_poe_frozen = torch.cat(eps_poe_frozen_list, dim=0)  # (K, 4, H, W)
        eps_uncond_frozen = torch.cat(eps_uncond_frozen_list, dim=0)  # (K, 4, H, W)
        span_basis = torch.stack(span_basis_list, dim=0)        # (K, 2, D)

        # Per-sample timesteps, one copy per branch in the batch.
        timestep_b = torch.tensor(
            timestep_per_sample, device=device, dtype=torch.long,
        ).repeat_interleave(n_branches)                        # (n_branches*K,)

        # DDIM scale_model_input is a no-op for our scheduler; still call it
        # so any future swap to a non-DDIM scheduler stays correct.
        latent_input_3K = scheduler.scale_model_input(latent_input_3K, timestep_b[0])

        _seqs = [seq_a, seq_b] if _drop_null_row else [seq_a, seq_b, seq_e]
        pe_3 = torch.cat(_seqs, dim=0).to(device=device, dtype=train_dtype)
        encoder_3K = pe_3.repeat(K, 1, 1)                     # (n_branches*K, 77, 2048)
        from poe_repair.methods._sampling import add_time_ids
        _pools = [pool_a, pool_b] if _drop_null_row else [pool_a, pool_b, pool_e]
        pool_3 = torch.cat(_pools, dim=0).to(device=device, dtype=train_dtype)
        cond_3K = {
            "text_embeds": pool_3.repeat(K, 1),               # (n_branches*K, 1280)
            "time_ids": add_time_ids(
                height=H, width=W, batch_size=n_branches * K,
                device=device, dtype=train_dtype,
            ),
        }

    # LoRA-enabled 3K-wide forward. Cast UNet output to fp32 immediately —
    # fp16 cancellation otherwise crushes gradients.
    noise = unet(
        latent_input_3K, timestep_b, encoder_hidden_states=encoder_3K,
        added_cond_kwargs=cond_3K, timestep_cond=None,
    ).sample.float()                                          # (n_branches*K, 4, H, W)

    # Reshape into (K, n_branches, 4, H, W) → split A / B / ∅ along the branch axis.
    noise = noise.view(K, n_branches, *noise.shape[1:])
    eps_a_raw_l = noise[:, 0]
    eps_b_raw_l = noise[:, 1]
    # With the null row dropped there is no adapted empty branch, and the cached one is what every
    # reader of this name is entitled to: the composition already selects it under --freeze-null,
    # the undialled diagnostic stays computable, and the null-anchor drift is zero by construction
    # because the two tensors are now the same object.
    eps_uncond_l = eps_uncond_frozen.float() if _drop_null_row else noise[:, 2]

    # V6A and V6B need a FROZEN branch at this same state. Under V6 the state is a noised picked
    # render, not the cached PoE trajectory, so the cached eps_a_raw / eps_b_raw / eps_uncond are
    # at a different x_t and cannot be used: they would put two different latents in one
    # subtraction. The base model's answer here is a second forward with the adapter switched
    # off. That costs one extra pass rather than saving one.
    eps_a_base = eps_b_base = eps_u_base = None
    if _render_teacher and (_freeze_null or _adapt_null_only):
        def _adapters(enable: bool) -> None:
            try:
                if enable and hasattr(unet, "enable_adapters"):
                    unet.enable_adapters()
                elif (not enable) and hasattr(unet, "disable_adapters"):
                    unet.disable_adapters()
            except ValueError:
                pass
        _adapters(False)
        with torch.no_grad():
            _base = unet(
                latent_input_3K, timestep_b, encoder_hidden_states=encoder_3K,
                added_cond_kwargs=cond_3K, timestep_cond=None,
            ).sample.float().view(K, 3, *noise.shape[2:])
        _adapters(True)
        eps_a_base, eps_b_base, eps_u_base = _base[:, 0], _base[:, 1], _base[:, 2]
    # Which variation of the correction loss this run trains. At most one of these is on; each
    # names which branches the adapter may move and which reference it is scored against. The
    # equations live in
    # artifacts/ideas/designing-the-correction-loss/maths/objectives-and-variations.tex.
    if _adapt_null_only:
        # V3: u + w(a_bar - u) + w(b_bar - u). The mirror of V1. Both concept branches come from
        # the cache, so the only free tensor in the loss is the empty branch: one per picture,
        # shared by every pair.
        eps_poe_lora = _compose(eps_a_frozen.float(), eps_b_frozen.float(), eps_uncond_l,
                                gs, compose, kappa)
    elif _render_teacher:
        # V6 and its two children. No guidance weight on either side, so _compose is not used.
        #   V6   (no freeze flag): all three branches adapted, z - (a + b - u)
        #   V6A  (--freeze-null):      the empty branch frozen,   z - (a + b - u_base)
        #   V6B  (--adapt-null-only):  both concepts frozen,      z - (a_base + b_base - u)
        if _freeze_null:
            eps_poe_lora = eps_a_raw_l + eps_b_raw_l - eps_u_base
        elif _adapt_null_only:
            eps_poe_lora = eps_a_base + eps_b_base - eps_uncond_l
        else:
            eps_poe_lora = eps_a_raw_l + eps_b_raw_l - eps_uncond_l
    elif _no_dial:
        # V4: the raw (a + b - u), guidance weight gone from this side too. With --freeze-null the
        # empty branch comes from the cache, which makes this V1 up to the constant w^2 and is what
        # the switch's help text promises. Before this read of _freeze_null the branch always used
        # the adapted null, so --no-dial --freeze-null trained with the null free while the sampler
        # detached it: the train/sample mismatch plan 04 exists to prevent.
        _u_no_dial = eps_uncond_frozen.float() if _freeze_null else eps_uncond_l
        eps_poe_lora = eps_a_raw_l + eps_b_raw_l - _u_no_dial
    else:
        # V0 (no flag) and V1 (--freeze-null) differ only in which empty branch the composition
        # reads: the adapted u, or the cached u_bar. Under V1 the u_bar terms cancel and the
        # loss becomes w^2 * ||(a + b - u_bar) - eps_J_bar||^2.
        _eps_u_for_compose = eps_uncond_frozen.float() if _freeze_null else eps_uncond_l
        eps_poe_lora = _compose(eps_a_raw_l, eps_b_raw_l, _eps_u_for_compose, gs, compose, kappa)
    # Per-sample noise-space MSE, then the loss-space weight (1 for the original trainer; the
    # clipped clean-estimate factor under --loss-space x0, plan 16 of scope 01).
    # What each variation is scored against. V6's target is the noise it drew; V4's is the raw
    # cached joint prediction; everything else uses the guided one.
    if _render_teacher:
        _target = drawn_noise.float()
    elif _no_dial:
        _target = eps_j_raw.float()
    else:
        _target = eps_j_target.float()
    err = eps_poe_lora - _target
    # The error the guidance weight hides: the three adapted branches summed at weight 1,
    # against the raw cached joint prediction. Diagnostic only, never added to `loss`, and
    # computed on the same fp32 tensors so no fp16 path is reintroduced.
    with torch.no_grad():
        err_undialled = (eps_a_raw_l + eps_b_raw_l - eps_uncond_l).float() - eps_j_raw.float()
        loss_undialled = (err_undialled ** 2).mean()
    orth_w = float(getattr(cfg, "orth_weight", 1.0))
    if orth_w == 1.0:
        per_sample = (err ** 2).mean(dim=(1, 2, 3))                # (K,)
        orth_frac = float("nan")
    else:
        # Split the error into the part the two experts could supply and the part they could not,
        # then charge the second one orth_w times. At orth_w == 1 this is identical to the plain
        # mean-square error, which is why that case short-circuits: same arithmetic, no projection.
        flat = err.reshape(K, 1, -1)                               # (K, 1, D)
        coeff = (flat * span_basis).sum(dim=2, keepdim=True)       # (K, 2, 1)
        in_span = (coeff * span_basis).sum(dim=1)                  # (K, D)
        out_span = flat.squeeze(1) - in_span                       # (K, D)
        D = flat.shape[-1]
        per_sample = ((in_span ** 2).sum(dim=1) + orth_w * (out_span ** 2).sum(dim=1)) / D
        with torch.no_grad():
            tot = (in_span ** 2).sum() + (out_span ** 2).sum()
            orth_frac = float(((out_span ** 2).sum() / tot.clamp_min(1e-12)).item())
    weights = _loss_weights(
        timestep_per_sample, scheduler.alphas_cumprod,
        loss_space=str(getattr(cfg, "loss_space", "eps")),
        cap=float(getattr(cfg, "loss_weight_cap", 22.0)),
        num_inference_steps=int(cfg.sampler.num_inference_steps),
    ).to(per_sample.device)
    loss_fit = (weights * per_sample).mean()
    loss_eps = per_sample.mean()
    # The running cost (plan 19 of scope 01): beta times the control energy of the adapter's own
    # correction r_hat = eps_PoE_lora - eps_PoE_frozen, per-element mean square weighted by the
    # normalised Girsanov weight of the sample's timestep. beta 0 is the original trainer.
    energy_beta = float(getattr(cfg, "energy_penalty", 0.0))
    r_hat = eps_poe_lora - eps_poe_frozen.float()
    energy_w = control_energy_weights_normalised(
        timestep_per_sample, scheduler.alphas_cumprod,
        num_inference_steps=int(cfg.sampler.num_inference_steps),
    ).to(per_sample.device)
    loss_energy = (energy_w * (r_hat ** 2).mean(dim=(1, 2, 3))).mean()
    # The null anchor. The fit loss constrains only the combination eps_1 + eps_2 - eps_null, so
    # adding the same perturbation to a concept branch and to the null branch leaves it exactly
    # unchanged. Inference reads the null branch a second time on its own, at coefficient
    # -(w - 1), so that invisible perturbation moves the sampled image by 6.5 times its size at
    # w = 7.5. Pinning the adapted null to the frozen one removes the degenerate direction. The
    # frozen null is already in the cache, so this costs no extra forward pass. mu 0 is the
    # original trainer.
    null_anchor_mu = float(getattr(cfg, "null_anchor", 0.0))
    null_drift = eps_uncond_l - eps_uncond_frozen.float()
    loss_null_anchor = (null_drift ** 2).mean(dim=(1, 2, 3)).mean()
    # The contrast term (scope 09). The haze in the corrected renders is a contrast collapse over
    # the first ten steps: the picture the composition is heading for is measurably narrower in
    # grey levels than the frozen one, most of all at the start. This charges the adapter for that
    # narrowing and never for widening, so it can always add contrast for free.
    #
    # Two deliberate choices. The spread is read on the predicted clean picture, not on the noise
    # prediction, because added high-frequency grain would raise the spread of the noise while
    # making the picture worse, and the edge-counting sharpness measure this project uses would
    # reward that rather than catch it. And it is a ratio, not a difference: the Tweedie scale
    # 1/sqrt(abar) is about seven times larger at step 0 than at step 10, so a difference of
    # standard deviations would silently weight the first step seven times the last. The measured
    # quantity is a percentage narrower, which is what the ratio form expresses.
    # Always measured, charged only when nu > 0. A run cannot tell us how far it flattens the
    # picture if the number only exists once it is being punished for it, and the two standard
    # deviations cost a multiply-add over the batch, so every run reports its own shrink.
    contrast_nu = float(getattr(cfg, "contrast_weight", 0.0))
    _ab = scheduler.alphas_cumprod.to(eps_poe_lora.device)[
        torch.tensor(timestep_per_sample, device=eps_poe_lora.device, dtype=torch.long)
    ].float().view(-1, 1, 1, 1)
    _xt_k = latent_input_3K.view(K, n_branches, *latent_input_3K.shape[1:])[:, 0].float()

    def _x0_of(eps):
        return (_xt_k - (1.0 - _ab).sqrt() * eps) / _ab.sqrt().clamp_min(1e-8)

    # When nothing is being charged the term is a pure diagnostic, so it is built without a graph:
    # an unused graph would pin activations for no reason and change nothing about the gradients.
    with torch.set_grad_enabled(contrast_nu > 0.0):
        sd_lora = _x0_of(eps_poe_lora).std(dim=(1, 2, 3))
    with torch.no_grad():
        sd_frozen = _x0_of(eps_poe_frozen.float()).std(dim=(1, 2, 3))
    shrink = torch.relu(1.0 - sd_lora / sd_frozen.clamp_min(1e-8))
    loss_contrast = (shrink ** 2).mean()
    contrast_shrink_mean = float(shrink.detach().mean().item())

    loss = loss_fit
    if contrast_nu > 0.0:
        loss = loss + contrast_nu * loss_contrast
    if energy_beta > 0.0:
        loss = loss + energy_beta * loss_energy
    if null_anchor_mu > 0.0:
        loss = loss + null_anchor_mu * loss_null_anchor

    info = {
        "loss_fit": float(loss_fit.detach().item()),
        # Diagnostic, never in `loss`: see task 1.1 of the four instrument fixes.
        "loss_undialled": float(loss_undialled.item()),
        "loss_energy": float(loss_energy.detach().item()),
        "loss_contrast": float(loss_contrast.detach().item()),
        "contrast_weight": contrast_nu,
        # The fraction narrower the adapter's clean estimate is than the frozen one, averaged over
        # the batch. 0 means it is no narrower. This is the quantity the term charges for, so it is
        # the one to watch: it should fall, and the renders must be read by eye alongside it.
        "contrast_shrink": contrast_shrink_mean,
        "energy_penalty_beta": energy_beta,
        "energy_weight_mean": float(energy_w.mean().item()),
        "loss_null_anchor": float(loss_null_anchor.detach().item()),
        "null_anchor_mu": null_anchor_mu,
        # How far the adapted null branch has moved from the frozen one, in the same units as
        # delta_hat_norm. Logged whether or not the anchor is on, so a run with mu = 0 still
        # shows the drift the anchor exists to prevent.
        "null_drift_norm": float(null_drift.norm().item() / K),
        "step_indices": [int(e.step_index) for e in step_entries],
        "timesteps": timestep_per_sample,
        "batch_size": K,
        "loss_eps": float(loss_eps.detach().item()),
        "loss_weight_mean": float(weights.mean().item()),
        # What fraction of this batch's error sits outside the two experts' own plane. Reported
        # every step so a projection that has collapsed is visible immediately rather than at the
        # end of a run: a flag whose target group is empty is a silent no-op.
        "orth_weight": orth_w,
        "orth_frac": orth_frac,
        "delta_target_norm": float(sum(delta_target_norms) / K),
        "delta_hat_norm": float(
            (eps_poe_lora - eps_poe_frozen.float()).norm().item() / K
        ),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Top-level training driver
# ---------------------------------------------------------------------------


@dataclass
class TrainerState:
    optimizer_step: int = 0
    epoch: int = 0
    bucket_loss_running: dict[str, float] | None = None
    bucket_count_running: dict[str, int] | None = None
    initial_commit_loss: float | None = None
    aborted_reason: str | None = None


def make_optimizer(unet: torch.nn.Module, cfg: RunConfig) -> AdamW:
    lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
    return AdamW(
        lora_params,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )


def make_grad_scaler(
    enabled: bool = True,
    *,
    init_scale: float = 2.0 ** 16,
):
    """fp16 LoRA training underflows in the backward path through 70
    cross-attn blocks of fp16 matmul. GradScaler scales the loss by a
    large factor pre-backward, then unscales before the optimizer step.
    """
    # torch.amp.GradScaler in torch ≥ 2.4; fall back to the older path on
    # earlier torch versions.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=init_scale)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale)


def _global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)


def train_epoch(
    *,
    unet: torch.nn.Module,
    scheduler,
    optimizer: AdamW,
    dataset: list[CachedStep],
    seq_a: torch.Tensor, pool_a: torch.Tensor,
    seq_b: torch.Tensor, pool_b: torch.Tensor,
    seq_e: torch.Tensor, pool_e: torch.Tensor,
    cfg: RunConfig,
    state: TrainerState,
    device: torch.device,
    train_dtype: torch.dtype,
    rng: torch.Generator,
    grad_scaler: "torch.cuda.amp.GradScaler | None" = None,
    logger_callback=None,
) -> bool:
    """One epoch = ``cfg.schedule.epoch_size`` optimizer steps over uniformly-sampled
    cached steps. Returns ``True`` if training should continue, ``False`` if
    a kill criterion triggered.
    """
    unet.train()
    if state.bucket_loss_running is None:
        state.bucket_loss_running = {"early": 0.0, "commit": 0.0, "late": 0.0}
        state.bucket_count_running = {"early": 0, "commit": 0, "late": 0}

    n_steps = int(cfg.schedule.epoch_size)
    grad_clip = float(cfg.optim.grad_clip)

    step_t0 = time.time()
    batch_K = int(cfg.schedule.train_batch_size)
    for _ in range(n_steps):
        idxs = torch.randint(0, len(dataset), (batch_K,), generator=rng).tolist()
        entries = [dataset[int(i)] for i in idxs]
        loss, info = _train_one_step(
            unet=unet, scheduler=scheduler, step_entries=entries,
            seq_a=seq_a, pool_a=pool_a,
            seq_b=seq_b, pool_b=pool_b,
            seq_e=seq_e, pool_e=pool_e,
            cfg=cfg, device=device, train_dtype=train_dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = _global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            lora_params = [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]
            grad_norm_pre = _global_grad_norm(lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

        loss_val = float(loss.detach().item())
        # Attribute the loss to whichever bucket is most represented in the
        # batch (with batch=1 this is just the single entry's bucket).
        bucket_counts = {"early": 0, "commit": 0, "late": 0}
        for s_idx in info["step_indices"]:
            bucket_counts[_bucket(int(s_idx), commit_window=cfg.probe.commit_window)] += 1
        bucket = max(bucket_counts, key=bucket_counts.get)
        state.bucket_loss_running[bucket] = (
            0.95 * state.bucket_loss_running[bucket] + 0.05 * loss_val
            if state.bucket_count_running[bucket] > 0 else loss_val
        )
        state.bucket_count_running[bucket] += 1

        state.optimizer_step += 1

        if logger_callback is not None:
            logger_callback(
                {
                    "train/loss": loss_val,
                    "train/loss_bucket/early": state.bucket_loss_running["early"],
                    "train/loss_bucket/commit": state.bucket_loss_running["commit"],
                    "train/loss_bucket/late": state.bucket_loss_running["late"],
                    "train/grad_norm": grad_norm_pre,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/batch_size": int(info["batch_size"]),
                    "train/epoch": state.epoch,
                    "train/optimizer_step": state.optimizer_step,
                    "train/delta_target_norm": info["delta_target_norm"],
                    "train/delta_hat_norm": info["delta_hat_norm"],
                    "train/orth_weight": info.get("orth_weight", 1.0),
                    "train/orth_frac": info.get("orth_frac", float("nan")),
                    # The null branch's drift from its frozen counterpart, and the anchor that
                    # penalises it. Logged at mu = 0 too: the drift is the quantity the sampler
                    # amplifies by (w - 1), so it is worth watching on runs that do not pin it.
                    "train/loss_null_anchor": info.get("loss_null_anchor", 0.0),
                    "train/null_anchor_mu": info.get("null_anchor_mu", 0.0),
                    "train/null_drift_norm": info.get("null_drift_norm", 0.0),
                    # How much narrower, as a fraction, the clean picture the composition is
                    # heading for is than the frozen one, and the charge for it. Logged at nu = 0
                    # too, for the same reason the drift is: it is the quantity the contrast term
                    # exists to remove, so every run should say how much of it it has.
                    "train/contrast_shrink": info.get("contrast_shrink", 0.0),
                    "train/loss_contrast": info.get("loss_contrast", 0.0),
                    "train/contrast_weight": info.get("contrast_weight", 0.0),
                }
            )

        # Kill criteria.
        if (
            state.optimizer_step >= int(cfg.kill.after_steps)
            and state.bucket_loss_running["commit"] > float(cfg.kill.loss_threshold)
            and state.bucket_count_running["commit"] >= 20
        ):
            state.aborted_reason = (
                f"commit-bucket loss {state.bucket_loss_running['commit']:.4f} > "
                f"{cfg.kill.loss_threshold} after {state.optimizer_step} steps"
            )
            log.warning("kill: %s", state.aborted_reason)
            return False
        if (
            state.optimizer_step == 200
            and state.bucket_count_running["commit"] >= 20
        ):
            state.initial_commit_loss = state.bucket_loss_running["commit"]
        if (
            state.initial_commit_loss is not None
            and state.optimizer_step >= int(cfg.kill.commit_bucket_halve_after_steps)
            and state.bucket_loss_running["commit"]
            > 0.5 * state.initial_commit_loss
        ):
            state.aborted_reason = (
                f"commit-bucket loss {state.bucket_loss_running['commit']:.4f} did not "
                f"halve from initial {state.initial_commit_loss:.4f} by step "
                f"{state.optimizer_step}"
            )
            log.warning("kill: %s", state.aborted_reason)
            return False

    state.epoch += 1

    if logger_callback is not None:
        elapsed = max(1e-6, time.time() - step_t0)
        logger_callback(
            {
                "train/throughput_steps_per_sec": float(n_steps / elapsed),
                "train/peak_vram_gb": (
                    float(torch.cuda.max_memory_allocated() / 1024**3)
                    if torch.cuda.is_available() else 0.0
                ),
            }
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    return True
