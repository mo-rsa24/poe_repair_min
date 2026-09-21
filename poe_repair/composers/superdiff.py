"""SuperDiff (AND) composer: continuity-equation composition, not naive score addition.

Skreta et al., arXiv 2412.17762. Reimplements the sampling loop from the official SDXL port
(`superdiff/superdiff-sdxl-v1-0`, `pipeline.py`, read in full before this file was written)
rather than loading it via `trust_remote_code`, because exposing the per-step prediction
`eps_M` and clamping the blending weight `kappa` both require access to the loop's internals,
which an opaque pipeline call does not give.

This pipeline has no DDIM mode. `_forward` is a fixed Euler-Maruyama stochastic integrator: it
injects fresh Gaussian noise every step except the last three, unconditionally, with no
scheduler object and no `eta` parameter anywhere in the source. Step count (`num_inference_steps`)
is the one sampling setting it exposes; guidance scale is the other.

`kappa` ships with no clamp anywhere in the source. This module adds one, toggleable, since an
unclamped spike has less room to recover from at low step counts (larger `dsigma` per step) than
at the pipeline's own default of 200.
"""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import torch

from poe_repair.composers._helpers import cell_output_dir
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, ensure_dir, write_json

METHOD_NAME = "superdiff"
SUPERDIFF_MODEL_ID = "superdiff/superdiff-sdxl-v1-0"

# Not applicable at this pipeline's own 200-step default (the regime the authors validated);
# added because this project's own settings put a render at 50 steps, where dsigma is larger
# per step and an unclamped kappa spike has less room to recover from.
KAPPA_CLAMP_MIN = -0.5
KAPPA_CLAMP_MAX = 1.5
KAPPA_CLAMP_WARMUP_FRACTION = 0.10  # hold kappa at 0.5 for the first ~10% of steps

_MODELS_CACHE: dict | None = None


def _default_hf_cache_dir() -> Path:
    """Where the ~7GB SuperDiff checkpoint downloads to.

    Never $HOME: this repo's home filesystem hit 100% once and silently killed
    checkpointing (poe-disk-001). Override with SUPERDIFF_HF_CACHE.
    """
    env = os.environ.get("SUPERDIFF_HF_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path(
        "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/hf_cache"
    )


def _load_superdiff_models(device: torch.device, dtype: torch.dtype) -> dict:
    """Load SuperDiff's own SDXL components, cached at module scope for this process.

    This is a separate checkpoint from this repo's own ``ctx.models`` (loaded from
    ``stabilityai/stable-diffusion-xl-base-1.0`` by ``load_sdxl_models``): SuperDiff's port
    ships its own copy of the base weights under its own Hub repo, per its ``model_index.json``.
    """
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    cache_dir = _default_hf_cache_dir()
    ensure_dir(cache_dir)

    st_kwargs = (
        {"use_safetensors": True}
        if "use_safetensors" in inspect.signature(CLIPTextModel.from_pretrained).parameters
        else {}
    )

    vae = AutoencoderKL.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="vae", torch_dtype=dtype,
        use_safetensors=True, cache_dir=str(cache_dir),
    ).to(device)
    # SDXL's fp16 VAE is numerically unstable without this; load_sdxl_models applies the same
    # upcast for this repo's own SDXL base load, per this repo's fp16-upcast convention.
    if getattr(vae.config, "force_upcast", False) and vae.dtype != torch.float32:
        vae = vae.to(dtype=torch.float32)

    unet = UNet2DConditionModel.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="unet", torch_dtype=dtype,
        use_safetensors=True, cache_dir=str(cache_dir),
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="text_encoder", torch_dtype=dtype,
        cache_dir=str(cache_dir), **st_kwargs,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype,
        cache_dir=str(cache_dir), **st_kwargs,
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="tokenizer", cache_dir=str(cache_dir),
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        SUPERDIFF_MODEL_ID, subfolder="tokenizer_2", cache_dir=str(cache_dir),
    )

    _MODELS_CACHE = {
        "vae": vae, "unet": unet,
        "text_encoder": text_encoder, "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer, "tokenizer_2": tokenizer_2,
    }
    return _MODELS_CACHE


def _get_scaled_coeffs() -> tuple[float, float]:
    beta_min, beta_max = 0.85, 12.0
    return beta_min**0.5, beta_max**0.5 - beta_min**0.5


def _beta(t: torch.Tensor) -> torch.Tensor:
    a, b = _get_scaled_coeffs()
    return (a + t * b) ** 2


def _int_beta(t: torch.Tensor) -> torch.Tensor:
    a, b = _get_scaled_coeffs()
    return ((a + b * t) ** 3 - a**3) / (3 * b)


def _sigma(t: torch.Tensor) -> torch.Tensor:
    return torch.expm1(_int_beta(t)) ** 0.5


def _prepare_prompt_input(
    models: dict, prompt_1: str, prompt_2: str, height: int, width: int, device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Reproduces `SuperDiffSDXLPipeline.prepare_prompt_input` for batch_size=1.

    Negative/unconditional embeddings are zeros, not an encoded empty string: that is the
    shipped pipeline's own choice, reproduced here rather than "corrected", since this plan
    tests that pipeline as published.
    """
    tokenizer, tokenizer_2 = models["tokenizer"], models["tokenizer_2"]
    text_encoder, text_encoder_2 = models["text_encoder"], models["text_encoder_2"]

    def _encode(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids_1 = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        ids_2 = tokenizer_2(
            prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            out_1 = text_encoder(ids_1, output_hidden_states=True)
            out_2 = text_encoder_2(ids_2, output_hidden_states=True)
        embeds = torch.cat((out_1.hidden_states[-2], out_2.hidden_states[-2]), dim=-1)
        pooled = out_2[0]
        return embeds, pooled

    embeds_1, pooled_1 = _encode(prompt_1)
    embeds_2, pooled_2 = _encode(prompt_2)
    negative_embeds = torch.zeros_like(embeds_1)
    negative_pooled = torch.zeros_like(pooled_1)

    time_ids = torch.tensor([(height, width, 0, 0, height, width)], device=device)

    prompt_embeds = torch.cat([negative_embeds, embeds_1, embeds_2], dim=0)
    add_text_embeds = torch.cat([negative_pooled, pooled_1, pooled_2], dim=0)
    add_time_ids = torch.cat([time_ids, time_ids, time_ids], dim=0)

    return prompt_embeds, {"text_embeds": add_text_embeds, "time_ids": add_time_ids}


def _encode_joint_prompt(
    models: dict, joint_prompt: str, height: int, width: int, device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single-conditioning SDXL text embedding for the literal joint prompt (e.g. "a cat and a
    dog"), the same dual-encoder concatenation `_prepare_prompt_input` uses per branch."""
    tokenizer, tokenizer_2 = models["tokenizer"], models["tokenizer_2"]
    text_encoder, text_encoder_2 = models["text_encoder"], models["text_encoder_2"]

    ids_1 = tokenizer(
        joint_prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    ids_2 = tokenizer_2(
        joint_prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        out_1 = text_encoder(ids_1, output_hidden_states=True)
        out_2 = text_encoder_2(ids_2, output_hidden_states=True)
    embeds = torch.cat((out_1.hidden_states[-2], out_2.hidden_states[-2]), dim=-1)
    pooled = out_2[0]
    time_ids = torch.tensor([(height, width, 0, 0, height, width)], device=device)
    return embeds, {"text_embeds": pooled, "time_ids": time_ids}


def _run_sampling_loop(
    models: dict,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    latents: torch.Tensor,
    generator: torch.Generator,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    kappa_clamp: bool,
    device: torch.device,
    joint_embeds: torch.Tensor | None = None,
    joint_added_cond_kwargs: dict | None = None,
    kappa_override: float | None = None,
    lam: float = 0.0,
    lora_adapter_name: str | None = None,
    lambda_value: float = 0.0,
) -> tuple[torch.Tensor, list[float], list[float], list[float], list[float], list[float]]:
    """Reproduces `SuperDiffSDXLPipeline._forward`'s AND branch, with a kappa clamp and an
    eps_M hook added.

    When ``joint_embeds`` is given, a 4th conditioning row (a single CFG branch on the literal
    joint prompt, `poe_repair`'s own `eps_J`) rides in the same batched UNet call, and
    ``r_t^SD = eps_J - eps_M`` is formed at every step on the actual trajectory this render
    visits, not a separately-sampled one.

    When ``kappa_override`` is given, the pipeline's own kappa is still computed and recorded
    (``kappa_raw``) but not used: every step blends with the fixed value instead, and the clamp
    is bypassed. This is the kappa-sweep figure's knob, SuperDiff's own analogue of this
    project's correction-amount axis. At 0 the blend is pure prompt_2 guidance, at 1 pure
    prompt_1, following how the shipped formula weights them.

    Returns (final_latents, eps_m_norms, kappa_raw, kappa_used, r_t_sd_norms). The last list is
    empty when ``joint_embeds`` is not given.
    """
    unet = models["unet"]
    t = torch.tensor(1.0)
    dt = 1.0 / num_inference_steps
    train_number_steps = 1000
    warmup_steps = max(1, round(KAPPA_CLAMP_WARMUP_FRACTION * num_inference_steps))
    form_r_t = joint_embeds is not None
    use_adapter = lora_adapter_name is not None

    # Adapter toggles, copied from poe_repair/methods/_sampling.py's
    # run_lora_residual_inject_masked: diffusers' PeftAdapterMixin methods, wrapped so a UNet
    # with no adapter attached is a harmless no-op ("No adapter loaded" is a ValueError).
    def _adapter_disable():
        try:
            if hasattr(unet, "disable_adapters"):
                unet.disable_adapters()
            elif hasattr(unet, "disable_adapter_layers"):
                unet.disable_adapter_layers()
        except ValueError:
            pass

    def _adapter_enable():
        try:
            if hasattr(unet, "enable_adapters"):
                unet.enable_adapters()
            elif hasattr(unet, "enable_adapter_layers"):
                unet.enable_adapter_layers()
        except ValueError:
            pass
        if hasattr(unet, "set_adapter"):
            try:
                unet.set_adapter(lora_adapter_name)
            except Exception:
                pass

    if form_r_t:
        batched_embeds = torch.cat([prompt_embeds, joint_embeds], dim=0)
        batched_cond = {
            "text_embeds": torch.cat(
                [added_cond_kwargs["text_embeds"], joint_added_cond_kwargs["text_embeds"]], dim=0
            ),
            "time_ids": torch.cat(
                [added_cond_kwargs["time_ids"], joint_added_cond_kwargs["time_ids"]], dim=0
            ),
        }
    else:
        batched_embeds, batched_cond = prompt_embeds, added_cond_kwargs

    latents = latents * (_sigma(t) ** 2 + 1) ** 0.5
    eps_m_norms: list[float] = []
    kappa_raw: list[float] = []
    kappa_used: list[float] = []
    r_t_sd_norms: list[float] = []
    delta_hat_norms: list[float] = []
    delta_hat_cos_r_t: list[float] = []  # cosine(Δ̂, r_t^SD) per step, when both are formed

    with torch.no_grad():
        for i in range(num_inference_steps):
            n_rows = 4 if form_r_t else 3
            latent_model_input = torch.cat([latents] * n_rows)
            sigma_t = _sigma(t)
            dsigma = _sigma(t - dt) - sigma_t
            latent_model_input = latent_model_input / (sigma_t**2 + 1) ** 0.5

            def _forward():
                out = unet(
                    latent_model_input, t * train_number_steps,
                    encoder_hidden_states=batched_embeds, added_cond_kwargs=batched_cond,
                    return_dict=False,
                )[0]
                return out.chunk(4) if form_r_t else (*out.chunk(3), None)

            # The adapter-off pass is the baseline on every step, adapter attached or not.
            if use_adapter:
                _adapter_disable()
            noise_pred_uncond, noise_pred_1, noise_pred_2, noise_pred_joint = _forward()
            if use_adapter:
                _adapter_enable()
                on_uncond, on_1, on_2, _ = _forward()
                _adapter_disable()

            noise = torch.sqrt(2 * torch.abs(dsigma) * sigma_t) * torch.empty_like(
                latents, device=device
            ).normal_(generator=generator)

            dx_ind = (
                2 * dsigma * (noise_pred_uncond + guidance_scale * (noise_pred_2 - noise_pred_uncond))
                + noise
            )
            kappa = (
                torch.abs(dsigma) * (noise_pred_2 - noise_pred_1) * (noise_pred_2 + noise_pred_1)
            ).sum((1, 2, 3)) - (dx_ind * (noise_pred_1 - noise_pred_2)).sum((1, 2, 3))
            kappa = kappa / (
                2 * dsigma * guidance_scale * ((noise_pred_1 - noise_pred_2) ** 2).sum((1, 2, 3))
            )
            kappa_raw.append(kappa.detach().float().mean().item())

            if kappa_override is not None:
                kappa = torch.full_like(kappa, float(kappa_override))
            elif kappa_clamp:
                if i < warmup_steps:
                    kappa = torch.full_like(kappa, 0.5)
                else:
                    kappa = kappa.clamp(KAPPA_CLAMP_MIN, KAPPA_CLAMP_MAX)
            kappa_used.append(kappa.detach().float().mean().item())

            eps_m = noise_pred_uncond + guidance_scale * (
                (noise_pred_2 - noise_pred_uncond)
                + kappa[:, None, None, None] * (noise_pred_1 - noise_pred_2)
            )
            eps_m_norms.append(eps_m.detach().float().norm().item())

            if form_r_t:
                # Single CFG branch on the literal joint prompt, this project's own eps_J
                # (poe_repair/composers/_helpers.py's get_joint_embeds), evaluated at this
                # step's actual latent so r_t^SD sits on the trajectory this render visits.
                eps_j = noise_pred_uncond + guidance_scale * (noise_pred_joint - noise_pred_uncond)
                r_t_sd = eps_j - eps_m
                r_t_sd_norms.append(r_t_sd.detach().float().norm().item())
                # The generalised amount axis from step 29, applied to this rule: add back a
                # fraction lam of the residual. lam=0 is SuperDiff alone, lam=1 is eps_J exactly.
                eps_used = eps_m + lam * r_t_sd if lam != 0.0 else eps_m
            else:
                eps_used = eps_m

            if use_adapter:
                # Same grammar as run_lora_residual_inject_masked with eps_M for eps_PoE:
                # Δ̂ = eps_M(adapter on) − eps_M(adapter off), at the same kappa, and the step
                # uses eps_M(off) + λ·Δ̂. On every step, no window.
                eps_m_on = on_uncond + guidance_scale * (
                    (on_2 - on_uncond) + kappa[:, None, None, None] * (on_1 - on_2)
                )
                delta_hat = eps_m_on - eps_m
                delta_hat_norms.append(delta_hat.detach().float().norm().item())
                if form_r_t:
                    a = delta_hat.detach().float().flatten()
                    b = r_t_sd.detach().float().flatten()
                    delta_hat_cos_r_t.append(
                        (torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)).item()
                    )
                eps_used = eps_used + lambda_value * delta_hat

            if i < num_inference_steps - 3:
                latents = latents + 2 * dsigma * eps_used + noise
            else:
                latents = latents + dsigma * eps_used

            t = t - dt

    return latents, eps_m_norms, kappa_raw, kappa_used, r_t_sd_norms, delta_hat_norms, delta_hat_cos_r_t


def _decode(models: dict, latents: torch.Tensor) -> "Image.Image":
    from PIL import Image

    vae = models["vae"]
    latents = latents / vae.config.scaling_factor
    latents = latents.to(vae.dtype)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    arr = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(arr)


def run(
    cell: PairSeedCell,
    ctx: MethodCtx,
    *,
    exp_name: str = "tmp",
    overwrite: bool = False,
    kappa_clamp: bool = True,
    form_r_t: bool = False,
    kappa_override: float | None = None,
    lam: float = 0.0,
    lora_adapter_name: str | None = None,
    lambda_value: float = 0.0,
    lora_tag: str = "",
) -> Path:
    """Wire SuperDiffSDXLPipeline as a composer, matching this repo's ``poe.py`` interface.

    ``ctx.num_inference_steps`` and ``ctx.guidance_scale`` are reused directly; build a
    ``MethodCtx`` at 200 or at 50 to select which cell of the grid this call produces.

    ``form_r_t=True`` adds a 4th conditioning row (a single CFG branch on the literal joint
    prompt) to every step's UNet call, forms ``r_t^SD = eps_J - eps_M`` at that step's actual
    latent, and writes its per-step norm into the sidecar. Off by default: the 8-cell grid and
    the kappa-sweep figure only need ``eps_M``; this is for verifying the hook isn't a
    placeholder, and for anything downstream that wants ``r_t^SD`` from this render directly
    rather than recomputing it from ``eps_M`` alone.
    """
    if kappa_override is not None:
        kappa_tag = f"kappa{kappa_override:.2f}"
    else:
        kappa_tag = "clamped" if kappa_clamp else "unclamped"
    method_name = f"{METHOD_NAME}_{ctx.num_inference_steps}steps_{kappa_tag}"
    if lam != 0.0:
        form_r_t = True  # the residual has to be formed to be added back
        method_name += f"_lam{lam:.2f}"
    elif form_r_t:
        method_name += "_with_rt"
    if lora_adapter_name is not None:
        # The adapter must already be attached to models["unet"] by the caller (see
        # scripts/showcase/lora_dose_sweep.py:_attach_and_load_lora); lora_tag names which one.
        method_name += f"_lora{lora_tag}_lv{lambda_value:.2f}"
    out_dir = cell_output_dir(ctx, exp_name, method_name, cell)
    image_path = out_dir / f"{method_name}.png"
    sidecar_path = out_dir / f"{method_name}.json"
    if image_path.exists() and sidecar_path.exists() and not overwrite:
        return image_path

    models = _load_superdiff_models(ctx.device, ctx.dtype)
    generator = torch.cuda.manual_seed(cell.seed)

    latents = torch.randn(
        (1, models["unet"].config.in_channels, cell.height // 8, cell.width // 8),
        generator=generator, dtype=ctx.dtype, device=ctx.device,
    )
    prompt_embeds, added_cond_kwargs = _prepare_prompt_input(
        models, cell.prompt_a, cell.prompt_b, cell.height, cell.width, ctx.device,
    )
    joint_embeds = joint_added_cond_kwargs = None
    if form_r_t:
        from poe_repair.composers._helpers import joint_text_for

        joint_text = joint_text_for(cell, ctx)
        joint_embeds, joint_added_cond_kwargs = _encode_joint_prompt(
            models, joint_text, cell.height, cell.width, ctx.device,
        )

    start = time.perf_counter()
    final_latents, eps_m_norms, kappa_raw, kappa_used, r_t_sd_norms, delta_hat_norms, delta_hat_cos_r_t = _run_sampling_loop(
        models, prompt_embeds, added_cond_kwargs, latents, generator,
        num_inference_steps=ctx.num_inference_steps, guidance_scale=ctx.guidance_scale,
        kappa_clamp=kappa_clamp, device=ctx.device,
        joint_embeds=joint_embeds, joint_added_cond_kwargs=joint_added_cond_kwargs,
        kappa_override=kappa_override, lam=lam,
        lora_adapter_name=lora_adapter_name, lambda_value=lambda_value,
    )
    wall_time_s = time.perf_counter() - start

    image = _decode(models, final_latents)
    image.save(str(image_path))
    sidecar = {
        "method": method_name,
        "pair_slug": cell.pair_slug, "seed": cell.seed,
        "prompt_a": cell.prompt_a, "prompt_b": cell.prompt_b,
        "num_inference_steps": ctx.num_inference_steps,
        "guidance_scale": ctx.guidance_scale,
        "kappa_clamp": kappa_clamp,
        "kappa_override": kappa_override,
        "lam": lam,
        "lora_adapter_name": lora_adapter_name,
        "lora_tag": lora_tag,
        "lambda_value": lambda_value,
        "wall_time_s": wall_time_s,
        "steps": eps_m_norms,  # eps_M per-step norm, per the plan's naming
        "kappa_raw": kappa_raw,
        "kappa_used": kappa_used,
    }
    if form_r_t:
        sidecar["r_t_sd_norms"] = r_t_sd_norms
    if lora_adapter_name is not None:
        sidecar["delta_hat_norms"] = delta_hat_norms
        if form_r_t:
            sidecar["delta_hat_cos_r_t"] = delta_hat_cos_r_t
    write_json(sidecar_path, sidecar)
    return image_path
