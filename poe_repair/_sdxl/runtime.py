"""Vendored SDXL model loading + prompt encoding + initial-latent helpers.

Pruned from `prototypes/oracle_gap_sdxl/runtime.py`. Only the helpers reached
by the four samplers and the synthesizer trainer are kept.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poe_repair._sdxl.sdipc_utils import decode_latents_to_tensor, load_shared_init_latents


@dataclass
class PairContext:
    pair_dir: Path
    pair_slug: str
    prompt_a: str
    prompt_b: str
    prompt_joint: str
    seed: int
    model_id: str
    taxonomy_group_key: str
    taxonomy_group_label: str
    height: int
    width: int
    metadata: dict[str, Any]


class LatentTrajectoryCollector:
    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        z_channels: int,
        latent_height: int,
        latent_width: int,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shape = (z_channels, latent_height, latent_width)
        self.trajectories = torch.zeros(
            (num_steps + 1, batch_size, z_channels, latent_height, latent_width)
        )
        self.velocities = torch.zeros(
            (num_steps, batch_size, z_channels, latent_height, latent_width)
        )
        self.sigmas = torch.zeros(num_steps + 1)
        self.timesteps = torch.zeros(num_steps)

    def store_step(
        self,
        step: int,
        latents: torch.Tensor,
        velocity: torch.Tensor | None,
        sigma: float,
        timestep: float | int | None,
    ) -> None:
        self.trajectories[step] = latents.detach().cpu()
        if velocity is not None:
            self.velocities[step] = velocity.detach().cpu()
        self.sigmas[step] = float(sigma)
        if timestep is not None:
            self.timesteps[step] = float(timestep)

    def store_final(self, latents: torch.Tensor, sigma: float = 0.0) -> None:
        self.trajectories[-1] = latents.detach().cpu()
        self.sigmas[-1] = float(sigma)


def infer_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = mapping[dtype_arg.lower()]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def load_sdxl_models(
    *,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    try:
        from transformers import CLIPTextModelWithProjection
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CLIPTextModelWithProjection is required for SDXL.") from exc

    transformers_safetensors = "use_safetensors" in inspect.signature(CLIPTextModel.from_pretrained).parameters
    st_kwargs = {"use_safetensors": True} if transformers_safetensors else {}
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
    if getattr(vae.config, "force_upcast", False) and vae.dtype != torch.float32:
        vae = vae.to(dtype=torch.float32)

    return {
        "vae": vae,
        "tokenizer": CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype, **st_kwargs
        ).to(device),
        "tokenizer_2": CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2"),
        "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=dtype, **st_kwargs
        ).to(device),
        "unet": UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True
        ).to(device),
    }


def load_ddim_scheduler(model_id: str):
    from diffusers import DDIMScheduler

    return DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")


def load_shared_latents(
    *,
    pair_dir: Path,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    models: dict[str, Any] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, float]:
    euler_sigma = 1.0
    try:
        latents = load_shared_init_latents(
            pair_dir=pair_dir,
            euler_init_noise_sigma=euler_sigma,
            device=device,
            dtype=dtype,
            reference_condition="poe",
        )
        return latents, euler_sigma
    except FileNotFoundError:
        if models is None:
            raise
        unet = models["unet"]
        in_channels = int(getattr(unet.config, "in_channels", 4))
        latent_h = int(height // 8) if height is not None else int(getattr(unet.config, "sample_size", 128))
        latent_w = int(width // 8) if width is not None else int(getattr(unet.config, "sample_size", 128))
        cpu_dtype = dtype if dtype != torch.bfloat16 else torch.float32
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            1, in_channels, latent_h, latent_w, device="cpu", dtype=cpu_dtype, generator=generator
        )
        return latents.to(device=device, dtype=dtype), euler_sigma


@torch.no_grad()
def encode_prompt_sdxl(
    prompt: str,
    *,
    models: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]

    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_2 = tokenizer_2(
        [prompt],
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    enc_1 = text_encoder(text_input.input_ids.to(device), output_hidden_states=True)
    enc_2 = text_encoder_2(text_input_2.input_ids.to(device), output_hidden_states=True)
    prompt_embeds = torch.cat(
        [enc_1.hidden_states[-2], enc_2.hidden_states[-2]], dim=-1
    ).to(device=device, dtype=dtype)
    pooled = enc_2[0]
    if pooled.ndim != 2:
        pooled = enc_2.text_embeds
    return prompt_embeds, pooled.to(device=device, dtype=dtype)


def decode_latents(models: dict[str, Any], latents: torch.Tensor) -> torch.Tensor:
    vae = models["vae"]
    with torch.no_grad():
        return decode_latents_to_tensor(vae, latents.to(vae.dtype))
