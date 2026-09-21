"""Zero-order search over the initial noise on top of the rank-32 adapter (scope 06, plan 12).

Same search as ``search.py`` (candidates ``z' = (z + sigma u)/sqrt(1 + sigma^2)`` around the seed's
cached noise, one round, N candidates), but the sampler is PoE plus the trained correction:

    eps_t = eps_PoE(adapter off) + lambda * (eps_PoE(adapter on) - eps_PoE(adapter off))

which is the on-step of ``run_lora_residual_inject_masked`` with every step on, batched over K
candidates. The adapter is the rank-32 checkpoint at step 30050, lambda 1.2, and it composes 7 of 8
held-out cat x dog seeds already, so the count alone cannot pick; the verifier here is the count
first and then sharpness (Laplacian variance of the finished image, plan 07's function), so the
search looks for the crispest two-animal render near each seed.

Every threshold the review file judges against sits here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean
from poe_repair.experiments.twisted_smc.sampler import ddim_step_eta
from poe_repair.methods._sampling import add_time_ids

# ---------------------------------------------------------------------------
# The bars (pre-registered in the review file; do not move after the run)
# ---------------------------------------------------------------------------
#   sharpness ratio: per seed, kept image's Laplacian variance over the adapter pivot's; the
#   statistic is the median over the 8 seeds.
#   support:      compose holds (kept rate >= adapter pivot rate - MAX_COMPOSE_DROP) and the median
#                 ratio >= 1 + MIN_SHARPNESS_GAIN at either sigma
#   null:         compose holds and |median ratio - 1| <= NULL_SHARPNESS_BAND at both sigmas
#   breaks:       kept rate < adapter pivot rate - MAX_COMPOSE_DROP at a sigma (reported per sigma)
#   inconclusive: anything else
MAX_COMPOSE_DROP = 0.125        # one seed of eight
MIN_SHARPNESS_GAIN = 0.10
NULL_SHARPNESS_BAND = 0.05

# ---------------------------------------------------------------------------
# The adapter and the method's fixed settings
# ---------------------------------------------------------------------------
CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
LORA_RANK = 32
LORA_ALPHA = 32
LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"
LAMBDA = 1.2
SIGMAS = (0.1, 0.3)
N_CANDIDATES = 8
ETA = 0.0


def attach_and_load_lora(unet: torch.nn.Module) -> dict:
    """Attach the rank-32 adapter and load the step-30050 weights (the probe helper's recipe)."""
    from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig

    cfg = LoRAConfig(rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0, target_modules=LORA_TARGET_MODULES,
                     init="gaussian", adapter_name=LORA_ADAPTER_NAME)
    info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=cfg))
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state")
    if state is None:
        raise KeyError(f"{CHECKPOINT} has no 'lora_state' key (found: {list(ckpt.keys())})")
    lora_trainer.load_lora_state(unet, state)
    info["n_loaded"] = len(state)
    info["checkpoint_step"] = int(ckpt.get("step", -1))
    return info


def adapter_disable(unet) -> None:
    if hasattr(unet, "disable_adapters"):
        unet.disable_adapters()
    elif hasattr(unet, "disable_adapter_layers"):
        unet.disable_adapter_layers()


def adapter_enable(unet) -> None:
    if hasattr(unet, "enable_adapters"):
        unet.enable_adapters()
    elif hasattr(unet, "enable_adapter_layers"):
        unet.enable_adapter_layers()
    if hasattr(unet, "set_adapter"):
        try:
            unet.set_adapter(LORA_ADAPTER_NAME)
        except Exception:
            pass


@torch.no_grad()
def run_adapter_candidates(
    *,
    init_latents: torch.Tensor,          # (K, 4, h, w), already divided by the euler sigma
    models: dict,
    scheduler,
    seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
    lambda_value: float = LAMBDA,
    unet_chunk: int = 2,
) -> torch.Tensor:
    """PoE plus lambda times the adapter's correction on every step, DDIM eta 0, K candidates at
    once. ``lambda_value`` 0 is plain PoE through the same code path (the adapter is still
    evaluated and discarded). Returns the final latents (K, 4, h, w). Leaves the adapter enabled,
    as the repo's LoRA samplers do; render references before attaching, or disable it yourself."""
    scheduler.set_timesteps(num_inference_steps)
    latents = init_latents.to(device=device, dtype=dtype)
    unet = models["unet"]
    nb = 3
    pe_one = torch.cat([seq_a, seq_b, seq_e], dim=0)
    pool_one = torch.cat([pool_a, pool_b, pool_e], dim=0)

    def eps_for(lat: torch.Tensor, timestep) -> torch.Tensor:
        out = []
        for s in range(0, lat.shape[0], unet_chunk):
            chunk = lat[s: s + unet_chunk]
            k = chunk.shape[0]
            latent_input = scheduler.scale_model_input(chunk.repeat_interleave(nb, dim=0), timestep)
            cond = {"text_embeds": pool_one.repeat(k, 1),
                    "time_ids": add_time_ids(height=height, width=width, batch_size=nb * k, device=device, dtype=dtype)}
            noise = unet(latent_input, timestep, encoder_hidden_states=pe_one.repeat(k, 1, 1),
                         added_cond_kwargs=cond, timestep_cond=None).sample
            noise = noise.view(k, nb, *noise.shape[1:])
            ea = guided_eps(noise[:, 0], noise[:, 2], guidance_scale)
            eb = guided_eps(noise[:, 1], noise[:, 2], guidance_scale)
            out.append(poe_eps(ea, eb, noise[:, 2]))
        return torch.cat(out)

    for step_index, timestep in enumerate(scheduler.timesteps):
        adapter_disable(unet)
        eps_frozen = eps_for(latents, timestep)
        adapter_enable(unet)
        eps_lora = eps_for(latents, timestep)
        eps_t = eps_frozen + float(lambda_value) * (eps_lora - eps_frozen)
        ab_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, ab_t, eps_t)
        latents, _ = ddim_step_eta(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                   x_t=latents, x0=x0, eps=eps_t, eta=ETA, generator=None)
    adapter_enable(unet)
    return latents


def laplacian_var(img_path: Path) -> float:
    """Sharpness: variance of the second differences of the greyscale image. Plan 07's
    ``lambda_window_grid._laplacian_var``, copied so this package does not import a script."""
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(img)
    return float(np.var(gx[1:] - gx[:-1]) + np.var(gy[:, 1:] - gy[:, :-1]))


def pick_count_then_sharpness(counts: list[int], sharp: list[float]) -> int:
    """Highest count (clipped at 2), then highest sharpness, then lowest index."""
    return max(range(len(counts)), key=lambda i: (min(counts[i], 2), sharp[i], -i))


def verdict(*, kept_rate_by_sigma: dict[float, float], adapter_pivot_rate: float,
            median_ratio_by_sigma: dict[float, float]) -> dict:
    compose_ok = {s: r >= adapter_pivot_rate - MAX_COMPOSE_DROP - 1e-9 for s, r in kept_rate_by_sigma.items()}
    gain = {s: median_ratio_by_sigma[s] >= 1.0 + MIN_SHARPNESS_GAIN - 1e-9 for s in kept_rate_by_sigma}
    flat = {s: abs(median_ratio_by_sigma[s] - 1.0) <= NULL_SHARPNESS_BAND + 1e-9 for s in kept_rate_by_sigma}
    if any(compose_ok[s] and gain[s] for s in compose_ok):
        v = "support"
    elif not all(compose_ok.values()):
        v = "breaks"
    elif all(flat.values()):
        v = "null"
    else:
        v = "inconclusive"
    return {"verdict": v, "compose_ok_by_sigma": {str(s): b for s, b in compose_ok.items()},
            "sharpness_gain_by_sigma": {str(s): b for s, b in gain.items()},
            "max_compose_drop": MAX_COMPOSE_DROP, "min_sharpness_gain": MIN_SHARPNESS_GAIN,
            "null_sharpness_band": NULL_SHARPNESS_BAND}
