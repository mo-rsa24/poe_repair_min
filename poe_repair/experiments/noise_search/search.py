"""Zero-order search over the initial noise, over the plain product-of-experts sampler.

Ma et al., "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps"
(arXiv 2501.09732), section 3.2: with a deterministic sampler the initial noise fixes the image,
so inference compute can be spent choosing the noise. Random search is best of N independent
draws. Zero-order search keeps a pivot noise, draws N candidates in its neighbourhood, and keeps
the best by a verifier. The paper defines the neighbourhood only as "distance lambda from the
pivot" and never writes how a candidate is drawn; this module uses the variance-preserving mixture

    z' = (z + sigma * u) / sqrt(1 + sigma^2),   u ~ N(0, I)

so every candidate is still a unit-variance Gaussian and its cosine to the pivot is
1 / sqrt(1 + sigma^2) (0.995 at sigma 0.1, 0.958 at sigma 0.3). One round only: the pivot is the
seed's cached noise, N candidates are drawn around it, and the best candidate is kept.

The verifier is the validated instance-count compose scorer, read two ways on the same
candidates: on the finished image (the paper's setting) and on the model's clean-image estimate
at ``VERIFIER_EARLY_STEP`` (to see whether the verifier works where the compose decision is made).

Every threshold the review file judges against sits here, so moving one after the answer is
visible shows up in a diff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean
from poe_repair.experiments.twisted_smc.sampler import ddim_step_eta
from poe_repair.methods._sampling import add_time_ids

# ---------------------------------------------------------------------------
# The bars (pre-registered in the review file; do not move after the run)
# ---------------------------------------------------------------------------
#   support:      the kept image composes at least PASS_MARGIN more often than the unperturbed
#                 seed at either sigma (final-image verifier)
#   null:         |kept - unperturbed| <= NULL_MARGIN at both sigmas
#   inconclusive: anything else (one sigma between the two margins, or one above and one below)
PASS_MARGIN = 0.25
NULL_MARGIN = 0.10

# ---------------------------------------------------------------------------
# The method's fixed settings
# ---------------------------------------------------------------------------
SIGMAS = (0.1, 0.3)          # perturbation size, see the module docstring
N_CANDIDATES = 8             # candidates per seed per sigma, one round
VERIFIER_EARLY_STEP = 10     # step index (of 50) whose x0-hat the second verifier reads
ETA = 0.0                    # deterministic DDIM: the noise is the whole story


def perturb(pivot: torch.Tensor, sigma: float, n: int, generator: torch.Generator) -> torch.Tensor:
    """N variance-preserving perturbations of one pivot noise (4, h, w) -> (N, 4, h, w), fp32 CPU."""
    u = torch.randn((n, *pivot.shape), generator=generator, dtype=torch.float32)
    return (pivot[None].float() + sigma * u) / math.sqrt(1.0 + sigma * sigma)


def cosine_to_pivot(cands: torch.Tensor, pivot: torch.Tensor) -> list[float]:
    p = pivot.float().flatten()
    return [float(torch.nn.functional.cosine_similarity(c.float().flatten(), p, dim=0)) for c in cands]


@dataclass
class CandidateOutputs:
    latents: torch.Tensor                                   # (K, 4, h, w) final
    x0_at: dict[int, torch.Tensor] = field(default_factory=dict)   # step index -> (K, 4, h, w) Tweedie estimate


@torch.no_grad()
def run_poe_candidates(
    *,
    init_latents: torch.Tensor,          # (K, 4, h, w), already divided by the euler sigma
    models: dict,
    scheduler,
    seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
    capture_steps: tuple[int, ...] = (VERIFIER_EARLY_STEP,),
    unet_chunk: int = 2,
) -> CandidateOutputs:
    """Plain PoE (eps_a + eps_b - eps_uncond, each guided) at DDIM eta 0 over K candidates at once."""
    scheduler.set_timesteps(num_inference_steps)
    latents = init_latents.to(device=device, dtype=dtype)
    unet = models["unet"]
    branches = [(seq_a, pool_a), (seq_b, pool_b), (seq_e, pool_e)]
    nb = len(branches)
    pe_one = torch.cat([s for s, _ in branches], dim=0)
    pool_one = torch.cat([p for _, p in branches], dim=0)

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

    res = CandidateOutputs(latents=latents)
    for step_index, timestep in enumerate(scheduler.timesteps):
        eps_t = eps_for(latents, timestep)
        ab_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, ab_t, eps_t)
        if step_index in capture_steps:
            res.x0_at[step_index] = x0.detach().clone()
        latents, _ = ddim_step_eta(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                   x_t=latents, x0=x0, eps=eps_t, eta=ETA, generator=None)
    res.latents = latents
    return res


def pick(scores: list[tuple[int, float]]) -> int:
    """Index of the best candidate: highest count, then highest summed box confidence, then lowest index."""
    return max(range(len(scores)), key=lambda i: (scores[i][0], scores[i][1], -i))


def verdict(*, kept_rate_by_sigma: dict[float, float], pivot_rate: float) -> dict:
    deltas = {s: r - pivot_rate for s, r in kept_rate_by_sigma.items()}
    if any(d >= PASS_MARGIN - 1e-9 for d in deltas.values()):
        v = "support"
    elif all(abs(d) <= NULL_MARGIN + 1e-9 for d in deltas.values()):
        v = "null"
    else:
        v = "inconclusive"
    return {"verdict": v, "delta_by_sigma": {str(s): d for s, d in deltas.items()},
            "pass_margin": PASS_MARGIN, "null_margin": NULL_MARGIN}
