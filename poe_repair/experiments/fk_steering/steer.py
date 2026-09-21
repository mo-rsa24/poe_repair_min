"""Feynman-Kac steering with a detector reward, over the plain PoE sampler.

The sampler follows Singhal et al., "A General Framework for Inference-time Scaling and
Steering of Diffusion Models" (arXiv 2501.06848), Algorithm 1 with the max potential:

    G_t = exp(lambda * max_{s <= t} r(x0hat_s))

where ``x0hat_s`` is the Tweedie estimate of the clean latent at step ``s``, decoded through
the VAE, and ``r`` is the validated instance-count compose scorer's count clipped at
``REWARD_CLIP``. The proposal is the base sampler itself (plain PoE, DDIM at ``eta`` 1.0, the
paper's setting). Particles are resampled at ``RESAMPLE_STEP_INDICES`` only (the paper's five
evenly spaced steps, [0, 20, 40, 60, 80] of 100, scaled to this repo's 50 steps).

One departure from the paper, stated here so it is not mistaken for a bug: resampling is
systematic rather than multinomial. With integer rewards most weight vectors tie, and
systematic resampling of a uniform weight vector returns every particle exactly once, where
multinomial would duplicate and drop particles for no reason.

Every threshold the review file judges against sits here, so moving one after the answer is
visible shows up in a diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean
from poe_repair._sdxl.runtime import decode_latents
from poe_repair.experiments.twisted_smc.sampler import ddim_step_eta, systematic_resample
from poe_repair.methods._sampling import add_time_ids, write_decoded_image

# ---------------------------------------------------------------------------
# The bars (pre-registered in the review file; do not move after the run)
# ---------------------------------------------------------------------------
#   support:      FK at K 16 composes at least PASS_MARGIN more often than the unweighted control
#   null:         |FK - control| <= NULL_MARGIN at both K
#   inconclusive: the reward read on x0hat at step REWARD_BLIND_STEP disagrees with the final
#                 detector verdict on more than MAX_REWARD_DISAGREEMENT of the final particles
PASS_MARGIN = 0.25
NULL_MARGIN = 0.10
MAX_REWARD_DISAGREEMENT = 0.50
REWARD_BLIND_STEP = 10          # the last reward read inside the compose-decisive window (steps 0 to 10)

# ---------------------------------------------------------------------------
# The method's fixed settings (the paper's, scaled to 50 steps)
# ---------------------------------------------------------------------------
LAMBDA = 10.0
REWARD_CLIP = 2                 # r = min(instance count, 2): 0, 1 or 2
RESAMPLE_STEP_INDICES = (0, 10, 20, 30, 40)
K_VALUES = (4, 16)
ETA = 1.0

# ---------------------------------------------------------------------------
# Plan 11: the same steering on top of the rank-32 correction (the adapter as proposal)
# ---------------------------------------------------------------------------
#   The proposal is eps_PoE_frozen + lambda_adapter * (eps_PoE_lora - eps_PoE_frozen), the
#   showcase renders' composition. Judged at ADAPTER_JUDGED_LAMBDA, where the adapter alone
#   composes only part of the seeds, so selection has room to show.
#   support:      FK at K 16 composes at least PASS_MARGIN more often than the adapter-alone
#                 control (particle 0, same lambda) at ADAPTER_JUDGED_LAMBDA
#   null:         |FK - control| <= NULL_MARGIN at every lambda in ADAPTER_LAMBDAS
#   inconclusive: the same reward-blind check as above
ADAPTER_CHECKPOINT = "/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt"
ADAPTER_RANK = 32
ADAPTER_LAMBDAS = (0.5, 1.2)
ADAPTER_JUDGED_LAMBDA = 0.5
ADAPTER_K = 16

# Fidelity tie-breaker (plan 11). The compose count decides the level; a human-preference score
# (ImageReward, the paper's own text-to-image reward) orders particles within a level, so among
# two-animal particles the cleanest is kept. The weight is under 1 so it can never outrank a count.
#   r = min(count, 2) + FIDELITY_WEIGHT * sigmoid(ImageReward score)
FIDELITY_WEIGHT = 0.5


@dataclass
class SteerOutputs:
    latents: torch.Tensor                       # (K, 4, h, w) final
    rewards_by_step: dict[int, list[float]] = field(default_factory=dict)    # step -> per-particle reward (pre-resample order)
    counts_by_step: dict[int, list[int]] = field(default_factory=dict)       # step -> raw detector count
    fidelity_by_step: dict[int, list[float]] = field(default_factory=dict)   # step -> raw fidelity score, if a fidelity model is used
    lineage_max: torch.Tensor | None = None     # (K,) running max reward per surviving particle
    resampled_at: list[int] = field(default_factory=list)
    ancestry: list[list[int]] = field(default_factory=list)                  # per resample: idx[new] = old
    weights_by_step: dict[int, list[float]] = field(default_factory=dict)    # normalised weights used at each resample
    xhat_paths: dict[int, list[str]] = field(default_factory=dict)


def clipped_reward(count: int, fidelity: float | None = None) -> float:
    """``fidelity`` is a raw ImageReward score (about -2 to 2) or None for count only."""
    r = float(min(int(count), REWARD_CLIP))
    if fidelity is not None:
        r += FIDELITY_WEIGHT * float(torch.sigmoid(torch.tensor(float(fidelity))).item())
    return r


@torch.no_grad()
def run_fk_steering(
    *,
    init_latents: torch.Tensor,             # (K, 4, h, w), already divided by euler sigma
    models: dict,
    scheduler,
    seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
    eta: float = ETA,
    generator: torch.Generator | None = None,
    resample_generator: torch.Generator | None = None,   # separate stream, so the DDIM noise draws match the control's
    reward_fn=None,                         # (image_path) -> int count, or (count, fidelity); None = the control
    resample_steps: tuple[int, ...] = RESAMPLE_STEP_INDICES,
    lam: float = LAMBDA,
    xhat_dir: Path | None = None,           # where the decoded x0hat images go (required when reward_fn is set)
    unet_chunk: int = 2,
    adapter_lambda: float | None = None,    # None or 0: plain PoE; else eps_frozen + lambda * (eps_lora - eps_frozen)
    adapter_name: str = "lora",
) -> SteerOutputs:
    """K particles on the plain PoE score. With ``reward_fn`` set, FK steering with the max
    potential at ``resample_steps``; with ``reward_fn`` None, the same K particles left alone,
    which is the unweighted control at the same ``eta``, K and noise draws."""
    scheduler.set_timesteps(num_inference_steps)
    K = init_latents.shape[0]
    latents = init_latents.to(device=device, dtype=dtype)
    unet = models["unet"]
    branches = [(seq_a, pool_a), (seq_b, pool_b), (seq_e, pool_e)]
    nb = len(branches)
    pe_one = torch.cat([s for s, _ in branches], dim=0)
    pool_one = torch.cat([p for _, p in branches], dim=0)

    use_adapter = adapter_lambda is not None and float(adapter_lambda) != 0.0

    def _adapter(on: bool) -> None:
        try:
            if on:
                unet.enable_adapters()
                if hasattr(unet, "set_adapter"):
                    unet.set_adapter(adapter_name)
            else:
                unet.disable_adapters()
        except ValueError:
            pass    # no adapter attached: the frozen forward is plain PoE regardless

    def poe_forward(chunk: torch.Tensor, timestep) -> torch.Tensor:
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
        ea = guided_eps(noise[:, 0], noise[:, 2], guidance_scale)
        eb = guided_eps(noise[:, 1], noise[:, 2], guidance_scale)
        return poe_eps(ea, eb, noise[:, 2])

    def eps_for(lat: torch.Tensor, timestep) -> torch.Tensor:
        out = []
        for s in range(0, lat.shape[0], unet_chunk):
            chunk = lat[s: s + unet_chunk]
            _adapter(False)
            eps_frozen = poe_forward(chunk, timestep)
            if use_adapter:
                _adapter(True)
                eps_lora = poe_forward(chunk, timestep)
                _adapter(False)
                out.append(eps_frozen + float(adapter_lambda) * (eps_lora - eps_frozen))
            else:
                out.append(eps_frozen)
        return torch.cat(out)

    res = SteerOutputs(latents=latents)
    steering = reward_fn is not None
    lineage_max = torch.full((K,), -1.0, device="cpu")

    for step_index, timestep in enumerate(scheduler.timesteps):
        eps_t = eps_for(latents, timestep)
        ab_t = scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)
        x0 = tweedie_mean(latents, ab_t, eps_t)

        if steering and step_index in resample_steps:
            assert xhat_dir is not None
            step_dir = xhat_dir / f"step_{step_index:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            counts, fids, paths = [], [], []
            for k in range(K):
                p = step_dir / f"p{k}.png"
                write_decoded_image(decode_latents(models, x0[k: k + 1]).cpu(), p)
                r = reward_fn(p)
                c, f = (r if isinstance(r, tuple) else (r, None))
                counts.append(int(c)); fids.append(None if f is None else float(f))
                paths.append(str(p))
            rewards = [clipped_reward(c, f) for c, f in zip(counts, fids)]
            res.counts_by_step[step_index] = counts
            if any(f is not None for f in fids):
                res.fidelity_by_step[step_index] = fids
            res.rewards_by_step[step_index] = rewards
            res.xhat_paths[step_index] = paths
            lineage_max = torch.maximum(lineage_max, torch.tensor(rewards, dtype=torch.float32))
            log_w = lam * lineage_max                       # the max potential, log scale
            w = torch.softmax(log_w, dim=0)
            res.weights_by_step[step_index] = w.tolist()
            idx = systematic_resample(log_w, resample_generator if resample_generator is not None else generator)
            res.resampled_at.append(step_index)
            res.ancestry.append(idx.tolist())
            latents = latents[idx.to(device)]
            x0 = x0[idx.to(device)]
            eps_t = eps_t[idx.to(device)]
            lineage_max = lineage_max[idx]

        latents, _ = ddim_step_eta(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x_t=latents, x0=x0, eps=eps_t, eta=eta, generator=generator,
        )

    res.latents = latents
    res.lineage_max = lineage_max if steering else None
    return res


def trace_ancestor(ancestry: list[list[int]], resampled_at: list[int], final_index: int, at_step: int) -> int:
    """Index, in the pre-resample population at ``at_step``, of the state a final particle
    descends from. ``ancestry[i][new] = old`` for the resample at ``resampled_at[i]``."""
    idx = final_index
    for step, anc in zip(reversed(resampled_at), reversed(ancestry)):
        if step < at_step:
            break
        idx = anc[idx]
    return idx


def verdict(*, fk16: float, fk4: float, ctrl16: float, ctrl4: float, disagreement: float | None,
            k_high: int = 16, k_low: int = 4) -> str:
    """``fk16``/``ctrl16`` are the rates at the larger K, ``fk4``/``ctrl4`` at the smaller."""
    if disagreement is not None and disagreement > MAX_REWARD_DISAGREEMENT:
        return ("inconclusive: the reward on x0hat at step %d disagrees with the final detector verdict "
                "on %.2f of the final particles (bar %.2f)" % (REWARD_BLIND_STEP, disagreement, MAX_REWARD_DISAGREEMENT))
    if fk16 - ctrl16 >= PASS_MARGIN:
        return "support: FK steering at K %d composes %.3f more often than the unweighted control (bar %.2f)" % (k_high, fk16 - ctrl16, PASS_MARGIN)
    if abs(fk16 - ctrl16) <= NULL_MARGIN and abs(fk4 - ctrl4) <= NULL_MARGIN:
        return "null: FK steering is within %.2f of the unweighted control at both K %d and K %d" % (NULL_MARGIN, k_low, k_high)
    return "neither bar met: K %d gap %.3f, K %d gap %.3f" % (k_high, fk16 - ctrl16, k_low, fk4 - ctrl4)


def dump(res: SteerOutputs, path: Path, extra: dict | None = None) -> None:
    payload = {
        "rewards_by_step": res.rewards_by_step, "counts_by_step": res.counts_by_step,
        "fidelity_by_step": res.fidelity_by_step,
        "weights_by_step": res.weights_by_step, "resampled_at": res.resampled_at,
        "ancestry": res.ancestry,
        "lineage_max": None if res.lineage_max is None else res.lineage_max.tolist(),
        "xhat_paths": res.xhat_paths,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1))
