"""Per-step noise search on the corrected sampler, in the window where the run commits.

The method follows Ramesh and Mardani, "Test-Time Scaling of Diffusion Models via Noise
Trajectory Search" (arXiv 2506.03164): the injected noise of a stochastic sampler is a search
variable at every step, and a greedy contextual-bandit search over it (a few candidates per
step, judged on the model's running estimate of the finished image, local perturbations of the
current best plus an occasional fresh draw) matches tree search at a fraction of its cost.

Here the sampler is product-of-experts plus the rank-32 correction (checkpoint step 30050) at
``lambda`` on every step, run as stochastic DDIM at ``eta`` 1.0, and the search runs only over
``SEARCH_STEPS`` (the steps where the corrected run commits to two animals and where its
running estimate goes soft). Outside the window the injected noise is the control's own draw,
so the searched run and the unsearched control share every noise draw except the chosen ones.

The reward on a candidate is the one plan 11 chose: the validated instance count, clipped at
2, plus half a sigmoid of ImageReward (a human-preference score against the joint prompt), so
a candidate that loses an animal is never preferred to one that keeps two, and among
two-animal candidates the cleanest wins. The judge is separate from the reward: sharpness
(Laplacian variance, paired per seed against the control), the compose count, and both-ness
in DINOv2 space.

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
from poe_repair.experiments.twisted_smc.sampler import _alpha_prev
from poe_repair.methods._sampling import add_time_ids, write_decoded_image

# ---------------------------------------------------------------------------
# The bars (pre-registered in the review file; do not move after the run)
# ---------------------------------------------------------------------------
#   Judged on cat x dog, seeds 9 to 16, searched run against the eta-1 control at the same lambda.
#   support:      compose_n(search) >= compose_n(control) - SUPPORT_MAX_COMPOSE_LOSS
#                 and the searched render is sharper than the control's on at least
#                 SUPPORT_MIN_SHARPER_SEEDS of 8 seeds (paired Laplacian variance at 1024 px)
#   null:         compose_n(search) <= compose_n(control) - NULL_MIN_COMPOSE_LOSS
#                 or sharper on at most NULL_MAX_SHARPER_SEEDS of 8 (a coin flip or worse)
#   inconclusive: otherwise (sharper on exactly 5 of 8 with the animals kept)
#   control pair: intact if presence_n(search) >= presence_n(control) - CONTROL_MAX_PRESENCE_LOSS
SUPPORT_MAX_COMPOSE_LOSS = 1
NULL_MIN_COMPOSE_LOSS = 2
SUPPORT_MIN_SHARPER_SEEDS = 6
NULL_MAX_SHARPER_SEEDS = 4
CONTROL_MAX_PRESENCE_LOSS = 1
BUTTERFLY_PRESENT_CONF = 0.30

# ---------------------------------------------------------------------------
# The method's fixed settings
# ---------------------------------------------------------------------------
ETA = 1.0
LAMBDA_ADAPTER = 1.2                      # the shipped setting, every step
SEARCH_STEPS = tuple(range(8, 26))        # step indices searched (t 821 down to t 481 at 50 steps)
NUM_CANDIDATES = 3                        # fresh candidates per round, beside the current pivot
NUM_ROUNDS = 3                            # rounds per step; the best of a round is the next pivot
EPSILON = 0.25                            # probability a candidate is a fresh draw rather than local
LOCAL_SIGMA = 0.3                         # local candidate: (pivot + sigma u) / sqrt(1 + sigma^2), u ~ N(0, I)
REWARD_CLIP = 2
FIDELITY_WEIGHT = 0.5
FRAME_STEPS = (0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 49)

ADAPTER_CHECKPOINT = "/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt"
ADAPTER_RANK = 32


@dataclass
class SearchOutputs:
    latents: torch.Tensor                                   # (1, 4, h, w) final
    frames: dict[int, str] = field(default_factory=dict)     # step -> decoded running-estimate png
    pivot_reward_by_step: dict[int, float] = field(default_factory=dict)
    chosen_reward_by_step: dict[int, float] = field(default_factory=dict)
    chosen_kind_by_step: dict[int, str] = field(default_factory=dict)   # pivot / local / fresh
    chosen_count_by_step: dict[int, int] = field(default_factory=dict)
    chosen_fidelity_by_step: dict[int, float] = field(default_factory=dict)
    candidates_by_step: dict[int, list[dict]] = field(default_factory=dict)
    evaluations: int = 0


def clipped_reward(count: int, fidelity: float | None) -> float:
    r = float(min(int(count), REWARD_CLIP))
    if fidelity is not None:
        r += FIDELITY_WEIGHT * float(torch.sigmoid(torch.tensor(float(fidelity))).item())
    return r


def local_candidate(pivot: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    u = torch.randn(pivot.shape, generator=generator, dtype=torch.float32)
    return (pivot + sigma * u) / float((1.0 + sigma * sigma) ** 0.5)


@torch.no_grad()
def run_search(
    *,
    init_latents: torch.Tensor,             # (1, 4, h, w), already divided by euler sigma
    models: dict,
    scheduler,
    seq_a, pool_a, seq_b, pool_b, seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    device: torch.device, dtype: torch.dtype,
    adapter_lambda: float = LAMBDA_ADAPTER,
    eta: float = ETA,
    noise_generator: torch.Generator,       # the control's stream: one draw per step, searched or not
    search_generator: torch.Generator | None = None,   # the candidates' stream
    reward_fn=None,                         # (png path) -> (count, fidelity or None); None = the control
    search_steps: tuple[int, ...] = SEARCH_STEPS,
    num_candidates: int = NUM_CANDIDATES,
    num_rounds: int = NUM_ROUNDS,
    epsilon: float = EPSILON,
    local_sigma: float = LOCAL_SIGMA,
    work_dir: Path | None = None,           # candidate estimates and frames go here
    frame_steps: tuple[int, ...] = FRAME_STEPS,
    frame_size: int = 256,
    adapter_name: str = "lora",
) -> SearchOutputs:
    """One trajectory of the corrected sampler at ``eta``. With ``reward_fn`` None this is the
    control: the injected noise at every step is the next draw of ``noise_generator``. With a
    reward, at every step in ``search_steps`` that draw becomes the pivot of a greedy search:
    ``num_rounds`` rounds of ``num_candidates`` candidates (each fresh with probability
    ``epsilon``, else local to the pivot at ``local_sigma``), every candidate judged by the
    reward on the decoded running estimate one step ahead, the best of a round becoming the
    next pivot, and the final pivot injected. The forward pass that judged the winner is the
    next step's forward pass, so a step costs ``1 + num_rounds * num_candidates`` corrected
    forwards inside the window and one outside it."""
    scheduler.set_timesteps(num_inference_steps)
    latents = init_latents.to(device=device, dtype=dtype)
    unet = models["unet"]
    branches = [(seq_a, pool_a), (seq_b, pool_b), (seq_e, pool_e)]
    nb = len(branches)
    pe_one = torch.cat([s for s, _ in branches], dim=0)
    pool_one = torch.cat([p for _, p in branches], dim=0)
    use_adapter = adapter_lambda is not None and float(adapter_lambda) != 0.0
    timesteps = list(scheduler.timesteps)

    def _adapter(on: bool) -> None:
        try:
            if on:
                unet.enable_adapters()
                if hasattr(unet, "set_adapter"):
                    unet.set_adapter(adapter_name)
            else:
                unet.disable_adapters()
        except ValueError:
            pass

    def poe_forward(lat: torch.Tensor, timestep) -> torch.Tensor:
        k = lat.shape[0]
        latent_input = scheduler.scale_model_input(lat.repeat_interleave(nb, dim=0), timestep)
        cond = {"text_embeds": pool_one.repeat(k, 1),
                "time_ids": add_time_ids(height=height, width=width, batch_size=nb * k, device=device, dtype=dtype)}
        noise = unet(latent_input, timestep, encoder_hidden_states=pe_one.repeat(k, 1, 1),
                     added_cond_kwargs=cond, timestep_cond=None).sample
        noise = noise.view(k, nb, *noise.shape[1:])
        ea = guided_eps(noise[:, 0], noise[:, 2], guidance_scale)
        eb = guided_eps(noise[:, 1], noise[:, 2], guidance_scale)
        return poe_eps(ea, eb, noise[:, 2])

    def eps_for(lat: torch.Tensor, timestep) -> torch.Tensor:
        _adapter(False)
        eps_frozen = poe_forward(lat, timestep)
        if not use_adapter:
            return eps_frozen
        _adapter(True)
        eps_lora = poe_forward(lat, timestep)
        _adapter(False)
        return eps_frozen + float(adapter_lambda) * (eps_lora - eps_frozen)

    def ab_of(timestep) -> torch.Tensor:
        return scheduler.alphas_cumprod[int(timestep.item())].to(device=device, dtype=dtype)

    def step_coefs(step_index: int, timestep):
        ab_t = ab_of(timestep)
        ab_prev, _ = _alpha_prev(scheduler, step_index, latents)
        sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / ab_prev)
        dir_coef = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0))
        return ab_prev, dir_coef, sigma

    def x_prev_of(x0, eps_t, z, ab_prev, dir_coef, sigma):
        return torch.sqrt(ab_prev) * x0 + dir_coef * eps_t + sigma * z.to(device=device, dtype=dtype)

    def save_estimate(x0_lat: torch.Tensor, path: Path, size: int | None) -> None:
        img = decode_latents(models, x0_lat).cpu()
        if size is not None and size != img.shape[-1]:
            img = torch.nn.functional.interpolate(img.float(), size=(size, size), mode="area")
        write_decoded_image(img, path)

    res = SearchOutputs(latents=latents)
    searching = reward_fn is not None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = (work_dir / "frames") if work_dir is not None else None
    if frames_dir is not None:
        frames_dir.mkdir(exist_ok=True)

    eps_t = eps_for(latents, timesteps[0])
    for step_index, timestep in enumerate(timesteps):
        ab_t = ab_of(timestep)
        x0 = tweedie_mean(latents, ab_t, eps_t)
        if frames_dir is not None and step_index in frame_steps:
            p = frames_dir / f"step_{step_index:02d}.png"
            save_estimate(x0, p, frame_size)
            res.frames[step_index] = str(p)
        ab_prev, dir_coef, sigma = step_coefs(step_index, timestep)
        pivot = torch.randn(latents.shape, generator=noise_generator, dtype=torch.float32)
        last = step_index + 1 >= len(timesteps)
        if last:
            # the same final step as ddim_step_eta: at eta 1 it carries a small last draw
            latents = x_prev_of(x0, eps_t, pivot, ab_prev, dir_coef, sigma)
            break
        next_t = timesteps[step_index + 1]

        if not (searching and step_index in search_steps):
            latents = x_prev_of(x0, eps_t, pivot, ab_prev, dir_coef, sigma)
            eps_t = eps_for(latents, next_t)
            continue

        # -- the search at this step -------------------------------------------------------
        assert work_dir is not None and search_generator is not None
        step_dir = work_dir / "candidates" / f"step_{step_index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        def evaluate(z: torch.Tensor, tag: str) -> dict:
            x_next = x_prev_of(x0, eps_t, z, ab_prev, dir_coef, sigma)
            eps_next = eps_for(x_next, next_t)
            x0_next = tweedie_mean(x_next, ab_of(next_t), eps_next)
            png = step_dir / f"{tag}.png"
            save_estimate(x0_next, png, None)
            count, fid = reward_fn(png)
            res.evaluations += 1
            return {"tag": tag, "z": z, "x_next": x_next, "eps_next": eps_next,
                    "count": int(count), "fidelity": None if fid is None else float(fid),
                    "reward": clipped_reward(count, fid), "png": str(png)}

        best = evaluate(pivot, "pivot")
        best["kind"] = "pivot"
        res.pivot_reward_by_step[step_index] = best["reward"]
        log_rows = [{k: best[k] for k in ("tag", "kind", "count", "fidelity", "reward", "png")}]
        for r in range(num_rounds):
            for c in range(num_candidates):
                fresh = torch.rand((), generator=search_generator).item() < epsilon
                if fresh:
                    z = torch.randn(latents.shape, generator=search_generator, dtype=torch.float32)
                else:
                    z = local_candidate(best["z"], local_sigma, search_generator)
                cand = evaluate(z, f"r{r}c{c}")
                cand["kind"] = "fresh" if fresh else "local"
                log_rows.append({k: cand[k] for k in ("tag", "kind", "count", "fidelity", "reward", "png")})
                if cand["reward"] > best["reward"]:
                    best = cand
        res.chosen_reward_by_step[step_index] = best["reward"]
        res.chosen_kind_by_step[step_index] = best["kind"]
        res.chosen_count_by_step[step_index] = best["count"]
        if best["fidelity"] is not None:
            res.chosen_fidelity_by_step[step_index] = best["fidelity"]
        res.candidates_by_step[step_index] = log_rows
        latents, eps_t = best["x_next"], best["eps_next"]

    res.latents = latents
    return res


def dump(res: SearchOutputs, path: Path, extra: dict | None = None) -> None:
    payload = {"frames": res.frames, "pivot_reward_by_step": res.pivot_reward_by_step,
               "chosen_reward_by_step": res.chosen_reward_by_step, "chosen_kind_by_step": res.chosen_kind_by_step,
               "chosen_count_by_step": res.chosen_count_by_step, "chosen_fidelity_by_step": res.chosen_fidelity_by_step,
               "candidates_by_step": res.candidates_by_step, "evaluations": res.evaluations}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1))


def verdict(*, compose_search: int, compose_control: int, sharper_seeds: int, n_seeds: int) -> str:
    loss = compose_control - compose_search
    if loss >= NULL_MIN_COMPOSE_LOSS:
        return "null: the animals go with the search (compose %d against the control's %d of %d)" % (compose_search, compose_control, n_seeds)
    if sharper_seeds <= NULL_MAX_SHARPER_SEEDS:
        return "null: the searched render is sharper than the control on only %d of %d seeds (bar %d)" % (sharper_seeds, n_seeds, SUPPORT_MIN_SHARPER_SEEDS)
    if loss <= SUPPORT_MAX_COMPOSE_LOSS and sharper_seeds >= SUPPORT_MIN_SHARPER_SEEDS:
        return "support: animals kept (%d against %d of %d) and sharper on %d of %d seeds (bar %d)" % (compose_search, compose_control, n_seeds, sharper_seeds, n_seeds, SUPPORT_MIN_SHARPER_SEEDS)
    return "inconclusive: animals kept, sharper on %d of %d seeds, between the bars" % (sharper_seeds, n_seeds)


def control_verdict(*, presence_search: int, presence_control: int) -> str:
    if presence_control - presence_search > CONTROL_MAX_PRESENCE_LOSS:
        return "breaks the control: the butterfly goes (%d against %d)" % (presence_search, presence_control)
    return "control intact (%d against %d)" % (presence_search, presence_control)
