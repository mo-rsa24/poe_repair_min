#!/usr/bin/env python
"""Correct early, then clean up (plan 14, 01-showcase-the-trained-lora).

Does a correction confined to the early denoising steps, followed by a plain product-of-experts
tail, give two animals at plain-PoE sharpness? Five stages, run in this order:

  --sharpness-over-steps   no GPU. Laplacian variance of every saved running-estimate frame under
                           FRAMES_ROOT (six conditions, seeds 9 to 16, 14 saved steps), one thin
                           line per seed and a thick mean per condition, with the pre-registered
                           read at EARLY_STEP and LATE_STEP.
  --detach-check           GPU. Plain PoE on seed 9 before the adapter is attached, a hard-cut run,
                           then plain PoE again through a sampler that never touches the adapter.
                           The two plain renders must agree within DETACH_MAX_MEAN_ABS_DIFF.
  --render --stage schedule   mono, plain PoE, full window, hard cut, decay; 8 seeds; both pairs.
  --render --stage tail       the best schedule (pick_best_schedule) and plain PoE at STEPS_TAIL.
  --render --stage renoise    the best schedule to RENOISE_FROM_STEP, its running estimate noised
                              back to T_RENOISE, plain PoE from RENOISE_TAIL_START_STEP to the end.
  --render --stage renoise-sweep   the re-noise level swept (the knob Consistency Trajectory Models'
                              gamma-sampling and Restart sampling turn): the best schedule's committed
                              image, from two sources (its running estimate at RENOISE_FROM_STEP and its
                              finished render), noised back to each t in RENOISE_LEVELS_T, plain PoE from
                              there to the end. One head render per seed, one short tail per level.
  --score                  compose (instance count), sharpness at 1024 px, both-ness on the landing
                           finding's cloud axes (cat x dog), butterfly presence (control pair),
                           results.json, one sheet per cell, W&B. Also the post-hoc per-seed paired
                           sharpness read (seeds where the cell is sharper than the full window / plain PoE).
  --strips                 no GPU. One strip per seed and pair: Mono | plain PoE | full window | best
                           schedule | best re-noise cell, the scorer's numbers under each tile, logged to
                           the same W&B run as images and as a per-seed table.

Every schedule is a function of the denoising timestep t (1000 noise, 0 image), so the 200-step
tail switches the correction off at the same noise level as the 50-step run. At 50 DDIM steps
the sampler visits t = 981 - 20k at step index k: step 10 is t 781, step 20 is t 581, step 35 is
t 281.

Thresholds live here as named constants so a change shows up in a diff.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/showcase"))

# ----------------------------------------------------------------------------- constants
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/correct_early_then_clean_up")
RESULTS_DIR = REPO_ROOT / "artifacts/results/does-correcting-early-then-cleaning-up-restore-sharpness"
FRAMES_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames")
TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
CLOUD_FEATS = REPO_ROOT / "artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy"
CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
LORA_RANK = 32
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"

SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
# slug -> (prompt_a, prompt_b, joint prompt, detector query a, detector query b)
PAIRS = {
    "a_cat__x__a_dog": ("a cat", "a dog", "a cat and a dog", "cat", "dog"),
    "a_butterfly__x__a_flower_meadow": ("a butterfly", "a flower meadow",
                                        "a butterfly and a flower meadow", "butterfly", "flower meadow"),
}
MAIN_PAIR = "a_cat__x__a_dog"
CONTROL_PAIR = "a_butterfly__x__a_flower_meadow"
GUIDANCE_SCALE = 7.5
STEPS_MAIN = 50
STEPS_TAIL = 200
EULER_INIT_NOISE_SIGMA = 1.0

LAMBDA_FULL = 1.2
T_CUT = 781          # step 10 at 50 steps: lambda is LAMBDA_FULL for t >= T_CUT
T_DECAY_END = 581    # step 20 at 50 steps: the decay reaches 0 here
T_RENOISE = 281      # step 35 at 50 steps: the re-noise target level
RENOISE_FROM_STEP = 20        # the running estimate taken at this step index (50-step grid)
RENOISE_TAIL_START_STEP = 35  # plain PoE resumes at this step index (50-step grid)

# task 1, the per-step read
EARLY_STEP = 20
LATE_STEP = 50

# the bar (review file, "The question written before the run")
SUPPORT_MAX_COMPOSE_LOSS = 1   # seeds of 8 a cell may lose against the full window and still support
NULL_MIN_COMPOSE_LOSS = 2      # losing this many is a null
# the control pair
BUTTERFLY_PRESENT_CONF = 0.30
CONTROL_MAX_PRESENCE_LOSS = 1
# the instrument checks, in mean absolute grey levels over the 1024 px image
DETACH_MAX_MEAN_ABS_DIFF = 1.0
IDENTITY_MAX_MEAN_ABS_DIFF = 6.0

SCHEDULE_CANDIDATES = ("hardcut_10", "decay_10_20")

# the re-noise level sweep: how far back the committed image is noised before plain PoE finishes it
RENOISE_LEVELS_T = (481, 381, 281, 181)   # steps 25, 30, 35, 40 on the 50-step grid (t = 981 - 20k)
RENOISE_SOURCES = ("x0s20", "final")      # x0s20: Tweedie mean at RENOISE_FROM_STEP; final: the finished schedule render
STRIP_TILE = 320


def lambda_of_t(name: str, t: float) -> float:
    if name == "poe":
        return 0.0
    if name == "full_1.2":
        return LAMBDA_FULL
    if name == "hardcut_10":
        return LAMBDA_FULL if t >= T_CUT else 0.0
    if name == "decay_10_20":
        if t >= T_CUT:
            return LAMBDA_FULL
        if t <= T_DECAY_END:
            return 0.0
        return LAMBDA_FULL * (t - T_DECAY_END) / (T_CUT - T_DECAY_END)
    raise KeyError(name)


DESCRIPTIONS = {
    "mono": "the joint prompt, plain CFG",
    "poe": "plain PoE, no correction",
    "full_1.2": f"PoE + {LAMBDA_FULL} x correction on all steps",
    "hardcut_10": f"lambda {LAMBDA_FULL} while t >= {T_CUT} (steps 0 to 10), 0 after",
    "decay_10_20": f"lambda {LAMBDA_FULL} while t >= {T_CUT}, straight line to 0 at t {T_DECAY_END} (step 20), 0 after",
}


# ----------------------------------------------------------------------------- helpers
def _laplacian_var(img_path: Path) -> float:
    from lambda_window_grid import _laplacian_var as f  # the same function plan 07 used
    return f(img_path)


def mean_abs_diff(a: Path, b: Path) -> float:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if x.shape != y.shape:
        y = np.asarray(Image.open(b).convert("RGB").resize((x.shape[1], x.shape[0]), Image.LANCZOS), dtype=np.float32)
    return float(np.abs(x - y).mean())


def _log(msg: str) -> None:
    print(f"[correct_early {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================= task 1
def sharpness_over_steps() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ("solo_a", "solo_b", "joint", "poe", "lora_1.0", "lora_1.2")
    labels = {"solo_a": "a cat alone", "solo_b": "a dog alone", "joint": '"a cat and a dog"',
              "poe": "PoE, no correction", "lora_1.0": "PoE + 1.0 x correction",
              "lora_1.2": "PoE + 1.2 x correction"}
    colors = {"solo_a": "#d95f02", "solo_b": "#1b9e77", "joint": "#7570b3",
              "poe": "#111111", "lora_1.0": "#666666", "lora_1.2": "#e7298a"}
    steps = sorted(int(p.stem.split("_")[1]) for p in (FRAMES_ROOT / "poe" / "seed_9").glob("step_*.png"))
    sharp = {c: {s: [ _laplacian_var(FRAMES_ROOT / c / f"seed_{s}" / f"step_{k:03d}.png") for k in steps]
                 for s in SEEDS} for c in conds}
    mean = {c: [float(np.mean([sharp[c][s][i] for s in SEEDS])) for i in range(len(steps))] for c in conds}
    band_lo = [float(min(sharp["poe"][s][i] for s in SEEDS)) for i in range(len(steps))]
    band_hi = [float(max(sharp["poe"][s][i] for s in SEEDS)) for i in range(len(steps))]

    ie, il = steps.index(EARLY_STEP), steps.index(LATE_STEP)
    m = mean["lora_1.2"]
    inside_early = band_lo[ie] <= m[ie] <= band_hi[ie]
    below_early = m[ie] < band_lo[ie]
    below_late = m[il] < band_lo[il]
    if inside_early and below_late:
        verdict = "late"
    elif below_early:
        verdict = "committed"
    else:
        verdict = "inconclusive"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(steps, band_lo, band_hi, color="#bbbbbb", alpha=0.35, lw=0,
                    label="plain-PoE seed band (min to max of 8 seeds)")
    for c in conds:
        for s in SEEDS:
            ax.plot(steps, sharp[c][s], color=colors[c], lw=0.7, alpha=0.3)
        ax.plot(steps, mean[c], color=colors[c], lw=2.6, label=f"{labels[c]} (mean of 8 seeds)")
    for k in (EARLY_STEP, LATE_STEP):
        ax.axvline(k, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("denoising step (0 = pure noise, 50 = finished image); one point per saved frame")
    ax.set_ylabel("sharpness: Laplacian variance of the 256 px running estimate (higher = crisper)")
    ax.set_title("cat x dog, held-out seeds 9 to 16: sharpness of each run's running estimate over the 50 steps\n"
                 f"thin lines one seed each, thick lines the condition mean; read at steps {EARLY_STEP} and {LATE_STEP}: {verdict}",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "sharpness-over-denoising-steps.png", dpi=160)

    out = {
        "source_frames": str(FRAMES_ROOT), "frame_px": 256,
        "sharpness": "Laplacian variance of the greyscale image, lambda_window_grid._laplacian_var (plan 07's function)",
        "steps": steps, "seeds": list(SEEDS),
        "EARLY_STEP": EARLY_STEP, "LATE_STEP": LATE_STEP,
        "rule": "late if the lambda 1.2 mean is inside the plain-PoE seed band at EARLY_STEP and below its lower edge at LATE_STEP; committed if below the lower edge at EARLY_STEP; inconclusive otherwise",
        "verdict": verdict,
        "read": {"lora_1.2_mean_at_early": m[ie], "poe_band_at_early": [band_lo[ie], band_hi[ie]],
                 "lora_1.2_mean_at_late": m[il], "poe_band_at_late": [band_lo[il], band_hi[il]],
                 "poe_mean_at_early": mean["poe"][ie], "poe_mean_at_late": mean["poe"][il]},
        "mean_by_condition": mean, "poe_band_lo": band_lo, "poe_band_hi": band_hi,
        "per_seed": {c: {str(s): sharp[c][s] for s in SEEDS} for c in conds},
    }
    (RESULTS_DIR / "sharpness-over-denoising-steps.json").write_text(json.dumps(out, indent=2))
    _log(f"verdict={verdict} read={json.dumps(out['read'])}")
    return out


# ============================================================================= samplers
def _make_ctx(num_steps: int):
    from poe_repair.run import make_ctx
    return make_ctx(num_inference_steps=num_steps, guidance_scale=GUIDANCE_SCALE)


class Sampler:
    """Holds the SDXL context, the encoded prompts and the pinned initial noise per cell."""

    def __init__(self):
        from poe_repair.runtime import encode_prompt_sdxl
        self.ctx = _make_ctx(STEPS_MAIN)
        ctx = self.ctx
        enc = lambda p: encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)  # noqa: E731
        self.seq_e, self.pool_e = enc("")
        self.prompts = {}
        for slug, (pa, pb, pj, _, _) in PAIRS.items():
            self.prompts[slug] = {"a": enc(pa), "b": enc(pb), "j": enc(pj)}
        self.attached = False
        self.attach_info = None

    def init_latents(self, slug: str, seed: int) -> torch.Tensor:
        from poe_repair.training_cache import CellPath
        from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
        try:
            cell = CellPath.from_root(slug, seed, split="heldout", cache_root=TRAINING_CACHE)
            cell.step_files()[0]
        except (FileNotFoundError, IndexError):
            # The cache stores one initial noise per seed, shared across pairs (checked: the step-0
            # x_t of cat x dog and butterfly x meadow are identical for seeds 9 to 12). The control
            # pair's cache stops at seed 12, so its seeds 13 to 16 take the noise from the main pair.
            cell = CellPath.from_root(MAIN_PAIR, seed, split="heldout", cache_root=TRAINING_CACHE)
            _log(f"init noise for {slug} seed {seed} taken from the {MAIN_PAIR} cache (same per-seed noise)")
        return load_pinned_init_latents(cell, device=self.ctx.device, dtype=self.ctx.dtype,
                                        euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA)

    def common(self, num_steps: int) -> dict:
        ctx = self.ctx
        return dict(models=ctx.models, scheduler=ctx.scheduler, seq_e=self.seq_e, pool_e=self.pool_e,
                    guidance_scale=GUIDANCE_SCALE, num_inference_steps=num_steps,
                    height=1024, width=1024, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
                    device=ctx.device, dtype=ctx.dtype)

    def attach(self) -> None:
        import lambda_boundary_probe as lbp
        lbp.LORA_RANK = lbp.LORA_ALPHA = LORA_RANK
        self.attach_info = lbp._attach_and_load_lora(self.ctx.models["unet"], CHECKPOINT)
        self.attached = True
        _log(f"LoRA rank={LORA_RANK} attached: n_matched={self.attach_info['n_matched']} "
             f"n_loaded={self.attach_info['n_loaded']} checkpoint_step={self.attach_info['checkpoint_step']}")

    # -- the reference samplers, which never touch the adapter --------------------------------
    def mono(self, slug: str, seed: int, num_steps: int = STEPS_MAIN):
        from poe_repair.methods._sampling import run_cfg
        sj, pj = self.prompts[slug]["j"]
        return run_cfg(init_latents=self.init_latents(slug, seed), seq_cond=sj, pool_cond=pj,
                       **self.common(num_steps))

    def plain_poe_untouched(self, slug: str, seed: int, num_steps: int = STEPS_MAIN):
        """poe_repair's own PoE sampler: it has no adapter code at all, so whatever state the
        adapter was left in is what this renders with. That is the detachment probe."""
        from poe_repair.methods._sampling import run_cfg_poe
        (sa, pa), (sb, pb) = self.prompts[slug]["a"], self.prompts[slug]["b"]
        return run_cfg_poe(init_latents=self.init_latents(slug, seed), seq_a=sa, pool_a=pa,
                           seq_b=sb, pool_b=pb, **self.common(num_steps))

    # -- the scheduled sampler ------------------------------------------------------------------
    @torch.no_grad()
    def scheduled(self, slug: str, seed: int, schedule: str, *, num_steps: int = STEPS_MAIN,
                  start_index: int = 0, end_index: int | None = None,
                  latents_at_start: torch.Tensor | None = None):
        """PoE with lambda(t) from `schedule`. Off-window steps (lambda 0) run the frozen guided
        PoE forward with the adapter disabled; on-window steps add lambda * (PoE with adapter -
        PoE frozen). The adapter is disabled again before returning, so a later render through a
        sampler that never toggles it is clean.

        start_index / latents_at_start let the re-noise cell resume at a later step from a latent
        already at that step's noise level; end_index stops the loop early (exclusive)."""
        from poe_repair._sdxl.metrics import ddim_prev_from_x0_eps, guided_eps, poe_eps, tweedie_mean
        from poe_repair._sdxl.runtime import LatentTrajectoryCollector, decode_latents
        from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
        import lambda_boundary_probe as lbp

        ctx = self.ctx
        device, dtype = ctx.device, ctx.dtype
        scheduler = ctx.scheduler
        scheduler.set_timesteps(num_steps)
        timesteps = scheduler.timesteps
        end_index = len(timesteps) if end_index is None else end_index
        if latents_at_start is None:
            latents = (self.init_latents(slug, seed) / EULER_INIT_NOISE_SIGMA).to(device=device, dtype=dtype)
        else:
            latents = latents_at_start.to(device=device, dtype=dtype)
        tracker = LatentTrajectoryCollector(num_steps, 1, latents.shape[1], latents.shape[2], latents.shape[3])
        (sa, pa), (sb, pb) = self.prompts[slug]["a"], self.prompts[slug]["b"]
        pe_3 = torch.cat([sa, sb, self.seq_e], dim=0)
        pool_3 = torch.cat([pa, pb, self.pool_e], dim=0)
        cond_3 = {"text_embeds": pool_3,
                  "time_ids": add_time_ids(height=1024, width=1024, batch_size=3, device=device, dtype=dtype)}
        unet = ctx.models["unet"]

        def _disable():
            if hasattr(unet, "disable_adapters"):
                unet.disable_adapters()
            elif hasattr(unet, "disable_adapter_layers"):
                unet.disable_adapter_layers()

        def _enable():
            if hasattr(unet, "enable_adapters"):
                unet.enable_adapters()
            elif hasattr(unet, "enable_adapter_layers"):
                unet.enable_adapter_layers()
            if hasattr(unet, "set_adapter"):
                try:
                    unet.set_adapter(lbp.LORA_ADAPTER_NAME)
                except Exception:
                    pass

        def _forward(timestep):
            x = scheduler.scale_model_input(latents.repeat(3, 1, 1, 1), timestep)
            return unet(x, timestep, encoder_hidden_states=pe_3, added_cond_kwargs=cond_3,
                        timestep_cond=None).sample.chunk(3)

        lam_per_step, t_per_step = [], []
        for step_index in range(start_index, end_index):
            timestep = timesteps[step_index]
            t = int(timestep.item())
            lam = lambda_of_t(schedule, t) if self.attached else 0.0
            if self.attached:
                _disable()
            ea, eb, eu = _forward(timestep)
            eps_poe = poe_eps(guided_eps(ea, eu, GUIDANCE_SCALE), guided_eps(eb, eu, GUIDANCE_SCALE), eu)
            if lam > 0.0:
                _enable()
                la, lb, lu = _forward(timestep)
                eps_lora = poe_eps(guided_eps(la, lu, GUIDANCE_SCALE), guided_eps(lb, lu, GUIDANCE_SCALE), lu)
                eps_t = eps_poe + float(lam) * (eps_lora - eps_poe)
            else:
                eps_t = eps_poe
            lam_per_step.append(float(lam)); t_per_step.append(t)
            ab = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
            x0 = tweedie_mean(latents, ab, eps_t)
            tracker.store_step(step_index, latents, eps_t, float(step_index) / float(num_steps), t)
            latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                            x0=x0, eps=eps_t)
        if self.attached:
            _disable()
        tracker.store_final(latents)
        image = decode_latents(ctx.models, latents).cpu() if end_index == len(timesteps) else None
        return SamplerOutputs(latents=latents, image=image, tracker=tracker,
                              extras={"schedule": schedule, "lambda_per_step": lam_per_step,
                                      "timestep_per_step": t_per_step, "num_steps": num_steps,
                                      "start_index": start_index, "end_index": end_index})

    @torch.no_grad()
    def renoise(self, slug: str, seed: int, schedule: str):
        """Run `schedule` up to and including RENOISE_FROM_STEP, take that step's running estimate
        (the Tweedie mean), add Gaussian noise back to the T_RENOISE level with a generator seeded
        by the seed, then plain PoE from RENOISE_TAIL_START_STEP to the end. All on the 50-step grid."""
        from poe_repair._sdxl.metrics import tweedie_mean
        from poe_repair._sdxl.runtime import decode_latents
        ctx = self.ctx
        head = self.scheduled(slug, seed, schedule, num_steps=STEPS_MAIN, end_index=RENOISE_FROM_STEP + 1)
        k = RENOISE_FROM_STEP
        z = head.tracker.trajectories[k].to(device=ctx.device, dtype=ctx.dtype)
        eps = head.tracker.velocities[k].to(device=ctx.device, dtype=ctx.dtype)
        t_k = int(head.tracker.timesteps[k].item())
        ab_k = ctx.scheduler.alphas_cumprod[t_k].to(device=ctx.device, dtype=ctx.dtype)
        x0 = tweedie_mean(z, ab_k, eps)
        ctx.scheduler.set_timesteps(STEPS_MAIN)
        t_re = int(ctx.scheduler.timesteps[RENOISE_TAIL_START_STEP].item())
        assert t_re == T_RENOISE, (t_re, T_RENOISE)
        ab_re = ctx.scheduler.alphas_cumprod[t_re].to(device=ctx.device, dtype=ctx.dtype)
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + RENOISE_TAIL_START_STEP)
        noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32).to(device=ctx.device, dtype=ctx.dtype)
        x_re = torch.sqrt(ab_re) * x0 + torch.sqrt(1.0 - ab_re) * noise
        tail = self.scheduled(slug, seed, "poe", num_steps=STEPS_MAIN, start_index=RENOISE_TAIL_START_STEP,
                              latents_at_start=x_re)
        tail.extras.update({"head_schedule": schedule, "renoise_from_step": k, "renoise_from_t": t_k,
                            "renoise_to_t": t_re, "x0_source": "Tweedie mean at renoise_from_step",
                            "head_lambda_per_step": head.extras["lambda_per_step"]})
        # the running estimate itself, for the sheet's provenance
        tail.extras["x0_image"] = decode_latents(ctx.models, x0).cpu()
        return tail


    @torch.no_grad()
    def renoise_heads(self, slug: str, seed: int, schedule: str):
        """The two committed images the sweep re-noises, from one full run of `schedule`: the running
        estimate (Tweedie mean) at RENOISE_FROM_STEP, which is the plan's re-noise cell's source, and
        the finished render's latent, which is the Restart-sampling source. Returns
        {source: (x0 latents, t the source was taken at)} and the full run's outputs."""
        from poe_repair._sdxl.metrics import tweedie_mean
        ctx = self.ctx
        full = self.scheduled(slug, seed, schedule, num_steps=STEPS_MAIN)
        k = RENOISE_FROM_STEP
        z = full.tracker.trajectories[k].to(device=ctx.device, dtype=ctx.dtype)
        eps = full.tracker.velocities[k].to(device=ctx.device, dtype=ctx.dtype)
        t_k = int(full.tracker.timesteps[k].item())
        ab_k = ctx.scheduler.alphas_cumprod[t_k].to(device=ctx.device, dtype=ctx.dtype)
        return {"x0s20": (tweedie_mean(z, ab_k, eps), t_k), "final": (full.latents, 0)}, full

    @torch.no_grad()
    def renoise_tail(self, slug: str, seed: int, x0: torch.Tensor, to_t: int):
        """Noise the clean latent x0 to the level `to_t` (a timestep on the 50-step grid) with a
        generator seeded by the seed and the start index, then plain PoE from that index to the end.
        At to_t == T_RENOISE the noise is the same draw the plan's re-noise cell used."""
        ctx = self.ctx
        ctx.scheduler.set_timesteps(STEPS_MAIN)
        ts = [int(x) for x in ctx.scheduler.timesteps.tolist()]
        start = ts.index(int(to_t))
        ab_re = ctx.scheduler.alphas_cumprod[int(to_t)].to(device=ctx.device, dtype=ctx.dtype)
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + start)
        noise = torch.randn(x0.shape, generator=gen, dtype=torch.float32).to(device=ctx.device, dtype=ctx.dtype)
        x_re = torch.sqrt(ab_re) * x0.to(device=ctx.device, dtype=ctx.dtype) + torch.sqrt(1.0 - ab_re) * noise
        tail = self.scheduled(slug, seed, "poe", num_steps=STEPS_MAIN, start_index=start, latents_at_start=x_re)
        tail.extras.update({"renoise_to_t": int(to_t), "tail_start_index": start})
        return tail


# ============================================================================= rendering
def _manifest_path() -> Path:
    return OUT_ROOT / "render_manifest.json"


def _load_manifest() -> list[dict]:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else []


def _save_manifest(rows: list[dict]) -> None:
    _manifest_path().write_text(json.dumps(rows, indent=2))


def _render_path(slug: str, cond: str, seed: int) -> Path:
    return OUT_ROOT / "renders" / slug / cond / f"seed_{seed}.png"


def _write(img, path: Path) -> None:
    from poe_repair.methods._sampling import write_decoded_image
    path.parent.mkdir(parents=True, exist_ok=True)
    write_decoded_image(img, path)


def _run_header() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def detach_check() -> dict:
    """Three renders of cat x dog seed 9 in one process. Writes detach_check.json."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    d = OUT_ROOT / "detach_check"; d.mkdir(exist_ok=True)
    S = Sampler()
    slug, seed = MAIN_PAIR, 9
    before = d / "poe_before_attach.png"
    _write(S.plain_poe_untouched(slug, seed).image, before)
    S.attach()
    hard = d / "hardcut_10_seed_9.png"
    _write(S.scheduled(slug, seed, "hardcut_10").image, hard)
    after = d / "poe_after_windowed_run_untouched_sampler.png"
    _write(S.plain_poe_untouched(slug, seed).image, after)
    lam0 = d / "poe_lambda0_through_scheduled_sampler.png"
    _write(S.scheduled(slug, seed, "poe").image, lam0)
    cached = TRAINING_CACHE / "heldout" / slug / f"seed_{seed}" / "poe.png"
    res = {
        **_run_header(),
        "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK, "attach_info": {k: v for k, v in S.attach_info.items() if isinstance(v, (int, float, str))},
        "mean_abs_diff_before_vs_after": mean_abs_diff(before, after),
        "mean_abs_diff_before_vs_lambda0": mean_abs_diff(before, lam0),
        "mean_abs_diff_before_vs_cached_poe": mean_abs_diff(before, cached),
        "mean_abs_diff_after_vs_cached_poe": mean_abs_diff(after, cached),
        "mean_abs_diff_hardcut_vs_before": mean_abs_diff(hard, before),
        "DETACH_MAX_MEAN_ABS_DIFF": DETACH_MAX_MEAN_ABS_DIFF,
        "IDENTITY_MAX_MEAN_ABS_DIFF": IDENTITY_MAX_MEAN_ABS_DIFF,
        "cached_poe": str(cached),
    }
    res["pass_detach"] = res["mean_abs_diff_before_vs_after"] <= DETACH_MAX_MEAN_ABS_DIFF
    res["pass_lambda0_same_sampler"] = res["mean_abs_diff_before_vs_lambda0"] <= DETACH_MAX_MEAN_ABS_DIFF
    res["pass_identity_vs_cached"] = res["mean_abs_diff_after_vs_cached_poe"] <= IDENTITY_MAX_MEAN_ABS_DIFF
    res["pass"] = bool(res["pass_detach"] and res["pass_lambda0_same_sampler"] and res["pass_identity_vs_cached"])
    (OUT_ROOT / "detach_check.json").write_text(json.dumps(res, indent=2))
    _log(f"detach check: {json.dumps({k: v for k, v in res.items() if k.startswith(('mean_abs', 'pass'))})}")
    return res


def pick_best_schedule(results: dict) -> str:
    """Highest compose count on the main pair among SCHEDULE_CANDIDATES; ties to the higher mean sharpness."""
    cells = results["summary"][MAIN_PAIR]
    return max(SCHEDULE_CANDIDATES, key=lambda c: (cells[c]["compose_n"], cells[c]["sharpness_mean"]))


def renoise_conditions(best: str) -> list[tuple[str, int, str]]:
    return [(src, t, f"{best}_renoise_{src}_t{t}") for src in RENOISE_SOURCES for t in RENOISE_LEVELS_T]


def pick_best_renoise(results: dict) -> str | None:
    """Among the re-noise cells on the main pair (the plan's cell and the sweep's), the highest compose
    count; ties to the higher mean sharpness. None if no re-noise cell is scored yet."""
    cells = results["summary"][MAIN_PAIR]
    cands = [c for c in cells if "_renoise" in c and isinstance(cells[c], dict)]
    if not cands:
        return None
    return max(cands, key=lambda c: (cells[c]["compose_n"], cells[c]["sharpness_mean"]))


def render(stage: str, seeds=SEEDS, pairs=tuple(PAIRS)) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest()
    done = {(r["pair"], r["condition"], r["seed"]) for r in rows}
    S = Sampler()
    _log(f"render stage={stage} header={_run_header()}")

    def _record(slug, cond, seed, out, num_steps, t0, extra=None):
        p = _render_path(slug, cond, seed)
        _write(out.image, p)
        row = {"pair": slug, "condition": cond, "seed": seed, "png": str(p), "num_steps": num_steps,
               "description": DESCRIPTIONS.get(cond, cond), "elapsed_s": round(time.time() - t0, 1),
               "lambda_per_step": out.extras.get("lambda_per_step"), "timestep_per_step": out.extras.get("timestep_per_step")}
        if extra:
            row.update(extra)
        rows.append(row); done.add((slug, cond, seed)); _save_manifest(rows)
        _log(f"{slug} {cond} seed={seed} steps={num_steps} {row['elapsed_s']}s -> {p.name}")

    if stage == "schedule":
        # pass 1: mono before the adapter exists on the UNet
        for slug in pairs:
            for seed in seeds:
                if (slug, "mono", seed) in done:
                    continue
                t0 = time.time(); _record(slug, "mono", seed, S.mono(slug, seed), STEPS_MAIN, t0)
        S.attach()
        for slug in pairs:
            for cond in ("poe", "full_1.2", "hardcut_10", "decay_10_20"):
                for seed in seeds:
                    if (slug, cond, seed) in done:
                        continue
                    t0 = time.time(); _record(slug, cond, seed, S.scheduled(slug, seed, cond), STEPS_MAIN, t0)
    elif stage in ("tail", "renoise"):
        results = json.loads((OUT_ROOT / "results.json").read_text())
        best = pick_best_schedule(results)
        _log(f"best schedule by pick_best_schedule: {best}")
        S.attach()
        if stage == "tail":
            S.ctx.scheduler.set_timesteps(STEPS_TAIL)
            ts = [int(x) for x in S.ctx.scheduler.timesteps.tolist()]
            _log(f"{STEPS_TAIL}-step grid contains T_CUT={T_CUT}: {T_CUT in ts}; T_DECAY_END={T_DECAY_END}: {T_DECAY_END in ts}")
            for slug in pairs:
                for cond, sched in ((f"poe_{STEPS_TAIL}", "poe"), (f"{best}_{STEPS_TAIL}", best)):
                    for seed in seeds:
                        if (slug, cond, seed) in done:
                            continue
                        t0 = time.time()
                        out = S.scheduled(slug, seed, sched, num_steps=STEPS_TAIL)
                        _record(slug, cond, seed, out, STEPS_TAIL, t0,
                                {"base_schedule": sched, "description": f"{DESCRIPTIONS[sched]}, {STEPS_TAIL} DDIM steps"})
        else:
            cond = f"{best}_renoise"
            for slug in pairs:
                for seed in seeds:
                    if (slug, cond, seed) in done:
                        continue
                    t0 = time.time()
                    out = S.renoise(slug, seed, best)
                    x0p = OUT_ROOT / "renders" / slug / cond / f"seed_{seed}_x0_at_step_{RENOISE_FROM_STEP}.png"
                    _write(out.extras["x0_image"], x0p)
                    _record(slug, cond, seed, out, STEPS_MAIN, t0,
                            {"base_schedule": best, "renoise_from_step": RENOISE_FROM_STEP,
                             "renoise_from_t": out.extras["renoise_from_t"], "renoise_to_t": out.extras["renoise_to_t"],
                             "x0_png": str(x0p),
                             "description": f"{DESCRIPTIONS[best]} to step {RENOISE_FROM_STEP}; its running estimate noised back to t {T_RENOISE}; plain PoE from step {RENOISE_TAIL_START_STEP}"})
    elif stage == "renoise-sweep":
        results = json.loads((OUT_ROOT / "results.json").read_text())
        best = pick_best_schedule(results)
        _log(f"best schedule by pick_best_schedule: {best}")
        S.attach()
        conds = renoise_conditions(best)
        src_words = {"x0s20": f"its running estimate at step {RENOISE_FROM_STEP}", "final": "its finished render"}
        for slug in pairs:
            for seed in seeds:
                todo = [(src, t, c) for src, t, c in conds if (slug, c, seed) not in done]
                if not todo:
                    continue
                t0 = time.time()
                heads, _full = S.renoise_heads(slug, seed, best)
                _log(f"{slug} seed={seed} heads of {best} rendered in {time.time() - t0:.1f}s")
                for src, t, c in todo:
                    t1 = time.time()
                    x0, from_t = heads[src]
                    out = S.renoise_tail(slug, seed, x0, t)
                    start = out.extras["tail_start_index"]
                    _record(slug, c, seed, out, STEPS_MAIN, t1,
                            {"base_schedule": best, "renoise_source": src, "renoise_from_t": from_t,
                             "renoise_to_t": t, "tail_start_index": start,
                             "description": f"{DESCRIPTIONS[best]}; {src_words[src]} noised back to t {t} (step {start}); plain PoE from step {start} to the end"})
    else:
        raise SystemExit(f"unknown stage {stage}")
    _log("render done")


# ============================================================================= scoring
def _cloud_axes():
    """Rebuild the landing finding's cloud axes from its saved 48 x 384 features (rows: solo_a,
    solo_b, joint, poe, lora_1.0, lora_1.2; 8 seeds each)."""
    feats = np.load(CLOUD_FEATS)
    ca, cb, cj = feats[0:8].mean(0), feats[8:16].mean(0), feats[16:24].mean(0)
    origin = 0.5 * (ca + cb)
    u1 = cb - ca; u1 /= np.linalg.norm(u1)
    u2 = cj - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    ref = {"poe": [float((f - origin) @ u2) for f in feats[24:32]],
           "lora_1.2": [float((f - origin) @ u2) for f in feats[40:48]],
           "joint": [float((f - origin) @ u2) for f in feats[16:24]]}
    return origin, u1, u2, ref


def verdict_for_cell(cell: dict, full: dict, poe: dict, premise_holds: bool) -> str:
    if not premise_holds:
        return "inconclusive: no softness to fix (full window already inside the plain band)"
    loss = full["compose_n"] - cell["compose_n"]
    if loss >= NULL_MIN_COMPOSE_LOSS:
        return "null: the animals go with the correction"
    if cell["sharpness_mean"] <= full["sharpness_mean"]:
        return "null: no sharpness gained over the full window"
    if loss <= SUPPORT_MAX_COMPOSE_LOSS and cell["sharpness_mean"] >= poe["sharpness_min"]:
        return "support"
    return "inconclusive: animals stay, sharpness rises, band not reached"


def control_verdict(cell: dict, poe: dict) -> str:
    if poe["presence_n"] - cell["presence_n"] > CONTROL_MAX_PRESENCE_LOSS:
        return "breaks the control: the butterfly goes"
    if cell["sharpness_mean"] < poe["sharpness_min"]:
        return "breaks the control: softer than its plain band"
    return "control intact"


def score() -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances)
    from scripts.build_lora_inspector_mds_semantic import DinoEmbedder

    rows = _load_manifest()
    if not rows:
        raise SystemExit("nothing rendered yet")
    prev = json.loads((OUT_ROOT / "results.json").read_text()) if (OUT_ROOT / "results.json").exists() else {"rows": []}
    cache = {(r["pair"], r["condition"], r["seed"]): r for r in prev.get("rows", [])}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origin, u1, u2, ref_both = _cloud_axes()
    # on CPU DINOv2 routes through an xformers kernel that only supports CUDA (known failure poe-mem-002)
    embedder = DinoEmbedder(device=dev)

    def _embed(p: Path) -> np.ndarray:
        im = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        return embedder.embed_decoded_batch(im)[0]

    scored = []
    for r in rows:
        key = (r["pair"], r["condition"], r["seed"])
        if key in cache and cache[key].get("png") == r["png"]:
            scored.append(cache[key]); continue
        qa, qb = PAIRS[r["pair"]][3], PAIRS[r["pair"]][4]
        inst = instance_score_to_dict(score_output_instances(Path(r["png"]), qa, qb, device=dev))
        row = dict(r)
        row.update({"compose": inst["label"] == "compose", "n_instances": inst["n_instances"],
                    "conf_a": inst["conf_a"], "conf_b": inst["conf_b"],
                    "presence_a": inst["conf_a"] >= BUTTERFLY_PRESENT_CONF,
                    "sharpness": _laplacian_var(Path(r["png"]))})
        if r["pair"] == MAIN_PAIR:
            f = _embed(Path(r["png"]))
            row["both_ness"] = float((f - origin) @ u2)
            row["which_animal"] = float((f - origin) @ u1)
        scored.append(row)
        _log(f"scored {key}: compose={row['compose']} n={row['n_instances']} sharp={row['sharpness']:.1f}"
             + (f" both={row['both_ness']:.3f}" if "both_ness" in row else f" conf_a={row['conf_a']:.2f}"))

    summary = {}
    for slug in PAIRS:
        conds = sorted({r["condition"] for r in scored if r["pair"] == slug})
        summary[slug] = {}
        for c in conds:
            rs = [r for r in scored if r["pair"] == slug and r["condition"] == c]
            sh = [r["sharpness"] for r in rs]
            cell = {"n": len(rs), "num_steps": rs[0]["num_steps"], "description": rs[0]["description"],
                    "compose_n": sum(r["compose"] for r in rs), "compose_rate": sum(r["compose"] for r in rs) / len(rs),
                    "presence_n": sum(r["presence_a"] for r in rs),
                    "sharpness_mean": float(np.mean(sh)), "sharpness_min": float(min(sh)), "sharpness_max": float(max(sh)),
                    "sharpness_per_seed": {str(r["seed"]): r["sharpness"] for r in rs},
                    "compose_per_seed": {str(r["seed"]): r["compose"] for r in rs}}
            if slug == MAIN_PAIR:
                cell["both_ness_mean"] = float(np.mean([r["both_ness"] for r in rs]))
                cell["both_ness_per_seed"] = {str(r["seed"]): r["both_ness"] for r in rs}
            summary[slug][c] = cell
        # verdicts
        s = summary[slug]
        if slug == MAIN_PAIR and "poe" in s and "full_1.2" in s:
            premise = s["full_1.2"]["sharpness_mean"] < s["poe"]["sharpness_min"]
            s["premise_full_window_softer_than_plain_band"] = premise
            for c in conds:
                if c in ("mono", "poe", "full_1.2"):
                    continue
                poe_ref = s[f"poe_{STEPS_TAIL}"] if c.endswith(f"_{STEPS_TAIL}") and f"poe_{STEPS_TAIL}" in s else s["poe"]
                s[c]["verdict"] = verdict_for_cell(s[c], s["full_1.2"], poe_ref, premise)
                s[c]["plain_band_used"] = [poe_ref["sharpness_min"], poe_ref["sharpness_max"]]
                # post-hoc, beside the bar and never replacing it: the per-seed paired read
                if s[c]["num_steps"] == STEPS_MAIN:
                    sp = s[c]["sharpness_per_seed"]
                    s[c]["sharper_than_full_n"] = sum(sp[k] > s["full_1.2"]["sharpness_per_seed"][k] for k in sp)
                    s[c]["sharper_than_poe_n"] = sum(sp[k] > s["poe"]["sharpness_per_seed"][k] for k in sp)
                    s[c]["kept_full_window_seeds_n"] = sum(
                        s[c]["compose_per_seed"][k] for k in sp if s["full_1.2"]["compose_per_seed"][k])
        elif slug == CONTROL_PAIR and "poe" in s:
            for c in conds:
                if c in ("mono", "poe"):
                    continue
                poe_ref = s[f"poe_{STEPS_TAIL}"] if c.endswith(f"_{STEPS_TAIL}") and f"poe_{STEPS_TAIL}" in s else s["poe"]
                s[c]["verdict"] = control_verdict(s[c], poe_ref)

    results = {
        "pairs": {k: {"prompt_a": v[0], "prompt_b": v[1], "joint_prompt": v[2]} for k, v in PAIRS.items()},
        "seeds": list(SEEDS), "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK,
        "scorer": "instance count: GroundingDINO query 'animal', NMS iou<0.5, conf>=0.30, compose iff count>=2 (context/world/compose-rate.md); on the control pair 'presence_a' is conf('butterfly') >= BUTTERFLY_PRESENT_CONF and the compose rule does not apply",
        "sharpness": "Laplacian variance on the 1024 px greyscale render, lambda_window_grid._laplacian_var",
        "both_ness": "projection onto the landing finding's cloud axis (solo midpoint toward the joint-prompt centroid), DINOv2 ViT-S/14 CLS; axes rebuilt from " + str(CLOUD_FEATS.relative_to(REPO_ROOT)),
        "reference_both_ness_from_landing_finding": {k: float(np.mean(v)) for k, v in ref_both.items()},
        "constants": {"LAMBDA_FULL": LAMBDA_FULL, "T_CUT": T_CUT, "T_DECAY_END": T_DECAY_END, "T_RENOISE": T_RENOISE,
                      "RENOISE_FROM_STEP": RENOISE_FROM_STEP, "RENOISE_TAIL_START_STEP": RENOISE_TAIL_START_STEP,
                      "STEPS_MAIN": STEPS_MAIN, "STEPS_TAIL": STEPS_TAIL,
                      "SUPPORT_MAX_COMPOSE_LOSS": SUPPORT_MAX_COMPOSE_LOSS, "NULL_MIN_COMPOSE_LOSS": NULL_MIN_COMPOSE_LOSS,
                      "BUTTERFLY_PRESENT_CONF": BUTTERFLY_PRESENT_CONF, "CONTROL_MAX_PRESENCE_LOSS": CONTROL_MAX_PRESENCE_LOSS},
        "rule": "support if compose_n >= full_window compose_n - SUPPORT_MAX_COMPOSE_LOSS and sharpness_mean >= plain-PoE sharpness_min; null if compose loss >= NULL_MIN_COMPOSE_LOSS or sharpness_mean <= full_window sharpness_mean; inconclusive otherwise; every cell inconclusive if the full window is not softer than the plain band",
        "summary": summary, "rows": scored,
    }
    if (OUT_ROOT / "detach_check.json").exists():
        results["detach_check"] = json.loads((OUT_ROOT / "detach_check.json").read_text())
    # the sweep's x0s20 cell at T_RENOISE repeats the plan's re-noise cell with the same noise draw;
    # the two renders should agree to fp16 drift, and the difference is recorded
    dup = {}
    for slug in PAIRS:
        for sched in SCHEDULE_CANDIDATES:
            a, b = f"{sched}_renoise", f"{sched}_renoise_x0s20_t{T_RENOISE}"
            for seed in SEEDS:
                pa, pb = _render_path(slug, a, seed), _render_path(slug, b, seed)
                if pa.exists() and pb.exists():
                    dup[f"{slug}/{sched}/seed_{seed}"] = mean_abs_diff(pa, pb)
    if dup:
        results["renoise_duplicate_check"] = {"mean_abs_diff_grey_levels": dup, "max": max(dup.values()),
                                             "bar": DETACH_MAX_MEAN_ABS_DIFF}
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    _write_table(results)
    sheets = _sheets(results)
    _wandb(results, sheets)
    return results


def _write_table(results: dict) -> None:
    lines = ["# Cell table", "",
             "One row per cell. compose is seeds of 8 with two or more animal instances (cat x dog) or with a butterfly box at conf >= "
             f"{BUTTERFLY_PRESENT_CONF} (control pair); sharpness is Laplacian variance at 1024 px, mean over 8 seeds, with the plain-PoE band "
             "(min to max of the 8 plain seeds at the same step count); both-ness is the mean projection toward the joint-prompt cloud (cat x dog only). "
             "From `results.json` beside this file.", ""]
    for slug, s in results["summary"].items():
        lines += [f"## {slug}", "", "| cell | steps | compose or presence (of 8) | mean sharpness | plain band | both-ness mean | verdict |", "|---|---|---|---|---|---|---|"]
        for c, cell in s.items():
            if not isinstance(cell, dict):
                continue
            n = cell["compose_n"] if slug == MAIN_PAIR else cell["presence_n"]
            band = cell.get("plain_band_used")
            band_s = f"{band[0]:.1f} to {band[1]:.1f}" if band else ""
            lines.append(f"| {c} | {cell['num_steps']} | {n} | {cell['sharpness_mean']:.1f} | {band_s} | "
                         f"{cell.get('both_ness_mean', float('nan')):.3f} | {cell.get('verdict', '')} |")
        if "premise_full_window_softer_than_plain_band" in s:
            lines.append(f"\npremise (full window softer than the plain band): {s['premise_full_window_softer_than_plain_band']}")
        lines.append("")
    (RESULTS_DIR / "cell-table.md").write_text("\n".join(lines))


def _sheets(results: dict) -> list[Path]:
    T, pad, cap, left, top = 256, 8, 34, 90, 60
    sheets = []
    rows_by = {(r["pair"], r["condition"], r["seed"]): r for r in results["rows"]}
    out_dir = OUT_ROOT / "sheets"; out_dir.mkdir(exist_ok=True)
    for slug, s in results["summary"].items():
        for c in s:
            if c in ("mono", "poe", "full_1.2") or not isinstance(s[c], dict):
                continue
            cols = [("mono", "Mono"), ("poe", "plain PoE"), (c, c), ("full_1.2", "full window lambda 1.2")]
            W = left + 4 * (T + pad) + pad
            H = top + len(SEEDS) * (T + cap + pad) + pad
            im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
            dr.text((8, 8), f"{slug}   {c}: {s[c]['description']}   steps {s[c]['num_steps']}   verdict: {s[c].get('verdict', '')}", fill="black")
            for j, (_, lab) in enumerate(cols):
                dr.text((left + pad + j * (T + pad), top - 18), lab, fill="black")
            for i, seed in enumerate(SEEDS):
                y0 = top + pad + i * (T + cap + pad)
                dr.text((8, y0 + T // 2), f"seed {seed}", fill="black")
                for j, (cc, _) in enumerate(cols):
                    r = rows_by.get((slug, cc, seed))
                    x0 = left + pad + j * (T + pad)
                    if r is None:
                        dr.rectangle([x0, y0, x0 + T, y0 + T], outline="#999"); continue
                    im.paste(Image.open(r["png"]).convert("RGB").resize((T, T), Image.LANCZOS), (x0, y0))
                    tag = f"n={r['n_instances']}" if slug == MAIN_PAIR else f"butterfly conf {r['conf_a']:.2f}"
                    both = f"  both {r['both_ness']:.2f}" if "both_ness" in r else ""
                    dr.text((x0, y0 + T + 2), f"{tag}  sharp {r['sharpness']:.0f}{both}", fill="black")
            p = out_dir / f"sheet-{slug}-{c}.png"
            im.save(p); sheets.append(p)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            im.save(RESULTS_DIR / p.name)
    _log(f"wrote {len(sheets)} sheets")
    return sheets


def strip_columns(results: dict) -> list[tuple[str, str]]:
    """Mono, plain PoE, the full window, the best schedule, the best re-noise cell (the last two only
    once scored)."""
    cols = [("mono", "Mono: the joint prompt"), ("poe", "plain PoE"), ("full_1.2", f"PoE + lambda {LAMBDA_FULL}, all steps")]
    s = results["summary"].get(MAIN_PAIR, {})
    if all(c in s for c in SCHEDULE_CANDIDATES):
        best = pick_best_schedule(results)
        cols.append((best, f"correct early: {best}"))
    best_re = pick_best_renoise(results)
    if best_re:
        cols.append((best_re, f"re-noised: {best_re.split('_renoise_', 1)[-1] or 'plan cell'}"))
    return cols


def _strips(results: dict) -> dict:
    """One strip per (pair, seed): the columns of strip_columns, the scorer's numbers under each tile."""
    T, pad, cap, top = STRIP_TILE, 10, 46, 30
    rows_by = {(r["pair"], r["condition"], r["seed"]): r for r in results["rows"]}
    cols = strip_columns(results)
    out_dir = OUT_ROOT / "strips"; out_dir.mkdir(exist_ok=True)
    (RESULTS_DIR / "strips").mkdir(parents=True, exist_ok=True)
    paths = {}
    for slug in results["summary"]:
        for seed in SEEDS:
            W = pad + len(cols) * (T + pad)
            H = top + T + cap + pad
            im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
            dr.text((pad, 8), f"{slug}   seed {seed}   checkpoint step 30050, rank {LORA_RANK}", fill="black")
            for j, (cc, lab) in enumerate(cols):
                x0 = pad + j * (T + pad); y0 = top
                r = rows_by.get((slug, cc, seed))
                if r is None:
                    dr.rectangle([x0, y0, x0 + T, y0 + T], outline="#999")
                    dr.text((x0, y0 + T + 4), lab, fill="black"); dr.text((x0, y0 + T + 20), "not rendered", fill="#999")
                    continue
                im.paste(Image.open(r["png"]).convert("RGB").resize((T, T), Image.LANCZOS), (x0, y0))
                if slug == MAIN_PAIR:
                    nums = f"animals counted {r['n_instances']}   sharpness {r['sharpness']:.0f}   both-ness {r.get('both_ness', float('nan')):.2f}"
                else:
                    nums = f"butterfly conf {r['conf_a']:.2f}   sharpness {r['sharpness']:.0f}"
                dr.text((x0, y0 + T + 4), lab, fill="black")
                dr.text((x0, y0 + T + 20), nums, fill="black")
                if cc not in ("mono", "poe", "full_1.2"):
                    dr.text((x0, y0 + T + 33), (results["summary"][slug][cc].get("verdict", "") or "")[:60], fill="#444")
            p = out_dir / f"strip-{slug}-seed_{seed}.png"
            im.save(p); im.save(RESULTS_DIR / "strips" / p.name)
            paths[(slug, seed)] = p
    _log(f"wrote {len(paths)} strips with columns {[c for c, _ in cols]}")
    return paths


def _wandb_strips(results: dict, paths: dict) -> None:
    """Log the strips to the plan's W&B run: one image per (pair, seed) and one table whose row is a
    seed and whose columns are the tiles, so the Media tab compares Mono, PoE and the solution per seed."""
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    idf = OUT_ROOT / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else None
    run = wandb.init(entity=entity, project=project, name="correct_early_then_clean_up_r32_030050",
                     id=run_id, resume="allow", tags=["plan-14", "scope-01", "lambda-schedule"])
    idf.write_text(run.id)
    rows_by = {(r["pair"], r["condition"], r["seed"]): r for r in results["rows"]}
    cols = strip_columns(results)
    log = {}
    for (slug, seed), p in paths.items():
        log[f"strips/{slug}/seed_{seed}"] = wandb.Image(str(p), caption=" | ".join(lab for _, lab in cols))
    tab_cols = ["pair", "seed"] + [c for c, _ in cols] + [f"{c}_n" for c, _ in cols] + [f"{c}_sharp" for c, _ in cols]
    tab = wandb.Table(columns=tab_cols)
    for slug in results["summary"]:
        for seed in SEEDS:
            tiles, ns, shs = [], [], []
            for cc, lab in cols:
                r = rows_by.get((slug, cc, seed))
                if r is None:
                    tiles.append(None); ns.append(None); shs.append(None); continue
                im = Image.open(r["png"]).convert("RGB").resize((512, 512), Image.LANCZOS)
                tiles.append(wandb.Image(im, caption=f"{lab}: n={r['n_instances']} sharp={r['sharpness']:.0f}"))
                ns.append(r["n_instances"]); shs.append(round(r["sharpness"], 1))
            tab.add_data(slug, seed, *tiles, *ns, *shs)
    log["strips_by_seed"] = tab
    run.log(log)
    _log(f"wandb strips logged to run {run.id} url {run.url}")
    run.finish()


def _wandb(results: dict, sheets: list[Path]) -> None:
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    idf = OUT_ROOT / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else None
    run = wandb.init(entity=entity, project=project, name="correct_early_then_clean_up_r32_030050",
                     id=run_id, resume="allow", config=results["constants"],
                     tags=["plan-14", "scope-01", "lambda-schedule"])
    idf.write_text(run.id)
    log = {}
    for p in sheets:
        log[f"sheets/{p.stem}"] = wandb.Image(str(p))
    cols = ["pair", "cell", "steps", "compose_or_presence_n", "sharpness_mean", "band_lo", "band_hi", "both_ness_mean", "verdict"]
    tab = wandb.Table(columns=cols)
    for slug, s in results["summary"].items():
        for c, cell in s.items():
            if not isinstance(cell, dict):
                continue
            n = cell["compose_n"] if slug == MAIN_PAIR else cell["presence_n"]
            band = cell.get("plain_band_used") or [None, None]
            tab.add_data(slug, c, cell["num_steps"], n, cell["sharpness_mean"], band[0], band[1],
                         cell.get("both_ness_mean"), cell.get("verdict", ""))
    log["cells"] = tab
    for slug, s in results["summary"].items():
        for c, cell in s.items():
            if isinstance(cell, dict):
                log[f"{slug}/{c}/compose_n"] = cell["compose_n"]
                log[f"{slug}/{c}/sharpness_mean"] = cell["sharpness_mean"]
    run.log(log)
    art = wandb.Artifact("correct_early_then_clean_up_results", type="results")
    art.add_file(str(OUT_ROOT / "results.json"))
    if (OUT_ROOT / "detach_check.json").exists():
        art.add_file(str(OUT_ROOT / "detach_check.json"))
    run.log_artifact(art)
    _log(f"wandb run id {run.id} url {run.url}")
    results.setdefault("wandb", {})["run_id"] = run.id
    results["wandb"]["url"] = run.url
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    run.finish()


# ============================================================================= main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharpness-over-steps", action="store_true")
    ap.add_argument("--detach-check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--stage", default="schedule", choices=["schedule", "tail", "renoise", "renoise-sweep"])
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--strips", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--pairs", default=",".join(PAIRS))
    a = ap.parse_args(argv)
    if not any([a.sharpness_over_steps, a.detach_check, a.render, a.score, a.strips]):
        ap.error("pass a stage")
    if a.sharpness_over_steps:
        sharpness_over_steps()
    if a.detach_check:
        detach_check()
    if a.render:
        render(a.stage, seeds=[int(s) for s in a.seeds.split(",")], pairs=[p for p in a.pairs.split(",")])
    if a.score:
        score()
    if a.strips:
        results = json.loads((OUT_ROOT / "results.json").read_text())
        _wandb_strips(results, _strips(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
