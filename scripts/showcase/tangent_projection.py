#!/usr/bin/env python
"""Does keeping the rank-32 correction tangent to the base model's manifold restore fidelity?

Plan 15 of ``plans/01-showcase-the-trained-lora``. The instrument is the one Saito and
Matsubara (arXiv 2510.05509) build for interpolation: the Jacobian of the score is
rank-deficient, its null directions are the tangent space of the data manifold at ``x_t``,
so a direction ``v`` with small ``||J v||`` moves along the manifold and one with large
``||J v||`` pushes off it. Here the direction is a *correction* added to the
product-of-experts prediction, and the manifold is the frozen base model's own, read
through its unconditional denoiser.

In the epsilon parametrisation the split is one finite-difference forward pass:

    J_eps v      ~=  (eps_0(x_t + h v_hat) - eps_0(x_t)) / h * ||v||      one extra fp32 forward
    normal(v)     =  sigma_t * J_eps v                                    (I - P) v in the ideal case
    tangent(v)    =  v - normal(v)                                         P v
    normal share  =  ||normal(v)|| / ||v||                                 sin of the angle to the tangent space

because for data on a manifold with tangent projector P the score is -(I - P)(x - Px)/sigma^2,
so sigma * d eps / dx = I - P. This is the paper's Proposition 1 written on the denoiser
instead of on the score, and the same object is both the read (how normal is each
correction) and the fix (drop the normal part before adding the correction).

Stages, each one process (the render stage attaches the adapter; the cache read attaches it
too, so neither shares a process with the plain references, which are already on disk):

    --cache-read   cat x dog, seeds 9 to 16, all 50 cached steps. The normal share of the true
                   correction r_t, the adapter's correction, a random direction, the PoE
                   prediction and the adapter's error, under the unconditional denoiser's
                   Jacobian (fp32, finite difference at FD_REL_STEP and a linearity check at
                   FD_REL_STEP_CHECK) and under the joint-prompt denoiser's; the fit of the
                   adapter to r_t before and after both are projected. Writes cache_read.json.
    --h-sweep      the instrument check: one seed at the frame steps, the normal share against six
                   finite-difference steps, and after one and two projections. Writes h_sweep.json.
    --render       both pairs, seeds 9 to 16: the adapter alone at lambda 1.2 on all 50 steps,
                   the same with the correction projected on every step, and projected from
                   step PROJECT_LATE_FROM on. Tweedie frames at FRAME_STEPS. One identity check
                   of the adapter-alone render against the render the corrector tail stage made.
    --score        compose count (validated instance count on cat x dog), DINOv2 distance to the
                   seed's joint-prompt render, Laplacian sharpness, both-ness on the fitted
                   cat x dog cloud axes, the frame tracks, and the verdict by the constants
                   below. Writes results.json.
    --figures      the graphs, the manifold pictures and the sheets, each with a .json sidecar.
    --wandb        log curves, tables, images and sidecars to W&B; print the run id.
    --smoke        with --cache-read or --render: one seed, three cached steps or one three-step
                   render under OUT/smoke/, proving the process runs on the pinned device
                   before the long stages start.

The bars sit here as constants, written before any number existed (2026-09-06).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))
sys.path.insert(0, str(REPO / "scripts"))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._poe_langevin import _adapter_disable, _adapter_enable  # noqa: E402
from poe_repair.methods._sampling import add_time_ids, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import (  # noqa: E402
    ddim_prev_from_x0_eps, decode_latents, guided_eps, poe_eps, tweedie_mean,
)

import correction_span_common as C  # noqa: E402  (the cache reader and the guided rule)

# ---------------------------------------------------------------------------
# Pre-registered constants (the review file names every one of them)
# ---------------------------------------------------------------------------

FAILING_PAIR = "a_cat__x__a_dog"
COMPOSING_PAIR = "a_butterfly__x__a_flower_meadow"
PAIRS = (FAILING_PAIR, COMPOSING_PAIR)
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
CONCEPT_QUERIES = {FAILING_PAIR: ("cat", "dog"), COMPOSING_PAIR: ("butterfly", "flowers")}
CONCEPT_CONF = 0.30

ADAPTER_CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
ADAPTER_RANK = 32
LAMBDA = 1.2                      # the shipped setting
GUIDANCE = 7.5

# The Jacobian probe. The perturbation has norm FD_REL_STEP * ||x_t||; the probe runs the
# frozen UNet in float32 with TF32 off, because a float16 difference of two forwards at
# this step size is the size of the rounding noise.
FD_REL_STEP = 0.01
FD_REL_STEP_CHECK = 0.02          # the same read at twice the step; agreement says the difference is linear
FD_MAX_LINEARITY_GAP = 0.15       # |share(h2) - share(h1)| / share(h1) above this on the mean marks the probe unreliable
EARLY_STEPS = range(0, 11)        # the window the rung-1 statistic averages over, as in plan 05/tests/07

# Question 1 (the read): is the adapter's correction more normal than the true one?
# statistic = mean over seeds of the per-seed mean over EARLY_STEPS of share(adapter) / share(true)
NORMAL_RATIO_SUPPORT = 1.25       # the adapter pushes off the manifold at least 25% harder per unit norm
NORMAL_RATIO_NULL = 1.05          # within 5%: the adapter's softness is not an off-manifold push

# Question 2 (the fix): does dropping the normal part bring the render nearer the joint image?
PROJECT_LATE_FROM = 10            # the second condition projects only from this step, leaving the early decision alone
MIN_MONO_GAIN = 0.05              # same value as CLEAN_MIN_MONO_GAIN in scripts/corrector_window_sweep.py
MAX_COMPOSE_LOSS_SEEDS = 1        # composed seeds may fall by at most one against the adapter alone
IDENTITY_MAX_MEAN_ABS_DIFF = 6.0  # grey levels; the adapter-alone render here against the corrector tail's k=0 render (cross-device fp16 drift ~2)

FRAME_STEPS = (0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49)
H_SWEEP_RELS = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)   # the step-size sweep, one seed, at FRAME_STEPS
H_SWEEP_SEED = 9
ARROW_STEPS = (0, 2, 5, 10, 20, 30, 49)
NUM_STEPS = 50
RANDOM_SEED = 20260906

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/tangent_projection")
CORRECTOR_OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector")
CACHE_JSON = OUT / "cache_read.json"
H_SWEEP_JSON = OUT / "h_sweep.json"
RENDERS_JSON = OUT / "renders.json"
RESULTS_JSON = OUT / "results.json"
RESULTS_DIR = REPO / "artifacts/results/does-keeping-the-correction-on-the-manifold-restore-fidelity"
LANDING_DIR = REPO / "artifacts/results/where-does-each-condition-land"
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"
CONDITIONS = ("adapter", "projected", "projected_late")
FOLLOW_ON_CONDITIONS = ("projected_ls",)          # raised after the step-size sweep; rendered by --conditions
COND_SPEC = {"adapter": (None, "subtract"), "projected": (0, "subtract"), "projected_late": (PROJECT_LATE_FROM, "subtract"),
             "projected_ls": (0, "linesearch")}   # (project from step, how)
COND_TITLE = {"adapter": f"adapter λ{LAMBDA}\nall 50 steps",
              "projected": f"adapter λ{LAMBDA}\nnormal part subtracted\nevery step",
              "projected_late": f"adapter λ{LAMBDA}\nnormal part subtracted\nfrom step {PROJECT_LATE_FROM}",
              "projected_ls": f"adapter λ{LAMBDA}\nnormal energy minimised\n(line search) every step"}
GREEN_FRAME_RULE = "compose == 1, i.e. the validated instance count (GroundingDINO 'animal', conf >= 0.30, NMS iou < 0.5) >= 2"


def _host() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    if 1.0 - st.f_bavail / max(st.f_blocks, 1) >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at or over 90%, refusing to write")


def _slim_vae(ctx) -> None:
    """Tiled, sliced decoding: a 1024 px decode then peaks at about 2 GB instead of 6, which is
    what lets the fp16 UNet, the fp32 probe and the decoder share a 24 GB card."""
    vae = ctx.models["vae"]
    for fn in ("enable_tiling", "enable_slicing"):
        if hasattr(vae, fn):
            getattr(vae, fn)()


def _offload_text_encoders(ctx, to: str) -> None:
    for k_ in ("text_encoder", "text_encoder_2"):
        ctx.models[k_].to(to)


def _norm(x: torch.Tensor) -> float:
    return float(x.float().reshape(-1).norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1); b = b.float().reshape(-1).to(a.device)
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------------------
# The probe: the frozen base model's denoiser in float32, its Jacobian by finite difference
# ---------------------------------------------------------------------------


class Probe:
    """``eps(x, t)`` for the unconditional and the joint prompt on a float32 copy of the frozen
    UNet, and ``jv`` = J_eps v by one extra forward. No adapter is ever attached to this copy."""

    def __init__(self, model_id: str, device: torch.device, height: int, width: int):
        from diffusers import UNet2DConditionModel
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32, use_safetensors=True,
        ).to(device).eval()
        self.time_ids = add_time_ids(height=height, width=width, batch_size=1, device=device, dtype=torch.float32)
        self.conds: dict[str, tuple[torch.Tensor, dict]] = {}
        self.n_forwards = 0

    def set_prompts(self, *, seq_e, pool_e, seq_j, pool_j) -> None:
        self.conds = {
            "uncond": (seq_e.to(self.device, torch.float32), {"text_embeds": pool_e.to(self.device, torch.float32), "time_ids": self.time_ids}),
            "joint": (seq_j.to(self.device, torch.float32), {"text_embeds": pool_j.to(self.device, torch.float32), "time_ids": self.time_ids}),
        }

    @torch.no_grad()
    def eps(self, x32: torch.Tensor, t_int: int, which: str = "uncond") -> torch.Tensor:
        seq, cond = self.conds[which]
        tb = torch.full((1,), int(t_int), device=self.device, dtype=torch.long)
        self.n_forwards += 1
        return self.unet(x32.to(self.device, torch.float32), tb, encoder_hidden_states=seq, added_cond_kwargs=cond).sample

    @torch.no_grad()
    def jv(self, x32: torch.Tensor, eps0: torch.Tensor, t_int: int, v: torch.Tensor,
           rel: float = FD_REL_STEP, which: str = "uncond") -> torch.Tensor:
        """J_eps v by a forward difference with a step of norm rel * ||x||."""
        v32 = v.to(self.device, torch.float32)
        vn = v32.reshape(-1).norm()
        h = float(rel) * x32.reshape(-1).norm()
        e1 = self.eps(x32 + h * v32 / (vn + 1e-12), t_int, which)
        return (e1 - eps0) * (vn / h)


def split(v: torch.Tensor, sigma_t: float, jv: torch.Tensor) -> dict:
    """The normal part sigma * J v, the tangent part v minus it, and the normal share."""
    v32 = v.float(); normal = float(sigma_t) * jv.float().to(v32.device)
    tangent = v32 - normal
    return {"normal": normal, "tangent": tangent,
            "normal_share": _norm(normal) / max(_norm(v32), 1e-12),
            "tangent_share": _norm(tangent) / max(_norm(v32), 1e-12)}


def project_linesearch(v: torch.Tensor, sigma_t: float, x32: torch.Tensor, eps0: torch.Tensor, t_int: int, probe: "Probe",
                       rel: float = FD_REL_STEP) -> dict:
    """The paper's Proposition 1 as a one-dimensional minimisation: with n = sigma J v the normal
    part, move along n by the alpha that minimises ||J (v - alpha n)||, alpha = <Jv, Jn> / ||Jn||^2.
    Exact on any eigen-direction of sigma J whatever its eigenvalue (alpha = 1/lambda), so it never
    overshoots where the plain subtraction does. Two probe forwards."""
    jv = probe.jv(x32, eps0, t_int, v, rel=rel).float()
    n = float(sigma_t) * jv.to(v.device)
    jn = probe.jv(x32, eps0, t_int, n, rel=rel).float().to(v.device)
    a = jv.reshape(-1).to(v.device); b = jn.reshape(-1)
    alpha = float(a @ b / (b @ b + 1e-12))
    out = v.float() - alpha * n
    return {"tangent": out, "alpha": alpha, "normal_share_before": _norm(n) / max(_norm(v), 1e-12),
            "normal_energy_before": _norm(jv), "normal_energy_after": _norm(a - alpha * b),
            "norm_after_over_before": _norm(out) / max(_norm(v), 1e-12)}


def sigma_of(scheduler, t_int: int) -> float:
    return math.sqrt(max(1.0 - float(scheduler.alphas_cumprod[int(t_int)]), 1e-12))


# ---------------------------------------------------------------------------
# The adapter at a state
# ---------------------------------------------------------------------------


def attach_adapter(ctx) -> dict:
    import lambda_boundary_probe as lbp
    lbp.LORA_RANK = lbp.LORA_ALPHA = ADAPTER_RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], ADAPTER_CHECKPOINT)
    ctx.models["unet"].eval()
    info["adapter_name"] = lbp.LORA_ADAPTER_NAME
    return info


class ThreeBranch:
    """The guided PoE prediction with the adapter off (frozen) and on, at one state."""

    def __init__(self, ctx, emb: dict, height: int, width: int, adapter_name: str):
        self.unet = ctx.models["unet"]; self.name = adapter_name
        self.scheduler = ctx.scheduler; self.dtype = ctx.dtype; self.device = ctx.device
        self.pe3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0).to(ctx.device, ctx.dtype)
        pool3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0).to(ctx.device, ctx.dtype)
        self.cond3 = {"text_embeds": pool3,
                      "time_ids": add_time_ids(height=height, width=width, batch_size=3, device=ctx.device, dtype=ctx.dtype)}

    @torch.no_grad()
    def _poe(self, x16: torch.Tensor, timestep) -> torch.Tensor:
        latent_input = self.scheduler.scale_model_input(x16.repeat(3, 1, 1, 1), timestep)
        noise = self.unet(latent_input, timestep, encoder_hidden_states=self.pe3,
                          added_cond_kwargs=self.cond3, timestep_cond=None).sample
        ea, eb, eu = noise.chunk(3)
        return poe_eps(guided_eps(ea, eu, GUIDANCE), guided_eps(eb, eu, GUIDANCE), eu)

    def frozen_and_adapter(self, x16: torch.Tensor, timestep) -> tuple[torch.Tensor, torch.Tensor]:
        _adapter_disable(self.unet)
        eps_frozen = self._poe(x16, timestep)
        _adapter_enable(self.unet, self.name)
        eps_lora = self._poe(x16, timestep)
        _adapter_disable(self.unet)
        return eps_frozen, eps_lora


# ---------------------------------------------------------------------------
# Stage: the cache read
# ---------------------------------------------------------------------------


def cache_read(*, smoke: bool = False) -> int:
    _disk_guard(OUT)
    ctx = make_ctx()
    info = attach_adapter(ctx)
    cell = cell_from_slug(FAILING_PAIR, SEEDS[0])
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    from poe_repair.config import DEFAULT_MODEL_ID
    probe = Probe(DEFAULT_MODEL_ID, ctx.device, cell.height, cell.width)
    probe.set_prompts(seq_e=emb["seq_e"], pool_e=emb["pool_e"], seq_j=seq_j, pool_j=pool_j)
    tb = ThreeBranch(ctx, emb, cell.height, cell.width, info["adapter_name"])
    for k_ in ("text_encoder", "text_encoder_2", "vae"):
        ctx.models[k_].to("cpu")
    torch.cuda.empty_cache()
    print(f"cache read: adapter rank {ADAPTER_RANK} attached, n_matched={info['n_matched']} step={info['checkpoint_step']}; "
          f"probe fp32 UNet loaded; host={_host()}", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    seeds = SEEDS[:1] if smoke else SEEDS
    rows = []
    t0 = time.time()
    for seed in seeds:
        n = 3 if smoke else C.num_steps(seed)
        for k in range(n):
            s = C.load_step(seed, k)
            t_int = s.timestep; sigma = sigma_of(ctx.scheduler, t_int)
            x16 = s.x_t.to(ctx.device, ctx.dtype)
            ts = torch.tensor(t_int, device=ctx.device)
            eps_frozen, eps_lora = tb.frozen_and_adapter(x16, ts)
            r_hat = (eps_lora - eps_frozen).float()                       # the adapter's correction at lambda 1
            r_t = s.r_t.to(ctx.device)                                     # the true correction at the same state
            z = torch.randn(r_t.shape, generator=gen).to(ctx.device) * (_norm(r_t) / math.sqrt(r_t.numel()))
            err = r_hat - r_t
            x32 = s.x_t.to(ctx.device, torch.float32)
            e0 = probe.eps(x32, t_int, "uncond")
            J = {name: probe.jv(x32, e0, t_int, v) for name, v in
                 (("true", r_t), ("adapter", r_hat), ("random", z), ("poe", s.eps_poe.to(ctx.device)), ("error", err))}
            sp = {name: split(v, sigma, J[name]) for name, v in
                  (("true", r_t), ("adapter", r_hat), ("random", z), ("poe", s.eps_poe.to(ctx.device)), ("error", err))}
            J2 = {name: probe.jv(x32, e0, t_int, v, rel=FD_REL_STEP_CHECK) for name, v in (("true", r_t), ("adapter", r_hat))}
            sp2 = {name: split(v, sigma, J2[name]) for name, v in (("true", r_t), ("adapter", r_hat))}
            ej = probe.eps(x32, t_int, "joint")
            Jj = {name: probe.jv(x32, ej, t_int, v, which="joint") for name, v in (("true", r_t), ("adapter", r_hat))}
            spj = {name: split(v, sigma, Jj[name]) for name, v in (("true", r_t), ("adapter", r_hat))}
            row = {"seed": seed, "step": k, "timestep": t_int, "sigma_t": sigma,
                   "norm_true": _norm(r_t), "norm_adapter": _norm(r_hat), "norm_poe": _norm(s.eps_poe),
                   "normal_share_true": sp["true"]["normal_share"], "normal_share_adapter": sp["adapter"]["normal_share"],
                   "normal_share_random": sp["random"]["normal_share"], "normal_share_poe": sp["poe"]["normal_share"],
                   "normal_share_error": sp["error"]["normal_share"],
                   "normal_share_true_h2": sp2["true"]["normal_share"], "normal_share_adapter_h2": sp2["adapter"]["normal_share"],
                   "normal_share_true_joint": spj["true"]["normal_share"], "normal_share_adapter_joint": spj["adapter"]["normal_share"],
                   "tangent_norm_true": _norm(sp["true"]["tangent"]), "normal_norm_true": _norm(sp["true"]["normal"]),
                   "tangent_norm_adapter": _norm(sp["adapter"]["tangent"]), "normal_norm_adapter": _norm(sp["adapter"]["normal"]),
                   "cos_adapter_true": _cos(r_hat, r_t),
                   "cos_adapter_tan_true_tan": _cos(sp["adapter"]["tangent"], sp["true"]["tangent"]),
                   "cos_adapter_tan_true": _cos(sp["adapter"]["tangent"], r_t),
                   "cos_adapter_normal_true_normal": _cos(sp["adapter"]["normal"], sp["true"]["normal"]),
                   "cos_random_true": _cos(z, r_t),
                   "sanity_cos_live_frozen_vs_cached_poe": _cos(eps_frozen, s.eps_poe)}
            if k in FRAME_STEPS:
                # idempotence: how normal is the projected correction, read the same way once more
                Jt = probe.jv(x32, e0, t_int, sp["adapter"]["tangent"])
                row["normal_share_adapter_after_projection"] = split(sp["adapter"]["tangent"], sigma, Jt)["normal_share"]
            rows.append(row)
            print(f"[{time.strftime('%H:%M:%S')}] seed {seed} step {k:02d} t={t_int}: share true {row['normal_share_true']:.3f} "
                  f"adapter {row['normal_share_adapter']:.3f} random {row['normal_share_random']:.3f} poe {row['normal_share_poe']:.3f} "
                  f"cos(adapter,true) {row['cos_adapter_true']:.3f} -> tan {row['cos_adapter_tan_true_tan']:.3f} "
                  f"({probe.n_forwards} probe forwards, {time.time() - t0:.0f}s)", flush=True)
            del J, sp, J2, sp2, Jj, spj
        gc.collect(); torch.cuda.empty_cache()
    result = cache_verdict(rows)
    result.update({"checkpoint": str(ADAPTER_CHECKPOINT), "adapter": info, "seeds": list(seeds), "smoke": smoke,
                   "probe": {"model": "frozen SDXL base UNet, float32, TF32 off, unconditional prompt (primary) and joint prompt (secondary)",
                             "fd_rel_step": FD_REL_STEP, "fd_rel_step_check": FD_REL_STEP_CHECK,
                             "rule": "normal(v) = sigma_t * (eps(x + h v_hat) - eps(x)) / h * ||v||; tangent(v) = v - normal(v); share = ||normal|| / ||v||"},
                   "directions": {"true": "r_t = guided joint minus guided PoE at the cached state (scripts/showcase/correction_span_common.py)",
                                  "adapter": "PoE prediction with the rank-32 adapter on minus with it off, same state, lambda 1",
                                  "random": "isotropic Gaussian scaled to ||r_t||", "poe": "the guided PoE prediction itself",
                                  "error": "adapter minus true"},
                   "rows": rows, "host": _host()})
    _write(CACHE_JSON if not smoke else OUT / "smoke" / "cache_read.json", result)
    print(json.dumps({k_: result[k_] for k_ in ("branch", "reasons", "statistic")}, indent=1), flush=True)
    return 0


def cache_verdict(rows: list[dict]) -> dict:
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        early = [r for r in rows if r["seed"] == seed and r["step"] in EARLY_STEPS]
        if not early:
            continue
        ratios = [r["normal_share_adapter"] / max(r["normal_share_true"], 1e-12) for r in early]
        per_seed[seed] = {"ratio_adapter_over_true_early": float(np.mean(ratios)),
                          "normal_share_true_early": float(np.mean([r["normal_share_true"] for r in early])),
                          "normal_share_adapter_early": float(np.mean([r["normal_share_adapter"] for r in early])),
                          "normal_share_random_early": float(np.mean([r["normal_share_random"] for r in early]))}
    ratio = float(np.mean([v["ratio_adapter_over_true_early"] for v in per_seed.values()])) if per_seed else float("nan")
    h1 = np.array([r["normal_share_adapter"] for r in rows]); h2 = np.array([r["normal_share_adapter_h2"] for r in rows])
    lin = float(np.mean(np.abs(h2 - h1) / np.maximum(h1, 1e-12))) if len(rows) else float("nan")
    linear = bool(lin <= FD_MAX_LINEARITY_GAP)
    stat = {"ratio_adapter_over_true_early_mean": ratio, "per_seed": per_seed,
            "linearity_gap_mean": lin, "linearity_ok": linear,
            "mean_over_all_steps": {k_: float(np.mean([r[k_] for r in rows])) for k_ in
                                    ("normal_share_true", "normal_share_adapter", "normal_share_random", "normal_share_poe", "normal_share_error",
                                     "cos_adapter_true", "cos_adapter_tan_true_tan", "cos_adapter_tan_true")} if rows else {}}
    if not per_seed:
        branch, reasons = "not ready", ["no early-step rows"]
    elif not linear:
        branch = "inconclusive"; reasons = [f"the finite-difference read is not linear: mean relative gap between steps {FD_REL_STEP} and {FD_REL_STEP_CHECK} is {lin:.2f}, bar {FD_MAX_LINEARITY_GAP}"]
    elif ratio >= NORMAL_RATIO_SUPPORT:
        branch = "support"; reasons = [f"adapter over true normal share, steps 0 to 10, mean of {len(per_seed)} seeds: {ratio:.3f} (bar {NORMAL_RATIO_SUPPORT})"]
    elif ratio <= NORMAL_RATIO_NULL:
        branch = "null"; reasons = [f"adapter over true normal share, steps 0 to 10: {ratio:.3f}, at or under {NORMAL_RATIO_NULL}"]
    else:
        branch = "inconclusive"; reasons = [f"ratio {ratio:.3f} between {NORMAL_RATIO_NULL} and {NORMAL_RATIO_SUPPORT}"]
    return {"branch": branch, "reasons": reasons, "statistic": stat,
            "thresholds": {"NORMAL_RATIO_SUPPORT": NORMAL_RATIO_SUPPORT, "NORMAL_RATIO_NULL": NORMAL_RATIO_NULL,
                           "FD_MAX_LINEARITY_GAP": FD_MAX_LINEARITY_GAP, "EARLY_STEPS": list(EARLY_STEPS)}}


def h_sweep() -> int:
    """The instrument check the smoke asked for: the normal share of the true correction, the
    adapter's and a random direction against the finite-difference step, one seed, at the frame
    steps, under both probes; plus the share after one and two projections at each step. Says
    where, if anywhere, the read is linear in h and how far from a projector the split is."""
    _disk_guard(OUT)
    ctx = make_ctx()
    info = attach_adapter(ctx)
    cell = cell_from_slug(FAILING_PAIR, H_SWEEP_SEED)
    emb = encode_pair(cell, ctx); seq_j, pool_j = get_joint_embeds(cell, ctx)
    from poe_repair.config import DEFAULT_MODEL_ID
    probe = Probe(DEFAULT_MODEL_ID, ctx.device, cell.height, cell.width)
    probe.set_prompts(seq_e=emb["seq_e"], pool_e=emb["pool_e"], seq_j=seq_j, pool_j=pool_j)
    tb = ThreeBranch(ctx, emb, cell.height, cell.width, info["adapter_name"])
    for k_ in ("text_encoder", "text_encoder_2", "vae"):
        ctx.models[k_].to("cpu")
    torch.cuda.empty_cache()
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    rows = []
    t0 = time.time()
    for k in FRAME_STEPS:
        s = C.load_step(H_SWEEP_SEED, k)
        t_int = s.timestep; sigma = sigma_of(ctx.scheduler, t_int)
        eps_frozen, eps_lora = tb.frozen_and_adapter(s.x_t.to(ctx.device, ctx.dtype), torch.tensor(t_int, device=ctx.device))
        r_hat = (eps_lora - eps_frozen).float(); r_t = s.r_t.to(ctx.device)
        z = torch.randn(r_t.shape, generator=gen).to(ctx.device) * (_norm(r_t) / math.sqrt(r_t.numel()))
        x32 = s.x_t.to(ctx.device, torch.float32)
        e0 = probe.eps(x32, t_int, "uncond"); ej = probe.eps(x32, t_int, "joint")
        for rel in H_SWEEP_RELS:
            row = {"seed": H_SWEEP_SEED, "step": k, "timestep": t_int, "sigma_t": sigma, "h_rel": rel, "h_abs": rel * float(x32.reshape(-1).norm())}
            for name, v in (("true", r_t), ("adapter", r_hat), ("random", z)):
                sp = split(v, sigma, probe.jv(x32, e0, t_int, v, rel=rel))
                row[f"normal_share_{name}"] = sp["normal_share"]
                if name != "random":
                    row[f"normal_share_{name}_joint"] = split(v, sigma, probe.jv(x32, ej, t_int, v, rel=rel, which="joint"))["normal_share"]
                    # one and two projections, read once more at the same h
                    p1 = sp["tangent"]; sp1 = split(p1, sigma, probe.jv(x32, e0, t_int, p1, rel=rel))
                    p2 = sp1["tangent"]; sp2 = split(p2, sigma, probe.jv(x32, e0, t_int, p2, rel=rel))
                    row[f"normal_share_{name}_after_1"] = sp1["normal_share"]; row[f"normal_share_{name}_after_2"] = sp2["normal_share"]
                    row[f"norm_{name}_after_1_over_before"] = _norm(p1) / max(_norm(v), 1e-12)
                    row[f"norm_{name}_after_2_over_before"] = _norm(p2) / max(_norm(v), 1e-12)
                    row[f"cos_{name}_after_2_vs_after_1"] = _cos(p2, p1)
                    ls = project_linesearch(v, sigma, x32, e0, t_int, probe, rel=rel)
                    q = ls["tangent"]; spq = split(q, sigma, probe.jv(x32, e0, t_int, q, rel=rel))
                    row[f"normal_share_{name}_after_ls"] = spq["normal_share"]; row[f"alpha_{name}_ls"] = ls["alpha"]
                    row[f"normal_energy_{name}_ls_after_over_before"] = ls["normal_energy_after"] / max(ls["normal_energy_before"], 1e-12)
                    row[f"norm_{name}_after_ls_over_before"] = ls["norm_after_over_before"]
            rows.append(row)
            print(f"[{time.strftime('%H:%M:%S')}] step {k:02d} h_rel {rel}: true {row['normal_share_true']:.3f} -> {row['normal_share_true_after_1']:.3f} -> {row['normal_share_true_after_2']:.3f}; "
                  f"adapter {row['normal_share_adapter']:.3f} -> {row['normal_share_adapter_after_1']:.3f} -> {row['normal_share_adapter_after_2']:.3f}; "
                  f"line search: true {row['normal_share_true_after_ls']:.3f} (α {row['alpha_true_ls']:.2f}, energy x{row['normal_energy_true_ls_after_over_before']:.2f}) "
                  f"adapter {row['normal_share_adapter_after_ls']:.3f} (α {row['alpha_adapter_ls']:.2f}, energy x{row['normal_energy_adapter_ls_after_over_before']:.2f}); "
                  f"random {row['normal_share_random']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    _write(H_SWEEP_JSON, {"seed": H_SWEEP_SEED, "steps": list(FRAME_STEPS), "h_rels": list(H_SWEEP_RELS), "rows": rows,
                          "what": "normal share against the finite-difference step, and after one and two projections at the same step; a flat line in h is the linear regime, and after_1 near 0 is what a projector would give",
                          "host": _host()})
    return 0


def fig_h_sweep() -> Path | None:
    plt = _mpl()
    if not H_SWEEP_JSON.exists():
        return None
    d = json.loads(H_SWEEP_JSON.read_text()); rows = d["rows"]
    steps = d["steps"]; rels = d["h_rels"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(steps):
        rr = sorted([r for r in rows if r["step"] == k], key=lambda r: r["h_rel"])
        col = cmap(i / max(len(steps) - 1, 1))
        axes[0].plot([r["h_rel"] for r in rr], [r["normal_share_true"] for r in rr], "-o", color=col, ms=3, label=f"step {k}")
        axes[1].plot([r["h_rel"] for r in rr], [r["normal_share_adapter"] for r in rr], "-o", color=col, ms=3)
        axes[2].plot([r["h_rel"] for r in rr], [r["normal_share_adapter_after_1"] for r in rr], "--s", color=col, ms=3, lw=1.0)
        if "normal_share_adapter_after_ls" in rr[0]:
            axes[2].plot([r["h_rel"] for r in rr], [r["normal_share_adapter_after_ls"] for r in rr], "-x", color=col, ms=5, lw=1.8)
    top = 1.05 * max(max(r[k_] for r in rows) for k_ in ("normal_share_true", "normal_share_adapter", "normal_share_adapter_after_1"))
    for ax, title in zip(axes, ("true correction r_t", "adapter's correction",
                                "adapter's after one subtraction (dashed)\nand after the line search (solid, x)")):
        ax.set_xscale("log"); ax.set_xlabel("finite-difference step, as a fraction of ‖x_t‖"); ax.set_ylim(0, top); ax.set_title(title, fontsize=9)
        ax.axvline(FD_REL_STEP, color="0.6", lw=0.8, ls="--")
        ax.axhline(1.0, color="0.75", lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0] * 1.05, 1.02, "1 = a random direction (early)", fontsize=6.5, color="0.4", va="bottom")
    axes[0].set_ylabel("normal share  ‖σ_t J v‖ / ‖v‖"); axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(f"The instrument against its step size: cat × dog seed {d['seed']}, frozen unconditional denoiser in fp32; dashed grey = the step the run used", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "normal-share-against-the-finite-difference-step.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170); plt.close(fig)
    _write(out.with_suffix(".json"), {"drawn_from": str(H_SWEEP_JSON), "x": "finite-difference step as a fraction of ||x_t||, log scale", "y": "normal share",
                                      "panels": ["true correction", "adapter's correction", "adapter's correction after one and two projections"],
                                      "rows": rows, "read": "a line flat in h is the linear regime; a share after one projection near zero is what a true projector gives"})
    return out


# ---------------------------------------------------------------------------
# Stage: the renders
# ---------------------------------------------------------------------------


def run_projected(*, ctx, probe: Probe, tb: ThreeBranch, init_latents, euler_sigma, height, width,
                  lam: float, project_from: int | None, frames_dir: Path | None,
                  mode: str = "subtract") -> tuple[torch.Tensor, list[dict]]:
    """The adapter run, op for op the corrected sampler's k=0 path, with one addition: from
    ``project_from`` on, the adapter's correction has its normal part (read on the frozen
    unconditional denoiser) removed before it is scaled by lambda and added."""
    scheduler = ctx.scheduler
    scheduler.set_timesteps(NUM_STEPS)
    latents = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
    per_step = []
    for k, timestep in enumerate(scheduler.timesteps):
        t_int = int(timestep.item())
        alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=ctx.device, dtype=ctx.dtype)
        sigma = sigma_of(scheduler, t_int)
        eps_frozen, eps_lora = tb.frozen_and_adapter(latents, timestep)
        delta = eps_lora - eps_frozen
        row = {"step": k, "timestep": t_int, "norm_delta": _norm(delta), "projected": False}
        if project_from is not None and k >= project_from:
            x32 = latents.float()
            e0 = probe.eps(x32, t_int, "uncond")
            if mode == "linesearch":
                ls = project_linesearch(delta, sigma, x32, e0, t_int, probe)
                used = ls["tangent"]
                row.update({"projected": True, "mode": mode, "normal_share_before": ls["normal_share_before"], "alpha": ls["alpha"],
                            "normal_energy_before": ls["normal_energy_before"], "normal_energy_after": ls["normal_energy_after"],
                            "norm_used_over_delta": ls["norm_after_over_before"]})
            else:
                sp = split(delta, sigma, probe.jv(x32, e0, t_int, delta))
                used = sp["tangent"]
                row.update({"projected": True, "mode": mode, "normal_share_before": sp["normal_share"],
                            "norm_used_over_delta": _norm(used) / max(_norm(delta), 1e-12)})
            eps_t = (eps_frozen.float() + float(lam) * used).to(ctx.dtype)
        else:
            eps_t = eps_frozen + float(lam) * delta
        per_step.append(row)
        x0 = tweedie_mean(latents, alpha_bar_t, eps_t)
        if frames_dir is not None and k in FRAME_STEPS:
            frames_dir.mkdir(parents=True, exist_ok=True)
            img = decode_latents(ctx.models, x0).cpu()
            from PIL import Image
            arr = (img[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            Image.fromarray(arr).resize((256, 256), Image.LANCZOS).save(frames_dir / f"step_{k:02d}.png")
        latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=k, x0=x0, eps=eps_t)
    _adapter_disable(tb.unet)
    return latents, per_step


def render(*, smoke: bool = False, conditions: tuple[str, ...] = CONDITIONS) -> int:
    global NUM_STEPS
    _disk_guard(OUT)
    root = OUT / "smoke" if smoke else OUT
    if smoke:
        NUM_STEPS = 3
    ctx = make_ctx()
    _slim_vae(ctx)
    from poe_repair.config import DEFAULT_MODEL_ID
    info = attach_adapter(ctx)
    print(f"render: adapter rank {ADAPTER_RANK} attached, n_matched={info['n_matched']} step={info['checkpoint_step']}; host={_host()}", flush=True)
    pairs = PAIRS[:1] if smoke else PAIRS
    seeds = SEEDS[:1] if smoke else SEEDS
    conds = ("projected",) if smoke else tuple(conditions)
    # every prompt encoded up front, then the text encoders leave the card for the whole run
    embs = {}
    for pair in pairs:
        cell = cell_from_slug(pair, seeds[0])
        emb = encode_pair(cell, ctx); seq_j, pool_j = get_joint_embeds(cell, ctx)
        embs[pair] = (emb, seq_j, pool_j, cell.height, cell.width)
    _offload_text_encoders(ctx, "cpu"); gc.collect(); torch.cuda.empty_cache()
    probe = None
    rows = []
    for pair in pairs:
        emb, seq_j, pool_j, height, width = embs[pair]
        if probe is None:
            probe = Probe(DEFAULT_MODEL_ID, ctx.device, height, width)
        probe.set_prompts(seq_e=emb["seq_e"], pool_e=emb["pool_e"], seq_j=seq_j, pool_j=pool_j)
        tb = ThreeBranch(ctx, emb, height, width, info["adapter_name"])
        for seed in seeds:
            cell = cell_from_slug(pair, seed)
            init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
            for cond in conds:
                d = root / "renders" / pair / cond
                d.mkdir(parents=True, exist_ok=True)
                png = d / f"seed_{seed}.png"
                frames = root / "frames" / pair / cond / f"seed_{seed}"
                t0 = time.time()
                if not png.exists():
                    latents, per_step = run_projected(
                        ctx=ctx, probe=probe, tb=tb, init_latents=init_latents, euler_sigma=euler_sigma,
                        height=cell.height, width=cell.width, lam=LAMBDA,
                        project_from=COND_SPEC[cond][0], mode=COND_SPEC[cond][1],
                        frames_dir=frames if pair == FAILING_PAIR else None)
                    write_decoded_image(decode_latents(ctx.models, latents).cpu(), png)
                    _write(d / f"seed_{seed}_per_step.json", {"pair": pair, "seed": seed, "condition": cond, "lambda": LAMBDA,
                                                              "per_step": per_step, "host": _host()})
                    del latents
                    gc.collect(); torch.cuda.empty_cache()
                rows.append({"pair": pair, "seed": seed, "condition": cond, "png": str(png),
                             "per_step_json": str(d / f"seed_{seed}_per_step.json"), "seconds": round(time.time() - t0, 1)})
                print(f"[{time.strftime('%H:%M:%S')}] {pair} seed {seed} {cond} ({time.time() - t0:.0f}s)", flush=True)
    if not smoke:
        # every condition already on disk from an earlier render call joins the table
        for pair in pairs:
            for cond in sorted((root / "renders" / pair).glob("*")):
                if cond.name in conds or not cond.is_dir():
                    continue
                for seed in seeds:
                    png = cond / f"seed_{seed}.png"
                    if png.exists():
                        rows.append({"pair": pair, "seed": seed, "condition": cond.name, "png": str(png),
                                     "per_step_json": str(cond / f"seed_{seed}_per_step.json"), "seconds": None})
    all_conds = [c for c in list(CONDITIONS) + list(FOLLOW_ON_CONDITIONS) if any(r["condition"] == c for r in rows)]
    identity = identity_check(rows) if not smoke else None
    _write(RENDERS_JSON if not smoke else root / "renders.json",
           {"lambda": LAMBDA, "conditions": all_conds, "project_late_from": PROJECT_LATE_FROM, "frame_steps": list(FRAME_STEPS),
            "spec": {c: {"project_from": COND_SPEC[c][0], "mode": COND_SPEC[c][1]} for c in all_conds},
            "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, 1024 square, the seed's cached initial noise; frozen and adapter three-branch forwards in fp16, the probe in fp32",
            "identity_check": identity, "rows": rows, "checkpoint": str(ADAPTER_CHECKPOINT), "host": _host()})
    if identity:
        print("identity check:", json.dumps(identity["summary"]), flush=True)
    return 0


def identity_check(rows: list[dict]) -> dict:
    """The adapter-alone render here against the corrector tail stage's k=0 render of the same
    configuration (a different process, usually a different card)."""
    from PIL import Image
    cells = []
    for r in rows:
        if r["condition"] != "adapter":
            continue
        ref = CORRECTOR_OUT / "tail" / r["pair"] / f"seed_{r['seed']}" / "lambda_1.2_k000_w35-50.png"
        if not ref.exists() or not Path(r["png"]).exists():
            cells.append({"pair": r["pair"], "seed": r["seed"], "reference": str(ref), "mean_abs_diff": None}); continue
        a = np.asarray(Image.open(r["png"]).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32)
        cells.append({"pair": r["pair"], "seed": r["seed"], "reference": str(ref), "mean_abs_diff": float(np.abs(a - b).mean())})
    vals = [c["mean_abs_diff"] for c in cells if c["mean_abs_diff"] is not None]
    return {"bar": IDENTITY_MAX_MEAN_ABS_DIFF, "cells": cells,
            "summary": {"n_compared": len(vals), "max_mean_abs_diff": max(vals) if vals else None,
                        "pass": bool(vals and max(vals) <= IDENTITY_MAX_MEAN_ABS_DIFF)}}


# ---------------------------------------------------------------------------
# Stage: the scores
# ---------------------------------------------------------------------------

_DEV = None
_EMB = None


def _dev():
    global _DEV
    if _DEV is None:
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEV


def embed_dino(paths: list[Path]) -> np.ndarray:
    global _EMB
    from poe_repair.experiments.compose_scorer_validation.scorer import _Embedders
    if _EMB is None:
        _EMB = _Embedders(device=_dev())
    return _EMB.dino(paths)


def mono_png(pair: str, seed: int) -> Path:
    return CORRECTOR_OUT / "sheet" / "references" / pair / f"seed_{seed}" / "mono.png"


def poe_png(pair: str, seed: int) -> Path:
    return CORRECTOR_OUT / "sheet" / "pairs" / pair / f"seed_{seed}" / "poe" / "poe.png"


def score_png(png: Path, pair: str) -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
    from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics
    from lambda_window_grid import _laplacian_var
    n, _boxes = count_instances(png, device=_dev())
    qa, qb = CONCEPT_QUERIES[pair]
    dets = vmetrics.detect_boxes(png, [qa, qb], device=_dev())
    conf = {}
    for q in (qa, qb):
        c = [d["confidence"] for d in dets if d.get("label", "").strip().lower() == q]
        conf[q] = float(max(c)) if c else 0.0
    both = bool(conf[qa] >= CONCEPT_CONF and conf[qb] >= CONCEPT_CONF)
    row = {"n_instances": int(n), "concept_conf": conf, "both_concepts_present": both,
           "sharpness_laplacian_var": float(_laplacian_var(png))}
    if pair == FAILING_PAIR:
        row["compose_read"] = "validated instance count"; row["composed"] = bool(n >= 2)
    else:
        row["compose_read"] = "unvalidated: both concepts detected (the validated rule counts animals, a meadow is not one)"
        row["composed"] = both
    return row


def cloud_axes() -> dict:
    """The cat x dog cloud axes refit from the landing finding's saved DINOv2 features: x from
    the cat-alone centroid to the dog-alone centroid, y from the solo midpoint toward the
    joint-prompt centroid, orthogonalised against x. Reproduces the sidecar's numbers."""
    feats = np.load(LANDING_DIR / "cat-x-dog-in-dino-space-dino-feats.npy")
    side = json.loads((LANDING_DIR / "cat-x-dog-in-dino-space.json").read_text())
    pts = side["points"]
    def cen(c):
        return feats[[i for i, p in enumerate(pts) if p["condition"] == c]].mean(0)
    ca, cb, cj = cen("solo_a"), cen("solo_b"), cen("joint")
    mid = (ca + cb) / 2
    x = cb - ca; x = x / np.linalg.norm(x)
    y = cj - mid; y = y - (y @ x) * x; y = y / np.linalg.norm(y)
    stored = [{"condition": p["condition"], "seed": p["seed"], "which_animal": float((feats[i] - mid) @ x),
               "both_ness": float((feats[i] - mid) @ y)} for i, p in enumerate(pts)]
    chk = [abs(s["both_ness"] - p["cloud_axes"]["both_ness"]) for s, p in zip(stored, pts)]
    return {"mid": mid, "x": x, "y": y, "stored": stored, "refit_max_abs_gap_vs_sidecar": float(max(chk))}


def project_axes(ax: dict, feats: np.ndarray) -> list[dict]:
    return [{"which_animal": float((f - ax["mid"]) @ ax["x"]), "both_ness": float((f - ax["mid"]) @ ax["y"])} for f in feats]


def score() -> int:
    d = json.loads(RENDERS_JSON.read_text())
    ax = cloud_axes()
    rows = []
    for r in d["rows"]:
        png = Path(r["png"])
        if not png.exists():
            continue
        row = dict(r); row.update(score_png(png, r["pair"]))
        mono = mono_png(r["pair"], r["seed"])
        if mono.exists():
            e = embed_dino([png, mono])
            row["dino_dist_to_mono"] = float(1.0 - e[0] @ e[1])
            if r["pair"] == FAILING_PAIR:
                row.update(project_axes(ax, e[:1])[0])
        rows.append(row)
    refs = []
    for pair in PAIRS:
        for seed in SEEDS:
            for name, fn in (("mono", mono_png), ("poe", poe_png)):
                p = fn(pair, seed)
                if not p.exists():
                    continue
                rr = {"pair": pair, "seed": seed, "condition": name, "png": str(p)}
                rr.update(score_png(p, pair))
                m = mono_png(pair, seed)
                if m.exists():
                    e = embed_dino([p, m]); rr["dino_dist_to_mono"] = float(1.0 - e[0] @ e[1])
                    if pair == FAILING_PAIR:
                        rr.update(project_axes(ax, e[:1])[0])
                refs.append(rr)
    # the frame tracks on the cloud axes, cat x dog only
    tracks = []
    for cond in d["conditions"]:
        for seed in SEEDS:
            fd = OUT / "frames" / FAILING_PAIR / cond / f"seed_{seed}"
            paths = [fd / f"step_{k:02d}.png" for k in FRAME_STEPS if (fd / f"step_{k:02d}.png").exists()]
            if not paths:
                continue
            e = embed_dino(paths)
            pts = project_axes(ax, e)
            tracks.append({"condition": cond, "seed": seed, "steps": [int(p.stem.split("_")[1]) for p in paths], "points": pts})
    result = fix_verdict(rows, refs)
    result.update({"lambda": LAMBDA, "conditions": d["conditions"], "identity_check": d.get("identity_check"),
                   "cloud_axes": {"refit_max_abs_gap_vs_sidecar": ax["refit_max_abs_gap_vs_sidecar"],
                                  "source": str(LANDING_DIR / "cat-x-dog-in-dino-space-dino-feats.npy")},
                   "rows": rows, "references": refs, "tracks": tracks, "green_frame_rule": GREEN_FRAME_RULE, "host": _host()})
    _write(RESULTS_JSON, result)
    print(json.dumps({k_: result[k_] for k_ in ("branch", "reasons", "summary")}, indent=1), flush=True)
    return 0


def fix_verdict(rows: list[dict], refs: list[dict]) -> dict:
    conds = [c for c in list(CONDITIONS) + list(FOLLOW_ON_CONDITIONS) if any(r["condition"] == c for r in rows)]
    summary = {"conditions": conds}
    for pair in PAIRS:
        summary[pair] = {}
        for cond in conds:
            pr = [r for r in rows if r["pair"] == pair and r["condition"] == cond]
            dm = [r["dino_dist_to_mono"] for r in pr if r.get("dino_dist_to_mono") is not None]
            summary[pair][cond] = {"composed_of_8": sum(int(r["composed"]) for r in pr), "n": len(pr),
                                   "mean_dino_dist_to_mono": float(np.mean(dm)) if dm else None,
                                   "mean_sharpness_laplacian_var": float(np.mean([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                                   "mean_both_ness": float(np.mean([r["both_ness"] for r in pr if "both_ness" in r])) if pair == FAILING_PAIR and pr else None}
        for name in ("mono", "poe"):
            pr = [r for r in refs if r["pair"] == pair and r["condition"] == name]
            dm = [r["dino_dist_to_mono"] for r in pr if r.get("dino_dist_to_mono") is not None]
            summary[pair][name] = {"composed_of_8": sum(int(r["composed"]) for r in pr), "n": len(pr),
                                   "mean_dino_dist_to_mono": float(np.mean(dm)) if dm else None,
                                   "mean_sharpness_laplacian_var": float(np.mean([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                                   "mean_both_ness": float(np.mean([r["both_ness"] for r in pr if "both_ness" in r])) if pair == FAILING_PAIR and pr else None}
    base = summary[FAILING_PAIR]["adapter"]; b0 = base["mean_dino_dist_to_mono"]
    supp, breaks, ctrl_fail = [], [], []
    for cond in [c for c in conds if c != "adapter"]:
        v = summary[FAILING_PAIR][cond]
        gain = (b0 - v["mean_dino_dist_to_mono"]) if (b0 is not None and v["mean_dino_dist_to_mono"] is not None) else None
        loss = base["composed_of_8"] - v["composed_of_8"]
        ctrl_loss = summary[COMPOSING_PAIR]["adapter"]["composed_of_8"] - summary[COMPOSING_PAIR][cond]["composed_of_8"]
        v.update({"mono_gain_vs_adapter_alone": gain, "compose_loss_vs_adapter_alone": loss, "control_compose_loss": ctrl_loss})
        nearer = gain is not None and gain >= MIN_MONO_GAIN
        if nearer and loss <= MAX_COMPOSE_LOSS_SEEDS and ctrl_loss <= MAX_COMPOSE_LOSS_SEEDS:
            supp.append(cond)
        elif nearer and loss > MAX_COMPOSE_LOSS_SEEDS:
            breaks.append(cond)
        elif ctrl_loss > MAX_COMPOSE_LOSS_SEEDS:
            ctrl_fail.append(cond)
    if b0 is None or base["n"] == 0:
        branch, reasons = "not ready", ["the adapter-alone renders or the joint-prompt references are missing"]
    elif supp:
        branch = "support"; reasons = [f"{c}: DINOv2 distance to the joint render {summary[FAILING_PAIR][c]['mean_dino_dist_to_mono']:.3f} against the adapter's {b0:.3f} "
                                       f"(gain {summary[FAILING_PAIR][c]['mono_gain_vs_adapter_alone']:+.3f}, bar {MIN_MONO_GAIN}), composed {summary[FAILING_PAIR][c]['composed_of_8']} of 8 against {base['composed_of_8']}" for c in supp]
    elif breaks:
        branch = "composition breaks"; reasons = [f"{c}: nearer the joint render but loses {summary[FAILING_PAIR][c]['compose_loss_vs_adapter_alone']} composed seeds" for c in breaks]
    elif ctrl_fail:
        branch = "inconclusive"; reasons = [f"{c}: the control pair lost {summary[FAILING_PAIR][c]['control_compose_loss']} composed seeds" for c in ctrl_fail]
    else:
        best = max([c for c in conds if c != "adapter"], key=lambda c: summary[FAILING_PAIR][c]["mono_gain_vs_adapter_alone"] or -1)
        branch = "null"; reasons = [f"no projected condition moves at least {MIN_MONO_GAIN} nearer the joint render while holding composition; "
                                    f"best gain {summary[FAILING_PAIR][best]['mono_gain_vs_adapter_alone']:+.3f} at {best} (adapter alone {b0:.3f})"]
    return {"branch": branch, "reasons": reasons, "summary": summary,
            "thresholds": {"MIN_MONO_GAIN": MIN_MONO_GAIN, "MAX_COMPOSE_LOSS_SEEDS": MAX_COMPOSE_LOSS_SEEDS, "PROJECT_LATE_FROM": PROJECT_LATE_FROM},
            "follow_on": {c: "raised after the step-size sweep, judged by the same bar, and said so" for c in conds if c in FOLLOW_ON_CONDITIONS},
            "primary_read": "DINOv2 ViT-S/14 cosine distance between the render and the seed's joint-prompt render (the compose scorer's embedder); lower is nearer the clean image",
            "condition": f"rank-{ADAPTER_RANK} adapter at step 30050, lambda {LAMBDA}; 'projected' drops the correction's normal part on every step, "
                         f"'projected_late' from step {PROJECT_LATE_FROM}; 'projected_ls' minimises the normal energy by a line search along it on every step; "
                     f"'adapter' is the unprojected baseline rendered in the same process"}


# ---------------------------------------------------------------------------
# Stage: the figures
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _by_step(rows, key, seeds):
    steps = sorted({r["step"] for r in rows})
    per = {s: [next((r[key] for r in rows if r["seed"] == s and r["step"] == k), np.nan) for k in steps] for s in seeds}
    mean = [float(np.nanmean([per[s][i] for s in seeds])) for i in range(len(steps))]
    return steps, per, mean


def fig_normal_share() -> Path:
    plt = _mpl()
    d = json.loads(CACHE_JSON.read_text()); rows = d["rows"]; seeds = d["seeds"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    colours = {"normal_share_true": ("#1a7a3c", "true correction  r_t = joint − PoE"),
               "normal_share_adapter": ("#c2410c", "adapter's correction (rank 32, step 30050)"),
               "normal_share_random": ("0.55", "random direction"),
               "normal_share_poe": ("#1d4ed8", "the PoE prediction itself"),
               "normal_share_error": ("#a21caf", "adapter minus true")}
    for key, (col, lab) in colours.items():
        steps, per, mean = _by_step(rows, key, seeds)
        for s in seeds:
            ax.plot(steps, per[s], color=col, alpha=0.18, lw=0.8)
        ax.plot(steps, mean, color=col, lw=2.2, label=lab)
    ax.axvspan(EARLY_STEPS[0], EARLY_STEPS[-1], color="0.92", zorder=0)
    ax.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel("denoising step (0 = noise, 49 = image)"); ax.set_ylabel("‖σ_t J v‖ / ‖v‖   (above 1: the denoiser amplifies v)")
    ax.set_title("How much of each direction points off the base model's manifold\n(frozen unconditional denoiser, fp32 finite difference; thin = seed, thick = mean of 8)", fontsize=9)
    ax.legend(fontsize=7.5, loc="best"); ax.set_ylim(bottom=0)
    ax2 = axes[1]
    for key, col, lab in (("cos_adapter_true", "0.3", "cos(adapter, true)"),
                          ("cos_adapter_tan_true_tan", "#c2410c", "cos(tangent parts)"),
                          ("cos_adapter_normal_true_normal", "#1d4ed8", "cos(normal parts)")):
        steps, per, mean = _by_step(rows, key, seeds)
        for s in seeds:
            ax2.plot(steps, per[s], color=col, alpha=0.18, lw=0.8)
        ax2.plot(steps, mean, color=col, lw=2.2, label=lab)
    ax2.set_xlabel("denoising step"); ax2.set_ylabel("cosine"); ax2.set_ylim(-0.1, 1.0)
    ax2.set_title("Fit of the adapter to the true correction,\nwhole vectors and each part separately", fontsize=9)
    ax2.legend(fontsize=7.5)
    st = d["statistic"]
    fig.suptitle(f"cat × dog, seeds 9 to 16, cached plain-PoE states. Steps 0 to 10: adapter/true normal-share ratio {st['ratio_adapter_over_true_early_mean']:.3f} "
                 f"(support ≥ {NORMAL_RATIO_SUPPORT}, null ≤ {NORMAL_RATIO_NULL}); printed branch \"{d['branch']}\"", fontsize=9, y=1.0)
    fig.tight_layout()
    out = RESULTS_DIR / "normal-share-over-denoising-steps.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    _write(out.with_suffix(".json"), {
        "drawn_from": str(CACHE_JSON), "pair": FAILING_PAIR, "seeds": seeds,
        "left_panel": {"y": "normal share ||sigma_t J v|| / ||v|| of a direction v under the frozen unconditional denoiser's Jacobian, 0 = tangent, 1 = fully normal in the ideal case",
                       "x": "denoising step 0 to 49", "lines": {k: v[1] for k, v in colours.items()}, "band": "steps 0 to 10, the pre-registered window"},
        "right_panel": {"y": "cosine between the adapter's correction and the true one", "lines": ["whole", "tangent parts", "normal parts"]},
        "statistic": st, "branch": d["branch"], "reasons": d["reasons"], "thresholds": d["thresholds"], "probe": d["probe"],
        "caption_owes": ["the share is 'sine of the angle to the tangent space' only where the manifold picture holds; the random-direction line is the reference for how big the normal space reads",
                         "the probe is the unconditional denoiser, so 'the manifold' is the base model's image manifold, not the cat-and-dog one; the joint-prompt read is in cache_read.json",
                         "the finite difference is linear only where the linearity check passes (statistic.linearity_ok)"]})
    return out


def fig_arrows() -> Path:
    """The manifold picture: at each step, the true and the adapter's correction as arrows in the
    (tangent, normal) plane, the manifold drawn as the horizontal axis."""
    plt = _mpl()
    d = json.loads(CACHE_JSON.read_text()); rows = d["rows"]; seeds = d["seeds"]
    steps = [k for k in ARROW_STEPS if any(r["step"] == k for r in rows)]
    fig, axes = plt.subplots(1, len(steps), figsize=(2.3 * len(steps), 3.4), squeeze=False)
    for ax, k in zip(axes[0], steps):
        rr = [r for r in rows if r["step"] == k]
        for r in rr:
            ax.annotate("", xy=(r["tangent_norm_true"], r["normal_norm_true"]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#1a7a3c", alpha=0.25, lw=0.9))
            ax.annotate("", xy=(r["tangent_norm_adapter"], r["normal_norm_adapter"]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#c2410c", alpha=0.25, lw=0.9))
        mt = (np.mean([r["tangent_norm_true"] for r in rr]), np.mean([r["normal_norm_true"] for r in rr]))
        ma = (np.mean([r["tangent_norm_adapter"] for r in rr]), np.mean([r["normal_norm_adapter"] for r in rr]))
        ax.annotate("", xy=mt, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#1a7a3c", lw=2.4))
        ax.annotate("", xy=ma, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#c2410c", lw=2.4))
        lim = 1.15 * max(mt[0], mt[1], ma[0], ma[1], max(r["tangent_norm_true"] for r in rr), max(r["tangent_norm_adapter"] for r in rr))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=2.0); ax.text(lim * 0.98, lim * 0.02, "manifold (tangent)", ha="right", va="bottom", fontsize=6.5)
        ax.set_title(f"step {k}  (t = {rr[0]['timestep']})", fontsize=8)
        ax.set_xlabel("along the manifold  ‖P v‖", fontsize=7); ax.tick_params(labelsize=6)
        if k == steps[0]:
            ax.set_ylabel("off the manifold  ‖(I−P) v‖", fontsize=7)
    fig.suptitle("Each correction split into its part along the base model's manifold and its part off it, cat × dog, cached states\n"
                 "green = true correction (joint − PoE), orange = the rank-32 adapter's; thin arrows = seeds 9 to 16, thick = mean; latent-space norms\n"
                 "(read with the step-size sweep: from step 10 on σ_t J amplifies, so the 'off' axis is the denoiser's response, not a distance from a manifold)",
                 fontsize=8.5, wrap=True)
    fig.tight_layout()
    out = RESULTS_DIR / "corrections-split-along-and-off-the-manifold.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    _write(out.with_suffix(".json"), {"drawn_from": str(CACHE_JSON), "steps": steps, "seeds": seeds,
                                      "x": "norm of the tangent part v - sigma_t J v, latent units", "y": "norm of the normal part sigma_t J v, latent units",
                                      "one_arrow": "one seed's correction at that step; the thick arrow is the mean of the eight endpoints",
                                      "per_step_means": {k: {"true": [float(np.mean([r["tangent_norm_true"] for r in rows if r["step"] == k])), float(np.mean([r["normal_norm_true"] for r in rows if r["step"] == k]))],
                                                             "adapter": [float(np.mean([r["tangent_norm_adapter"] for r in rows if r["step"] == k])), float(np.mean([r["normal_norm_adapter"] for r in rows if r["step"] == k]))]} for k in steps},
                                      "caption_owes": ["the horizontal axis is the local tangent space at the cached state, a different subspace at every step and seed; the picture aligns them by construction"]})
    return out


def fig_landing() -> Path | None:
    """The DINOv2 cloud axes with the new endpoints and the frame tracks."""
    plt = _mpl()
    if not RESULTS_JSON.exists():
        return None
    d = json.loads(RESULTS_JSON.read_text())
    ax_ = cloud_axes()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    cloud_col = {"solo_a": "#7c3aed", "solo_b": "#0891b2", "joint": "#1a7a3c", "poe": "#a33", "lora_1.2": "0.5"}
    cloud_lab = {"solo_a": "a cat", "solo_b": "a dog", "joint": "a cat and a dog", "poe": "plain PoE (landing finding)", "lora_1.2": "adapter λ1.2 (landing finding)"}
    for ax in axes:
        for c, col in cloud_col.items():
            pts = [p for p in ax_["stored"] if p["condition"] == c]
            ax.scatter([p["which_animal"] for p in pts], [p["both_ness"] for p in pts], s=22, color=col, alpha=0.35, label=cloud_lab[c], edgecolor="none")
        ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlabel("which animal  (← cat        dog →)"); ax.set_ylabel("both-ness  (toward the joint-prompt cloud)")
    cond_col = {"adapter": "#f59e0b", "projected": "#c2410c", "projected_late": "#be123c", "projected_ls": "#0f766e"}
    cond_lab = {"adapter": f"adapter λ{LAMBDA} (this run)", "projected": "normal part subtracted, every step", "projected_late": f"normal part subtracted from step {PROJECT_LATE_FROM}",
                "projected_ls": "normal energy minimised (line search), every step"}
    for c in d["conditions"]:
        pts = [r for r in d["rows"] if r["pair"] == FAILING_PAIR and r["condition"] == c and "both_ness" in r]
        axes[0].scatter([p["which_animal"] for p in pts], [p["both_ness"] for p in pts], s=48, marker="D", color=cond_col[c],
                        edgecolor="k", lw=0.5, label=f"{cond_lab[c]}: mean both-ness {np.mean([p['both_ness'] for p in pts]):.3f}" if pts else cond_lab[c])
    axes[0].set_title("Where the final renders land (◆ = this run, seeds 9 to 16)", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="lower left")
    for tr in d.get("tracks", []):
        col = cond_col.get(tr["condition"], "0.3")
        xs = [p["which_animal"] for p in tr["points"]]; ys = [p["both_ness"] for p in tr["points"]]
        axes[1].plot(xs, ys, color=col, alpha=0.5, lw=1.0)
        axes[1].scatter(xs[-1:], ys[-1:], color=col, s=26, edgecolor="k", lw=0.4, zorder=3)
        axes[1].scatter(xs[:1], ys[:1], color=col, s=10, marker="s", zorder=3)
    for c in d["conditions"]:
        axes[1].plot([], [], color=cond_col[c], label=cond_lab[c])
    axes[1].set_title(f"Tracks of the running estimate (Tweedie mean) at steps {', '.join(map(str, FRAME_STEPS))}\n■ = step 0, ● = step 49", fontsize=9)
    axes[1].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("cat × dog in the compose scorer's DINOv2 space, on the cloud axes fitted for the landing finding", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / "cat-x-dog-in-dino-space-with-the-projected-correction.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    _write(out.with_suffix(".json"), {"drawn_from": [str(RESULTS_JSON), str(LANDING_DIR / "cat-x-dog-in-dino-space.json")],
                                      "axes": "x: unit vector from the cat-alone centroid to the dog-alone centroid; y: from the solo midpoint toward the joint-prompt centroid, orthogonalised; origin the solo midpoint; DINOv2 ViT-S/14 CLS, L2-normed",
                                      "refit_max_abs_gap_vs_sidecar": ax_["refit_max_abs_gap_vs_sidecar"],
                                      "endpoints": {c: [{k_: r[k_] for k_ in ("seed", "which_animal", "both_ness", "composed")} for r in d["rows"] if r["pair"] == FAILING_PAIR and r["condition"] == c and "both_ness" in r] for c in d["conditions"]},
                                      "means": {c: d["summary"][FAILING_PAIR][c].get("mean_both_ness") for c in d["conditions"]},
                                      "tracks": d.get("tracks", []),
                                      "caption_owes": ["two axes of a 384-dimensional space; the off-plane norm is about 0.8 in the landing finding, so distances here understate distances in the space",
                                                       "the frames are 256 px thumbnails of the running estimate, embedded with the same encoder as the 1024 px renders"]})
    return out


def fig_run_share() -> Path | None:
    plt = _mpl()
    if not RENDERS_JSON.exists():
        return None
    d = json.loads(RENDERS_JSON.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    cond_col = {"projected": "#c2410c", "projected_late": "#be123c", "projected_ls": "#0f766e"}
    for cond in ("projected", "projected_late", "projected_ls"):
        allrows = []
        for r in d["rows"]:
            if r["pair"] != FAILING_PAIR or r["condition"] != cond or not Path(r["per_step_json"]).exists():
                continue
            ps = json.loads(Path(r["per_step_json"]).read_text())["per_step"]
            ys = [p.get("normal_share_before", np.nan) for p in ps]
            ax.plot([p["step"] for p in ps], ys, color=cond_col[cond], alpha=0.2, lw=0.8)
            allrows.append(ys)
        if allrows:
            ax.plot(range(len(allrows[0])), np.nanmean(np.array(allrows, dtype=float), axis=0), color=cond_col[cond], lw=2.2, label=cond.replace("_", " "))
    ax.set_xlabel("denoising step"); ax.set_ylabel("normal share of the adapter's correction\nbefore it is dropped, on the live trajectory")
    ax.set_title("cat × dog, seeds 9 to 16: how normal the adapter's push was at each step of the projected runs", fontsize=9)
    ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = RESULTS_DIR / "normal-share-during-the-projected-runs.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    _write(out.with_suffix(".json"), {"drawn_from": str(RENDERS_JSON), "y": "||sigma_t J delta|| / ||delta|| of the adapter's correction at the live state, before projection", "x": "denoising step"})
    return out


def _frame(ax, composed, title=""):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        if composed is None:
            s.set_color("0.7"); s.set_linewidth(0.8)
        else:
            s.set_color("#1a7a3c" if composed else "#a33"); s.set_linewidth(2.2)
    if title:
        ax.set_title(title, fontsize=6.5, pad=2.0, color=("#1a7a3c" if composed else ("#a33" if composed is False else "0.4")))


def sheets() -> list[Path]:
    plt = _mpl()
    from PIL import Image
    if not RESULTS_JSON.exists():
        return []
    d = json.loads(RESULTS_JSON.read_text())
    outs = []
    short = {FAILING_PAIR: "cat-dog", COMPOSING_PAIR: "butterfly-meadow"}
    for pair in PAIRS:
        titles = ["joint prompt\n(Mono)", "plain PoE"] + [COND_TITLE.get(c, c) for c in d["conditions"]]
        tiles = {}
        for r in d["references"]:
            if r["pair"] != pair:
                continue
            ct = titles[0] if r["condition"] == "mono" else titles[1]
            lab = f"{r['n_instances']} inst · sharp {r['sharpness_laplacian_var']:.0f}"
            if r.get("dino_dist_to_mono") is not None and r["condition"] == "poe":
                lab += f" · d(joint) {r['dino_dist_to_mono']:.2f}"
            tiles[(r["seed"], ct)] = {"png": r["png"], "composed": r["composed"], "label": lab}
        for r in d["rows"]:
            if r["pair"] != pair:
                continue
            ct = COND_TITLE.get(r["condition"], r["condition"])
            lab = f"{r['n_instances']} inst · d(joint) {r.get('dino_dist_to_mono', float('nan')):.2f} · sharp {r['sharpness_laplacian_var']:.0f}"
            tiles[(r["seed"], ct)] = {"png": r["png"], "composed": r["composed"], "label": lab}
        n_rows, n_cols = len(SEEDS), len(titles)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, 1.75 * n_rows + 0.9), squeeze=False)
        for ri, seed in enumerate(SEEDS):
            for ci, ct in enumerate(titles):
                ax = axes[ri][ci]; t = tiles.get((seed, ct))
                if t and Path(t["png"]).is_file():
                    ax.imshow(Image.open(t["png"]).resize((384, 384))); _frame(ax, bool(t["composed"]), t["label"])
                else:
                    _frame(ax, None); ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                if ri == 0:
                    ax.set_title(f"{ct}\n{ax.get_title()}", fontsize=6.5, pad=3)
                if ci == 0:
                    ax.set_ylabel(f"seed {seed}", fontsize=8)
        s = d["summary"][pair]
        counts = " · ".join(f"{n}: {v['composed_of_8']} of 8" + (f", d(joint) {v['mean_dino_dist_to_mono']:.3f}" if v.get("mean_dino_dist_to_mono") is not None else "")
                            for n, v in s.items() if n != "mono" and isinstance(v, dict))
        fig.suptitle(f"{pair.replace('__x__', ' × ').replace('_', ' ')}: {counts}\n"
                     f"frame = {'validated instance count' if pair == FAILING_PAIR else 'both concepts detected (unvalidated)'}; printed branch \"{d['branch']}\"", fontsize=8.5)
        fig.subplots_adjust(top=1 - 0.95 / (1.75 * n_rows + 0.9), bottom=0.01, left=0.04, right=0.995, hspace=0.2, wspace=0.03)
        out = RESULTS_DIR / f"mono-vs-poe-vs-adapter-vs-projected-{short[pair]}-eight-seed-sheet.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        _write(out.with_suffix(".json"), {"drawn_from": str(RESULTS_JSON), "pair": pair, "seeds": list(SEEDS), "columns": titles,
                                          "condition": d["condition"], "thresholds": d["thresholds"], "branch": d["branch"], "reasons": d["reasons"],
                                          "summary": s, "primary_read": d["primary_read"], "green_frame_rule": GREEN_FRAME_RULE,
                                          "tiles": [{k_: r.get(k_) for k_ in ("seed", "condition", "png", "n_instances", "composed", "dino_dist_to_mono", "sharpness_laplacian_var", "both_ness", "concept_conf")} for r in d["rows"] if r["pair"] == pair]})
        outs.append(out)
    return outs


def figures() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    if CACHE_JSON.exists():
        outs += [fig_normal_share(), fig_arrows()]
    outs += [p for p in (fig_h_sweep(), fig_run_share(), fig_landing()) if p]
    outs += sheets()
    for p in outs:
        print(p)
    return 0


# ---------------------------------------------------------------------------
# Stage: W&B
# ---------------------------------------------------------------------------


def log_wandb() -> str:
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    run = wandb.init(project=project, entity=entity, name=f"scope01-tangent-projection-{time.strftime('%Y%m%d-%H%M')}",
                     job_type="tangent-projection",
                     config={"scope": "01-showcase-the-trained-lora", "plan": "15-keep-the-correction-on-the-manifold",
                             "checkpoint": str(ADAPTER_CHECKPOINT), "rank": ADAPTER_RANK, "lambda": LAMBDA,
                             "fd_rel_step": FD_REL_STEP, "project_late_from": PROJECT_LATE_FROM,
                             "thresholds": {"NORMAL_RATIO_SUPPORT": NORMAL_RATIO_SUPPORT, "NORMAL_RATIO_NULL": NORMAL_RATIO_NULL,
                                            "MIN_MONO_GAIN": MIN_MONO_GAIN, "MAX_COMPOSE_LOSS_SEEDS": MAX_COMPOSE_LOSS_SEEDS}})
    summ = {}
    per_step: dict[int, dict] = {}
    tables = {}
    if CACHE_JSON.exists():
        d = json.loads(CACHE_JSON.read_text()); rows = d["rows"]
        keys = ("normal_share_true", "normal_share_adapter", "normal_share_random", "normal_share_poe", "normal_share_error",
                "normal_share_true_joint", "normal_share_adapter_joint", "cos_adapter_true", "cos_adapter_tan_true_tan",
                "cos_adapter_normal_true_normal", "norm_true", "norm_adapter")
        for k in sorted({r["step"] for r in rows}):
            rr = [r for r in rows if r["step"] == k]
            per_step.setdefault(k, {}).update({f"cache/{key}_mean": float(np.mean([r[key] for r in rr])) for key in keys})
            per_step[k]["cache/ratio_adapter_over_true_mean"] = float(np.mean([r["normal_share_adapter"] / max(r["normal_share_true"], 1e-12) for r in rr]))
        cols = list(rows[0].keys())
        tables["cache/per_seed_per_step"] = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
        summ["cache_read"] = {"branch": d["branch"], "reasons": d["reasons"], "statistic": {k_: v for k_, v in d["statistic"].items() if k_ != "per_seed"}}
    if RENDERS_JSON.exists():
        d = json.loads(RENDERS_JSON.read_text())
        for cond in ("projected", "projected_late", "projected_ls"):
            series = []
            for r in d["rows"]:
                if r["pair"] == FAILING_PAIR and r["condition"] == cond and Path(r["per_step_json"]).exists():
                    ps = json.loads(Path(r["per_step_json"]).read_text())["per_step"]
                    series.append([p.get("normal_share_before", np.nan) for p in ps])
            if series:
                m = np.nanmean(np.array(series, dtype=float), axis=0)
                for k, v in enumerate(m):
                    if not np.isnan(v):
                        per_step.setdefault(k, {})[f"run/{cond}/normal_share_before_projection_mean"] = float(v)
        summ["identity_check"] = (d.get("identity_check") or {}).get("summary")
    for k in sorted(per_step):
        run.log(per_step[k], step=k)
    if tables:
        run.log(tables)
    if RESULTS_JSON.exists():
        d = json.loads(RESULTS_JSON.read_text())
        cols = ["pair", "seed", "condition", "n_instances", "composed", "dino_dist_to_mono", "sharpness_laplacian_var", "both_ness", "png"]
        tab = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in d["rows"] + d["references"]])
        run.log({"scores/per_render": tab})
        summ["fix"] = {"branch": d["branch"], "reasons": d["reasons"], "summary": d["summary"]}
    imgs = {}
    for p in sorted(RESULTS_DIR.glob("*.png")):
        imgs[("sheets/" if "sheet" in p.stem else "figures/") + p.stem] = wandb.Image(str(p))
    if imgs:
        run.log(imgs)
    art = wandb.Artifact("scope01-tangent-projection-sidecars", type="results")
    for p in [CACHE_JSON, H_SWEEP_JSON, RENDERS_JSON, RESULTS_JSON] + sorted(RESULTS_DIR.glob("*.json")):
        if p.exists():
            art.add_file(str(p), name=p.name)
    run.log_artifact(art)
    run.summary.update(summ)
    rid = run.id; run.finish()
    (OUT / "wandb_run_id.txt").write_text(f"{WANDB_PROJECT}/{rid}\n")
    print("wandb run:", f"{WANDB_PROJECT}/{rid}")
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("cache-read", "h-sweep", "render", "score", "figures", "wandb", "smoke"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--conditions", default=None, help="comma-separated conditions for --render (default: the three pre-registered ones)")
    args = ap.parse_args()
    if args.cache_read:
        cache_read(smoke=args.smoke)
    if args.h_sweep:
        h_sweep()
    if args.render:
        render(smoke=args.smoke, conditions=tuple(args.conditions.split(",")) if args.conditions else CONDITIONS)
    if args.score:
        score()
    if args.figures:
        figures()
    if args.wandb:
        log_wandb()
    if not any(v for k_, v in vars(args).items() if k_ != "smoke"):
        ap.error("pass at least one stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
