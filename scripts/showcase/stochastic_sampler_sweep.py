#!/usr/bin/env python
"""Stochastic sampler sweep (plan 19, 01-showcase-the-trained-lora).

Does a stochastic DDIM sampler (eta above 0) sharpen the corrected render without losing the
two animals? Every condition at every eta shares the same initial noise per seed and the same
per-step noise sample per seed, so the comparison between conditions is pathwise: only the
noise prediction (and eta, which scales the fresh noise) differs.

  --render   mono (the joint prompt, plain CFG), plain PoE, and PoE + LAMBDA_FULL x the rank-32
             step-30050 correction on every step, at each eta in ETAS, 8 seeds, both pairs.
             Mono is rendered before the adapter is attached.
  --score    compose (instance count), sharpness at 1024 px, DINOv2 drift against the mono and
             plain-PoE renders at the same eta, both-ness on the landing finding's cloud axes
             (cat x dog), butterfly presence (control pair); results.json, sheets, a figure, W&B.

The DDIM update with eta (Song et al. 2020, eq. 12): with x0 the running estimate and
ab the cumulative alpha,

  sigma_t   = eta * sqrt((1 - ab_prev) / (1 - ab_t)) * sqrt(1 - ab_t / ab_prev)
  x_{t-1}   = sqrt(ab_prev) * x0 + sqrt(1 - ab_prev - sigma_t^2) * eps + sigma_t * z,  z ~ N(0, I)

eta 0 is the deterministic update the rest of this repo uses (ddim_prev_from_x0_eps); eta 1 is
the ancestral (DDPM-like) sampler on the same 50-step grid.

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

import correct_early_then_clean_up as ce  # noqa: E402  the SDXL context, prompts, pinned noise, attach

# ----------------------------------------------------------------------------- constants
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/stochastic_sampler_sweep")
RESULTS_DIR = REPO_ROOT / "artifacts/results/does-a-stochastic-sampler-sharpen-the-corrected-render"
TRAINING_CACHE = ce.TRAINING_CACHE
CLOUD_FEATS = ce.CLOUD_FEATS
CHECKPOINT = ce.CHECKPOINT
LORA_RANK = ce.LORA_RANK
WANDB_PROJECT = ce.WANDB_PROJECT

SEEDS = ce.SEEDS
PAIRS = ce.PAIRS
MAIN_PAIR = ce.MAIN_PAIR
CONTROL_PAIR = ce.CONTROL_PAIR
GUIDANCE_SCALE = ce.GUIDANCE_SCALE
STEPS = ce.STEPS_MAIN
EULER_INIT_NOISE_SIGMA = ce.EULER_INIT_NOISE_SIGMA

LAMBDA_FULL = 1.2
ETAS = (0.0, 0.5, 1.0)
CONDITIONS = ("mono", "poe", "corrected")
NOISE_SEED_STRIDE = 100_003   # per-step noise generator seed = seed * stride + step index

# the bar (review file, "The question written before the run"); every eta > 0 cell of the
# corrected run is judged against the corrected run at eta 0 on the same seeds
SUPPORT_MAX_COMPOSE_LOSS = 1     # seeds of 8 the eta cell may lose against eta 0 and still support
NULL_MIN_COMPOSE_LOSS = 2        # losing this many is a null
SUPPORT_MIN_SEEDS_SHARPER = 6    # of 8 seeds sharper than at eta 0, per seed
NULL_MAX_SEEDS_SHARPER = 3       # this few sharper is a null
SUPPORT_MAX_DRIFT_RISE = 0.02    # DINOv2 drift may not rise above the eta 0 value by more than this
# the sampler-side read for scope 06: plain PoE gaining this many seeds at eta > 0 is reported
POE_SAMPLER_GAIN_SEEDS = 2
# the control pair
BUTTERFLY_PRESENT_CONF = ce.BUTTERFLY_PRESENT_CONF
CONTROL_MAX_PRESENCE_LOSS = 1
# the instrument check: plain PoE at eta 0 on cat x dog seed 9 against the cached poe.png,
# mean absolute grey levels over the 1024 px image (fp16 across cards is about 2)
IDENTITY_MAX_MEAN_ABS_DIFF = ce.IDENTITY_MAX_MEAN_ABS_DIFF


def cond_name(cond: str, eta: float) -> str:
    return f"{cond}_eta{eta:g}"


DESCRIPTIONS = {
    "mono": "the joint prompt, plain CFG",
    "poe": "plain PoE, no correction",
    "corrected": f"PoE + {LAMBDA_FULL} x the rank-{LORA_RANK} step-30050 correction on all {STEPS} steps",
}


def _log(msg: str) -> None:
    print(f"[stochastic_sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- the step
def noise_for(seed: int, step_index: int, shape) -> torch.Tensor:
    """The fresh noise for one step of one seed, identical for every condition and eta."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * NOISE_SEED_STRIDE + int(step_index))
    return torch.randn(shape, generator=g, dtype=torch.float32)


def stochastic_ddim_step(scheduler, timesteps, k: int, x: torch.Tensor, eps: torch.Tensor,
                         eta: float, z: torch.Tensor) -> tuple[torch.Tensor, float]:
    """One DDIM update with eta, in float32, cast back to x's dtype. Returns (x_prev, sigma)."""
    t = int(timesteps[k].item())
    ab_t = float(scheduler.alphas_cumprod[t])
    if k + 1 < len(timesteps):
        ab_p = float(scheduler.alphas_cumprod[int(timesteps[k + 1].item())])
    else:
        ab_p = float(scheduler.final_alpha_cumprod)
    x32, e32 = x.float(), eps.float()
    x0 = (x32 - (1.0 - ab_t) ** 0.5 * e32) / ab_t ** 0.5
    sigma = float(eta) * ((1.0 - ab_p) / (1.0 - ab_t)) ** 0.5 * max(1.0 - ab_t / ab_p, 0.0) ** 0.5
    dir_coef = max(1.0 - ab_p - sigma ** 2, 0.0) ** 0.5
    x_prev = ab_p ** 0.5 * x0 + dir_coef * e32 + sigma * z.to(device=x32.device, dtype=torch.float32)
    return x_prev.to(x.dtype), sigma


def _disable(unet) -> None:
    if hasattr(unet, "disable_adapters"):
        unet.disable_adapters()
    elif hasattr(unet, "disable_adapter_layers"):
        unet.disable_adapter_layers()


def _enable(unet) -> None:
    import lambda_boundary_probe as lbp
    if hasattr(unet, "enable_adapters"):
        unet.enable_adapters()
    elif hasattr(unet, "enable_adapter_layers"):
        unet.enable_adapter_layers()
    if hasattr(unet, "set_adapter"):
        try:
            unet.set_adapter(lbp.LORA_ADAPTER_NAME)
        except Exception:
            pass


@torch.no_grad()
def sample(S: ce.Sampler, slug: str, seed: int, cond: str, eta: float):
    """One render. `cond` is mono, poe or corrected. The adapter is disabled on every frozen
    forward and on exit, so a later mono or PoE render is clean."""
    from poe_repair._sdxl.metrics import guided_eps, poe_eps
    from poe_repair._sdxl.runtime import decode_latents
    from poe_repair.methods._sampling import add_time_ids

    ctx = S.ctx
    device, dtype = ctx.device, ctx.dtype
    sch = ctx.scheduler
    sch.set_timesteps(STEPS)
    ts = sch.timesteps
    latents = (S.init_latents(slug, seed) / EULER_INIT_NOISE_SIGMA).to(device=device, dtype=dtype)
    unet = ctx.models["unet"]
    (sa, pa), (sb, pb), (sj, pj) = S.prompts[slug]["a"], S.prompts[slug]["b"], S.prompts[slug]["j"]
    if cond == "mono":
        pe, pool, n = torch.cat([sj, S.seq_e], 0), torch.cat([pj, S.pool_e], 0), 2
    else:
        pe, pool, n = torch.cat([sa, sb, S.seq_e], 0), torch.cat([pa, pb, S.pool_e], 0), 3
    added = {"text_embeds": pool,
             "time_ids": add_time_ids(height=1024, width=1024, batch_size=n, device=device, dtype=dtype)}

    def fwd(t):
        x = sch.scale_model_input(latents.repeat(n, 1, 1, 1), t)
        return unet(x, t, encoder_hidden_states=pe, added_cond_kwargs=added, timestep_cond=None).sample.chunk(n)

    sigmas, delta_norms = [], []
    for k, t in enumerate(ts):
        if S.attached:
            _disable(unet)
        if cond == "mono":
            ej, eu = fwd(t)
            eps_t = guided_eps(ej, eu, GUIDANCE_SCALE)
        else:
            ea, eb, eu = fwd(t)
            eps_poe = poe_eps(guided_eps(ea, eu, GUIDANCE_SCALE), guided_eps(eb, eu, GUIDANCE_SCALE), eu)
            if cond == "corrected":
                if not S.attached:
                    raise RuntimeError("corrected render requested before the adapter was attached")
                _enable(unet)
                la, lb, lu = fwd(t)
                eps_l = poe_eps(guided_eps(la, lu, GUIDANCE_SCALE), guided_eps(lb, lu, GUIDANCE_SCALE), lu)
                delta = eps_l - eps_poe
                delta_norms.append(float(delta.float().norm().item()))
                eps_t = eps_poe + float(LAMBDA_FULL) * delta
            else:
                eps_t = eps_poe
        latents, sigma = stochastic_ddim_step(sch, ts, k, latents, eps_t, eta, noise_for(seed, k, latents.shape))
        sigmas.append(sigma)
    if S.attached:
        _disable(unet)
    image = decode_latents(ctx.models, latents).cpu()
    return image, {"sigma_per_step": sigmas, "delta_norm_per_step": delta_norms,
                   "timestep_per_step": [int(x) for x in ts.tolist()]}


# ----------------------------------------------------------------------------- rendering
def _manifest_path() -> Path:
    return OUT_ROOT / "render_manifest.json"


def _load_manifest() -> list[dict]:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else []


def _save_manifest(rows: list[dict]) -> None:
    _manifest_path().write_text(json.dumps(rows, indent=2))


def _render_path(slug: str, cname: str, seed: int) -> Path:
    return OUT_ROOT / "renders" / slug / cname / f"seed_{seed}.png"


def _run_header() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def identity_check(S: ce.Sampler) -> dict:
    """Plain PoE at eta 0 through this sampler against the cached poe.png, cat x dog seed 9.
    Runs before the adapter is attached so the sampler is judged on its own."""
    slug, seed = MAIN_PAIR, 9
    p = OUT_ROOT / "identity_check" / "poe_eta0_seed_9.png"
    img, _ = sample(S, slug, seed, "poe", 0.0)
    ce._write(img, p)
    cached = TRAINING_CACHE / "heldout" / slug / f"seed_{seed}" / "poe.png"
    res = {**_run_header(), "mean_abs_diff_vs_cached_poe": ce.mean_abs_diff(p, cached),
           "IDENTITY_MAX_MEAN_ABS_DIFF": IDENTITY_MAX_MEAN_ABS_DIFF, "cached_poe": str(cached), "png": str(p)}
    res["pass"] = res["mean_abs_diff_vs_cached_poe"] <= IDENTITY_MAX_MEAN_ABS_DIFF
    (OUT_ROOT / "identity_check.json").write_text(json.dumps(res, indent=2))
    _log(f"identity check: mean abs diff {res['mean_abs_diff_vs_cached_poe']:.2f} pass={res['pass']}")
    return res


def render(seeds=SEEDS, pairs=tuple(PAIRS), etas=ETAS) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest()
    done = {(r["pair"], r["condition"], r["seed"]) for r in rows}
    S = ce.Sampler()
    _log(f"render header={_run_header()}")
    if not (OUT_ROOT / "identity_check.json").exists():
        res = identity_check(S)
        if not res["pass"]:
            raise SystemExit("identity check failed: the sampler at eta 0 does not reproduce the cached plain PoE render")

    def _record(slug, cond, eta, seed, img, extras, t0):
        cname = cond_name(cond, eta)
        p = _render_path(slug, cname, seed)
        ce._write(img, p)
        row = {"pair": slug, "condition": cname, "cond": cond, "eta": float(eta), "seed": seed, "png": str(p),
               "num_steps": STEPS, "lambda": LAMBDA_FULL if cond == "corrected" else 0.0,
               "description": DESCRIPTIONS[cond] + f", eta {eta:g}", "elapsed_s": round(time.time() - t0, 1),
               "sigma_per_step": extras["sigma_per_step"], "timestep_per_step": extras["timestep_per_step"],
               "max_delta_norm": max(extras["delta_norm_per_step"]) if extras["delta_norm_per_step"] else None}
        rows.append(row); done.add((slug, cname, seed)); _save_manifest(rows)
        _log(f"{slug} {cname} seed={seed} {row['elapsed_s']}s -> {p.name}")

    # pass 1: mono at every eta, before the adapter exists on the UNet
    for eta in etas:
        for slug in pairs:
            for seed in seeds:
                if (slug, cond_name("mono", eta), seed) in done:
                    continue
                t0 = time.time()
                img, ex = sample(S, slug, seed, "mono", eta)
                _record(slug, "mono", eta, seed, img, ex, t0)
    S.attach()
    for eta in etas:
        for slug in pairs:
            for cond in ("poe", "corrected"):
                for seed in seeds:
                    if (slug, cond_name(cond, eta), seed) in done:
                        continue
                    t0 = time.time()
                    img, ex = sample(S, slug, seed, cond, eta)
                    _record(slug, cond, eta, seed, img, ex, t0)
    _log("render done")


# ----------------------------------------------------------------------------- scoring
def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verdict_for_eta(cell: dict, base: dict) -> str:
    """`cell` is the corrected run at eta > 0, `base` the corrected run at eta 0, same seeds."""
    loss = base["compose_n"] - cell["compose_n"]
    sharper = cell["seeds_sharper_than_eta0"]
    if loss >= NULL_MIN_COMPOSE_LOSS:
        return "null: the animals go with the noise"
    if sharper <= NULL_MAX_SEEDS_SHARPER:
        return "null: no sharper on most seeds"
    if (loss <= SUPPORT_MAX_COMPOSE_LOSS and sharper >= SUPPORT_MIN_SEEDS_SHARPER
            and cell["drift_same_eta_mean"] <= base["drift_same_eta_mean"] + SUPPORT_MAX_DRIFT_RISE):
        return "support"
    return "inconclusive: animals stay, sharpness moves, the drift or the seed count falls short"


def control_verdict(cell: dict, poe: dict) -> str:
    if poe["presence_n"] - cell["presence_n"] > CONTROL_MAX_PRESENCE_LOSS:
        return "breaks the control: the butterfly goes"
    return "control intact"


def score() -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances)
    from scripts.build_lora_inspector_mds_semantic import DinoEmbedder

    rows = _load_manifest()
    if not rows:
        raise SystemExit("nothing rendered yet")
    prev = json.loads((OUT_ROOT / "results.json").read_text()) if (OUT_ROOT / "results.json").exists() else {"rows": []}
    cache = {(r["pair"], r["condition"], r["seed"]): r for r in prev.get("rows", []) if "feat" in r}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origin, u1, u2, ref_both = ce._cloud_axes()
    embedder = DinoEmbedder(device=dev)   # on CPU DINOv2 hits a CUDA-only xformers kernel (poe-mem-002)

    def _embed(p: Path) -> np.ndarray:
        im = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        return np.asarray(embedder.embed_decoded_batch(im)[0], dtype=np.float32)

    scored = []
    for r in rows:
        key = (r["pair"], r["condition"], r["seed"])
        if key in cache and cache[key].get("png") == r["png"]:
            scored.append(cache[key]); continue
        qa, qb = PAIRS[r["pair"]][3], PAIRS[r["pair"]][4]
        inst = instance_score_to_dict(score_output_instances(Path(r["png"]), qa, qb, device=dev))
        row = dict(r)
        f = _embed(Path(r["png"]))
        row.update({"compose": inst["label"] == "compose", "n_instances": inst["n_instances"],
                    "conf_a": inst["conf_a"], "conf_b": inst["conf_b"],
                    "presence_a": inst["conf_a"] >= BUTTERFLY_PRESENT_CONF,
                    "sharpness": ce._laplacian_var(Path(r["png"])), "feat": f.tolist()})
        if r["pair"] == MAIN_PAIR:
            row["both_ness"] = float((f - origin) @ u2)
            row["which_animal"] = float((f - origin) @ u1)
        scored.append(row)
        _log(f"scored {key}: compose={row['compose']} n={row['n_instances']} sharp={row['sharpness']:.1f}")

    by = {(r["pair"], r["condition"], r["seed"]): r for r in scored}
    # DINOv2 drift: distance to the mono render minus distance to the plain-PoE render, at the
    # same eta (the pathwise read: same seed, same noise path) and against the eta 0 references
    for r in scored:
        f = np.asarray(r["feat"], dtype=np.float32)
        refs = {}
        for tag, eta_ref in (("same_eta", r["eta"]), ("eta0", 0.0)):
            m = by.get((r["pair"], cond_name("mono", eta_ref), r["seed"]))
            p = by.get((r["pair"], cond_name("poe", eta_ref), r["seed"]))
            if m is None or p is None:
                continue
            dm = _cos_dist(f, np.asarray(m["feat"], dtype=np.float32))
            dp = _cos_dist(f, np.asarray(p["feat"], dtype=np.float32))
            refs[f"d_mono_{tag}"] = dm
            refs[f"drift_{tag}"] = dm - dp
        r.update(refs)

    summary = {}
    for slug in PAIRS:
        summary[slug] = {}
        conds = sorted({r["condition"] for r in scored if r["pair"] == slug})
        for c in conds:
            rs = sorted([r for r in scored if r["pair"] == slug and r["condition"] == c], key=lambda r: r["seed"])
            sh = [r["sharpness"] for r in rs]
            cell = {"n": len(rs), "cond": rs[0]["cond"], "eta": rs[0]["eta"], "description": rs[0]["description"],
                    "compose_n": int(sum(r["compose"] for r in rs)), "presence_n": int(sum(r["presence_a"] for r in rs)),
                    "sharpness_mean": float(np.mean(sh)), "sharpness_min": float(min(sh)), "sharpness_max": float(max(sh)),
                    "sharpness_per_seed": {str(r["seed"]): r["sharpness"] for r in rs},
                    "compose_per_seed": {str(r["seed"]): r["compose"] for r in rs},
                    "drift_same_eta_mean": float(np.mean([r["drift_same_eta"] for r in rs if "drift_same_eta" in r] or [np.nan])),
                    "drift_eta0_mean": float(np.mean([r["drift_eta0"] for r in rs if "drift_eta0" in r] or [np.nan])),
                    "d_mono_same_eta_mean": float(np.mean([r["d_mono_same_eta"] for r in rs if "d_mono_same_eta" in r] or [np.nan])),
                    "drift_same_eta_per_seed": {str(r["seed"]): r.get("drift_same_eta") for r in rs}}
            if slug == MAIN_PAIR:
                cell["both_ness_mean"] = float(np.mean([r["both_ness"] for r in rs]))
                cell["both_ness_per_seed"] = {str(r["seed"]): r["both_ness"] for r in rs}
            summary[slug][c] = cell
        s = summary[slug]
        base = s.get(cond_name("corrected", 0.0))
        for eta in ETAS:
            c = cond_name("corrected", eta)
            if c not in s or base is None:
                continue
            per = s[c]["sharpness_per_seed"]
            s[c]["seeds_sharper_than_eta0"] = int(sum(per[k] > base["sharpness_per_seed"][k] for k in per))
            if eta > 0:
                if slug == MAIN_PAIR:
                    s[c]["verdict"] = verdict_for_eta(s[c], base)
                else:
                    s[c]["verdict"] = control_verdict(s[c], s[cond_name("poe", eta)])
        poe0 = s.get(cond_name("poe", 0.0))
        for eta in ETAS:
            c = cond_name("poe", eta)
            if c in s and poe0 is not None and eta > 0:
                gain = s[c]["compose_n"] - poe0["compose_n"]
                s[c]["compose_gain_over_eta0"] = int(gain)
                s[c]["sampler_side_read"] = ("plain PoE gains seeds with noise alone" if gain >= POE_SAMPLER_GAIN_SEEDS
                                             else "plain PoE does not gain seeds with noise alone")
    main = summary[MAIN_PAIR]
    eta_verdicts = {f"{eta:g}": main[cond_name("corrected", eta)].get("verdict") for eta in ETAS if eta > 0
                    and cond_name("corrected", eta) in main}
    if any(v == "support" for v in eta_verdicts.values()):
        plan_verdict = "support"
    elif eta_verdicts and all(v is not None and v.startswith("null") for v in eta_verdicts.values()):
        plan_verdict = "null"
    else:
        plan_verdict = "inconclusive"

    results = {
        "pairs": {k: {"prompt_a": v[0], "prompt_b": v[1], "joint_prompt": v[2]} for k, v in PAIRS.items()},
        "seeds": list(SEEDS), "etas": list(ETAS), "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK,
        "lambda": LAMBDA_FULL, "num_steps": STEPS,
        "scorer": "instance count: GroundingDINO, compose iff count>=2 (context/world/compose-rate.md); on the control pair presence_a is conf('butterfly') >= BUTTERFLY_PRESENT_CONF",
        "sharpness": "Laplacian variance on the 1024 px greyscale render, lambda_window_grid._laplacian_var",
        "drift": "DINOv2 ViT-S/14 CLS cosine distance to the mono render minus distance to the plain-PoE render of the same seed; same_eta uses the references rendered at the cell's eta on the same noise path, eta0 the eta 0 references; negative = nearer the joint-prompt image",
        "both_ness": "projection onto the landing finding's cloud axis, axes rebuilt from " + str(CLOUD_FEATS.relative_to(REPO_ROOT)),
        "reference_both_ness_from_landing_finding": {k: float(np.mean(v)) for k, v in ref_both.items()},
        "noise_path": f"per-step noise from torch.Generator(cpu).manual_seed(seed * {NOISE_SEED_STRIDE} + step), identical across conditions and etas",
        "constants": {"LAMBDA_FULL": LAMBDA_FULL, "ETAS": list(ETAS), "STEPS": STEPS,
                      "SUPPORT_MAX_COMPOSE_LOSS": SUPPORT_MAX_COMPOSE_LOSS, "NULL_MIN_COMPOSE_LOSS": NULL_MIN_COMPOSE_LOSS,
                      "SUPPORT_MIN_SEEDS_SHARPER": SUPPORT_MIN_SEEDS_SHARPER, "NULL_MAX_SEEDS_SHARPER": NULL_MAX_SEEDS_SHARPER,
                      "SUPPORT_MAX_DRIFT_RISE": SUPPORT_MAX_DRIFT_RISE, "POE_SAMPLER_GAIN_SEEDS": POE_SAMPLER_GAIN_SEEDS,
                      "BUTTERFLY_PRESENT_CONF": BUTTERFLY_PRESENT_CONF, "CONTROL_MAX_PRESENCE_LOSS": CONTROL_MAX_PRESENCE_LOSS,
                      "IDENTITY_MAX_MEAN_ABS_DIFF": IDENTITY_MAX_MEAN_ABS_DIFF},
        "rule": ("per eta > 0, corrected against corrected at eta 0 on cat x dog: null if compose loss >= NULL_MIN_COMPOSE_LOSS "
                 "or seeds sharper <= NULL_MAX_SEEDS_SHARPER; support if compose loss <= SUPPORT_MAX_COMPOSE_LOSS and seeds sharper "
                 ">= SUPPORT_MIN_SEEDS_SHARPER and drift_same_eta_mean <= eta 0 value + SUPPORT_MAX_DRIFT_RISE; inconclusive otherwise. "
                 "Plan: support if any eta supports, null if every eta is null, inconclusive otherwise"),
        "eta_verdicts": eta_verdicts, "plan_verdict": plan_verdict,
        "summary": summary, "rows": scored,
    }
    if (OUT_ROOT / "identity_check.json").exists():
        results["identity_check"] = json.loads((OUT_ROOT / "identity_check.json").read_text())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in results.items() if k != "rows"}
    slim["rows"] = [{k: v for k, v in r.items() if k != "feat"} for r in scored]
    (RESULTS_DIR / "results.json").write_text(json.dumps(slim, indent=2))
    _write_table(results)
    sheets = _sheets(results)
    fig = _figure(results)
    _wandb(results, sheets + [fig])
    _log(f"plan verdict: {plan_verdict}; per eta: {eta_verdicts}")
    return results


def _write_table(results: dict) -> None:
    lines = ["# Cell table", "",
             "One row per cell. compose is seeds of 8 with two or more animal instances (cat x dog) or with a butterfly box at conf >= "
             f"{BUTTERFLY_PRESENT_CONF} (control pair); sharpness is Laplacian variance at 1024 px, mean over 8 seeds; sharper is the number "
             "of seeds sharper than the corrected run at eta 0; drift is DINOv2 distance to mono minus distance to plain PoE at the same eta "
             "(negative = nearer the joint-prompt image); both-ness is the mean projection toward the joint-prompt cloud (cat x dog only). "
             "From `results.json` beside this file.", ""]
    for slug, s in results["summary"].items():
        lines += [f"## {slug}", "", "| cell | eta | compose or presence (of 8) | mean sharpness | sharper than eta 0 (of 8) | drift, same eta | both-ness mean | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for c, cell in s.items():
            n = cell["compose_n"] if slug == MAIN_PAIR else cell["presence_n"]
            lines.append(f"| {c} | {cell['eta']:g} | {n} | {cell['sharpness_mean']:.1f} | {cell.get('seeds_sharper_than_eta0', '')} | "
                         f"{cell['drift_same_eta_mean']:.3f} | {cell.get('both_ness_mean', float('nan')):.3f} | "
                         f"{cell.get('verdict', cell.get('sampler_side_read', ''))} |")
        lines.append("")
    lines.append(f"plan verdict: {results['plan_verdict']} (per eta: {results['eta_verdicts']})")
    (RESULTS_DIR / "cell-table.md").write_text("\n".join(lines))


def _sheets(results: dict) -> list[Path]:
    T, pad, cap, left, top = 256, 8, 34, 90, 60
    sheets = []
    rows_by = {(r["pair"], r["condition"], r["seed"]): r for r in results["rows"]}
    out_dir = OUT_ROOT / "sheets"; out_dir.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _sheet(slug, cols, title, name):
        W = left + len(cols) * (T + pad) + pad
        H = top + len(SEEDS) * (T + cap + pad) + pad
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((8, 8), title, fill="black")
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
                drift = f"  drift {r['drift_same_eta']:+.2f}" if "drift_same_eta" in r else ""
                dr.text((x0, y0 + T + 2), f"{tag}  sharp {r['sharpness']:.0f}{drift}", fill="black")
        p = out_dir / name
        im.save(p); im.save(RESULTS_DIR / name); sheets.append(p)

    for slug, s in results["summary"].items():
        for eta in ETAS:
            if eta == 0 or cond_name("corrected", eta) not in s:
                continue
            c = cond_name("corrected", eta)
            cols = [(cond_name("mono", eta), f"Mono, eta {eta:g}"), (cond_name("poe", eta), f"plain PoE, eta {eta:g}"),
                    (cond_name("corrected", 0.0), "corrected, eta 0"), (c, f"corrected, eta {eta:g}")]
            _sheet(slug, cols, f"{slug}   eta {eta:g}: {s[c]['description']}   verdict: {s[c].get('verdict', '')}",
                   f"sheet-{slug}-eta{eta:g}.png")
        cols = [(cond_name("mono", 0.0), "Mono, eta 0")] + [(cond_name("corrected", e), f"corrected, eta {e:g}") for e in ETAS]
        _sheet(slug, cols, f"{slug}   the corrected run across eta, same seed and same noise path per row", f"sheet-{slug}-across-eta.png")
    _log(f"wrote {len(sheets)} sheets")
    return sheets


def _figure(results: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = results["summary"][MAIN_PAIR]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"poe": "#111111", "corrected": "#e7298a", "mono": "#7570b3"}
    # left: sharpness per seed against eta, corrected and plain PoE
    ax = axes[0]
    for cond in ("poe", "corrected"):
        for seed in SEEDS:
            ys = [s[cond_name(cond, e)]["sharpness_per_seed"][str(seed)] for e in ETAS if cond_name(cond, e) in s]
            ax.plot(ETAS[:len(ys)], ys, color=colors[cond], lw=0.8, alpha=0.35)
        ys = [s[cond_name(cond, e)]["sharpness_mean"] for e in ETAS if cond_name(cond, e) in s]
        ax.plot(ETAS[:len(ys)], ys, color=colors[cond], lw=2.6, marker="o", label=f"{cond} (mean of 8, thin = one seed)")
    ax.set_yscale("log"); ax.set_xlabel("eta (0 = deterministic DDIM, 1 = ancestral)")
    ax.set_ylabel("sharpness: Laplacian variance at 1024 px (log)"); ax.legend(fontsize=8); ax.set_title("sharpness against eta, cat x dog", fontsize=10)
    # middle: DINOv2 drift against eta
    ax = axes[1]
    for cond in ("poe", "corrected"):
        for seed in SEEDS:
            ys = [s[cond_name(cond, e)]["drift_same_eta_per_seed"][str(seed)] for e in ETAS if cond_name(cond, e) in s]
            ax.plot(ETAS[:len(ys)], ys, color=colors[cond], lw=0.8, alpha=0.35)
        ys = [s[cond_name(cond, e)]["drift_same_eta_mean"] for e in ETAS if cond_name(cond, e) in s]
        ax.plot(ETAS[:len(ys)], ys, color=colors[cond], lw=2.6, marker="o", label=cond)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("eta"); ax.set_ylabel("DINOv2 drift: d(cell, mono) - d(cell, plain PoE), same eta\n(negative = nearer the joint-prompt image)")
    ax.legend(fontsize=8); ax.set_title("fidelity to the joint-prompt render against eta", fontsize=10)
    # right: compose count against eta
    ax = axes[2]
    for cond in ("poe", "corrected", "mono"):
        ys = [s[cond_name(cond, e)]["compose_n"] for e in ETAS if cond_name(cond, e) in s]
        ax.plot(ETAS[:len(ys)], ys, color=colors[cond], lw=2.6, marker="o", label=cond)
    ax.set_ylim(-0.3, 8.3); ax.set_xlabel("eta"); ax.set_ylabel("seeds of 8 with two or more animals counted")
    ax.legend(fontsize=8); ax.set_title("compose count against eta", fontsize=10)
    fig.suptitle(f"cat x dog, seeds 9 to 16, rank-{LORA_RANK} step-30050 correction at lambda {LAMBDA_FULL}, 50 DDIM steps; "
                 f"plan verdict: {results['plan_verdict']}", fontsize=10)
    fig.tight_layout()
    p = RESULTS_DIR / "sharpness-drift-and-compose-vs-eta.png"
    fig.savefig(p, dpi=160)
    return p


def _wandb(results: dict, images: list[Path]) -> None:
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    idf = OUT_ROOT / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else None
    run = wandb.init(entity=entity, project=project, name="stochastic_sampler_sweep_r32_030050",
                     id=run_id, resume="allow", config=results["constants"],
                     tags=["plan-19", "scope-01", "stochastic-sampler"])
    idf.write_text(run.id)
    log = {f"images/{p.stem}": wandb.Image(str(p)) for p in images}
    cols = ["pair", "cell", "eta", "compose_or_presence_n", "sharpness_mean", "seeds_sharper_than_eta0", "drift_same_eta_mean", "both_ness_mean", "verdict"]
    tab = wandb.Table(columns=cols)
    for slug, s in results["summary"].items():
        for c, cell in s.items():
            n = cell["compose_n"] if slug == MAIN_PAIR else cell["presence_n"]
            tab.add_data(slug, c, cell["eta"], n, cell["sharpness_mean"], cell.get("seeds_sharper_than_eta0"),
                         cell["drift_same_eta_mean"], cell.get("both_ness_mean"), cell.get("verdict", cell.get("sampler_side_read", "")))
            log[f"{slug}/{c}/compose_n"] = cell["compose_n"]
            log[f"{slug}/{c}/sharpness_mean"] = cell["sharpness_mean"]
            log[f"{slug}/{c}/drift_same_eta_mean"] = cell["drift_same_eta_mean"]
    log["cells"] = tab
    run.log(log)
    art = wandb.Artifact("stochastic_sampler_sweep_results", type="results")
    art.add_file(str(RESULTS_DIR / "results.json"))
    if (OUT_ROOT / "identity_check.json").exists():
        art.add_file(str(OUT_ROOT / "identity_check.json"))
    run.log_artifact(art)
    _log(f"wandb run id {run.id} url {run.url}")
    results.setdefault("wandb", {})["run_id"] = run.id
    results["wandb"]["url"] = run.url
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    slim = json.loads((RESULTS_DIR / "results.json").read_text()); slim["wandb"] = results["wandb"]
    (RESULTS_DIR / "results.json").write_text(json.dumps(slim, indent=2))
    run.finish()


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--etas", default=",".join(f"{e:g}" for e in ETAS))
    a = ap.parse_args(argv)
    if not (a.render or a.score):
        ap.error("pass --render and/or --score")
    if a.render:
        render(seeds=[int(s) for s in a.seeds.split(",")], pairs=[p for p in a.pairs.split(",")],
               etas=[float(e) for e in a.etas.split(",")])
    if a.score:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
