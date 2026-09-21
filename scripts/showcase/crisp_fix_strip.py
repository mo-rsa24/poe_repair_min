#!/usr/bin/env python
"""Three inference-time fixes for the softness of the corrected render (plan 15, 01-showcase-the-trained-lora).

The rank-32 adapter at step 30050 composes cat x dog on most held-out seeds, and the render comes
out softer and more washed than plain product-of-experts (PoE). Three published fixes for the same
symptom under classifier-free guidance are tried on the correction, each at the shipped setting
(lambda 1.2 on all 50 steps), so the only thing that changes against the full-window run is the fix:

  cfgpp     CFG++ (Chung et al. 2024, arXiv 2406.08070): the corrected prediction forms the running
            estimate x0, and the DDIM re-noising step uses the plain PoE prediction instead. The
            correction steers where the image goes; the noise put back is the model's own.
  apg       APG (Sadat et al. 2025, arXiv 2410.02416): the correction is split into the part parallel
            to the PoE prediction and the part orthogonal to it; the parallel part is scaled by
            APG_ETA (0 drops it). The orthogonal part carries the composition; the parallel part
            changes the prediction's norm, which is the damping the correction finding measured.
  normkeep  the norm argument on its own: the corrected prediction is rescaled to the PoE
            prediction's norm at every step, direction unchanged.

The reference columns Mono (joint prompt), plain PoE and the full window lambda 1.2 are plan 14's
renders of the same seeds through the same sampler, copied in; an identity render proves the new
code path equals plan 14's when the fix is switched off.

Stages: --identity (1 render), --render (24 renders), --score, --strips; --all runs them in order.
Per-step diagnostics are recorded on every render: ||delta||, the parallel share of delta, and the
norm ratio of the prediction actually used to the PoE prediction.

Thresholds live here as named constants so a change shows up in a diff.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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

import correct_early_then_clean_up as ce  # noqa: E402  plan 14's sampler, scorer helpers and constants

# ----------------------------------------------------------------------------- constants
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/crisp_fix_strip")
RESULTS_DIR = REPO_ROOT / "artifacts/results/does-an-inference-time-fix-make-the-corrected-render-crisp"
PLAN14_ROOT = ce.OUT_ROOT                    # where the reference renders come from
CHECKPOINT = ce.CHECKPOINT                   # rank 32, step 30050
LORA_RANK = ce.LORA_RANK
WANDB_PROJECT = ce.WANDB_PROJECT
WANDB_RUN_NAME = "crisp_fix_strip_r32_030050"
SEEDS = ce.SEEDS
PAIR = ce.MAIN_PAIR                          # cat x dog only
PROMPT_A, PROMPT_B, PROMPT_J, QUERY_A, QUERY_B = ce.PAIRS[PAIR]
GUIDANCE_SCALE = ce.GUIDANCE_SCALE
STEPS = ce.STEPS_MAIN
LAMBDA = ce.LAMBDA_FULL                      # 1.2, the shipped setting and plan 14's full window
EULER_INIT_NOISE_SIGMA = ce.EULER_INIT_NOISE_SIGMA

REFERENCE_CONDITIONS = ("mono", "poe", "full_1.2")
FIXES = ("cfgpp", "apg", "normkeep")
BASELINE = "full_1.2_sameloop"                # the full window rendered through this loop with the hook off: the bar's reference
APG_ETA = 0.0                                # weight on the parallel part; APG's recommended default

# the bar (review file, "The question written before the run"); per-seed paired reads because the
# plain band is set by three sketch seeds and a mean-against-band rule reads nothing on this pair
SUPPORT_MAX_COMPOSE_LOSS = 1                 # seeds of 8 a fix may lose against the full window and still support
NULL_MIN_COMPOSE_LOSS = 2                    # losing this many is a null
SUPPORT_MIN_SHARPER_SEEDS = 6                # of 8: the fix is sharper than the full window on this many seeds
NULL_MAX_SHARPER_SEEDS = 4                   # of 8: at or under this, the fix gained nothing
IDENTITY_MAX_MEAN_ABS_DIFF = ce.DETACH_MAX_MEAN_ABS_DIFF   # 1.0 grey levels: same process, same device, same code

STRIP_TILE = 300
STRIP_COLUMNS = (("mono", "Mono: the joint prompt"), ("poe", "plain PoE"),
                 (BASELINE, f"PoE + {LAMBDA} x correction, full window"),
                 ("cfgpp", "fix: CFG++ re-noising"), ("apg", f"fix: APG, parallel part x {APG_ETA:g}"),
                 ("normkeep", "fix: PoE norm kept"))
DESCRIPTIONS = {
    "mono": "the joint prompt, plain CFG (plan 14's render)",
    "poe": "plain PoE, no correction (plan 14's render)",
    "full_1.2": f"PoE + {LAMBDA} x correction on all steps (plan 14's render)",
    BASELINE: f"PoE + {LAMBDA} x correction on all steps, this loop with the hook off (the bar's reference)",
    "cfgpp": f"PoE + {LAMBDA} x correction forms x0; DDIM re-noises with the plain PoE prediction",
    "apg": f"PoE + {LAMBDA} x (correction with its part parallel to the PoE prediction scaled by {APG_ETA:g})",
    "normkeep": f"PoE + {LAMBDA} x correction, then rescaled to the PoE prediction's norm each step",
}


def _log(msg: str) -> None:
    print(f"[crisp_fix {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _render_path(cond: str, seed: int) -> Path:
    return OUT_ROOT / "renders" / cond / f"seed_{seed}.png"


def _manifest_path() -> Path:
    return OUT_ROOT / "render_manifest.json"


def _load_manifest() -> list[dict]:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else []


def _save_manifest(rows: list[dict]) -> None:
    _manifest_path().write_text(json.dumps(rows, indent=2))


def _run_header() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# ============================================================================= the fixes
def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() * b.float()).flatten(1).sum(dim=1)


def _norm(a: torch.Tensor) -> torch.Tensor:
    return a.float().flatten(1).norm(dim=1)


def apply_fix(fix: str | None, eps_poe: torch.Tensor, eps_lora: torch.Tensor, lam: float):
    """Returns (eps_for_x0, eps_for_renoise, diagnostics). fix None is plan 14's full window."""
    delta = (eps_lora.float() - eps_poe.float())
    par_coef = _dot(delta, eps_poe) / _dot(eps_poe, eps_poe).clamp_min(1e-12)
    par = par_coef.view(-1, 1, 1, 1) * eps_poe.float()
    orth = delta - par
    diag = {"delta_norm": float(_norm(delta)[0]), "poe_norm": float(_norm(eps_poe)[0]),
            "parallel_share": float((_norm(par)[0] / _norm(delta)[0].clamp_min(1e-12))),
            "parallel_coef": float(par_coef[0])}
    if fix is None:
        eps_t = eps_poe.float() + lam * delta
        eps_re = eps_t
    elif fix == "cfgpp":
        eps_t = eps_poe.float() + lam * delta
        eps_re = eps_poe.float()
    elif fix == "apg":
        eps_t = eps_poe.float() + lam * (orth + APG_ETA * par)
        eps_re = eps_t
    elif fix == "normkeep":
        raw = eps_poe.float() + lam * delta
        scale = (_norm(eps_poe) / _norm(raw).clamp_min(1e-12)).view(-1, 1, 1, 1)
        eps_t = raw * scale
        eps_re = eps_t
        diag["normkeep_scale"] = float(scale[0])
    else:
        raise KeyError(fix)
    diag["used_norm_ratio"] = float(_norm(eps_t)[0] / _norm(eps_poe)[0].clamp_min(1e-12))
    return eps_t.to(eps_poe.dtype), eps_re.to(eps_poe.dtype), diag


class FixSampler(ce.Sampler):
    """Plan 14's sampler with the fix hook between the two predictions and the DDIM step."""

    @torch.no_grad()
    def fixed(self, seed: int, fix: str | None):
        from poe_repair._sdxl.metrics import ddim_prev_from_x0_eps, guided_eps, poe_eps, tweedie_mean
        from poe_repair._sdxl.runtime import LatentTrajectoryCollector, decode_latents
        from poe_repair.methods._sampling import SamplerOutputs, add_time_ids
        import lambda_boundary_probe as lbp

        assert self.attached, "attach the adapter first"
        ctx = self.ctx
        device, dtype = ctx.device, ctx.dtype
        scheduler = ctx.scheduler
        scheduler.set_timesteps(STEPS)
        timesteps = scheduler.timesteps
        latents = (self.init_latents(PAIR, seed) / EULER_INIT_NOISE_SIGMA).to(device=device, dtype=dtype)
        tracker = LatentTrajectoryCollector(STEPS, 1, latents.shape[1], latents.shape[2], latents.shape[3])
        (sa, pa), (sb, pb) = self.prompts[PAIR]["a"], self.prompts[PAIR]["b"]
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

        diags = []
        for step_index in range(STEPS):
            timestep = timesteps[step_index]
            t = int(timestep.item())
            _disable()
            ea, eb, eu = _forward(timestep)
            eps_poe = poe_eps(guided_eps(ea, eu, GUIDANCE_SCALE), guided_eps(eb, eu, GUIDANCE_SCALE), eu)
            _enable()
            la, lb, lu = _forward(timestep)
            eps_lora = poe_eps(guided_eps(la, lu, GUIDANCE_SCALE), guided_eps(lb, lu, GUIDANCE_SCALE), lu)
            eps_t, eps_re, diag = apply_fix(fix, eps_poe, eps_lora, LAMBDA)
            diag.update({"step": step_index, "t": t})
            diags.append(diag)
            ab = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
            x0 = tweedie_mean(latents, ab, eps_t)
            tracker.store_step(step_index, latents, eps_t, float(step_index) / float(STEPS), t)
            latents = ddim_prev_from_x0_eps(scheduler=scheduler, timestep=timestep, step_index=step_index,
                                            x0=x0, eps=eps_re)
        _disable()
        tracker.store_final(latents)
        image = decode_latents(ctx.models, latents).cpu()
        return SamplerOutputs(latents=latents, image=image, tracker=tracker,
                              extras={"fix": fix, "lambda": LAMBDA, "num_steps": STEPS, "diagnostics": diags})


# ============================================================================= stages
def copy_references() -> list[dict]:
    """Plan 14's mono, plain PoE and full-window renders of the same seeds, copied in with their source."""
    src_rows = json.loads((PLAN14_ROOT / "render_manifest.json").read_text())
    rows = _load_manifest()
    done = {(r["condition"], r["seed"]) for r in rows}
    for cond in REFERENCE_CONDITIONS:
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            src = next(r for r in src_rows if r["pair"] == PAIR and r["condition"] == cond and r["seed"] == seed)
            dst = _render_path(cond, seed)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src["png"], dst)
            rows.append({"condition": cond, "seed": seed, "png": str(dst), "source_png": src["png"],
                         "num_steps": src["num_steps"], "description": DESCRIPTIONS[cond],
                         "lambda_per_step": src.get("lambda_per_step")})
            done.add((cond, seed))
    _save_manifest(rows)
    _log(f"references in place: {sum(1 for r in rows if r['condition'] in REFERENCE_CONDITIONS)} of {3 * len(SEEDS)}")
    return rows


def identity_check(S: FixSampler) -> dict:
    """fix=None through the new loop on seed 9 must equal plan 14's full_1.2 render of seed 9."""
    d = OUT_ROOT / "identity_check"; d.mkdir(parents=True, exist_ok=True)
    out = S.fixed(9, None)
    mine = d / "full_1.2_seed_9_through_fix_sampler.png"
    ce._write(out.image, mine)
    theirs = ce._render_path(PAIR, "full_1.2", 9)
    res = {**_run_header(), "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK,
           "mean_abs_diff_vs_plan14_full_1.2": ce.mean_abs_diff(mine, theirs),
           "IDENTITY_MAX_MEAN_ABS_DIFF": IDENTITY_MAX_MEAN_ABS_DIFF,
           "plan14_render": str(theirs), "this_render": str(mine),
           "note": "plan 14's render came from mscluster108 device 1; a run on another card sees fp16 drift of about 2 grey levels and the bar then reads against IDENTITY_MAX_MEAN_ABS_DIFF_CROSS_DEVICE"}
    res["IDENTITY_MAX_MEAN_ABS_DIFF_CROSS_DEVICE"] = ce.IDENTITY_MAX_MEAN_ABS_DIFF
    res["pass_same_device"] = res["mean_abs_diff_vs_plan14_full_1.2"] <= IDENTITY_MAX_MEAN_ABS_DIFF
    res["pass_cross_device"] = res["mean_abs_diff_vs_plan14_full_1.2"] <= ce.IDENTITY_MAX_MEAN_ABS_DIFF
    (OUT_ROOT / "identity_check.json").write_text(json.dumps(res, indent=2))
    _log(f"identity check: {res['mean_abs_diff_vs_plan14_full_1.2']:.3f} grey levels "
         f"(same-device bar {IDENTITY_MAX_MEAN_ABS_DIFF}, cross-device bar {ce.IDENTITY_MAX_MEAN_ABS_DIFF})")
    return res


def render(S: FixSampler, seeds=SEEDS, conditions=FIXES) -> None:
    rows = _load_manifest()
    done = {(r["condition"], r["seed"]) for r in rows}
    _log(f"render header={_run_header()} conditions={conditions}")
    for fix in conditions:
        for seed in seeds:
            if (fix, seed) in done:
                continue
            t0 = time.time()
            out = S.fixed(seed, None if fix == BASELINE else fix)
            p = _render_path(fix, seed)
            ce._write(out.image, p)
            diags = out.extras["diagnostics"]
            (OUT_ROOT / "renders" / fix / f"seed_{seed}_diagnostics.json").write_text(json.dumps(diags, indent=2))
            rows.append({"condition": fix, "seed": seed, "png": str(p), "num_steps": STEPS,
                         "description": DESCRIPTIONS[fix], "elapsed_s": round(time.time() - t0, 1),
                         "lambda": LAMBDA, "apg_eta": APG_ETA if fix == "apg" else None,
                         "hook": None if fix == BASELINE else fix,
                         "diag_mean": {k: float(np.mean([d[k] for d in diags])) for k in ("delta_norm", "parallel_share", "used_norm_ratio")},
                         "diag_early_mean_steps_0_9": {k: float(np.mean([d[k] for d in diags[:10]])) for k in ("parallel_share", "used_norm_ratio")}})
            done.add((fix, seed)); _save_manifest(rows)
            _log(f"{fix} seed={seed} {rows[-1]['elapsed_s']}s parallel_share={rows[-1]['diag_mean']['parallel_share']:.3f} "
                 f"norm_ratio={rows[-1]['diag_mean']['used_norm_ratio']:.3f} -> {p.name}")
    _log("render done")


def verdict_for_fix(cell: dict, full: dict) -> str:
    loss = full["compose_n"] - cell["compose_n"]
    if loss >= NULL_MIN_COMPOSE_LOSS:
        return "null: the animals go with the fix"
    if cell["sharper_than_full_n"] <= NULL_MAX_SHARPER_SEEDS:
        return "null: no sharpness gained over the full window"
    if loss <= SUPPORT_MAX_COMPOSE_LOSS and cell["sharper_than_full_n"] >= SUPPORT_MIN_SHARPER_SEEDS:
        return "support"
    return "inconclusive"


def score() -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances)
    from scripts.build_lora_inspector_mds_semantic import DinoEmbedder

    rows = _load_manifest()
    if not rows:
        raise SystemExit("nothing rendered yet")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origin, u1, u2, ref_both = ce._cloud_axes()
    embedder = DinoEmbedder(device=dev)   # GPU: on CPU DINOv2 hits the CUDA-only xformers kernel (poe-mem-002)

    def _embed(p: Path) -> np.ndarray:
        im = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        f = embedder.embed_decoded_batch(im)[0]
        return f / (np.linalg.norm(f) + 1e-12)

    feats = {}
    scored = []
    for r in rows:
        inst = instance_score_to_dict(score_output_instances(Path(r["png"]), QUERY_A, QUERY_B, device=dev))
        row = dict(r)
        f = _embed(Path(r["png"]))
        feats[(r["condition"], r["seed"])] = f
        row.update({"compose": inst["label"] == "compose", "n_instances": inst["n_instances"],
                    "conf_a": inst["conf_a"], "conf_b": inst["conf_b"],
                    "sharpness": ce._laplacian_var(Path(r["png"])),
                    "both_ness": float((f - origin) @ u2), "which_animal": float((f - origin) @ u1)})
        scored.append(row)
    for row in scored:
        fm = feats[("mono", row["seed"])]
        fp = feats[("poe", row["seed"])]
        row["d_mono"] = float(1.0 - feats[(row["condition"], row["seed"])] @ fm)   # DINOv2 cosine distance to the joint-prompt render
        row["d_poe"] = float(1.0 - feats[(row["condition"], row["seed"])] @ fp)
        _log(f"scored {row['condition']} seed={row['seed']}: compose={row['compose']} n={row['n_instances']} "
             f"sharp={row['sharpness']:.1f} both={row['both_ness']:.3f} d_mono={row['d_mono']:.3f}")

    summary = {}
    conds = [c for c in (*REFERENCE_CONDITIONS, BASELINE, *FIXES) if any(r["condition"] == c for r in scored)]
    for c in conds:
        rs = sorted([r for r in scored if r["condition"] == c], key=lambda r: r["seed"])
        summary[c] = {"n": len(rs), "description": rs[0]["description"],
                      "compose_n": sum(r["compose"] for r in rs),
                      "compose_per_seed": {str(r["seed"]): r["compose"] for r in rs},
                      "n_instances_per_seed": {str(r["seed"]): r["n_instances"] for r in rs},
                      "sharpness_mean": float(np.mean([r["sharpness"] for r in rs])),
                      "sharpness_per_seed": {str(r["seed"]): r["sharpness"] for r in rs},
                      "both_ness_mean": float(np.mean([r["both_ness"] for r in rs])),
                      "d_mono_mean": float(np.mean([r["d_mono"] for r in rs])),
                      "d_mono_per_seed": {str(r["seed"]): r["d_mono"] for r in rs}}
        if c in (*FIXES, BASELINE):
            summary[c]["diag_mean"] = {k: float(np.mean([r["diag_mean"][k] for r in rs])) for k in rs[0]["diag_mean"]}
            summary[c]["diag_early_mean_steps_0_9"] = {k: float(np.mean([r["diag_early_mean_steps_0_9"][k] for r in rs])) for k in rs[0]["diag_early_mean_steps_0_9"]}
    reference = BASELINE if BASELINE in summary else "full_1.2"
    full = summary[reference]
    # the fp16 floor: plan 14's full window against the same configuration through this loop, per seed
    floor = {str(seed): ce.mean_abs_diff(_render_path("full_1.2", seed), _render_path(BASELINE, seed))
             for seed in SEEDS if _render_path(BASELINE, seed).exists()}
    if floor:
        summary[BASELINE]["sharper_than_plan14_full_n"] = sum(
            summary[BASELINE]["sharpness_per_seed"][k] > summary["full_1.2"]["sharpness_per_seed"][k] for k in floor)
        summary[BASELINE]["compose_agrees_with_plan14_n"] = sum(
            summary[BASELINE]["compose_per_seed"][k] == summary["full_1.2"]["compose_per_seed"][k] for k in floor)
    for c in FIXES:
        if c not in summary:
            continue
        s = summary[c]
        keys = s["sharpness_per_seed"]
        s["sharper_than_full_n"] = sum(keys[k] > full["sharpness_per_seed"][k] for k in keys)
        s["sharper_than_poe_n"] = sum(keys[k] > summary["poe"]["sharpness_per_seed"][k] for k in keys)
        s["kept_full_window_seeds_n"] = sum(s["compose_per_seed"][k] for k in keys if full["compose_per_seed"][k])
        s["nearer_mono_than_full_n"] = sum(s["d_mono_per_seed"][k] < full["d_mono_per_seed"][k] for k in keys)
        s["verdict"] = verdict_for_fix(s, full)
    results = {
        "pair": PAIR, "prompts": {"a": PROMPT_A, "b": PROMPT_B, "joint": PROMPT_J},
        "seeds": list(SEEDS), "checkpoint": str(CHECKPOINT), "lora_rank": LORA_RANK, "lambda": LAMBDA,
        "scorer": "instance count: GroundingDINO query 'animal', NMS iou<0.5, conf>=0.30, compose iff count>=2 (context/world/compose-rate.md)",
        "sharpness": "Laplacian variance on the 1024 px greyscale render, lambda_window_grid._laplacian_var; compared per seed, never as a mean against a band",
        "d_mono": "DINOv2 ViT-S/14 CLS cosine distance from the render to the same seed's joint-prompt render (lower = nearer Mono)",
        "both_ness": "projection onto the landing finding's cloud axis; axes rebuilt from " + str(ce.CLOUD_FEATS.relative_to(REPO_ROOT)),
        "reference_full_window": reference,
        "fp16_floor_grey_levels": {"per_seed": floor, "mean": float(np.mean(list(floor.values()))) if floor else None,
                                   "what": "mean absolute pixel difference between plan 14's full-window render and the same configuration through this loop with the hook off; the two differ only in fp32-versus-fp16 combination of the two predictions"},
        "constants": {"LAMBDA": LAMBDA, "APG_ETA": APG_ETA, "STEPS": STEPS,
                      "SUPPORT_MAX_COMPOSE_LOSS": SUPPORT_MAX_COMPOSE_LOSS, "NULL_MIN_COMPOSE_LOSS": NULL_MIN_COMPOSE_LOSS,
                      "SUPPORT_MIN_SHARPER_SEEDS": SUPPORT_MIN_SHARPER_SEEDS, "NULL_MAX_SHARPER_SEEDS": NULL_MAX_SHARPER_SEEDS,
                      "IDENTITY_MAX_MEAN_ABS_DIFF": IDENTITY_MAX_MEAN_ABS_DIFF},
        "rule": ("the full window is the same-loop render (hook off) when present, else plan 14's; "
                 "support if compose_n >= full-window compose_n - SUPPORT_MAX_COMPOSE_LOSS and the fix is sharper than the full window "
                 "on >= SUPPORT_MIN_SHARPER_SEEDS of 8 seeds (paired per seed); null if compose loss >= NULL_MIN_COMPOSE_LOSS or "
                 "sharper on <= NULL_MAX_SHARPER_SEEDS seeds; inconclusive otherwise. nearer_mono_than_full_n and both_ness are "
                 "descriptive reads beside the bar, not part of it"),
        "summary": summary, "rows": scored,
    }
    if (OUT_ROOT / "identity_check.json").exists():
        results["identity_check"] = json.loads((OUT_ROOT / "identity_check.json").read_text())
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    _write_table(results)
    diag_fig = _diagnostics_figure(results)
    _wandb(results, diag_fig)
    return results


def _write_table(results: dict) -> None:
    s = results["summary"]
    lines = ["# Cell table", "",
             "One row per condition, cat x dog, held-out seeds 9 to 16, rank 32 adapter at step 30050, lambda "
             f"{LAMBDA} on all 50 steps. compose is seeds of 8 with two or more animal instances; sharpness is Laplacian variance "
             "at 1024 px (mean, and the paired per-seed count against the full window); d(Mono) is the DINOv2 cosine distance to the "
             "same seed's joint-prompt render, mean over seeds; both-ness is the projection toward the joint-prompt cloud. "
             "From `results.json` beside this file.", "",
             "| condition | compose (of 8) | kept the full window's seeds | sharpness mean | sharper than full window (seeds of 8) | d(Mono) mean | nearer Mono than full window (seeds of 8) | both-ness mean | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, cell in s.items():
        lines.append(f"| {c} | {cell['compose_n']} | {cell.get('kept_full_window_seeds_n', '')} | {cell['sharpness_mean']:.1f} | "
                     f"{cell.get('sharper_than_full_n', '')} | {cell['d_mono_mean']:.3f} | {cell.get('nearer_mono_than_full_n', '')} | "
                     f"{cell['both_ness_mean']:.3f} | {cell.get('verdict', '')} |")
    lines += ["", f"The bar's full-window reference is `{results['reference_full_window']}`; the fp16 floor between plan 14's render and the same-loop render is "
              f"{(results['fp16_floor_grey_levels']['mean'] or float('nan')):.2f} grey levels mean over seeds.", "",
              "Per-step diagnostics, mean over steps and seeds (the same numbers are recorded on every fix render, and are identical across fixes "
              "for the two raw predictions since the fixes only change what is done with them):", "",
              "| fix | mean ||delta|| | parallel share of delta (all steps) | parallel share (steps 0 to 9) | norm of the prediction used / norm of PoE (all steps) | same, steps 0 to 9 |",
              "|---|---|---|---|---|---|"]
    for c in (BASELINE, *FIXES):
        if c in s:
            dm, de = s[c]["diag_mean"], s[c]["diag_early_mean_steps_0_9"]
            lines.append(f"| {c} | {dm['delta_norm']:.2f} | {dm['parallel_share']:.3f} | {de['parallel_share']:.3f} | {dm['used_norm_ratio']:.3f} | {de['used_norm_ratio']:.3f} |")
    (RESULTS_DIR / "cell-table.md").write_text("\n".join(lines) + "\n")


def _diagnostics_figure(results: dict) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_fix = {}
    for fix in (BASELINE, *FIXES):
        ds = []
        for seed in SEEDS:
            p = OUT_ROOT / "renders" / fix / f"seed_{seed}_diagnostics.json"
            if p.exists():
                ds.append(json.loads(p.read_text()))
        if ds:
            per_fix[fix] = ds
    if not per_fix:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # panel 1: the parallel share of the correction, on the full-window path (taken from the cfgpp run,
    # whose x_t path is the closest to the full window's since only the re-noise term differs)
    ref = per_fix.get(BASELINE) or per_fix.get("cfgpp") or next(iter(per_fix.values()))
    steps = [d["step"] for d in ref[0]]
    for ds in ref:
        axes[0].plot(steps, [d["parallel_share"] for d in ds], color="#888", lw=0.7, alpha=0.5)
    axes[0].plot(steps, np.mean([[d["parallel_share"] for d in ds] for ds in ref], axis=0), color="black", lw=2.4, label="mean of 8 seeds")
    axes[0].set_title("share of the correction parallel to the PoE prediction\n||parallel part|| / ||correction||, per step (thin: one seed)", fontsize=10)
    axes[0].set_xlabel("denoising step (0 = noise, 49 = image)"); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8)
    for ds in ref:
        axes[1].plot(steps, [d["parallel_coef"] for d in ds], color="#888", lw=0.7, alpha=0.5)
    axes[1].plot(steps, np.mean([[d["parallel_coef"] for d in ds] for ds in ref], axis=0), color="black", lw=2.4)
    axes[1].axhline(0, color="#bbb", lw=0.8)
    axes[1].set_title("sign and size of the parallel part\n<correction, PoE> / ||PoE||^2 (negative = the correction damps PoE)", fontsize=10)
    axes[1].set_xlabel("denoising step")
    colors = {BASELINE: "#111111", "cfgpp": "#1b9e77", "apg": "#d95f02", "normkeep": "#7570b3"}
    for fix, ds in per_fix.items():
        axes[2].plot(steps, np.mean([[d["used_norm_ratio"] for d in one] for one in ds], axis=0), color=colors[fix], lw=2.2,
                     label="full window (hook off)" if fix == BASELINE else fix)
    axes[2].axhline(1.0, color="#bbb", lw=0.8)
    axes[2].set_title("norm of the prediction used for x0 / norm of the PoE prediction\nmean of 8 seeds, per fix (1 = PoE's own norm)", fontsize=10)
    axes[2].set_xlabel("denoising step"); axes[2].legend(fontsize=8)
    fig.suptitle(f"cat x dog, seeds 9 to 16, rank 32 step 30050, lambda {LAMBDA}: what the correction does to the prediction, per step", fontsize=11)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "correction-parallel-share-and-norm-per-step.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ============================================================================= strips
def _strips(results: dict) -> dict:
    T, pad, cap, top = STRIP_TILE, 10, 62, 30
    rows_by = {(r["condition"], r["seed"]): r for r in results["rows"]}
    out_dir = OUT_ROOT / "strips"; out_dir.mkdir(exist_ok=True)
    (RESULTS_DIR / "strips").mkdir(parents=True, exist_ok=True)
    paths = {}
    for seed in SEEDS:
        W = pad + len(STRIP_COLUMNS) * (T + pad)
        H = top + T + cap + pad
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((pad, 8), f"cat x dog   seed {seed}   rank {LORA_RANK} adapter at step 30050, lambda {LAMBDA} on all 50 steps", fill="black")
        for j, (cc, lab) in enumerate(STRIP_COLUMNS):
            x0 = pad + j * (T + pad); y0 = top
            r = rows_by.get((cc, seed))
            if r is None:
                dr.rectangle([x0, y0, x0 + T, y0 + T], outline="#999")
                dr.text((x0, y0 + T + 4), lab, fill="black"); dr.text((x0, y0 + T + 20), "not rendered", fill="#999")
                continue
            im.paste(Image.open(r["png"]).convert("RGB").resize((T, T), Image.LANCZOS), (x0, y0))
            color = (20, 130, 40) if r["compose"] else (190, 30, 30)
            dr.rectangle([x0, y0, x0 + T - 1, y0 + T - 1], outline=color, width=3)
            dr.text((x0, y0 + T + 4), lab, fill="black")
            dr.text((x0, y0 + T + 20), f"animals counted {r['n_instances']}   sharpness {r['sharpness']:.0f}", fill=color)
            dr.text((x0, y0 + T + 34), f"d(Mono) {r['d_mono']:.3f}   both-ness {r['both_ness']:.2f}", fill="black")
            if cc in FIXES:
                dr.text((x0, y0 + T + 48), (results["summary"][cc].get("verdict", "") or "")[:48], fill="#444")
        p = out_dir / f"strip-seed_{seed}.png"
        im.save(p); im.save(RESULTS_DIR / "strips" / p.name)
        paths[seed] = p
    _log(f"wrote {len(paths)} strips")
    return paths


def _wandb_run():
    import wandb
    entity, project = WANDB_PROJECT.split("/")
    idf = OUT_ROOT / "wandb_run_id.txt"
    run_id = idf.read_text().strip() if idf.exists() else None
    run = wandb.init(entity=entity, project=project, name=WANDB_RUN_NAME, id=run_id, resume="allow",
                     tags=["plan-15", "scope-01", "crisp-fix", "cfgpp", "apg"])
    idf.write_text(run.id)
    return run


def _wandb(results: dict, diag_fig: Path | None) -> None:
    import wandb
    run = _wandb_run()
    run.config.update(results["constants"], allow_val_change=True)
    log = {}
    cols = ["condition", "compose_n", "kept_full_window_seeds_n", "sharpness_mean", "sharper_than_full_n",
            "d_mono_mean", "nearer_mono_than_full_n", "both_ness_mean", "verdict"]
    tab = wandb.Table(columns=cols)
    for c, cell in results["summary"].items():
        tab.add_data(c, cell["compose_n"], cell.get("kept_full_window_seeds_n"), cell["sharpness_mean"],
                     cell.get("sharper_than_full_n"), cell["d_mono_mean"], cell.get("nearer_mono_than_full_n"),
                     cell["both_ness_mean"], cell.get("verdict", ""))
        for k in ("compose_n", "sharpness_mean", "d_mono_mean", "both_ness_mean"):
            log[f"{c}/{k}"] = cell[k]
        for k in ("sharper_than_full_n", "nearer_mono_than_full_n", "kept_full_window_seeds_n"):
            if k in cell:
                log[f"{c}/{k}"] = cell[k]
    log["cells"] = tab
    if diag_fig:
        log["diagnostics/correction_parallel_share_and_norm_per_step"] = wandb.Image(str(diag_fig))
    run.log(log)
    # per-step curves as W&B line plots, one series per fix, mean over seeds
    for fix in (BASELINE, *FIXES):
        ds = [json.loads((OUT_ROOT / "renders" / fix / f"seed_{s}_diagnostics.json").read_text())
              for s in SEEDS if (OUT_ROOT / "renders" / fix / f"seed_{s}_diagnostics.json").exists()]
        if not ds:
            continue
        for i in range(STEPS):
            run.log({"step_index": i,
                     f"per_step/{fix}/parallel_share": float(np.mean([d[i]["parallel_share"] for d in ds])),
                     f"per_step/{fix}/used_norm_ratio": float(np.mean([d[i]["used_norm_ratio"] for d in ds])),
                     f"per_step/{fix}/delta_norm": float(np.mean([d[i]["delta_norm"] for d in ds]))})
    art = wandb.Artifact("crisp_fix_strip_results", type="results")
    art.add_file(str(OUT_ROOT / "results.json"))
    if (OUT_ROOT / "identity_check.json").exists():
        art.add_file(str(OUT_ROOT / "identity_check.json"))
    run.log_artifact(art)
    results.setdefault("wandb", {})["run_id"] = run.id
    results["wandb"]["url"] = run.url
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    _log(f"wandb run id {run.id} url {run.url}")
    run.finish()


def _wandb_strips(results: dict, paths: dict) -> None:
    import wandb
    run = _wandb_run()
    rows_by = {(r["condition"], r["seed"]): r for r in results["rows"]}
    log = {}
    for seed, p in paths.items():
        log[f"strips/seed_{seed}"] = wandb.Image(str(p), caption=" | ".join(lab for _, lab in STRIP_COLUMNS))
    tab_cols = ["seed"] + [c for c, _ in STRIP_COLUMNS] + [f"{c}_n" for c, _ in STRIP_COLUMNS] + [f"{c}_sharp" for c, _ in STRIP_COLUMNS] + [f"{c}_d_mono" for c, _ in STRIP_COLUMNS]
    tab = wandb.Table(columns=tab_cols)
    for seed in SEEDS:
        tiles, ns, shs, dms = [], [], [], []
        for cc, lab in STRIP_COLUMNS:
            r = rows_by.get((cc, seed))
            if r is None:
                tiles.append(None); ns.append(None); shs.append(None); dms.append(None); continue
            im = Image.open(r["png"]).convert("RGB").resize((512, 512), Image.LANCZOS)
            tiles.append(wandb.Image(im, caption=f"{lab}: n={r['n_instances']} sharp={r['sharpness']:.0f} d(Mono)={r['d_mono']:.3f}"))
            ns.append(r["n_instances"]); shs.append(round(r["sharpness"], 1)); dms.append(round(r["d_mono"], 4))
        tab.add_data(seed, *tiles, *ns, *shs, *dms)
    log["strips_by_seed"] = tab
    run.log(log)
    _log(f"wandb strips logged to run {run.id} url {run.url}")
    run.finish()


# ============================================================================= main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-baseline", action="store_true", help=f"render {BASELINE}: the full window through this loop with the hook off")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--strips", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a = ap.parse_args(argv)
    if not any([a.identity, a.render, a.render_baseline, a.score, a.strips, a.all]):
        ap.error("pass a stage")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.identity or a.render or a.render_baseline or a.all:
        copy_references()
        S = FixSampler()
        S.attach()
        if a.identity or a.all:
            res = identity_check(S)
            if not res["pass_cross_device"]:
                raise SystemExit(f"identity check failed: {res['mean_abs_diff_vs_plan14_full_1.2']:.3f} grey levels; nothing else runs")
        if a.render or a.all:
            render(S, seeds=seeds)
        if a.render_baseline or a.all:
            render(S, seeds=seeds, conditions=(BASELINE,))
        del S
        torch.cuda.empty_cache()
    if a.score or a.all:
        score()
    if a.strips or a.all:
        results = json.loads((OUT_ROOT / "results.json").read_text())
        _wandb_strips(results, _strips(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
