"""Runner for the noise search on top of the rank-32 adapter (scope 06, plan 12).

Phase 1, adapter not yet attached: per pair and seed, the Mono reference and the plain-PoE pivot
(this package's batched sampler, so the plain band is measured by the same code). Phase 2: attach
the adapter, prove it inert when disabled (plain PoE seed 9 again, byte-close to phase 1). Phase 3:
per pair and seed, the adapter pivot (the cached noise at lambda 1.2), then per sigma N perturbed
candidates rendered with the adapter, every finished image scored by the compose scorer and by
sharpness, the best kept by count then sharpness. A sheet per pair, ``summary.json``,
``verdict.json``, W&B, and both-ness of every cat x dog tile.

    <co3 python> -m poe_repair.experiments.noise_search.run_adapter --smoke
    <co3 python> -m poe_repair.experiments.noise_search.run_adapter --pairs a_cat__x__a_dog
    <co3 python> -m poe_repair.experiments.noise_search.run_adapter --bothness-only --run-id <run>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from poe_repair._sdxl.runtime import (encode_prompt_sdxl, infer_device, infer_dtype,
                                      load_ddim_scheduler, load_sdxl_models)
from poe_repair.experiments.fk_steering.run import PROMPTS, Detector, draw_sheet, pinned_init
from poe_repair.experiments.noise_search import adapter as ad
from poe_repair.experiments.noise_search import search
from poe_repair.experiments.noise_search.run import (CONTROL_PAIR, JUDGED_PAIR, LANDING_SEEDS,
                                                     R32_RENDERS, bothness, mean_abs_diff, score)
from poe_repair.experiments.twisted_smc.sampler import decode_one, run_particles
from poe_repair.experiments.twisted_smc.train import WandB
from poe_repair.methods._sampling import write_decoded_image

log = logging.getLogger("noise_search_adapter")

DEFAULT_OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search")
DEFAULT_CACHE = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", nargs="+", default=[JUDGED_PAIR, CONTROL_PAIR])
    p.add_argument("--seeds", nargs="+", type=int, default=list(LANDING_SEEDS))
    p.add_argument("--sigmas", nargs="+", type=float, default=list(ad.SIGMAS))
    p.add_argument("--n-candidates", type=int, default=ad.N_CANDIDATES)
    p.add_argument("--lam", type=float, default=ad.LAMBDA)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--unet-chunk", type=int, default=2)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--run-id", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="poe-repair-animals-compose")
    p.add_argument("--wandb-entity", default="prime_lab")
    p.add_argument("--skip-bothness", action="store_true")
    p.add_argument("--bothness-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="cat x dog seed 9, N 2, sigma 0.3, 10 steps, 512²")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="adapter weights to load; defaults to the step-30050 checkpoint in adapter.py")
    a = p.parse_args(argv)
    if a.smoke:
        a.pairs, a.seeds, a.sigmas, a.n_candidates, a.steps = [JUDGED_PAIR], [9], [0.3], 2, 10
        a.size, a.wandb_mode, a.skip_bothness = 512, "disabled", True
    return a


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    a = parse_args(argv)
    if a.checkpoint is not None:
        ad.CHECKPOINT = Path(a.checkpoint)
        log.info("checkpoint overridden: %s", ad.CHECKPOINT)
    if a.bothness_only:
        run_dir = a.out_root / a.run_id
        summ = json.loads((run_dir / "summary.json").read_text())
        cols = summ["pairs"][JUDGED_PAIR]["tile_paths"]
        b = bothness(run_dir, {c: {int(s): p for s, p in v.items()} for c, v in cols.items()})
        summ["bothness"] = b
        (run_dir / "summary.json").write_text(json.dumps(summ, indent=1))
        log.info("both-ness means: %s", {c: v["mean_both_ness"] for c, v in b["by_column"].items()})
        return

    device = infer_device(a.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("torch.cuda.is_available() is False on the pinned device (poe-launch-002); aborting")
        log.info("device %s = %s", device, torch.cuda.get_device_name(device))
    dtype = infer_dtype(a.dtype, device)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = a.run_id or (("smoke_adapter_" if a.smoke else "zo_adapter_") + f"N{a.n_candidates}_s{a.steps}_{stamp}")
    run_dir = a.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    constants = {k: getattr(ad, k) for k in ("MAX_COMPOSE_DROP", "MIN_SHARPNESS_GAIN", "NULL_SHARPNESS_BAND", "LAMBDA",
                                             "LORA_RANK", "LORA_ALPHA", "SIGMAS", "N_CANDIDATES", "ETA")}
    constants["CHECKPOINT"] = str(ad.CHECKPOINT)
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "out_root": str(a.out_root), "cache_root": str(a.cache_root),
                                                    "run_id": run_id, "node": os.uname().nodename, "pid": os.getpid(),
                                                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                    "perturbation": "z' = (z + sigma u) / sqrt(1 + sigma^2), u ~ N(0, I), one round",
                                                    "verifier": "instance count clipped at 2, then Laplacian variance of the finished image, then lowest index",
                                                    "constants": constants}, indent=1, default=str))
    log.info("run dir %s (node %s, pid %d)", run_dir, os.uname().nodename, os.getpid())

    models = load_sdxl_models(model_id=a.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(a.model_id)
    det = Detector(device)
    wb = WandB(mode=a.wandb_mode, project=a.wandb_project, entity=a.wandb_entity, name=run_id, run_dir=run_dir,
               config=json.loads((run_dir / "config.json").read_text()), tags=["noise-search", "adapter", "scope-06", "plan-12"],
               group="noise-search-adapter")
    if wb.run is not None:
        log.info("wandb run id %s url %s", wb.run.id, wb.run.url)
    h = w = a.size // 8
    enc = lambda t: encode_prompt_sdxl(t, models=models, device=device, dtype=dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    common = dict(models=models, scheduler=scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=a.guidance_scale,
                  num_inference_steps=a.steps, height=a.size, width=a.size, device=device, dtype=dtype)
    prompts = {pair: tuple(enc(t) for t in PROMPTS[pair]) for pair in a.pairs}
    bases = {(pair, seed): pinned_init(a.cache_root, pair, seed, h, w) for pair in a.pairs for seed in a.seeds}

    # ---- phase 1: references, adapter not attached ----------------------------------------
    for pair in a.pairs:
        (seq_a, pool_a), (seq_b, pool_b), (seq_j, pool_j) = prompts[pair]
        for seed in a.seeds:
            sd = run_dir / pair / f"seed_{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            base, _ = bases[(pair, seed)]
            mono = sd / "mono_eta0.png"
            if not mono.exists():
                out = run_particles(init_latents=base[None], composition="mono", seq_a=seq_a, pool_a=pool_a, seq_b=seq_b,
                                    pool_b=pool_b, seq_j=seq_j, pool_j=pool_j, eta=0.0, generator=None, twist=None,
                                    unet_chunk=a.unet_chunk, **common)
                write_decoded_image(decode_one(models, out.latents[0]), mono)
            piv = sd / "poe_pivot.png"
            if not piv.exists():
                out = search.run_poe_candidates(init_latents=base[None], capture_steps=(), seq_a=seq_a, pool_a=pool_a,
                                                seq_b=seq_b, pool_b=pool_b, unet_chunk=a.unet_chunk, **common)
                write_decoded_image(decode_one(models, out.latents[0]), piv)
            log.info("[%s] seed %d references done", pair, seed)

    # ---- phase 2: attach, and prove the adapter inert when disabled ------------------------
    info = ad.attach_and_load_lora(models["unet"])
    log.info("LoRA rank=%d attached: n_matched=%s n_loaded=%s checkpoint_step=%s", ad.LORA_RANK,
             info.get("n_matched"), info.get("n_loaded"), info.get("checkpoint_step"))
    pair0, seed0 = a.pairs[0], a.seeds[0]
    (seq_a, pool_a), (seq_b, pool_b), _ = prompts[pair0]
    ad.adapter_disable(models["unet"])
    out = search.run_poe_candidates(init_latents=bases[(pair0, seed0)][0][None], capture_steps=(), seq_a=seq_a, pool_a=pool_a,
                                    seq_b=seq_b, pool_b=pool_b, unet_chunk=a.unet_chunk, **common)
    chk = run_dir / "detach_check_poe_after_attach.png"
    write_decoded_image(decode_one(models, out.latents[0]), chk)
    detach = {"pair": pair0, "seed": seed0,
              "mean_abs_diff_disabled_vs_before_attach": mean_abs_diff(chk, run_dir / pair0 / f"seed_{seed0}" / "poe_pivot.png")}
    log.info("detach check: %s", detach)
    (run_dir / "detach_check.json").write_text(json.dumps({**detach, "attach_info": {k: v for k, v in info.items() if isinstance(v, (int, float, str))}}, indent=1))

    # ---- phase 3: adapter pivot and the search ---------------------------------------------
    summary: dict = {"run_id": run_id, "pairs": {}, "constants": constants, "sigmas": a.sigmas, "n_candidates": a.n_candidates,
                     "detach_check": detach}
    step_counter = 0
    for pair in a.pairs:
        (seq_a, pool_a), (seq_b, pool_b), _ = prompts[pair]
        kw = dict(seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, unet_chunk=a.unet_chunk, lambda_value=a.lam, **common)
        rows = []
        for seed in a.seeds:
            t0 = time.time()
            sd = run_dir / pair / f"seed_{seed}"
            base, origin = bases[(pair, seed)]
            cell: dict = {"seed": seed, "init_noise_from": origin, "tiles": {}, "runs": {}}
            for tag in ("mono_eta0", "poe_pivot"):
                p = sd / f"{tag}.png"
                c, _ = score(det, p)
                cell["tiles"]["mono" if tag == "mono_eta0" else "poe_pivot"] = {"path": str(p), "count": c, "sharpness": ad.laplacian_var(p)}
            apiv = sd / f"adapter_pivot_lambda{a.lam:g}.png"
            if not apiv.exists():
                lat = ad.run_adapter_candidates(init_latents=base[None], **kw)
                write_decoded_image(decode_one(models, lat[0]), apiv)
            c, conf = score(det, apiv)
            cell["tiles"]["adapter_pivot"] = {"path": str(apiv), "count": c, "sharpness": ad.laplacian_var(apiv)}
            ref = R32_RENDERS / f"seed_{seed}_lambda_{a.lam:g}.png"
            if pair == JUDGED_PAIR and ref.exists() and a.size == 1024:
                cell["adapter_pivot_mean_abs_diff_vs_r32"] = mean_abs_diff(apiv, ref)
            for sigma in a.sigmas:
                name = f"sigma_{sigma:g}"
                rd = sd / name
                rd.mkdir(exist_ok=True)
                done = rd / "run.json"
                if done.exists():
                    rec = json.loads(done.read_text())
                else:
                    g = torch.Generator(device="cpu").manual_seed(1_000 * seed + int(round(sigma * 100)))
                    cands = search.perturb(base, sigma, a.n_candidates, g)   # same draws as plan 11 for the same seed and sigma
                    cos = search.cosine_to_pivot(cands, base)
                    lat = ad.run_adapter_candidates(init_latents=cands, **kw)
                    finals, sharp = [], []
                    for k in range(a.n_candidates):
                        pk = rd / f"c{k}.png"
                        write_decoded_image(decode_one(models, lat[k]), pk)
                        finals.append(score(det, pk)); sharp.append(ad.laplacian_var(pk))
                    counts = [f[0] for f in finals]
                    kp = ad.pick_count_then_sharpness(counts, sharp)
                    kc = search.pick(finals)
                    rec = {"sigma": sigma, "n": a.n_candidates, "cosine_to_pivot": cos, "final_scores": finals, "sharpness": sharp,
                           "pick_count_then_sharpness": kp, "pick_count_then_confidence": kc,
                           "kept_compose": int(counts[kp] >= 2), "kept_sharpness": sharp[kp],
                           "kept_by_confidence_compose": int(counts[kc] >= 2), "kept_by_confidence_sharpness": sharp[kc],
                           "candidate_compose_fraction": sum(cc >= 2 for cc in counts) / a.n_candidates,
                           "sharpest_candidate": int(np.argmax(sharp)), "max_sharpness": float(max(sharp)),
                           "final_pngs": [str(rd / f"c{k}.png") for k in range(a.n_candidates)]}
                    done.write_text(json.dumps(rec, indent=1))
                    del lat
                    torch.cuda.empty_cache()
                cell["runs"][name] = rec
                cell["tiles"][f"best_{name}"] = {"path": rec["final_pngs"][rec["pick_count_then_sharpness"]],
                                                 "count": rec["final_scores"][rec["pick_count_then_sharpness"]][0],
                                                 "sharpness": rec["kept_sharpness"]}
            ap_s = cell["tiles"]["adapter_pivot"]["sharpness"]
            cell["sharpness_ratio_plain_over_adapter"] = cell["tiles"]["poe_pivot"]["sharpness"] / ap_s if ap_s > 0 else None
            for sigma in a.sigmas:
                r = cell["runs"][f"sigma_{sigma:g}"]
                r["sharpness_ratio_kept_over_adapter_pivot"] = r["kept_sharpness"] / ap_s if ap_s > 0 else None
                r["kept_reaches_plain_band"] = int(r["kept_sharpness"] >= cell["tiles"]["poe_pivot"]["sharpness"])
                r["kept_reaches_mono_band"] = int(r["kept_sharpness"] >= cell["tiles"]["mono"]["sharpness"])
                r["sharpness_ratio_kept_over_mono"] = r["kept_sharpness"] / cell["tiles"]["mono"]["sharpness"] if cell["tiles"]["mono"]["sharpness"] > 0 else None
            (sd / "cell.json").write_text(json.dumps(cell, indent=1))
            rows.append(cell)
            log.info("[%s] seed %d done in %.0f s: counts %s sharpness %s", pair, seed, time.time() - t0,
                     {k: v["count"] for k, v in cell["tiles"].items()}, {k: round(v["sharpness"], 1) for k, v in cell["tiles"].items()})
            step_counter += 1
            wb.log({f"progress/{pair}/seed": seed}, step=step_counter)

        columns = [("mono", "Mono, joint prompt\nDDIM eta 0"), ("poe_pivot", "plain PoE, cached noise\n(eta 0)"),
                   ("adapter_pivot", f"PoE + rank-32 correction\nlambda {a.lam:g}, cached noise")]
        for sigma in a.sigmas:
            columns.append((f"best_sigma_{sigma:g}", f"best of {a.n_candidates}, sigma {sigma:g}\nadapter on; count then sharpness"))
        sheet_png = run_dir / f"noise_search_adapter_{pair}_sheet.png"
        side = draw_sheet(rows, columns, f"{pair.replace('__x__', ' x ').replace('_', ' ')}: zero-order search over the initial noise on top of the "
                          f"rank-32 step-30050 adapter at lambda {a.lam:g}, {a.steps} DDIM steps, guidance {a.guidance_scale:g}, {a.size}²; "
                          f"count on each tile, sharpness in the sidecar", sheet_png, a.thumb)
        rates = {key: sum(r["tiles"][key]["count"] >= 2 for r in rows) / len(rows) for key, _ in columns}
        sharp_mean = {key: float(np.mean([r["tiles"][key]["sharpness"] for r in rows])) for key, _ in columns}
        extras = {"median_ratio_plain_over_adapter": float(np.median([r["sharpness_ratio_plain_over_adapter"] for r in rows])),
                  "median_ratio_mono_over_adapter": float(np.median([r["tiles"]["mono"]["sharpness"] / r["tiles"]["adapter_pivot"]["sharpness"] for r in rows])),
                  "adapter_pivot_mean_abs_diff_vs_r32": [r.get("adapter_pivot_mean_abs_diff_vs_r32") for r in rows]}
        for sigma in a.sigmas:
            name = f"sigma_{sigma:g}"
            rs = [r["runs"][name] for r in rows]
            extras[f"{name}_median_ratio_kept_over_adapter_pivot"] = float(np.median([x["sharpness_ratio_kept_over_adapter_pivot"] for x in rs]))
            extras[f"{name}_kept_reaches_plain_band"] = sum(x["kept_reaches_plain_band"] for x in rs) / len(rs)
            extras[f"{name}_kept_reaches_mono_band"] = sum(x["kept_reaches_mono_band"] for x in rs) / len(rs)
            extras[f"{name}_median_ratio_kept_over_mono"] = float(np.median([x["sharpness_ratio_kept_over_mono"] for x in rs]))
            extras[f"{name}_candidate_compose_fraction"] = sum(x["candidate_compose_fraction"] for x in rs) / len(rs)
            extras[f"{name}_kept_by_confidence_compose_rate"] = sum(x["kept_by_confidence_compose"] for x in rs) / len(rs)
            extras[f"{name}_mean_cosine_to_pivot"] = float(np.mean([c for x in rs for c in x["cosine_to_pivot"]]))
        tile_paths = {key: {r["seed"]: r["tiles"][key]["path"] for r in rows} for key, _ in columns}
        for t in side["tiles"]:
            t["sharpness"] = next(r["tiles"][t["column"]]["sharpness"] for r in rows if r["seed"] == t["row_seed"])
        side.update({"columns": columns, "compose_rate_by_column": rates, "mean_sharpness_by_column": sharp_mean,
                     "n_seeds": len(rows), "secondary": extras,
                     "settings": {"steps": a.steps, "eta": ad.ETA, "guidance": a.guidance_scale, "size": a.size, "lambda": a.lam,
                                  "sigmas": a.sigmas, "n_candidates": a.n_candidates, "checkpoint": str(ad.CHECKPOINT)}})
        (run_dir / f"noise_search_adapter_{pair}_sheet.json").write_text(json.dumps(side, indent=1))
        summary["pairs"][pair] = {"compose_rate_by_column": rates, "mean_sharpness_by_column": sharp_mean, "secondary": extras,
                                  "tile_paths": tile_paths}
        wb.log_image(f"sheets/{pair}", sheet_png, step=step_counter, caption=f"compose rates {rates}; mean sharpness {sharp_mean}")
        payload = {f"compose/{pair}/{k}": v for k, v in rates.items()}
        payload.update({f"sharpness/{pair}/{k}": v for k, v in sharp_mean.items()})
        payload.update({f"secondary/{pair}/{k}": v for k, v in extras.items() if isinstance(v, (int, float))})
        wb.log(payload, step=step_counter)
        log.info("[%s] compose rates %s; mean sharpness %s; secondary %s", pair, rates, sharp_mean, extras)

    v = {"judged_pair": JUDGED_PAIR, "constants": constants, "detach_check": detach}
    if JUDGED_PAIR in summary["pairs"]:
        pr = summary["pairs"][JUDGED_PAIR]
        kept = {s: pr["compose_rate_by_column"][f"best_sigma_{s:g}"] for s in a.sigmas}
        ratio = {s: pr["secondary"][f"sigma_{s:g}_median_ratio_kept_over_adapter_pivot"] for s in a.sigmas}
        v.update({"adapter_pivot_compose_rate": pr["compose_rate_by_column"]["adapter_pivot"],
                  "plain_pivot_compose_rate": pr["compose_rate_by_column"]["poe_pivot"],
                  "kept_compose_rate_by_sigma": {str(s): r for s, r in kept.items()},
                  "median_sharpness_ratio_kept_over_adapter_by_sigma": {str(s): r for s, r in ratio.items()},
                  "median_sharpness_ratio_plain_over_adapter": pr["secondary"]["median_ratio_plain_over_adapter"]})
        v.update(ad.verdict(kept_rate_by_sigma=kept, adapter_pivot_rate=v["adapter_pivot_compose_rate"], median_ratio_by_sigma=ratio))
    summary["verdict"] = v
    if not a.skip_bothness and JUDGED_PAIR in summary["pairs"]:
        try:
            b = bothness(run_dir, {c: {int(s): p for s, p in v2.items()} for c, v2 in summary["pairs"][JUDGED_PAIR]["tile_paths"].items()})
            summary["bothness"] = b
        except Exception as exc:  # noqa: BLE001
            log.warning("both-ness failed (%s); rerun with --bothness-only --run-id %s on the session node with XFORMERS_DISABLED=1", exc, run_id)
            summary["bothness"] = {"error": str(exc)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "verdict.json").write_text(json.dumps(v, indent=1))
    art_dir = run_dir / "_artifact"
    art_dir.mkdir(exist_ok=True)
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.json")):
        (art_dir / f.name).write_bytes(f.read_bytes())
    wb.log_artifact_dir(art_dir, name=f"noise-search-adapter-{run_id}", kind="sheets", aliases=["latest"])
    wb.finish()
    log.info("verdict: %s", v.get("verdict"))
    log.info("[done] %s", run_dir)


if __name__ == "__main__":
    main()
