"""Runner for zero-order search over the initial noise (scope 06, plan 11).

Per pair and seed: the two references (Mono, plain PoE from the seed's cached noise, the pivot),
then per sigma N perturbed candidates rendered with plain PoE at DDIM eta 0, every candidate's
finished image and its x0-hat at the early step scored by the validated compose scorer, the best
kept under each verifier. A sheet per pair, ``summary.json``, ``verdict.json``, W&B, and the
both-ness of every cat x dog tile on the landing finding's cloud axes.

    <co3 python> -m poe_repair.experiments.noise_search.run --smoke
    <co3 python> -m poe_repair.experiments.noise_search.run --pairs a_cat__x__a_dog a_butterfly__x__a_flower_meadow
    <co3 python> -m poe_repair.experiments.noise_search.run --bothness-only --run-id <run>
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
from PIL import Image

from poe_repair._sdxl.runtime import (encode_prompt_sdxl, infer_device, infer_dtype,
                                      load_ddim_scheduler, load_sdxl_models)
from poe_repair.experiments.fk_steering.run import PROMPTS, Detector, draw_sheet, pinned_init
from poe_repair.experiments.noise_search import search
from poe_repair.experiments.twisted_smc.sampler import decode_one, run_particles
from poe_repair.experiments.twisted_smc.train import WandB
from poe_repair.methods._sampling import write_decoded_image

log = logging.getLogger("noise_search")

DEFAULT_OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search")
DEFAULT_CACHE = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
JUDGED_PAIR = "a_cat__x__a_dog"
CONTROL_PAIR = "a_butterfly__x__a_flower_meadow"
# the existing rank-32 lambda 1.2 render and the plain-PoE render the identity check reads against
R32_RENDERS = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/renders/full")
# the landing finding's DINOv2 features of the 48 cat x dog renders, rows: 6 conditions x seeds 9..16
LANDING_FEATS = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy")
LANDING_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", nargs="+", default=[JUDGED_PAIR, CONTROL_PAIR])
    p.add_argument("--seeds", nargs="+", type=int, default=list(LANDING_SEEDS))
    p.add_argument("--sigmas", nargs="+", type=float, default=list(search.SIGMAS))
    p.add_argument("--n-candidates", type=int, default=search.N_CANDIDATES)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--early-step", type=int, default=search.VERIFIER_EARLY_STEP)
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
    p.add_argument("--bothness-only", action="store_true", help="recompute both-ness for an existing run dir (needs --run-id)")
    p.add_argument("--smoke", action="store_true", help="cat x dog seed 9, N 2, sigma 0.3, 10 steps, 512², early step 2")
    a = p.parse_args(argv)
    if a.smoke:
        a.pairs, a.seeds, a.sigmas, a.n_candidates, a.steps = [JUDGED_PAIR], [9], [0.3], 2, 10
        a.early_step, a.size, a.wandb_mode, a.skip_bothness = 2, 512, "disabled", True
    return a


def score(det: Detector, png: Path) -> tuple[int, float]:
    """(instance count, summed confidence of the kept boxes)."""
    n, kept = det.fn(Path(png), device=det.device)
    return int(n), float(sum(b.get("confidence", 0.0) for b in kept))


def mean_abs_diff(a: Path, b: Path) -> float:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    return float(np.abs(x - y).mean()) if x.shape == y.shape else float("nan")


# ---------------------------------------------------------------------------
# Both-ness on the landing finding's cloud axes
# ---------------------------------------------------------------------------


def bothness(run_dir: Path, thumb_paths: dict[str, dict[int, str]]) -> dict:
    """Project each tile's DINOv2 CLS onto the landing finding's cloud axes (x: cat -> dog centroid,
    y: solo midpoint -> joint centroid, orthogonalised). Axes come from the saved 48-render features."""
    sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
    from scripts.build_lora_inspector_mds_semantic import DinoEmbedder  # noqa: E402
    feats = np.load(LANDING_FEATS)
    idx = lambda cond, s: {"solo_a": 0, "solo_b": 1, "joint": 2}[cond] * 8 + LANDING_SEEDS.index(s)  # noqa: E731
    cent = {c: feats[[idx(c, s) for s in LANDING_SEEDS]].mean(0) for c in ("solo_a", "solo_b", "joint")}
    origin = 0.5 * (cent["solo_a"] + cent["solo_b"])
    u1 = cent["solo_b"] - cent["solo_a"]; u1 /= np.linalg.norm(u1)
    u2 = cent["joint"] - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    emb = DinoEmbedder(device=torch.device("cpu"))

    def load(p):
        im = Image.open(p).convert("RGB")
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]

    out = {"axes_from": str(LANDING_FEATS), "x": "which animal: cat centroid to dog centroid, DINOv2 cosine units",
           "y": "both-ness: solo midpoint toward the joint-prompt centroid, orthogonal to x",
           "reference_means": {"poe_landing": None, "joint_landing": None}, "by_column": {}}
    joint_b = [(feats[idx("joint", s)] - origin) @ u2 for s in LANDING_SEEDS]
    out["reference_means"]["joint_landing"] = round(float(np.mean(joint_b)), 4)
    for col, per_seed in thumb_paths.items():
        rows = {}
        for s, p in per_seed.items():
            f = emb.embed_decoded_batch(load(p))[0]
            rows[int(s)] = {"which_animal": round(float((f - origin) @ u1), 4), "both_ness": round(float((f - origin) @ u2), 4)}
        out["by_column"][col] = {"per_seed": rows, "mean_both_ness": round(float(np.mean([r["both_ness"] for r in rows.values()])), 4)}
    (run_dir / "bothness.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    a = parse_args(argv)
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
    run_id = a.run_id or (("smoke_" if a.smoke else "zo_") + f"N{a.n_candidates}_s{a.steps}_{stamp}")
    run_dir = a.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    constants = {k: getattr(search, k) for k in ("PASS_MARGIN", "NULL_MARGIN", "SIGMAS", "N_CANDIDATES", "VERIFIER_EARLY_STEP", "ETA")}
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "out_root": str(a.out_root), "cache_root": str(a.cache_root),
                                                    "run_id": run_id, "node": os.uname().nodename, "pid": os.getpid(),
                                                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                    "perturbation": "z' = (z + sigma u) / sqrt(1 + sigma^2), u ~ N(0, I), one round",
                                                    "constants": constants}, indent=1, default=str))
    log.info("run dir %s (node %s, pid %d)", run_dir, os.uname().nodename, os.getpid())

    models = load_sdxl_models(model_id=a.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(a.model_id)
    det = Detector(device)
    wb = WandB(mode=a.wandb_mode, project=a.wandb_project, entity=a.wandb_entity, name=run_id, run_dir=run_dir,
               config=json.loads((run_dir / "config.json").read_text()), tags=["noise-search", "scope-06", "plan-11"],
               group="noise-search")
    if wb.run is not None:
        log.info("wandb run id %s url %s", wb.run.id, wb.run.url)
    h = w = a.size // 8
    enc = lambda t: encode_prompt_sdxl(t, models=models, device=device, dtype=dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    common = dict(models=models, scheduler=scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=a.guidance_scale,
                  num_inference_steps=a.steps, height=a.size, width=a.size, device=device, dtype=dtype)

    summary: dict = {"run_id": run_id, "pairs": {}, "constants": constants, "sigmas": a.sigmas, "n_candidates": a.n_candidates}
    step_counter = 0
    for pair in a.pairs:
        pa, pb, pj = PROMPTS[pair]
        seq_a, pool_a = enc(pa); seq_b, pool_b = enc(pb); seq_j, pool_j = enc(pj)
        poe_kw = dict(seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b, unet_chunk=a.unet_chunk, **common)
        pair_dir = run_dir / pair
        rows = []
        for seed in a.seeds:
            t0 = time.time()
            sd = pair_dir / f"seed_{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            base, origin = pinned_init(a.cache_root, pair, seed, h, w)
            cell: dict = {"seed": seed, "init_noise_from": origin, "tiles": {}, "runs": {}}
            # Mono reference, eta 0
            mono = sd / "mono_eta0.png"
            if not mono.exists():
                out = run_particles(init_latents=base[None], composition="mono", seq_a=seq_a, pool_a=pool_a, seq_b=seq_b,
                                    pool_b=pool_b, seq_j=seq_j, pool_j=pool_j, eta=0.0, generator=None, twist=None,
                                    unet_chunk=a.unet_chunk, **common)
                write_decoded_image(decode_one(models, out.latents[0]), mono)
            cell["tiles"]["mono"] = {"path": str(mono), "count": score(det, mono)[0]}
            # the pivot: plain PoE from the cached noise, with its early x0-hat
            piv = sd / "poe_pivot.png"
            piv_x0 = sd / f"poe_pivot_xhat_step{a.early_step:02d}.png"
            if not piv.exists() or not piv_x0.exists():
                out = search.run_poe_candidates(init_latents=base[None], capture_steps=(a.early_step,), **poe_kw)
                write_decoded_image(decode_one(models, out.latents[0]), piv)
                write_decoded_image(decode_one(models, out.x0_at[a.early_step][0]), piv_x0)
            pf, pe = score(det, piv), score(det, piv_x0)
            cell["tiles"]["poe_pivot"] = {"path": str(piv), "count": pf[0]}
            cell["pivot"] = {"final": pf, "early": pe, "final_compose": int(pf[0] >= 2), "early_compose": int(pe[0] >= 2)}
            ref = R32_RENDERS / f"seed_{seed}_lambda_0.0.png"
            if pair == JUDGED_PAIR and ref.exists() and a.size == 1024:
                cell["pivot"]["mean_abs_diff_vs_r32_lambda0"] = mean_abs_diff(piv, ref)
            l12 = R32_RENDERS / f"seed_{seed}_lambda_1.2.png"
            if pair == JUDGED_PAIR and l12.exists():
                cell["tiles"]["lora_1.2_existing"] = {"path": str(l12), "count": score(det, l12)[0]}
            # the candidates per sigma
            for sigma in a.sigmas:
                name = f"sigma_{sigma:g}"
                rd = sd / name
                rd.mkdir(exist_ok=True)
                done = rd / "run.json"
                if done.exists():
                    rec = json.loads(done.read_text())
                else:
                    g = torch.Generator(device="cpu").manual_seed(1_000 * seed + int(round(sigma * 100)))
                    cands = search.perturb(base, sigma, a.n_candidates, g)
                    cos = search.cosine_to_pivot(cands, base)
                    res = search.run_poe_candidates(init_latents=cands, capture_steps=(a.early_step,), **poe_kw)
                    finals, earlies = [], []
                    for k in range(a.n_candidates):
                        pf_k = rd / f"c{k}.png"; pe_k = rd / f"c{k}_xhat_step{a.early_step:02d}.png"
                        write_decoded_image(decode_one(models, res.latents[k]), pf_k)
                        write_decoded_image(decode_one(models, res.x0_at[a.early_step][k]), pe_k)
                        finals.append(score(det, pf_k)); earlies.append(score(det, pe_k))
                    kf, ke = search.pick(finals), search.pick(earlies)
                    rec = {"sigma": sigma, "n": a.n_candidates, "cosine_to_pivot": cos,
                           "final_scores": finals, "early_scores": earlies, "early_step": a.early_step,
                           "pick_final": kf, "pick_early": ke,
                           "kept_final_compose": int(finals[kf][0] >= 2),
                           "kept_early_compose_by_final": int(finals[ke][0] >= 2),
                           "kept_early_compose_by_early": int(earlies[ke][0] >= 2),
                           "any_candidate_composes": int(any(f[0] >= 2 for f in finals)),
                           "candidate_compose_fraction": sum(f[0] >= 2 for f in finals) / a.n_candidates,
                           "early_agrees_with_final": sum(int((e[0] >= 2) == (f[0] >= 2)) for e, f in zip(earlies, finals)) / a.n_candidates,
                           "final_pngs": [str(rd / f"c{k}.png") for k in range(a.n_candidates)],
                           "early_pngs": [str(rd / f"c{k}_xhat_step{a.early_step:02d}.png") for k in range(a.n_candidates)]}
                    done.write_text(json.dumps(rec, indent=1))
                    del res
                    torch.cuda.empty_cache()
                cell["runs"][name] = rec
                cell["tiles"][f"best_final_{name}"] = {"path": rec["final_pngs"][rec["pick_final"]], "count": rec["final_scores"][rec["pick_final"]][0]}
                cell["tiles"][f"best_early_{name}"] = {"path": rec["final_pngs"][rec["pick_early"]], "count": rec["final_scores"][rec["pick_early"]][0]}
            (sd / "cell.json").write_text(json.dumps(cell, indent=1))
            rows.append(cell)
            log.info("[%s] seed %d done in %.0f s: %s", pair, seed, time.time() - t0, {k: v["count"] for k, v in cell["tiles"].items()})
            step_counter += 1
            wb.log({f"progress/{pair}/seed": seed}, step=step_counter)

        columns = [("mono", "Mono, joint prompt\nDDIM eta 0"), ("poe_pivot", "plain PoE, cached noise\n(the pivot, eta 0)")]
        if all("lora_1.2_existing" in r["tiles"] for r in rows):
            columns.append(("lora_1.2_existing", "PoE + rank-32 correction\nlambda 1.2 (existing render)"))
        for sigma in a.sigmas:
            columns.append((f"best_final_sigma_{sigma:g}", f"best of {a.n_candidates}, sigma {sigma:g}\nverifier: final image"))
        for sigma in a.sigmas:
            columns.append((f"best_early_sigma_{sigma:g}", f"best of {a.n_candidates}, sigma {sigma:g}\nverifier: x0-hat at step {a.early_step}"))
        sheet_png = run_dir / f"noise_search_{pair}_sheet.png"
        side = draw_sheet(rows, columns, f"{pair.replace('__x__', ' x ').replace('_', ' ')}: zero-order search over the initial noise, "
                          f"{a.steps} DDIM steps, guidance {a.guidance_scale:g}, {a.size}², count of animal instances on each tile, compose = 2 or more",
                          sheet_png, a.thumb)
        rates = {key: sum(r["tiles"][key]["count"] >= 2 for r in rows if key in r["tiles"]) / len(rows) for key, _ in columns}
        extras = {}
        for sigma in a.sigmas:
            name = f"sigma_{sigma:g}"
            rs = [r["runs"][name] for r in rows]
            extras[f"{name}_any_candidate_composes"] = sum(x["any_candidate_composes"] for x in rs) / len(rs)
            extras[f"{name}_candidate_compose_fraction"] = sum(x["candidate_compose_fraction"] for x in rs) / len(rs)
            extras[f"{name}_kept_early_compose_by_early"] = sum(x["kept_early_compose_by_early"] for x in rs) / len(rs)
            extras[f"{name}_early_agrees_with_final"] = sum(x["early_agrees_with_final"] for x in rs) / len(rs)
            extras[f"{name}_mean_cosine_to_pivot"] = float(np.mean([c for x in rs for c in x["cosine_to_pivot"]]))
        extras["pivot_early_compose"] = sum(r["pivot"]["early_compose"] for r in rows) / len(rows)
        extras["pivot_mean_abs_diff_vs_r32_lambda0"] = [r["pivot"].get("mean_abs_diff_vs_r32_lambda0") for r in rows]
        tile_paths = {key: {r["seed"]: r["tiles"][key]["path"] for r in rows if key in r["tiles"]} for key, _ in columns}
        side.update({"columns": columns, "compose_rate_by_column": rates, "n_seeds": len(rows), "secondary": extras,
                     "settings": {"steps": a.steps, "eta": search.ETA, "guidance": a.guidance_scale, "size": a.size,
                                  "sigmas": a.sigmas, "n_candidates": a.n_candidates, "early_step": a.early_step}})
        (run_dir / f"noise_search_{pair}_sheet.json").write_text(json.dumps(side, indent=1))
        summary["pairs"][pair] = {"compose_rate_by_column": rates, "secondary": extras, "tile_paths": tile_paths}
        wb.log_image(f"sheets/{pair}", sheet_png, step=step_counter, caption=f"compose rates {rates}")
        payload = {f"compose/{pair}/{k}": v for k, v in rates.items()}
        payload.update({f"secondary/{pair}/{k}": v for k, v in extras.items() if isinstance(v, (int, float))})
        wb.log(payload, step=step_counter)
        log.info("[%s] compose rates %s", pair, rates)

    v = {"judged_pair": JUDGED_PAIR, "constants": constants}
    if JUDGED_PAIR in summary["pairs"]:
        pr = summary["pairs"][JUDGED_PAIR]
        kept = {sigma: pr["compose_rate_by_column"][f"best_final_sigma_{sigma:g}"] for sigma in a.sigmas}
        kept_early = {sigma: pr["compose_rate_by_column"][f"best_early_sigma_{sigma:g}"] for sigma in a.sigmas}
        piv = pr["compose_rate_by_column"]["poe_pivot"]
        v.update({"pivot_compose_rate": piv, "kept_final_verifier": {str(s): r for s, r in kept.items()},
                  "kept_early_verifier": {str(s): r for s, r in kept_early.items()},
                  "final_verifier": search.verdict(kept_rate_by_sigma=kept, pivot_rate=piv),
                  "early_verifier": search.verdict(kept_rate_by_sigma=kept_early, pivot_rate=piv)})
        v["verdict"] = v["final_verifier"]["verdict"]
        if CONTROL_PAIR in summary["pairs"]:
            cp = summary["pairs"][CONTROL_PAIR]["compose_rate_by_column"]
            v["control_pair"] = {k: cp.get(k) for k in cp}
    summary["verdict"] = v
    if not a.skip_bothness and JUDGED_PAIR in summary["pairs"]:
        try:
            b = bothness(run_dir, {c: {int(s): p for s, p in v2.items()} for c, v2 in summary["pairs"][JUDGED_PAIR]["tile_paths"].items()})
            summary["bothness"] = b
            wb.log({f"bothness/{c}": v2["mean_both_ness"] for c, v2 in b["by_column"].items()}, step=step_counter + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("both-ness failed (%s); rerun with --bothness-only --run-id %s on a node with network", exc, run_id)
            summary["bothness"] = {"error": str(exc)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "verdict.json").write_text(json.dumps(v, indent=1))
    art_dir = run_dir / "_artifact"
    art_dir.mkdir(exist_ok=True)
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.json")):
        (art_dir / f.name).write_bytes(f.read_bytes())
    wb.log_artifact_dir(art_dir, name=f"noise-search-{run_id}", kind="sheets", aliases=["latest"])
    wb.finish()
    log.info("verdict: %s", v.get("verdict"))
    log.info("[done] %s", run_dir)


if __name__ == "__main__":
    main()
