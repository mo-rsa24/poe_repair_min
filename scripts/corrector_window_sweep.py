#!/usr/bin/env python
"""Does the Langevin corrector compose in the same window the injected correction does?

Step 27 of ``plans/06-is-the-gap-the-samplers-or-the-models`` (plan 04), plus the
session read-out every parallel session shares: an eight-seed sheet per condition with the
joint prompt, plain product-of-experts and the corrector side by side, and the fidelity
addition, where the corrector runs only on the last fifteen steps on top of the rank-32
λ 1.2 adapter run.

Stages (each one process; the adapter stage attaches the adapter, so it never shares a
process with a pure-corrector stage):

    --window-sweep   cat × dog seeds 9 to 12, nine ten-step corrector windows plus an
                     all-50 column, at the k on the flat part of step 26's curve and the
                     c the step-size search picked. Writes window_curves_mcmc.json.
    --sheet          both pairs, seeds 9 to 16: the joint prompt, plain PoE (the composer at
                     k=0, byte-identical to the reference sampler) and the corrector on all
                     50 steps. Writes sheet_scores.json.
    --tail           both pairs, seeds 9 to 16, the rank-32 λ 1.2 run with k ∈ {0, 5, 20}
                     corrector steps on steps 35 to 49 only. Writes tail_fidelity.json with
                     compose and Laplacian-variance sharpness per render and the verdict.
    --clean-tail     the fidelity fix under test: the adapter on steps [0, cutoff) only, the
                     frozen model's plain PoE step after, and k corrector steps on the frozen
                     score inside steps 35 to 49; cutoff ∈ {20, 30}, k ∈ {0, 5, 20}. Bar: mean
                     sharpness back inside the plain-PoE seed band (mean minus one SD) with
                     composition within one seed of the adapter-alone run. Writes clean_tail.json.
    --figures        the sliding-window strip matched to the injected-correction figure,
                     and one sheet per pair per condition, each with a .json sidecar.
    --wandb          log every sheet and sidecar that exists to W&B and print the run id.

Scoring is the validated instance-count compose scorer for the animal pair. For the
control pair a butterfly × a flower meadow the validated rule does not apply (a meadow is
not an animal), so its compose read is a per-concept detection, a butterfly box and a
flower box both present, reported as unvalidated beside the raw animal count.

The thresholds for the fidelity read sit here as constants, written before the run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from poe_repair.composers import poe_langevin as cmp_lg  # noqa: E402
from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term import window_grid as wg  # noqa: E402
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants (plan 04's review file names them)
# ---------------------------------------------------------------------------

FAILING_PAIR = "a_cat__x__a_dog"
COMPOSING_PAIR = "a_butterfly__x__a_flower_meadow"
PAIRS = (FAILING_PAIR, COMPOSING_PAIR)
STRIP_SEEDS = (9, 10, 11, 12)                   # the injected-correction figure's rows
SHEET_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)   # the held-out seeds every session reports on
CONCEPT_QUERIES = {FAILING_PAIR: ("cat", "dog"), COMPOSING_PAIR: ("butterfly", "flowers")}
CONCEPT_CONF = 0.30                             # the validated scorer's own confidence floor

# The fidelity addition: the corrector on the tail of the adapter run.
TAIL_WINDOW = (35, 50)
TAIL_K = (0, 5, 20)
TAIL_LAMBDA = 1.2
ADAPTER_CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
ADAPTER_RANK = 32
FIDELITY_MIN_SHARPNESS_RISE = 0.10   # mean Laplacian variance at k=20 must be at least 10% above k=0
FIDELITY_MAX_COMPOSE_LOSS_SEEDS = 1  # composed seeds at k=20 may fall by at most one from k=0

# The clean-tail conditions (plan 04, task group 5): the adapter only on the early steps that
# set the composition, the frozen model's own PoE step after that, and a corrector on the
# frozen score at low noise. The bar is session B's: sharpness back inside the plain-PoE seed
# band while composition holds within one seed.
CLEAN_TAIL_CUTOFFS = (20, 30)            # the adapter is on for steps [0, cutoff)
CLEAN_TAIL_K = (0, 5, 20)                # corrector steps on the frozen score inside TAIL_WINDOW
CLEAN_TAIL_SCORE = "frozen"
CLEAN_MIN_MONO_GAIN = 0.05               # support: mean DINOv2 cosine distance to the seed's joint-prompt render
                                         # falls by at least this much against the adapter-alone run (8-seed means;
                                         # the size the training-longer finding read as a real move)
CLEAN_MAX_COMPOSE_LOSS_SEEDS = 1         # composed seeds may fall by at most one from the adapter-alone run
# Laplacian variance stays a reported secondary read: on the plain-PoE references it is heavy-tailed
# (cat x dog seeds 11, 14, 15 render grainy at 275 to 499 against about 15 elsewhere), so a band
# built from it cannot separate conditions. Measured 2026-09-06 02:40, before this grid ran.

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector")
SEARCH_JSON = OUT / "step_size_search.json"
CURVES_JSON = OUT / "residual_curves.json"
WINDOW_JSON = OUT / "window_curves_mcmc.json"
SHEET_JSON = OUT / "sheet_scores.json"
TAIL_JSON = OUT / "tail_fidelity.json"
CLEAN_JSON = OUT / "clean_tail.json"
FIG_DIR = REPO / "paper/iclr/figures/when-the-correction-arrives/mcmc"
STRIP_NAME = "samples-as-a-ten-step-corrector-window-slides"
RESULTS_DIR = REPO / "artifacts/results/is-the-gap-the-samplers-or-the-models"
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"
GREEN_FRAME_RULE = "compose == 1, i.e. the validated instance count (GroundingDINO 'animal', conf >= 0.30, NMS iou < 0.5) >= 2"


def _host() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    if 1.0 - st.f_bavail / max(st.f_blocks, 1) >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at or over 90%, refusing to write")


def picked_c() -> float:
    d = json.loads(SEARCH_JSON.read_text())
    if d.get("picked_c") is None:
        raise SystemExit("no picked c in the step-size search")
    return float(d["picked_c"])


def flat_k() -> int:
    from corrector_residual_curve import flat_k as _fk  # noqa: E402
    return int(_fk(CURVES_JSON))


def columns() -> list[tuple[str, tuple[int, int] | None]]:
    cols = [(f"{a}-{b}", (a, b)) for a, b in wg.windows()]
    cols.append(("all 50", None))
    return cols


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_DEVICE = None


def _dev():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def laplacian_var(png: Path) -> float:
    from lambda_window_grid import _laplacian_var
    return float(_laplacian_var(png))


_EMB = None


def dino_dist_to_mono(png: Path, pair: str, seed: int) -> float | None:
    """DINOv2 ViT-S/14 cosine distance between a render and the seed's joint-prompt render,
    the compose scorer's own embedder; lower is nearer the clean image."""
    global _EMB
    from poe_repair.experiments.compose_scorer_validation.scorer import _Embedders, _cosine_distance
    mono = OUT / "sheet" / "references" / pair / f"seed_{seed}" / "mono.png"
    if not mono.exists():
        return None
    if _EMB is None:
        _EMB = _Embedders(device=_dev())
    e = _EMB.dino([png, mono])
    return float(_cosine_distance(e[0], e[1]))


def score_png(png: Path, pair: str) -> dict:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
    from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics

    n, boxes = count_instances(png, device=_dev())
    qa, qb = CONCEPT_QUERIES[pair]
    dets = vmetrics.detect_boxes(png, [qa, qb], device=_dev())
    conf = {}
    for q in (qa, qb):
        c = [d["confidence"] for d in dets if d.get("label", "").strip().lower() == q]
        conf[q] = float(max(c)) if c else 0.0
    both = bool(conf[qa] >= CONCEPT_CONF and conf[qb] >= CONCEPT_CONF)
    row = {"n_instances": int(n), "compose": int(n >= 2),
           "concept_conf": conf, "both_concepts_present": both}
    if pair == FAILING_PAIR:
        row["compose_read"] = "validated instance count"
        row["composed"] = bool(n >= 2)
    else:
        row["compose_read"] = "unvalidated: both concepts detected (the validated rule counts animals, a meadow is not one)"
        row["composed"] = both
    return row


# ---------------------------------------------------------------------------
# The renders
# ---------------------------------------------------------------------------


def window_sweep(k: int, c: float) -> int:
    _disk_guard(OUT)
    ctx = make_ctx()
    cells = []
    print(f"window sweep: k={k} c={c} host={_host()}", flush=True)
    for seed in STRIP_SEEDS:
        cell = cell_from_slug(FAILING_PAIR, seed)
        for label, win in columns():
            t0 = time.time()
            png = cmp_lg.run(cell, ctx, k=k, c=c, corrector_window=win,
                             exp_name="interaction_term/corrector/window")
            cells.append({"pair": FAILING_PAIR, "seed": seed, "column": label,
                          "window": None if win is None else list(win), "k": k, "c": c,
                          "png": str(png), "seconds": round(time.time() - t0, 1)})
            print(f"[{time.strftime('%H:%M:%S')}] seed {seed} {label}: {png.name} ({time.time() - t0:.0f}s)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    for cl in cells:
        cl.update(score_png(Path(cl["png"]), cl["pair"]))
    per_col = {}
    for label, _ in columns():
        rows = [cl for cl in cells if cl["column"] == label]
        per_col[label] = {"composed_of_4": sum(int(cl["composed"]) for cl in rows), "n": len(rows)}
    WINDOW_JSON.write_text(json.dumps({
        "pair": FAILING_PAIR, "seeds": list(STRIP_SEEDS), "k": k, "c": c,
        "columns": [lab for lab, _ in columns()], "green_frame_rule": GREEN_FRAME_RULE,
        "corrector": "k unadjusted Langevin steps on the guided PoE score inside the window, delta_t = c * beta_t; plain PoE outside",
        "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, 1024 square, the seed's cached initial noise",
        "per_column": per_col, "cells": cells, "host": _host(),
    }, indent=1))
    print("per column composed of 4:", {k_: v["composed_of_4"] for k_, v in per_col.items()}, flush=True)
    return 0


def _ref_dir(pair: str, seed: int) -> Path:
    d = OUT / "sheet" / "references" / pair / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_references(ctx, pair: str, seed: int) -> dict[str, Path]:
    """The joint prompt (Mono) through plain CFG and plain PoE through the composer at k=0,
    from the seed's cached initial noise. Both skip if already on disk."""
    cell = cell_from_slug(pair, seed)
    d = _ref_dir(pair, seed)
    mono = d / "mono.png"
    if not mono.exists():
        init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
        emb = encode_pair(cell, ctx)
        seq_j, pool_j = get_joint_embeds(cell, ctx)
        out = run_cfg(init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                      seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                      guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                      height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                      device=ctx.device, dtype=ctx.dtype)
        write_decoded_image(out.image, mono)
        gc.collect(); torch.cuda.empty_cache()
    poe = cmp_lg.run(cell, ctx, k=0, c=0.0, exp_name="interaction_term/corrector/sheet",
                     method_name_override="poe")
    return {"mono": mono, "poe": poe}


def sheet(k: int, c: float) -> int:
    _disk_guard(OUT)
    ctx = make_ctx()
    print(f"sheet: k={k} c={c} seeds={SHEET_SEEDS} host={_host()}", flush=True)
    rows = []
    for pair in PAIRS:
        for seed in SHEET_SEEDS:
            t0 = time.time()
            refs = render_references(ctx, pair, seed)
            cell = cell_from_slug(pair, seed)
            corr = cmp_lg.run(cell, ctx, k=k, c=c, corrector_window=None,
                              exp_name="interaction_term/corrector/sheet")
            rows.append({"pair": pair, "seed": seed, "mono": str(refs["mono"]), "poe": str(refs["poe"]),
                         "corrector": str(corr), "k": k, "c": c, "seconds": round(time.time() - t0, 1)})
            print(f"[{time.strftime('%H:%M:%S')}] {pair} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    for r in rows:
        for col in ("mono", "poe", "corrector"):
            r[f"{col}_score"] = score_png(Path(r[col]), r["pair"])
    summary = {}
    for pair in PAIRS:
        pr = [r for r in rows if r["pair"] == pair]
        summary[pair] = {col: {"composed_of_8": sum(int(r[f'{col}_score']['composed']) for r in pr),
                               "compose_rate": sum(int(r[f'{col}_score']['composed']) for r in pr) / max(len(pr), 1),
                               "read": pr[0][f"{col}_score"]["compose_read"] if pr else None}
                         for col in ("mono", "poe", "corrector")}
    SHEET_JSON.write_text(json.dumps({
        "seeds": list(SHEET_SEEDS), "k": k, "c": c, "columns": ["mono", "poe", "corrector"],
        "corrector": f"k={k} unadjusted Langevin steps on the guided PoE score at every one of the 50 levels, delta_t = {c} * beta_t",
        "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, 1024 square, the seed's cached initial noise (from-seed fp16 randn, which the cache equals)",
        "summary": summary, "rows": rows, "host": _host(),
    }, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def tail(c: float) -> int:
    _disk_guard(OUT)
    import lambda_boundary_probe as lbp
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe

    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = ADAPTER_RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], ADAPTER_CHECKPOINT)
    print(f"tail: adapter rank {ADAPTER_RANK} attached, n_matched={info['n_matched']} "
          f"n_loaded={info['n_loaded']} checkpoint_step={info['checkpoint_step']}; c={c} host={_host()}", flush=True)
    rows = []
    for pair in PAIRS:
        for seed in SHEET_SEEDS:
            cell = cell_from_slug(pair, seed)
            init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            for k in TAIL_K:
                d = OUT / "tail" / pair / f"seed_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                png = d / f"lambda_{TAIL_LAMBDA}_k{k:03d}_w{TAIL_WINDOW[0]}-{TAIL_WINDOW[1]}.png"
                t0 = time.time()
                if not png.exists():
                    out = run_lora_langevin_windowed_poe(
                        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                        device=ctx.device, dtype=ctx.dtype,
                        lambda_value=TAIL_LAMBDA, k=k, c=c, corrector_window=TAIL_WINDOW,
                        noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
                    )
                    write_decoded_image(out.image, png)
                    (d / f"summary_k{k:03d}.json").write_text(json.dumps(
                        {kk: v for kk, v in out.extras.items() if kk != "per_step"} |
                        {"median_chain_disp_rel_in_window": float(np.median(
                            [r["chain_disp_rel"] for r in out.extras["per_step"] if r["k_applied"] > 0] or [0.0]))},
                        indent=1))
                    del out
                    gc.collect(); torch.cuda.empty_cache()
                rows.append({"pair": pair, "seed": seed, "k": k, "png": str(png),
                             "seconds": round(time.time() - t0, 1)})
                print(f"[{time.strftime('%H:%M:%S')}] {pair} seed {seed} k={k} ({time.time() - t0:.0f}s)", flush=True)
    for r in rows:
        r.update(score_png(Path(r["png"]), r["pair"]))
        r["sharpness_laplacian_var"] = laplacian_var(Path(r["png"]))
    result = tail_verdict(rows, c)
    TAIL_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1)); print("BRANCH:", result["branch"], result["reasons"], flush=True)
    return 0


def clean_tail(c: float) -> int:
    """Adapter λ 1.2 on steps [0, cutoff), the frozen model's plain PoE step after, and k
    corrector steps on the frozen score inside TAIL_WINDOW. Baseline: the adapter on all 50
    steps (the tail stage's k=0 render, reused). Target band: the plain-PoE references' sharpness."""
    _disk_guard(OUT)
    import lambda_boundary_probe as lbp
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe

    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = ADAPTER_RANK
    info = lbp._attach_and_load_lora(ctx.models["unet"], ADAPTER_CHECKPOINT)
    print(f"clean tail: adapter rank {ADAPTER_RANK} attached, n_matched={info['n_matched']} "
          f"checkpoint_step={info['checkpoint_step']}; c={c} host={_host()}", flush=True)
    rows = []
    for pair in PAIRS:
        for seed in SHEET_SEEDS:
            cell = cell_from_slug(pair, seed)
            init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            for cutoff in CLEAN_TAIL_CUTOFFS:
                for k in CLEAN_TAIL_K:
                    d = OUT / "clean_tail" / pair / f"seed_{seed}"
                    d.mkdir(parents=True, exist_ok=True)
                    png = d / f"lambda_{TAIL_LAMBDA}_on0-{cutoff}_k{k:03d}_{CLEAN_TAIL_SCORE}_w{TAIL_WINDOW[0]}-{TAIL_WINDOW[1]}.png"
                    t0 = time.time()
                    if not png.exists():
                        out = run_lora_langevin_windowed_poe(
                            init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                            seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                            seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                            guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                            device=ctx.device, dtype=ctx.dtype,
                            lambda_value=TAIL_LAMBDA, k=k, c=c, corrector_window=TAIL_WINDOW,
                            noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
                            lambda_window=(0, cutoff), corrector_score=CLEAN_TAIL_SCORE,
                        )
                        write_decoded_image(out.image, png)
                        del out
                        gc.collect(); torch.cuda.empty_cache()
                    rows.append({"pair": pair, "seed": seed, "cutoff": cutoff, "k": k, "png": str(png),
                                 "seconds": round(time.time() - t0, 1)})
                    print(f"[{time.strftime('%H:%M:%S')}] {pair} seed {seed} cutoff {cutoff} k={k} ({time.time() - t0:.0f}s)", flush=True)
    for r in rows:
        r.update(score_png(Path(r["png"]), r["pair"]))
        r["sharpness_laplacian_var"] = laplacian_var(Path(r["png"]))
        r["dino_dist_to_mono"] = dino_dist_to_mono(Path(r["png"]), r["pair"], r["seed"])
    result = clean_tail_verdict(rows, c)
    CLEAN_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps({k_: v for k_, v in result.items() if k_ in ("branch", "reasons", "band", "summary")}, indent=1), flush=True)
    return 0


def clean_tail_verdict(rows: list[dict], c: float) -> dict:
    """Baseline and band come from files the earlier stages wrote: the adapter-alone renders in
    tail_fidelity.json (k=0) and the plain-PoE references in sheet_scores.json."""
    tail = json.loads(TAIL_JSON.read_text()) if TAIL_JSON.exists() else None
    sheet = json.loads(SHEET_JSON.read_text()) if SHEET_JSON.exists() else None
    band, base = {}, {}
    for pair in PAIRS:
        poe_sharp = []
        if sheet:
            for r in sheet["rows"]:
                if r["pair"] == pair:
                    poe_sharp.append(laplacian_var(Path(r["poe"])))
        else:
            for seed in SHEET_SEEDS:
                q = OUT / "sheet" / "pairs" / pair / f"seed_{seed}" / "poe" / "poe.png"
                if q.exists():
                    poe_sharp.append(laplacian_var(q))
        band[pair] = {"plain_poe_mean": float(np.mean(poe_sharp)) if poe_sharp else None,
                      "plain_poe_sd": float(np.std(poe_sharp)) if poe_sharp else None,
                      "n": len(poe_sharp),
                      "median": float(np.median(poe_sharp)) if poe_sharp else None}
        if tail:
            t0 = [r for r in tail["rows"] if r["pair"] == pair and r["k"] == 0]
            dm = [dino_dist_to_mono(Path(r["png"]), pair, r["seed"]) for r in t0]
            dm = [x for x in dm if x is not None]
            base[pair] = {"composed_of_8": sum(int(r["composed"]) for r in t0),
                          "mean_sharpness_laplacian_var": float(np.mean([r["sharpness_laplacian_var"] for r in t0])) if t0 else None,
                          "mean_dino_dist_to_mono": float(np.mean(dm)) if dm else None,
                          "source": "tail_fidelity.json, k=0 (the adapter alone on all 50 steps)"}
        pp = []
        if sheet:
            for r in sheet["rows"]:
                if r["pair"] == pair:
                    v = dino_dist_to_mono(Path(r["poe"]), pair, r["seed"])
                    if v is not None:
                        pp.append(v)
        band[pair]["plain_poe_mean_dino_dist_to_mono"] = float(np.mean(pp)) if pp else None
    summary = {}
    for pair in PAIRS:
        summary[pair] = {}
        for cutoff in CLEAN_TAIL_CUTOFFS:
            for k in CLEAN_TAIL_K:
                pr = [r for r in rows if r["pair"] == pair and r["cutoff"] == cutoff and r["k"] == k]
                dm = [r["dino_dist_to_mono"] for r in pr if r.get("dino_dist_to_mono") is not None]
                summary[pair][f"cutoff{cutoff}_k{k}"] = {
                    "composed_of_8": sum(int(r["composed"]) for r in pr), "n": len(pr),
                    "mean_dino_dist_to_mono": float(np.mean(dm)) if dm else None,
                    "mean_sharpness_laplacian_var": float(np.mean([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                    "median_sharpness_laplacian_var": float(np.median([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                }
    b = base.get(FAILING_PAIR, {}); b0 = b.get("mean_dino_dist_to_mono")
    supp, breaks = [], []
    for name, v in summary[FAILING_PAIR].items():
        gain = (b0 - v["mean_dino_dist_to_mono"]) if (b0 is not None and v["mean_dino_dist_to_mono"] is not None) else None
        nearer = gain is not None and gain >= CLEAN_MIN_MONO_GAIN
        loss = (b.get("composed_of_8", 0) - v["composed_of_8"]) if b else None
        ctrl = summary[COMPOSING_PAIR][name]["composed_of_8"]
        ctrl_base = base.get(COMPOSING_PAIR, {}).get("composed_of_8", 8)
        v["mono_gain_vs_adapter_alone"] = gain; v["nearer_the_joint_render"] = bool(nearer)
        v["compose_loss_vs_adapter_alone"] = loss; v["control_compose_loss"] = ctrl_base - ctrl
        if nearer and loss is not None and loss <= CLEAN_MAX_COMPOSE_LOSS_SEEDS and (ctrl_base - ctrl) <= CLEAN_MAX_COMPOSE_LOSS_SEEDS:
            supp.append(name)
        elif nearer and loss is not None and loss > CLEAN_MAX_COMPOSE_LOSS_SEEDS:
            breaks.append(name)
    ctrl_fail = [n for n, v in summary[FAILING_PAIR].items() if v["control_compose_loss"] > CLEAN_MAX_COMPOSE_LOSS_SEEDS]
    if b0 is None or not b:
        branch, reasons = "not ready", ["the adapter-alone baseline or the joint-prompt references are missing"]
    elif supp:
        branch = "support"; reasons = [f"{n}: DINOv2 distance to the joint render {summary[FAILING_PAIR][n]['mean_dino_dist_to_mono']:.3f} against the adapter's {b0:.3f} (gain {summary[FAILING_PAIR][n]['mono_gain_vs_adapter_alone']:+.3f}, bar {CLEAN_MIN_MONO_GAIN}) with composed {summary[FAILING_PAIR][n]['composed_of_8']} of 8 against the adapter's {b['composed_of_8']}" for n in supp]
    elif breaks:
        branch = "composition breaks"; reasons = [f"{n}: nearer the joint render but loses {summary[FAILING_PAIR][n]['compose_loss_vs_adapter_alone']} composed seeds" for n in breaks]
    elif ctrl_fail:
        branch = "inconclusive"; reasons = [f"{n}: the control pair lost {summary[FAILING_PAIR][n]['control_compose_loss']} composed seeds" for n in ctrl_fail]
    else:
        best = max(summary[FAILING_PAIR].items(), key=lambda kv: kv[1]["mono_gain_vs_adapter_alone"] or -1)
        branch = "null"; reasons = [f"no condition moves at least {CLEAN_MIN_MONO_GAIN} nearer the joint render while holding composition; best gain "
                                    f"{best[1]['mono_gain_vs_adapter_alone']:+.3f} at {best[0]} (adapter alone {b0:.3f})"]
    return {
        "condition": f"rank-{ADAPTER_RANK} adapter at step 30050, lambda {TAIL_LAMBDA} on steps [0, cutoff), the frozen model's plain PoE step after, "
                     f"plus k Langevin steps on the FROZEN score inside steps {TAIL_WINDOW[0]} to {TAIL_WINDOW[1] - 1}",
        "checkpoint": str(ADAPTER_CHECKPOINT), "c": c, "cutoffs": list(CLEAN_TAIL_CUTOFFS), "k": list(CLEAN_TAIL_K),
        "window": list(TAIL_WINDOW), "lambda": TAIL_LAMBDA, "seeds": list(SHEET_SEEDS),
        "thresholds": {"CLEAN_MIN_MONO_GAIN": CLEAN_MIN_MONO_GAIN, "CLEAN_MAX_COMPOSE_LOSS_SEEDS": CLEAN_MAX_COMPOSE_LOSS_SEEDS},
        "primary_read": "DINOv2 ViT-S/14 cosine distance between the render and the seed's joint-prompt render (the compose scorer's embedder); lower is nearer the clean image",
        "sharpness": "Laplacian variance of the greyscale 1024x1024 render, a secondary read; on the plain-PoE references it is heavy-tailed and cannot set a band",
        "band": band, "baseline_adapter_alone": base, "branch": branch, "reasons": reasons,
        "summary": summary, "rows": rows, "host": _host(),
    }


def tail_verdict(rows: list[dict], c: float) -> dict:
    summary = {}
    for pair in PAIRS:
        summary[pair] = {}
        for k in TAIL_K:
            pr = [r for r in rows if r["pair"] == pair and r["k"] == k]
            summary[pair][str(k)] = {
                "composed_of_8": sum(int(r["composed"]) for r in pr), "n": len(pr),
                "mean_sharpness_laplacian_var": float(np.mean([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                "median_sharpness_laplacian_var": float(np.median([r["sharpness_laplacian_var"] for r in pr])) if pr else None,
                "n_seeds_sharper_than_k0": None,
            }
        for k in TAIL_K[1:]:
            n_up = 0
            for seed in SHEET_SEEDS:
                a = [r for r in rows if r["pair"] == pair and r["k"] == 0 and r["seed"] == seed]
                b = [r for r in rows if r["pair"] == pair and r["k"] == k and r["seed"] == seed]
                if a and b and b[0]["sharpness_laplacian_var"] > a[0]["sharpness_laplacian_var"]:
                    n_up += 1
            summary[pair][str(k)]["n_seeds_sharper_than_k0"] = n_up
    f = summary[FAILING_PAIR]
    g = summary[COMPOSING_PAIR]
    s0, s5, s20 = (f[str(k)]["mean_sharpness_laplacian_var"] for k in TAIL_K)
    rise = s20 / max(s0, 1e-12) - 1.0
    compose_loss = f["0"]["composed_of_8"] - f["20"]["composed_of_8"]
    control_loss = g["0"]["composed_of_8"] - g["20"]["composed_of_8"]
    reasons = []
    if control_loss > FIDELITY_MAX_COMPOSE_LOSS_SEEDS:
        reasons.append(f"the control pair lost {control_loss} of 8 composed seeds at k=20: the corrector breaks what already works")
        branch = "inconclusive"
    elif compose_loss > FIDELITY_MAX_COMPOSE_LOSS_SEEDS:
        branch = "composition breaks"
        reasons.append(f"cat x dog lost {compose_loss} composed seeds at k=20 (bar {FIDELITY_MAX_COMPOSE_LOSS_SEEDS})")
    elif rise >= FIDELITY_MIN_SHARPNESS_RISE and s5 >= s0:
        branch = "support"
        reasons.append(f"mean sharpness rose {rise:+.1%} from k=0 to k=20 (bar +{FIDELITY_MIN_SHARPNESS_RISE:.0%}), "
                       f"k=5 in between, composed seeds {f['0']['composed_of_8']} -> {f['20']['composed_of_8']}")
    elif abs(rise) < FIDELITY_MIN_SHARPNESS_RISE:
        branch = "null"
        reasons.append(f"mean sharpness changed {rise:+.1%} from k=0 to k=20, inside the {FIDELITY_MIN_SHARPNESS_RISE:.0%} band")
    else:
        branch = "no branch fired"
        reasons.append(f"mean sharpness changed {rise:+.1%} (k=5 {s5:.1f}, k=0 {s0:.1f}, k=20 {s20:.1f}); "
                       f"a fall past the band, or a non-monotone rise, is not one of the three branches")
    return {
        "condition": f"rank-{ADAPTER_RANK} adapter at step 30050, lambda {TAIL_LAMBDA} on all 50 steps, "
                     f"plus k Langevin steps on the corrected score inside steps {TAIL_WINDOW[0]} to {TAIL_WINDOW[1] - 1}",
        "checkpoint": str(ADAPTER_CHECKPOINT), "c": c, "k": list(TAIL_K), "window": list(TAIL_WINDOW),
        "lambda": TAIL_LAMBDA, "seeds": list(SHEET_SEEDS),
        "thresholds": {"FIDELITY_MIN_SHARPNESS_RISE": FIDELITY_MIN_SHARPNESS_RISE,
                       "FIDELITY_MAX_COMPOSE_LOSS_SEEDS": FIDELITY_MAX_COMPOSE_LOSS_SEEDS},
        "sharpness": "Laplacian variance of the greyscale 1024x1024 render (scripts/showcase/lambda_window_grid.py::_laplacian_var); higher is sharper",
        "branch": branch, "reasons": reasons, "sharpness_rise_k0_to_k20": rise,
        "summary": summary, "rows": rows, "host": _host(),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _frame(ax, composed: bool | None, title: str = "") -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        if composed is None:
            s.set_color("0.7"); s.set_linewidth(0.8)
        else:
            s.set_color("#1a7a3c" if composed else "#a33"); s.set_linewidth(2.2)
    if title:
        ax.set_title(title, fontsize=6.5, pad=2.0, color=("#1a7a3c" if composed else ("#a33" if composed is False else "0.4")))


def strip_figure() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    d = json.loads(WINDOW_JSON.read_text())
    cols = columns()
    n_rows, n_cols = len(STRIP_SEEDS), len(cols)
    header_in, row_in = 0.95, 2.0
    fig_h = header_in + row_in * n_rows
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(1.55 * n_cols, fig_h),
                             gridspec_kw={"height_ratios": [1, 0.13] * n_rows, "hspace": 0.55, "wspace": 0.04},
                             squeeze=False)
    by = {(c["seed"], c["column"]): c for c in d["cells"]}
    for r, seed in enumerate(STRIP_SEEDS):
        for ci, (label, win) in enumerate(cols):
            ax, tr = axes[2 * r][ci], axes[2 * r + 1][ci]
            cell = by.get((seed, label))
            if cell and Path(cell["png"]).is_file():
                ax.imshow(Image.open(cell["png"]))
                composed = bool(cell["composed"])
                _frame(ax, composed, f"{'composes' if composed else 'blended'} · {cell['n_instances']} inst")
            else:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=7, color="0.6", transform=ax.transAxes)
                _frame(ax, None)
            tr.set_xticks([]); tr.set_yticks([])
            for s in tr.spines.values():
                s.set_visible(False)
            tr.set_xlim(0, wg.NUM_STEPS); tr.set_ylim(0, 1)
            tr.add_patch(Rectangle((0, 0.15), wg.NUM_STEPS, 0.7, facecolor="0.88", edgecolor="none"))
            a, b = (0, wg.NUM_STEPS) if win is None else win
            tr.add_patch(Rectangle((a, 0.15), b - a, 0.7, facecolor="tab:green", edgecolor="none"))
            tr.set_xlabel(label if win is None else f"{a}–{b}", fontsize=7, labelpad=1.5)
        axes[2 * r][0].set_ylabel(f"seed {seed}", fontsize=8)
    fig.suptitle(f"The Langevin corrector (k = {d['k']}, δ_t = {d['c']}·β_t) switched on in one window only, a cat × a dog\n"
                 f"green bar = steps where it acts; frame colour = the detector's verdict (green: instance count ≥ 2)",
                 fontsize=9, y=1 - 0.10 / fig_h, verticalalignment="top")
    fig.subplots_adjust(top=1 - header_in / fig_h, bottom=0.06, left=0.05, right=0.995)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{STRIP_NAME}.png"
    fig.savefig(out, dpi=170); fig.savefig(FIG_DIR / f"{STRIP_NAME}.pdf")
    plt.close(fig)
    (FIG_DIR / f"{STRIP_NAME}.json").write_text(json.dumps({
        "drawn_from": str(WINDOW_JSON), "pair": FAILING_PAIR, "seeds": list(STRIP_SEEDS),
        "columns": [lab for lab, _ in cols], "k": d["k"], "c": d["c"],
        "green_frame_rule": GREEN_FRAME_RULE,
        "matched_to": "paper/iclr/figures/when-the-correction-arrives/poe/samples-as-a-ten-step-window-slides.png: same pair, same seeds, same nine windows, same border rule; one extra column with the corrector on all 50 steps",
        "per_column_composed_of_4": {k_: v["composed_of_4"] for k_, v in d["per_column"].items()},
        "cells": [{k_: c[k_] for k_ in ("seed", "column", "window", "png", "n_instances", "composed")} for c in d["cells"]],
        "caption_owes": ["the corrector runs at one k, which is a compute budget rather than a property of the problem",
                         "the green border encodes the detector's verdict, which disagrees with the eye on this pair often enough that the eye count is quoted beside it",
                         "this is a behavioural comparison and not an attribution: at steps 0 to 10 the sampler's and the model's errors cannot be told apart"],
    }, indent=1))
    return out


def sheet_figures() -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    outs = []
    short = {FAILING_PAIR: "cat-dog", COMPOSING_PAIR: "butterfly-meadow"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def draw(tiles, col_titles, name, title, side):
        n_rows, n_cols = len(SHEET_SEEDS), len(col_titles)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7 * n_cols, 1.7 * n_rows + 1.2), squeeze=False)
        for r, seed in enumerate(SHEET_SEEDS):
            for ci, ct in enumerate(col_titles):
                ax = axes[r][ci]
                t = tiles.get((seed, ct))
                if t and Path(t["png"]).is_file():
                    ax.imshow(Image.open(t["png"]).resize((384, 384)))
                    _frame(ax, bool(t["composed"]), t.get("label", ""))
                else:
                    _frame(ax, None); ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                if r == 0:
                    ax.set_title(f"{ct}\n{ax.get_title()}", fontsize=7, pad=3)
                if ci == 0:
                    ax.set_ylabel(f"seed {seed}", fontsize=8)
        fig.suptitle(title, fontsize=9, y=0.995)
        fig.subplots_adjust(top=1 - 1.1 / (1.7 * n_rows + 1.2), bottom=0.01, left=0.04, right=0.995, hspace=0.18, wspace=0.03)
        out = RESULTS_DIR / f"{name}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(side, indent=1))
        outs.append(out)

    if SHEET_JSON.exists():
        d = json.loads(SHEET_JSON.read_text())
        for pair in PAIRS:
            rows = [r for r in d["rows"] if r["pair"] == pair]
            tiles, titles = {}, ["joint prompt", "plain PoE", f"PoE + corrector k={d['k']}"]
            for r in rows:
                for col, ct in zip(("mono", "poe", "corrector"), titles):
                    sc = r[f"{col}_score"]
                    tiles[(r["seed"], ct)] = {"png": r[col], "composed": sc["composed"],
                                              "label": f"{sc['n_instances']} inst" + ("" if pair == FAILING_PAIR else f" · b {sc['concept_conf'].get('butterfly', 0):.2f} f {sc['concept_conf'].get('flowers', 0):.2f}")}
            s = d["summary"][pair]
            counts = " · ".join(f"{ct}: {s[col]['composed_of_8']} of 8" for col, ct in zip(("mono", "poe", "corrector"), titles))
            draw(tiles, titles, f"corrector-{short[pair]}-eight-seed-sheet",
                 f"{pair.replace('__x__', ' × ').replace('_', ' ')}: {counts}\n"
                 f"corrector k={d['k']}, δ_t={d['c']}·β_t on all 50 steps; frame = {'validated instance count' if pair == FAILING_PAIR else 'both concepts detected (unvalidated)'}",
                 {"drawn_from": str(SHEET_JSON), "pair": pair, "seeds": list(SHEET_SEEDS), "columns": titles,
                  "k": d["k"], "c": d["c"], "compose_rate": {ct: s[col] for col, ct in zip(("mono", "poe", "corrector"), titles)},
                  "sampler": d["sampler"], "frame_rule": rows[0]["corrector_score"]["compose_read"] if rows else None,
                  "tiles": [{"seed": r["seed"], col: r[col], f"{col}_score": r[f"{col}_score"]} for r in rows for col in ("mono", "poe", "corrector")]})
    if TAIL_JSON.exists() and SHEET_JSON.exists():
        t = json.loads(TAIL_JSON.read_text())
        d = json.loads(SHEET_JSON.read_text())
        for pair in PAIRS:
            tiles = {}
            titles = ["joint prompt", "plain PoE", f"adapter λ{TAIL_LAMBDA} (k=0)"] + \
                     [f"adapter + corrector k={k}\nsteps {TAIL_WINDOW[0]}-{TAIL_WINDOW[1] - 1}" for k in TAIL_K[1:]]
            for r in (x for x in d["rows"] if x["pair"] == pair):
                for col, ct in zip(("mono", "poe"), titles[:2]):
                    sc = r[f"{col}_score"]
                    tiles[(r["seed"], ct)] = {"png": r[col], "composed": sc["composed"], "label": f"{sc['n_instances']} inst"}
            for r in (x for x in t["rows"] if x["pair"] == pair):
                ct = titles[2 + TAIL_K.index(r["k"])]
                tiles[(r["seed"], ct)] = {"png": r["png"], "composed": r["composed"],
                                          "label": f"{r['n_instances']} inst · sharp {r['sharpness_laplacian_var']:.0f}"}
            s = t["summary"][pair]
            counts = " · ".join(f"k={k}: {s[str(k)]['composed_of_8']} of 8, sharp {s[str(k)]['mean_sharpness_laplacian_var']:.0f}" for k in TAIL_K)
            draw(tiles, titles, f"corrector-on-adapter-tail-{short[pair]}-eight-seed-sheet",
                 f"{pair.replace('__x__', ' × ').replace('_', ' ')}: {counts}\n"
                 f"rank-32 adapter, λ {TAIL_LAMBDA}, corrector on steps {TAIL_WINDOW[0]} to {TAIL_WINDOW[1] - 1} only; printed branch \"{t['branch']}\"",
                 {"drawn_from": [str(SHEET_JSON), str(TAIL_JSON)], "pair": pair, "seeds": list(SHEET_SEEDS), "columns": titles,
                  "condition": t["condition"], "c": t["c"], "thresholds": t["thresholds"], "branch": t["branch"], "reasons": t["reasons"],
                  "summary": s, "sharpness": t["sharpness"],
                  "tiles": [{k_: r[k_] for k_ in ("seed", "k", "png", "n_instances", "composed", "sharpness_laplacian_var", "concept_conf")} for r in t["rows"] if r["pair"] == pair]})
    if CLEAN_JSON.exists():
        t = json.loads(CLEAN_JSON.read_text())
        d = json.loads(SHEET_JSON.read_text()) if SHEET_JSON.exists() else None
        tf = json.loads(TAIL_JSON.read_text()) if TAIL_JSON.exists() else None
        for pair in PAIRS:
            tiles = {}
            titles = ["joint prompt", "plain PoE", f"adapter λ{TAIL_LAMBDA} all 50"]
            for cutoff in CLEAN_TAIL_CUTOFFS:
                for k in CLEAN_TAIL_K:
                    titles.append(f"adapter on 0-{cutoff - 1}\n+ k={k} frozen-score\ncorrector 35-49")
            if d:
                for r in (x for x in d["rows"] if x["pair"] == pair):
                    for col, ct in zip(("mono", "poe"), titles[:2]):
                        sc = r[f"{col}_score"]
                        tiles[(r["seed"], ct)] = {"png": r[col], "composed": sc["composed"],
                                                  "label": f"{sc['n_instances']} inst · sharp {laplacian_var(Path(r[col])):.0f}"}
            if tf:
                for r in (x for x in tf["rows"] if x["pair"] == pair and x["k"] == 0):
                    tiles[(r["seed"], titles[2])] = {"png": r["png"], "composed": r["composed"],
                                                     "label": f"{r['n_instances']} inst · sharp {r['sharpness_laplacian_var']:.0f}"}
            for r in (x for x in t["rows"] if x["pair"] == pair):
                ct = titles[3 + CLEAN_TAIL_CUTOFFS.index(r["cutoff"]) * len(CLEAN_TAIL_K) + CLEAN_TAIL_K.index(r["k"])]
                dd = r.get("dino_dist_to_mono")
                tiles[(r["seed"], ct)] = {"png": r["png"], "composed": r["composed"],
                                          "label": f"{r['n_instances']} inst · d(joint) {dd:.2f} · sharp {r['sharpness_laplacian_var']:.0f}" if dd is not None else f"{r['n_instances']} inst · sharp {r['sharpness_laplacian_var']:.0f}"}
            s_ = t["summary"][pair]; bd = t["band"][pair]; bs = t["baseline_adapter_alone"].get(pair, {})
            counts = " · ".join(f"{n}: {v['composed_of_8']} of 8, d(joint) {v['mean_dino_dist_to_mono']:.3f}" for n, v in s_.items() if v.get('mean_dino_dist_to_mono') is not None)
            draw(tiles, titles, f"corrector-clean-tail-{short[pair]}-eight-seed-sheet",
                 f"{pair.replace('__x__', ' × ').replace('_', ' ')}: DINOv2 distance to the joint render, adapter alone {bs.get('mean_dino_dist_to_mono', float('nan')):.3f}, plain PoE {bd.get('plain_poe_mean_dino_dist_to_mono', float('nan')):.3f}; printed branch \"{t['branch']}\"\n{counts}",
                 {"drawn_from": [str(CLEAN_JSON), str(SHEET_JSON), str(TAIL_JSON)], "pair": pair, "seeds": list(SHEET_SEEDS),
                  "columns": titles, "condition": t["condition"], "c": t["c"], "thresholds": t["thresholds"],
                  "band": bd, "baseline_adapter_alone": t["baseline_adapter_alone"].get(pair), "branch": t["branch"],
                  "reasons": t["reasons"], "summary": s_, "sharpness": t["sharpness"],
                  "tiles": [{k_: r[k_] for k_ in ("seed", "cutoff", "k", "png", "n_instances", "composed", "sharpness_laplacian_var", "concept_conf")} for r in t["rows"] if r["pair"] == pair]})
    return outs


def log_wandb() -> str:
    import wandb
    run = wandb.init(project=WANDB_PROJECT.split("/")[1], entity=WANDB_PROJECT.split("/")[0],
                     name=f"scope06-corrector-window-and-sheets-{time.strftime('%Y%m%d-%H%M')}",
                     job_type="corrector", config={"scope": "06-is-the-gap-the-samplers-or-the-models",
                                                   "plans": ["02", "03", "04"]})
    imgs = {}
    for p in sorted(RESULTS_DIR.glob("corrector-*eight-seed-sheet.png")):
        imgs[f"sheets/{p.stem}"] = wandb.Image(str(p))
    for p in (FIG_DIR / f"{STRIP_NAME}.png", FIG_DIR / "how-much-of-the-correction-a-corrector-removes.png"):
        if p.exists():
            imgs[f"figures/{p.stem}"] = wandb.Image(str(p))
    if imgs:
        run.log(imgs)
    art = wandb.Artifact("scope06-corrector-sidecars", type="results")
    for p in [SEARCH_JSON, CURVES_JSON, WINDOW_JSON, SHEET_JSON, TAIL_JSON, CLEAN_JSON] + sorted(OUT.glob("verdict_c*.json")) + \
             sorted(RESULTS_DIR.glob("corrector-*eight-seed-sheet.json")) + sorted(FIG_DIR.glob("*.json")):
        if p.exists():
            art.add_file(str(p), name=p.name)
    run.log_artifact(art)
    summ = {}
    if SHEET_JSON.exists():
        summ["sheet"] = json.loads(SHEET_JSON.read_text())["summary"]
    if TAIL_JSON.exists():
        t = json.loads(TAIL_JSON.read_text()); summ["tail"] = {"branch": t["branch"], "summary": t["summary"]}
    if CLEAN_JSON.exists():
        t = json.loads(CLEAN_JSON.read_text()); summ["clean_tail"] = {"branch": t["branch"], "summary": t["summary"], "band": t["band"]}
    if WINDOW_JSON.exists():
        summ["window"] = json.loads(WINDOW_JSON.read_text())["per_column"]
    if (OUT / "verdict.json").exists():
        v = json.loads((OUT / "verdict.json").read_text()); summ["residual_branch"] = v.get("branch")
    run.summary.update(summ)
    rid = run.id
    run.finish()
    (OUT / "wandb_run_id.txt").write_text(f"{WANDB_PROJECT}/{rid}\n")
    print("wandb run:", f"{WANDB_PROJECT}/{rid}")
    return rid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-sweep", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--clean-tail", action="store_true", help="adapter early, frozen tail, corrector on the frozen score")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--k", type=int, default=None, help="corrector count; default: the flat-part k from residual_curves.json")
    ap.add_argument("--c", type=float, default=None, help="step-size multiplier; default: the picked c")
    args = ap.parse_args()
    c = args.c if args.c is not None else picked_c()
    if args.window_sweep or args.sheet:
        k = args.k if args.k is not None else flat_k()
        if args.window_sweep:
            window_sweep(k, c)
        if args.sheet:
            sheet(k, c)
    if args.tail:
        tail(c)
    if args.clean_tail:
        clean_tail(c)
    if args.figures:
        if WINDOW_JSON.exists():
            print(strip_figure())
        for p in sheet_figures():
            print(p)
    if args.wandb:
        log_wandb()
    if not any([args.window_sweep, args.sheet, args.tail, args.clean_tail, args.figures, args.wandb]):
        ap.error("pass at least one stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
