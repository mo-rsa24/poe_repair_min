#!/usr/bin/env python
"""Does handing the denoising tail back to frozen SDXL make the adapter's renders crisper?

Stage A of scope 09's crispness question. The adapter is attached over denoising steps
[0, k) and detached over [k, 50), so the frozen model's plain product-of-experts step draws
the tail. k sweeps 25, 29, 33, 36, 40 and 50; k = 50 is the adapter on the whole path and is
the control every other k is measured against.

No training. Three checkpoints that already exist are loaded and rendered from.

This refines an earlier reading of the same question. `scripts/corrector_window_sweep.py`
--clean-tail asked it on one checkpoint at cutoffs 20 and 30 and read the answer pooled over
eight seeds, landing at a 0.038 mean gain against the 0.05 bar. Two things change here: the
cutoff grid is finer and sits where that curve was still moving, and every number is read per
seed. Pooling is wrong on this data. In pool43-all50, an_elephant x a_penguin seed 9 renders
as an engraved plate at every step while seed 10 is photographic at every step, so a mean over
the two describes neither.

Stages, each its own process, because the windowed sampler returns with the adapter enabled
and a plain-CFG render made after it in the same process carries the correction:

    --references   the joint-prompt (Mono) render per cell through plain CFG, with no adapter
                   ever attached in this process, plus the detector read that says whether
                   that reference is usable at all. Writes references.json.
    --render       one checkpoint, all cutoffs, all cells. Skips a render already on disk.
    --score        scores one checkpoint's renders and writes its per-seed verdict.
    --sheet        one sheet per pair per checkpoint, rows are seeds and columns are k.

Bars, fixed here before any render, read PER SEED and never pooled:

    Fidelity     cosine distance from the render to that seed's own Mono render in the compose
                 scorer's own DINOv2 embedding. A seed clears the bar at k if that distance is
                 at least MIN_MONO_GAIN nearer than the same checkpoint at k = 50.
    Composition  the validated instance count from the compose scorer. A seed holds if it does
                 not lose composition it had at k = 50.
    Laplacian    reported beside both and never a bar. It prefers drawn fur to a clean
                 photograph, which is the failure this project has already recorded.

The fidelity bar is only readable on a seed whose Mono reference actually shows both animals,
and on this data most of them do not. Three reads have to agree before a seed anchors the bar: the
detector, the project's existing eye-confirmed target_quality.json, and this run's own look at the
rendered references. The detector alone is too permissive, calling one of the two dogs on cat x dog
seed 10 a cat at confidence 0.35, above the scorer's own 0.30 floor.

The result: cat x dog anchors on seeds 9, 11, 12 and 16; seeds 10, 13 and 14 draw two of one animal
and seed 15 is ambiguous. an_elephant x a_penguin anchors on seed 10 and not on seed 9, which draws
an elephant alone. So the fidelity bar is readable on five of the ten cells.

A seed whose reference is broken or ambiguous is rendered and scored like every other and keeps its
composition and Laplacian numbers, because neither of those compares the render to the reference.
Only its distance-to-reference number is withheld from the count of seeds clearing the bar.
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

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._sampling import run_cfg, write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

EVAL_PAIR = "a_cat__x__a_dog"
SECOND_PAIR = "an_elephant__x__a_penguin"
CELLS: tuple[tuple[str, int], ...] = tuple(
    [(EVAL_PAIR, s) for s in range(9, 17)] + [(SECOND_PAIR, 9), (SECOND_PAIR, 10)]
)
CONCEPT_QUERIES = {EVAL_PAIR: ("cat", "dog"), SECOND_PAIR: ("elephant", "penguin")}
CONCEPT_CONF = 0.30                  # the validated scorer's own confidence floor

# Is a seed's joint-prompt reference usable as an anchor at all? Three sources have to agree,
# because the detector on its own is too permissive here: on cat x dog seed 10 it calls one of
# the two dogs a cat at confidence 0.35, which is above the scorer's own 0.30 floor.
#   detector       both concepts detected at conf >= 0.30 and the validated instance count >= 2
#   target_quality the project's existing eye-confirmed read, outputs/showcase/target_quality/
#                  target_quality.json, field both_present
#   eye            this run's own look at the rendered references, 2026-09-19
# A seed counts toward the fidelity bar only when it is "valid". "broken" and "ambiguous" seeds
# are still rendered and still carry composition and Laplacian numbers, because neither of those
# compares the render to the reference; only the distance-to-reference number is withheld.
REFERENCE_EYE_READ = {
    ("a_cat__x__a_dog", 9): "valid",       # a tabby cat beside a white dog
    ("a_cat__x__a_dog", 10): "broken",     # two dogs, no cat
    ("a_cat__x__a_dog", 11): "valid",      # a white cat and a grey dog, drawn
    ("a_cat__x__a_dog", 12): "valid",      # a white cat beside a brown dog
    ("a_cat__x__a_dog", 13): "broken",     # two dogs, no cat
    ("a_cat__x__a_dog", 14): "broken",     # two cats, no dog
    ("a_cat__x__a_dog", 15): "ambiguous",  # a collie-like dog with a cat-faced animal in front;
                                           # target_quality reads it as two cats
    ("a_cat__x__a_dog", 16): "valid",      # an orange cat beside a brown and white dog
    ("an_elephant__x__a_penguin", 9): "broken",   # an elephant alone
    ("an_elephant__x__a_penguin", 10): "valid",   # an elephant and a penguin
}
TARGET_QUALITY_JSON = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json")

CUTOFFS = (25, 29, 33, 36, 40, 50)   # adapter on for steps [0, k); 50 is the control
CONTROL_CUTOFF = 50
LAMBDA_VALUE = 1.0                   # the adapter's own correction, held fixed across the sweep

MIN_MONO_GAIN = 0.05                 # CLEAN_MIN_MONO_GAIN in scripts/corrector_window_sweep.py,
                                     # the same constant this question was asked with before
MAX_COMPOSE_LOSS_SEEDS = 1           # composed seeds may fall by at most this against k = 50

CHECKPOINTS = {
    "v1_freeze_null_r16_s0_25_30k": (
        "/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/"
        "v1_freeze_null_r16_s0_25/checkpoints/lora_step_030000.pt", 16),
    "pool43_all50_40k": (
        "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_040000.pt", 32),
    "pool43_all50_60k": (
        "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_060000.pt", 32),
}

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/clean_tail_k_sweep")
REF_DIR = OUT / "references"
REF_JSON = OUT / "references.json"
RESULTS_DIR = REPO / "artifacts/results/is-the-gap-the-samplers-or-the-models"
WANDB_PROJECT = "prime_lab/poe-repair-animals-compose"


def _host() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _git_stamp() -> dict:
    import subprocess
    def sh(*a):
        return subprocess.run(a, cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    return {"commit": sh("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(sh("git", "status", "--porcelain"))}


def _disk_guard(root: Path) -> None:
    """Guard the filesystem this script actually writes to, not some other mount."""
    p = root
    while not p.exists():
        p = p.parent
    st = os.statvfs(str(p))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")
    print(f"disk guard: {p} at {used:.0%}", flush=True)


def _device_guard() -> None:
    """Abort rather than share a device someone else is already using."""
    if not torch.cuda.is_available():
        raise SystemExit("device guard: no CUDA device visible")
    free, total = torch.cuda.mem_get_info(0)
    used_gb = (total - free) / 1e9
    if used_gb > 1.0:
        raise SystemExit(f"device guard: {used_gb:.1f} GB already in use on this device, refusing to share")
    print(f"device guard: device clear ({used_gb:.2f} GB used of {total/1e9:.0f} GB)", flush=True)


_DEVICE = None


def _dev():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def laplacian_var(png: Path) -> float:
    from lambda_window_grid import _laplacian_var
    return float(_laplacian_var(png))


_EMB = None


def dino_dist_to_mono(png: Path, pair: str, seed: int) -> float | None:
    """DINOv2 ViT-S/14 cosine distance to the seed's joint-prompt render, the compose
    scorer's own embedder. Lower is nearer the picture the adapter is trying to reach."""
    global _EMB
    from poe_repair.experiments.compose_scorer_validation.scorer import _Embedders, _cosine_distance
    mono = mono_path(pair, seed)
    if not mono.exists():
        return None
    if _EMB is None:
        _EMB = _Embedders(device=_dev())
    e = _EMB.dino([png, mono])
    return float(_cosine_distance(e[0], e[1]))


def score_png(png: Path, pair: str) -> dict:
    """The validated instance count, plus the per-concept detection beside it. Both animals
    in both pairs are animals, so the validated rule applies to both."""
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
    from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics

    n, _boxes = count_instances(png, device=_dev())
    qa, qb = CONCEPT_QUERIES[pair]
    dets = vmetrics.detect_boxes(png, [qa, qb], device=_dev())
    conf = {}
    for q in (qa, qb):
        c = [d["confidence"] for d in dets if d.get("label", "").strip().lower() == q]
        conf[q] = float(max(c)) if c else 0.0
    return {"n_instances": int(n), "composed": bool(n >= 2),
            "compose_read": "validated instance count (GroundingDINO 'animal', conf >= 0.30, NMS iou < 0.5) >= 2",
            "concept_conf": conf,
            "both_concepts_present": bool(conf[qa] >= CONCEPT_CONF and conf[qb] >= CONCEPT_CONF)}


# ---------------------------------------------------------------------------
# Stage 1: the references, in a process that never attaches an adapter
# ---------------------------------------------------------------------------

def mono_path(pair: str, seed: int) -> Path:
    return REF_DIR / pair / f"seed_{seed}" / "mono.png"


def references() -> int:
    _disk_guard(OUT)
    _device_guard()
    ctx = make_ctx()
    print(f"references: no adapter is attached in this process. host={_host()}", flush=True)
    rows = []
    for pair, seed in CELLS:
        png = mono_path(pair, seed)
        png.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if not png.exists():
            cell = cell_from_slug(pair, seed)
            init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
            emb = encode_pair(cell, ctx)
            seq_j, pool_j = get_joint_embeds(cell, ctx)
            out = run_cfg(init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                          seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                          guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                          height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                          device=ctx.device, dtype=ctx.dtype)
            write_decoded_image(out.image, png)
            del out
            gc.collect(); torch.cuda.empty_cache()
        rows.append({"pair": pair, "seed": seed, "png": str(png),
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{time.strftime('%H:%M:%S')}] reference {pair} seed {seed} ({time.time() - t0:.0f}s)", flush=True)

    # Is this reference usable? A joint-prompt render that does not show both animals cannot
    # anchor a distance, so the fidelity bar is unreadable on that seed.
    for r in rows:
        s = score_png(Path(r["png"]), r["pair"])
        r.update(s)
        r["reference_valid"] = bool(s["both_concepts_present"] and s["n_instances"] >= 2)

    REF_JSON.write_text(json.dumps({
        "what": "the joint-prompt (Mono) render per cell, plain CFG, no adapter in the process",
        "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, eta 0, 1024 square, the seed's cached initial noise",
        "reference_valid_rule": "the DETECTOR COMPONENT ONLY: both concepts at conf >= 0.30 and the validated instance count >= 2. On its own it is too permissive (it calls a dog a cat at 0.35 on cat x dog seed 10); the status the fidelity bar keys off is the three-way agreement in reference_validity()",
        "host": _host(), "git": _git_stamp(), "rows": rows,
    }, indent=1))
    ok = [f"{r['pair'].split('__x__')[0]}/{r['seed']}" for r in rows if r["reference_valid"]]
    bad = [f"{r['pair'].split('__x__')[0]}/{r['seed']}" for r in rows if not r["reference_valid"]]
    print(f"references usable ({len(ok)}): {ok}", flush=True)
    print(f"references BROKEN ({len(bad)}), fidelity unreadable there: {bad}", flush=True)
    return 0


def target_quality_read() -> dict[tuple[str, int], bool]:
    """The project's existing eye-confirmed read of which joint-prompt renders show both animals."""
    if not TARGET_QUALITY_JSON.exists():
        raise SystemExit(f"no {TARGET_QUALITY_JSON}; the reference check needs it")
    out = {}
    for r in json.loads(TARGET_QUALITY_JSON.read_text()):
        try:
            seed = int(str(r["seed"]).split("_")[1])
        except (KeyError, IndexError, ValueError):
            continue
        out[(r["pair"], seed)] = bool(r["both_present"])
    return out


def reference_validity() -> dict[tuple[str, int], dict]:
    """Per cell: the three reads and the status the fidelity bar keys off."""
    if not REF_JSON.exists():
        raise SystemExit(f"no {REF_JSON}; run --references first")
    d = json.loads(REF_JSON.read_text())
    tq = target_quality_read()
    out = {}
    for r in d["rows"]:
        key = (r["pair"], r["seed"])
        det = bool(r["reference_valid"])
        tq_ok = tq.get(key)
        eye = REFERENCE_EYE_READ.get(key, "ambiguous")
        if eye == "valid" and det and tq_ok:
            status = "valid"
        elif eye == "broken":
            status = "broken"
        else:
            status = "ambiguous"
        out[key] = {"status": status, "detector": det, "target_quality_both_present": tq_ok,
                    "eye": eye}
    return out


# ---------------------------------------------------------------------------
# Stage 2: the renders, one checkpoint per process
# ---------------------------------------------------------------------------

def render_path(name: str, pair: str, seed: int, cutoff: int) -> Path:
    return OUT / name / pair / f"seed_{seed}" / f"lambda_{LAMBDA_VALUE}_adapter_on_0-{cutoff}.png"


def render(name: str) -> int:
    _disk_guard(OUT)
    _device_guard()
    ckpt, rank = CHECKPOINTS[name]
    import lambda_boundary_probe as lbp
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe

    ctx = make_ctx()
    lbp.LORA_RANK = lbp.LORA_ALPHA = rank
    info = lbp._attach_and_load_lora(ctx.models["unet"], Path(ckpt))
    print(f"render {name}: rank {rank}, n_matched={info['n_matched']}, n_loaded={info['n_loaded']}, "
          f"checkpoint_step={info['checkpoint_step']}, lambda={LAMBDA_VALUE}, cutoffs={CUTOFFS}", flush=True)
    print(f"host={_host()} git={_git_stamp()}", flush=True)
    if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
        raise SystemExit(f"{name}: the adapter matched {info['n_matched']} modules and loaded "
                         f"{info['n_loaded']} tensors; a zero here renders the frozen model under "
                         f"an adapter's name")

    rows = []
    for pair, seed in CELLS:
        cell = cell_from_slug(pair, seed)
        init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
        emb = encode_pair(cell, ctx)
        for cutoff in CUTOFFS:
            png = render_path(name, pair, seed, cutoff)
            png.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if not png.exists():
                out = run_lora_langevin_windowed_poe(
                    init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                    seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                    seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                    seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                    guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                    height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                    device=ctx.device, dtype=ctx.dtype,
                    lambda_value=LAMBDA_VALUE, k=0, c=0.0, corrector_window=None,
                    noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
                    lambda_window=(0, cutoff), corrector_score="frozen",
                )
                # The flag has to have selected something: count the steps the adapter was
                # actually on, rather than trusting the name of the condition.
                on = sum(1 for r in out.extras["per_step"] if r["adapter_on"])
                if on != cutoff:
                    raise SystemExit(f"{name} {pair} seed {seed} cutoff {cutoff}: the adapter was on "
                                     f"for {on} steps, not {cutoff}")
                write_decoded_image(out.image, png)
                del out
                gc.collect(); torch.cuda.empty_cache()
                adapter_on_steps = cutoff
            else:
                adapter_on_steps = cutoff
            rows.append({"checkpoint_name": name, "checkpoint": ckpt, "rank": rank,
                         "lambda_value": LAMBDA_VALUE, "pair": pair, "seed": seed,
                         "cutoff": cutoff, "adapter_on_steps": adapter_on_steps,
                         "png": str(png), "seconds": round(time.time() - t0, 1)})
            print(f"[{time.strftime('%H:%M:%S')}] {name} {pair} seed {seed} k={cutoff} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    (OUT / name / "renders.json").write_text(json.dumps(
        {"checkpoint_name": name, "checkpoint": ckpt, "rank": rank, "lambda_value": LAMBDA_VALUE,
         "cutoffs": list(CUTOFFS), "control_cutoff": CONTROL_CUTOFF,
         "condition": "the adapter on denoising steps [0, k), the frozen model's plain PoE step on [k, 50)",
         "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, eta 0, 1024 square, the seed's cached initial noise",
         "attach_info": {k_: v for k_, v in info.items() if k_ != "config"},
         "host": _host(), "git": _git_stamp(), "rows": rows}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Stage 3: the per-seed read
# ---------------------------------------------------------------------------

def score(name: str) -> int:
    src = OUT / name / "renders.json"
    if not src.exists():
        raise SystemExit(f"no {src}; run --render {name} first")
    d = json.loads(src.read_text())
    valid = reference_validity()
    rows = d["rows"]
    for r in rows:
        png = Path(r["png"])
        r.update(score_png(png, r["pair"]))
        r["sharpness_laplacian_var"] = laplacian_var(png)
        r["dino_dist_to_mono"] = dino_dist_to_mono(png, r["pair"], r["seed"])
        v = valid.get((r["pair"], r["seed"]), {"status": "ambiguous"})
        r["reference_status"] = v["status"]
        r["reference_reads"] = v
        r["reference_valid"] = bool(v["status"] == "valid")
        print(f"[{time.strftime('%H:%M:%S')}] scored {r['pair']} seed {r['seed']} k={r['cutoff']}: "
              f"n={r['n_instances']} dist={r['dino_dist_to_mono']} lap={r['sharpness_laplacian_var']:.0f}",
              flush=True)

    by = {(r["pair"], r["seed"], r["cutoff"]): r for r in rows}
    per_seed = []
    for pair, seed in CELLS:
        ctrl = by[(pair, seed, CONTROL_CUTOFF)]
        entry = {"pair": pair, "seed": seed, "reference_valid": ctrl["reference_valid"],
                 "reference_status": ctrl["reference_status"],
                 "reference_reads": ctrl["reference_reads"],
                 "control_k": CONTROL_CUTOFF,
                 "control_dino_dist_to_mono": ctrl["dino_dist_to_mono"],
                 "control_n_instances": ctrl["n_instances"],
                 "control_composed": ctrl["composed"],
                 "control_sharpness_laplacian_var": ctrl["sharpness_laplacian_var"],
                 "k": {}}
        for cutoff in CUTOFFS:
            r = by[(pair, seed, cutoff)]
            gain = None
            if r["dino_dist_to_mono"] is not None and ctrl["dino_dist_to_mono"] is not None:
                gain = float(ctrl["dino_dist_to_mono"] - r["dino_dist_to_mono"])
            entry["k"][str(cutoff)] = {
                "dino_dist_to_mono": r["dino_dist_to_mono"],
                "mono_gain_vs_k50": gain,
                "fidelity_bar_cleared": (bool(gain is not None and gain >= MIN_MONO_GAIN)
                                         if entry["reference_valid"] else None),
                "n_instances": r["n_instances"], "composed": r["composed"],
                "composition_held_vs_k50": bool(r["composed"] or not ctrl["composed"]),
                "sharpness_laplacian_var": r["sharpness_laplacian_var"],
                "png": r["png"],
            }
        per_seed.append(entry)

    readable = [e for e in per_seed if e["reference_valid"]]
    by_k = {}
    for cutoff in CUTOFFS:
        cleared = [e for e in readable if e["k"][str(cutoff)]["fidelity_bar_cleared"]]
        held = [e for e in per_seed if e["k"][str(cutoff)]["composition_held_vs_k50"]]
        composed_n = sum(1 for e in per_seed if e["k"][str(cutoff)]["composed"])
        by_k[str(cutoff)] = {
            "seeds_clearing_fidelity_bar": len(cleared),
            "of_readable_seeds": len(readable),
            "which": [f"{e['pair']} seed {e['seed']}" for e in cleared],
            "seeds_holding_composition": len(held), "of_all_seeds": len(per_seed),
            "composed_seeds": composed_n,
            "composed_seeds_vs_control": composed_n - sum(
                1 for e in per_seed if e["k"][str(CONTROL_CUTOFF)]["composed"]),
        }

    control_composed = sum(1 for e in per_seed if e["control_composed"])
    best = max((c for c in CUTOFFS if c != CONTROL_CUTOFF),
               key=lambda c: by_k[str(c)]["seeds_clearing_fidelity_bar"])
    bk = by_k[str(best)]
    if len(readable) == 0:
        branch = "inconclusive"
        reasons = ["no seed has a usable joint-prompt reference, so the fidelity bar cannot be read"]
    elif bk["seeds_clearing_fidelity_bar"] == 0:
        branch = "null"
        reasons = [f"no k moves any readable seed at least {MIN_MONO_GAIN} nearer its joint-prompt "
                   f"render than k={CONTROL_CUTOFF}; best k={best} clears 0 of {len(readable)}"]
    elif control_composed - bk["composed_seeds"] > MAX_COMPOSE_LOSS_SEEDS:
        branch = "composition broke"
        reasons = [f"k={best} clears the fidelity bar on {bk['seeds_clearing_fidelity_bar']} of "
                   f"{len(readable)} readable seeds but composed seeds fall from {control_composed} "
                   f"to {bk['composed_seeds']}, more than {MAX_COMPOSE_LOSS_SEEDS}"]
    else:
        branch = "support"
        reasons = [f"k={best} moves {bk['seeds_clearing_fidelity_bar']} of {len(readable)} readable "
                   f"seeds at least {MIN_MONO_GAIN} nearer the joint-prompt render than "
                   f"k={CONTROL_CUTOFF}, with composed seeds {control_composed} -> {bk['composed_seeds']}"]

    out = {"checkpoint_name": name, "checkpoint": d["checkpoint"], "rank": d["rank"],
           "lambda_value": d["lambda_value"], "cutoffs": list(CUTOFFS),
           "control_cutoff": CONTROL_CUTOFF, "condition": d["condition"], "sampler": d["sampler"],
           "thresholds": {"MIN_MONO_GAIN": MIN_MONO_GAIN,
                          "MAX_COMPOSE_LOSS_SEEDS": MAX_COMPOSE_LOSS_SEEDS},
           "reads": {
               "fidelity": "DINOv2 cosine distance to the seed's own Mono render; gain is the control's distance minus this one's, so positive is nearer",
               "composition": "the validated instance count >= 2",
               "sharpness_laplacian_var": "reported, never a bar: it prefers drawn fur to a clean photograph",
           },
           "reference_rule": "a seed anchors the fidelity bar only when the detector, target_quality.json and this run's eye read all say the joint-prompt render shows both animals",
           "readable_seeds": [f"{e['pair']} seed {e['seed']}" for e in readable],
           "unreadable_seeds": [f"{e['pair']} seed {e['seed']} ({e['reference_status']})"
                                for e in per_seed if not e["reference_valid"]],
           "branch": branch, "reasons": reasons, "best_k": best,
           "by_k": by_k, "per_seed": per_seed,
           "host": _host(), "git": _git_stamp(), "rows": rows}
    (OUT / name / "verdict.json").write_text(json.dumps(out, indent=1))

    print(f"\n=== {name} (rank {d['rank']}, lambda {d['lambda_value']}) ===", flush=True)
    print(f"readable seeds ({len(readable)}): {out['readable_seeds']}", flush=True)
    print(f"unreadable (reference broken): {out['unreadable_seeds']}", flush=True)
    for cutoff in CUTOFFS:
        b = by_k[str(cutoff)]
        print(f"  k={cutoff:2d}  fidelity {b['seeds_clearing_fidelity_bar']}/{b['of_readable_seeds']}"
              f"  composed {b['composed_seeds']}/{len(per_seed)}"
              f"  composition held {b['seeds_holding_composition']}/{b['of_all_seeds']}", flush=True)
    print(f"BRANCH: {branch} :: {reasons}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Stage 4: the sheets
# ---------------------------------------------------------------------------

def sheet(name: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    v = json.loads((OUT / name / "verdict.json").read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for pair in (EVAL_PAIR, SECOND_PAIR):
        seeds = [e for e in v["per_seed"] if e["pair"] == pair]
        if not seeds:
            continue
        cols = [("mono", "joint prompt\n(the reference)")] + [
            (str(c), f"adapter on 0-{c - 1}\nfrozen tail {c}-49" if c != CONTROL_CUTOFF
             else "adapter on all 50\n(the control)") for c in CUTOFFS]
        fig, axes = plt.subplots(len(seeds), len(cols), squeeze=False,
                                 figsize=(2.5 * len(cols), 2.75 * len(seeds)))
        for i, e in enumerate(seeds):
            for j, (key, title) in enumerate(cols):
                ax = axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
                if key == "mono":
                    png = mono_path(pair, e["seed"])
                    cap = ("usable anchor" if e["reference_valid"]
                           else f"{e['reference_status'].upper()}: no distance readable")
                else:
                    k = e["k"][key]
                    png = Path(k["png"])
                    g = k["mono_gain_vs_k50"]
                    gtxt = "n/a" if (g is None or not e["reference_valid"]) else f"{g:+.3f}"
                    cap = f"n={k['n_instances']}  gain {gtxt}  lap {k['sharpness_laplacian_var']:.0f}"
                if Path(png).exists():
                    ax.imshow(Image.open(png))
                ax.set_xlabel(cap, fontsize=7)
                if i == 0:
                    ax.set_title(title, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"seed {e['seed']}", fontsize=9)
                for sp in ax.spines.values():
                    sp.set_linewidth(2.0)
                    sp.set_color("#2e7d32" if (key != "mono" and e["k"][key]["composed"])
                                  else ("#c62828" if key != "mono" else "#616161"))
        fig.suptitle(
            f"{pair.replace('__x__', ' x ')} — {name} (rank {v['rank']}, lambda {v['lambda_value']})\n"
            f"handing the tail back to frozen SDXL; gain is DINOv2 distance nearer the joint prompt "
            f"than the control, bar {MIN_MONO_GAIN}; green frame = instance count >= 2",
            fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = RESULTS_DIR / f"clean-tail-k-sweep-{name.replace('_', '-')}-{pair.replace('__x__', '-x-').replace('_', '-')}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        p.with_suffix(".json").write_text(json.dumps({
            "what": f"rows are held-out seeds, columns are k, the step after which the adapter is detached",
            "checkpoint": v["checkpoint"], "rank": v["rank"], "lambda_value": v["lambda_value"],
            "bar": {"MIN_MONO_GAIN": MIN_MONO_GAIN}, "branch": v["branch"], "by_k": v["by_k"],
            "tiles": [{"seed": e["seed"], "reference_valid": e["reference_valid"], "k": e["k"]}
                      for e in seeds]}, indent=1))
        made.append(p)
        print(f"wrote {p}", flush=True)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", action="store_true")
    ap.add_argument("--render", metavar="NAME")
    ap.add_argument("--score", metavar="NAME")
    ap.add_argument("--sheet", metavar="NAME")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.references:
        return references()
    if args.render:
        return render(args.render)
    if args.score:
        return score(args.score)
    if args.sheet:
        sheet(args.sheet)
        return 0
    ap.error("pick a stage")


if __name__ == "__main__":
    raise SystemExit(main())
