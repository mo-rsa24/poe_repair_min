#!/usr/bin/env python
"""Basin-boundary probe: does more correction (lambda > 1) or an early-window-only
injection keep cat x dog out of the one-animal basin?

Motivated by the analyst read on phase1_r8_450k (tracking.md): cat x dog seed 09 flips
between two-animal and one-animal renders from save to save while the LoRA's correction
direction stays flat, and plan 03's dose curve is still rising at lambda=1, the largest
value ever tested. So: fixed checkpoint (280k by default), all 8 held-out seeds, lambda in
{0 (plain PoE reference), 1.0, 1.5, 2.0}, two injection windows:

  full   every one of the 50 steps gets eps_PoE + lambda * delta_hat  (the tracking sampler)
  early  steps 0-9 get the correction, steps 10-49 are plain guided PoE (plan 03/07's window)

Init latents are the training cache's pinned x_0 for each (pair, seed), the same ones the
per-checkpoint tracking eval uses, so seed 09 here is the same image the analyst looked at.
Sampler is lambda_window_grid.run_lora_residual_inject_windowed_poe (off-window = plain
guided PoE), NOT run_lora_residual_inject_masked (off-window = unconditional only).

Scores: the validated instance-count scorer (compose = 2+ animals counted), DINOv2/CLIP
embedding drift = distance-to-mono minus distance-to-poe (negative = nearer the target),
max ||delta_hat|| per render (the over-correction warning from plan 03).

Outputs under OUT_ROOT: renders/<window>/seed_<s>_lambda_<l>.png, results.json, and
grid_full_window.png / grid_early_window.png (rows = seeds, cols = lambda, each tile
labelled with the compose verdict, instance count and DINOv2 drift).

Usage:
    python scripts/showcase/lambda_boundary_probe.py            # render + measure + grids
    python scripts/showcase/lambda_boundary_probe.py --measure  # re-score existing renders
    python scripts/showcase/lambda_boundary_probe.py --grid     # rebuild grids from results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (  # noqa: E402
    count_instances, instance_score_to_dict, score_output_instances,
)
from poe_repair.experiments.compose_scorer_validation.detection_scorer import vmetrics as _vm  # noqa: E402
_box_iou = _vm.box_iou
from poe_repair.experiments.compose_scorer_validation.scorer import (  # noqa: E402
    _Embedders, _cosine_distance,
)
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer  # noqa: E402
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents  # noqa: E402
from poe_repair.methods._sampling import write_decoded_image  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402
from poe_repair.runtime import encode_prompt_sdxl  # noqa: E402
from poe_repair.training_cache import CellPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambda_window_grid import run_lora_residual_inject_windowed_poe  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints/lora_step_280000.pt"
)
TRAINING_CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
OUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/lambda_boundary_probe")

PAIR_SLUG = "a_cat__x__a_dog"
PROMPT_A, PROMPT_B = "a cat", "a dog"
QUERY_A, QUERY_B = "cat", "dog"
HELD_OUT_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)

LAMBDAS = (1.0, 1.5, 2.0)          # lambda=0 is rendered once as the plain-PoE reference
WINDOWS = {"full": 50, "early": 10}  # name -> number of leading steps that get the correction
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
EULER_INIT_NOISE_SIGMA = 1.0         # cfg.sampler.euler_init_noise_sigma in every training run

LORA_RANK, LORA_ALPHA = 8, 8
LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"
LORA_KEY = "lora_state"   # or "lora_state_ema" for a checkpoint that carries the EMA weights (plan 20)


def _mask(on_steps: int) -> list[bool]:
    return [True] * on_steps + [False] * (NUM_INFERENCE_STEPS - on_steps)


def _attach_and_load_lora(unet: torch.nn.Module, checkpoint: Path, key: str | None = None) -> dict:
    from types import SimpleNamespace
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    key = LORA_KEY if key is None else key
    state = ckpt.get(key)
    if state is None:
        raise KeyError(f"{checkpoint} has no {key!r} key (found: {list(ckpt.keys())})")
    # A checkpoint trained with self-attention too carries attn1 keys; attach to what it covers,
    # or those weights find no module and the load silently drops them.
    targets = tuple(LORA_TARGET_MODULES)
    if any(".attn1." in k for k in state):
        targets = targets + tuple(t.replace("attn2.", "attn1.") for t in LORA_TARGET_MODULES)
    lora_cfg = LoRAConfig(
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0,
        target_modules=targets, init="gaussian",
        adapter_name=LORA_ADAPTER_NAME,
    )
    attach_info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=lora_cfg))
    attach_info["target_modules"] = list(targets)
    # The checkpoint's keys carry the adapter name they were saved under. Loading into an
    # adapter with a different name matches nothing, so rename first; without this the load
    # reports success and the adapter renders as if it were not there.
    if LORA_ADAPTER_NAME != "lora":
        state = {k.replace(".lora.", f".{LORA_ADAPTER_NAME}."): v for k, v in state.items()}
    attach_info["n_loaded"] = lora_trainer.load_lora_state(unet, state)
    attach_info["n_in_file"] = len(state)
    attach_info["lora_key"] = key
    attach_info["checkpoint_step"] = int(ckpt.get("step", -1))
    return attach_info


def _render_path(window: str, seed: int, lam: float) -> Path:
    return OUT_ROOT / "renders" / window / f"seed_{seed}_lambda_{lam}.png"


def render(checkpoint: Path, seeds: list[int]) -> list[dict]:
    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    info = _attach_and_load_lora(ctx.models["unet"], checkpoint)
    print(f"[probe] LoRA attached+loaded: n_matched={info['n_matched']} "
          f"n_loaded={info['n_loaded']} checkpoint_step={info['checkpoint_step']}", flush=True)

    seq_a, pool_a = encode_prompt_sdxl(PROMPT_A, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_b, pool_b = encode_prompt_sdxl(PROMPT_B, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = encode_prompt_sdxl("", models=ctx.models, device=ctx.device, dtype=ctx.dtype)

    jobs: list[tuple[str, float]] = [("full", 0.0)]  # plain PoE once; window is irrelevant at lambda=0
    for window in WINDOWS:
        jobs += [(window, lam) for lam in LAMBDAS]

    results = []
    for seed in seeds:
        cache_cell = CellPath.from_root(PAIR_SLUG, seed, split="heldout", cache_root=TRAINING_CACHE)
        init_latents = load_pinned_init_latents(
            cache_cell, device=ctx.device, dtype=ctx.dtype,
            euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
        )
        for window, lam in jobs:
            img_path = _render_path(window, seed, lam)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            out = run_lora_residual_inject_windowed_poe(
                init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                seq_e=seq_e, pool_e=pool_e,
                guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                cfg_mask=_mask(WINDOWS[window]),
                height=1024, width=1024, euler_init_noise_sigma=EULER_INIT_NOISE_SIGMA,
                device=ctx.device, dtype=ctx.dtype,
                lambda_value=lam, lora_adapter_name=LORA_ADAPTER_NAME,
            )
            write_decoded_image(out.image, img_path)
            max_delta_norm = max(out.extras["delta_norm_per_step"])
            results.append({
                "seed": seed, "window": window, "lambda": lam,
                "image_path": str(img_path), "max_delta_norm": max_delta_norm,
                "checkpoint": str(checkpoint), "checkpoint_step": info["checkpoint_step"],
            })
            print(f"[probe] seed={seed} window={window} lambda={lam} -> {img_path.name} "
                  f"max||delta||={max_delta_norm:.2f}", flush=True)

    (OUT_ROOT / "render_run.json").write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {len(results)} renders", flush=True)
    return results


def measure() -> dict:
    rows = json.loads((OUT_ROOT / "render_run.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedders = _Embedders(device=device)

    for row in rows:
        out_path = Path(row["image_path"])
        inst = instance_score_to_dict(score_output_instances(out_path, QUERY_A, QUERY_B, device=device))
        row["compose"] = bool(inst["n_instances"] >= 2)
        row["n_instances"] = inst["n_instances"]
        row["n_instances_head"] = inst.get("n_instances_head")
        n_cat, cat_boxes = count_instances(out_path, query=QUERY_A, device=device)
        n_dog, dog_boxes = count_instances(out_path, query=QUERY_B, device=device)
        row["n_cat"], row["n_dog"] = int(n_cat), int(n_dog)
        # A blended single animal answers BOTH text queries with the same box, so
        # "a cat and a dog" means a cat box and a dog box in different places.
        row["cat_and_dog"] = any(
            _box_iou(c["box"], d["box"]) < 0.5 for c in cat_boxes for d in dog_boxes
        )
        anchor = TRAINING_CACHE / "heldout" / PAIR_SLUG / f"seed_{row['seed']}"
        mono, poe = anchor / "mono.png", anchor / "poe.png"
        row["drift"] = {}
        for space in ("dino", "clip"):
            e_out, e_mono, e_poe = getattr(embedders, space)([out_path, mono, poe])
            d_mono, d_poe = _cosine_distance(e_out, e_mono), _cosine_distance(e_out, e_poe)
            row["drift"][space] = {"d_mono": d_mono, "d_poe": d_poe, "drift": d_mono - d_poe}
        print(f"[probe] seed={row['seed']} window={row['window']} lambda={row['lambda']} "
              f"n={row['n_instances']} compose={row['compose']} cat={row['n_cat']} dog={row['n_dog']} "
              f"dino_drift={row['drift']['dino']['drift']:+.3f}", flush=True)

    # Compose rate per (window, lambda), plus mean drift and mean max-delta.
    summary: dict[str, dict] = {}
    for window in WINDOWS:
        summary[window] = {}
        for lam in (0.0,) + LAMBDAS:
            w = "full" if lam == 0.0 else window
            cell = [r for r in rows if r["window"] == w and r["lambda"] == lam]
            if not cell:
                continue
            summary[window][str(lam)] = {
                "n": len(cell),
                "compose_rate": sum(r["compose"] for r in cell) / len(cell),
                "cat_and_dog_rate": sum(r["cat_and_dog"] for r in cell) / len(cell),
                "mean_dino_drift": sum(r["drift"]["dino"]["drift"] for r in cell) / len(cell),
                "mean_clip_drift": sum(r["drift"]["clip"]["drift"] for r in cell) / len(cell),
                "mean_max_delta_norm": sum(r["max_delta_norm"] for r in cell) / len(cell),
            }
    out = {
        "scorer": "instance_count (compose = n_instances >= 2)",
        "pair": PAIR_SLUG, "seeds": sorted({r["seed"] for r in rows}),
        "checkpoint": rows[0]["checkpoint"], "checkpoint_step": rows[0]["checkpoint_step"],
        "windows": WINDOWS, "lambdas": [0.0, *LAMBDAS],
        "summary": summary, "rows": rows,
    }
    (OUT_ROOT / "results.json").write_text(json.dumps(out, indent=2))
    print("[probe] compose rate by window x lambda:", flush=True)
    for window, by_lam in summary.items():
        print(f"  {window:5s} " + "  ".join(
            f"l={lam}: {v['compose_rate']:.3f} cat+dog {v['cat_and_dog_rate']:.3f} (drift {v['mean_dino_drift']:+.3f}, "
            f"max|d| {v['mean_max_delta_norm']:.1f})" for lam, v in by_lam.items()), flush=True)
    return out


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def grid() -> list[Path]:
    res = json.loads((OUT_ROOT / "results.json").read_text())
    rows = res["rows"]
    seeds = res["seeds"]
    lambdas = res["lambdas"]
    thumb, pad, label_h, header_h, row_label_w = 300, 8, 26, 64, 120
    f_head, f_lab, f_tile = _font(22), _font(18), _font(16)
    paths = []
    for window in WINDOWS:
        W = row_label_w + pad + (1 + len(lambdas)) * (thumb + pad)  # +1: mono reference column
        H = header_h + len(seeds) * (thumb + label_h + pad) + pad
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        on = WINDOWS[window]
        title = (f"cat x dog, checkpoint {res['checkpoint_step']}, window={window} "
                 f"(correction on steps 0-{on - 1} of {NUM_INFERENCE_STEPS}"
                 + (", plain PoE after)" if on < NUM_INFERENCE_STEPS else ")"))
        draw.text((pad, 8), title, fill="black", font=f_head)
        draw.text((row_label_w + pad, header_h - 28), "mono (joint prompt)", fill="black", font=f_lab)
        for j, lam in enumerate(lambdas):
            x = row_label_w + pad + (j + 1) * (thumb + pad)
            col = "lambda = 0  (plain PoE)" if lam == 0.0 else f"lambda = {lam}"
            draw.text((x, header_h - 28), col, fill="black", font=f_lab)
        for i, seed in enumerate(seeds):
            y = header_h + i * (thumb + label_h + pad)
            draw.text((pad, y + thumb // 2 - 10), f"seed {seed}", fill="black", font=f_lab)
            mono = TRAINING_CACHE / "heldout" / PAIR_SLUG / f"seed_{seed}" / "mono.png"
            x0 = row_label_w + pad
            if mono.exists():
                canvas.paste(Image.open(mono).convert("RGB").resize((thumb, thumb), Image.LANCZOS), (x0, y))
                draw.rectangle([x0, y, x0 + thumb - 1, y + thumb - 1], outline=(120, 120, 120), width=2)
            else:
                draw.rectangle([x0, y, x0 + thumb, y + thumb], outline="red")
            for j, lam in enumerate(lambdas):
                w = "full" if lam == 0.0 else window
                r = next((r for r in rows if r["seed"] == seed and r["window"] == w and r["lambda"] == lam), None)
                x = row_label_w + pad + (j + 1) * (thumb + pad)
                if r is None:
                    draw.rectangle([x, y, x + thumb, y + thumb], outline="red")
                    continue
                im = Image.open(r["image_path"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
                canvas.paste(im, (x, y))
                ok = r["compose"]
                strict = r.get("cat_and_dog", ok)
                color = (20, 130, 40) if (ok and strict) else (210, 140, 0) if ok else (190, 30, 30)
                draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline=color, width=4)
                tag = ("2 animals" if ok else "1 animal") + f"  (n={r['n_instances']})"
                draw.text((x, y + thumb + 4), tag, fill=color, font=f_tile)
        out_path = OUT_ROOT / f"grid_{window}_window.png"
        canvas.save(out_path)
        paths.append(out_path)
        print(f"[probe] wrote {out_path}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    global OUT_ROOT, LAMBDAS, LORA_RANK, LORA_ALPHA, WINDOWS, LORA_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--rank", type=int, default=LORA_RANK,
                    help="LoRA rank of the checkpoint (alpha = rank, per plan 09)")
    ap.add_argument("--lora-key", default=LORA_KEY, choices=("lora_state", "lora_state_ema"),
                    help="which weights to load from the checkpoint; lora_state_ema exists only on plan 20 runs")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="where renders/results/grids go; use one dir per checkpoint")
    ap.add_argument("--seeds", default=",".join(str(s) for s in HELD_OUT_SEEDS))
    ap.add_argument("--windows", default=",".join(WINDOWS),
                    help="comma-separated subset of full,early")
    ap.add_argument("--lambdas", default=",".join(str(l) for l in LAMBDAS),
                    help="comma-separated lambda values beyond the lambda=0 reference column")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args(argv)
    OUT_ROOT = Path(args.out_root)
    LORA_RANK = LORA_ALPHA = int(args.rank)
    LORA_KEY = str(args.lora_key)
    LAMBDAS = tuple(float(l) for l in args.lambdas.split(",") if l.strip())
    WINDOWS = {w: WINDOWS[w] for w in args.windows.split(",") if w.strip()}
    do_all = not (args.render or args.measure or args.grid)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if do_all or args.render:
        render(Path(args.checkpoint), seeds)
    if do_all or args.measure:
        measure()
    if do_all or args.grid:
        grid()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
