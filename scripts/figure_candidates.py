#!/usr/bin/env python
"""Joint prompt, plain product and one adapter, from the same starting noise, on pairs the
adapter never trained on. The candidate pool the paper's main figure is picked from.

Every render of a given seed starts from the same noise the rest of this project uses for that
seed: the cached starting latent of `a_cat__x__a_dog` at that seed, which the cache shares across
pairs. So a column differs from its neighbours only in what the column is, never in where it
started.

One column per process, over a list of cells. The joint-prompt column never has an adapter
attached, because an attached adapter can stay enabled and leak into a later plain render. The
plain-product and adapter columns attach the same checkpoint and differ only in the window the
adapter is switched on for (none of the fifty steps, or all of them), and each render checks the
window it actually got.

    python scripts/figure_candidates.py --column mono --pairs "a horse|an owl" --seeds 9 10
    python scripts/figure_candidates.py --column poe  --pairs "a horse|an owl" --seeds 9 10 --checkpoint <pt>
    python scripts/figure_candidates.py --column ours --pairs "a horse|an owl" --seeds 9 10 --checkpoint <pt>
    python scripts/figure_candidates.py --sheet --pairs "a horse|an owl" --seeds 9 10
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

OUT = Path(os.environ.get("FIGURE_CANDIDATES_OUT",
                          "/datasets/mmolefe/poe_repair_min/outputs/figure_candidates"))
NOISE_PAIR = "a_cat__x__a_dog"          # whose cached starting latent every seed borrows
COLUMNS = ("mono", "poe", "ours", "monolora")
LABELS = {"mono": "joint prompt", "poe": "plain product", "ours": "ours",
          "monolora": "joint prompt through the adapter"}


def parse_pair(s: str) -> tuple[str, str]:
    a, b = (p.strip() for p in s.split("|"))
    # The subject lists are a hard constraint of this project, and a render path that skips the
    # check is how a refused subject reaches a sheet anyway.
    from disallowed_subjects import check
    check(a)
    check(b)
    return a, b


def cell_dir(slug: str, seed: int) -> Path:
    return OUT / slug / f"seed{seed:02d}"


def render(column: str, pairs: list[tuple[str, str]], seeds: list[int], ckpt: Path | None,
           rank: int, windows: list[int | None] = (None,), tag: str = "",
           spread_match: float = 0.0, late_ckpt: Path | None = None, ema: bool = False,
           switch_at: int | None = None, langevin: dict | None = None,
           candidates: int = 1, jitter: float = 0.0, reward_guidance: dict | None = None,
           guidance: float | None = None, expert_weights=None, refine=None, contrast: dict | None = None) -> None:
    free, total = torch.cuda.mem_get_info(0)
    # Sharing a card slows both jobs and can run either out of memory, so it is refused unless the
    # caller says how much of someone else's memory this render may sit beside.
    max_used = float(os.environ.get("FIGURE_CANDIDATES_MAXUSED_GB", "1.0"))
    if (total - free) / 1e9 > max_used:
        raise SystemExit(f"device holds {(total - free) / 1e9:.1f} GB, refusing to share "
                         f"(raise FIGURE_CANDIDATES_MAXUSED_GB above it to allow)")

    from poe_repair.composers._helpers import encode_pair, get_joint_embeds, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
    from poe_repair.methods._sampling import run_cfg, write_decoded_image
    from poe_repair.run import make_ctx
    import lambda_boundary_probe as lbp

    ctx = make_ctx(guidance_scale=guidance) if guidance else make_ctx()
    if column != "mono":
        if ckpt is None or not ckpt.exists():
            raise SystemExit(f"--column {column} needs an existing --checkpoint, got {ckpt}")
        lbp.LORA_RANK = lbp.LORA_ALPHA = rank
        # The running average of the weights is saved beside the raw ones in every
        # checkpoint; it damps the swing between neighbouring checkpoints.
        info = lbp._attach_and_load_lora(ctx.models["unet"], ckpt,
                                         key="lora_state_ema" if ema else None)
        if int(info["n_matched"]) == 0 or int(info["n_loaded"]) == 0:
            raise SystemExit(f"adapter did not load: matched {info['n_matched']}, loaded {info['n_loaded']}")
        print(f"attached {ckpt} (step {info['checkpoint_step']}, rank {rank})", flush=True)
        if late_ckpt is not None:
            # The second adapter finishes the picture from switch_at on; attached under its own
            # name so the sampler can swap between the two per step.
            if switch_at is None:
                raise SystemExit("--late-checkpoint needs --switch-at")
            lbp.LORA_ADAPTER_NAME = "late"
            late = lbp._attach_and_load_lora(ctx.models["unet"], late_ckpt)
            lbp.LORA_ADAPTER_NAME = "lora"
            if int(late["n_loaded"]) == 0:
                raise SystemExit(f"late adapter did not load from {late_ckpt}")
            print(f"attached {late_ckpt} as the late adapter from step {switch_at}", flush=True)
    for window in windows:
        if window is None or column != "ours":
            window = {"poe": 0, "ours": 50}.get(column, 0)
        name = f"{column}_{tag}_w{window:02d}" if tag else column

        for a, b in pairs:
            for seed in seeds:
              for cand in range(candidates):
                cell = cell_for(a, b, seed)
                out = cell_dir(cell.pair_slug, seed) / f"{name}.png"
                if candidates > 1:
                    out = cell_dir(cell.pair_slug, seed) / f"{name}_c{cand:02d}.png"
                if out.exists():
                    print(f"{cell.pair_slug} seed {seed} {column}: already rendered", flush=True)
                    continue
                base_latents, euler_sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
                # Best of N: candidate 0 is the cell's own noise, the rest sit a short step away
                # from it, which is the neighbourhood the noise-search finding measured.
                init_latents = base_latents
                if cand > 0 and jitter > 0:
                    g = torch.Generator(device="cpu").manual_seed(1000 * seed + cand)
                    step = torch.randn(base_latents.shape, generator=g, dtype=torch.float32)
                    init_latents = base_latents + jitter * step.to(base_latents.device, base_latents.dtype)
                emb = encode_pair(cell, ctx)
                if column in ("mono", "monolora"):
                    seq_j, pool_j = get_joint_embeds(cell, ctx)
                    res = run_cfg(
                        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                        seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                        device=ctx.device, dtype=ctx.dtype,
                    )
                else:
                    res = run_lora_langevin_windowed_poe(
                        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
                        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                        guidance_scale=ctx.guidance_scale, num_inference_steps=ctx.num_inference_steps,
                        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
                        device=ctx.device, dtype=ctx.dtype, lambda_value=1.0,
                        k=(langevin or {}).get("k", 0), c=(langevin or {}).get("c", 0.0),
                        corrector_window=(langevin or {}).get("window"),
                        noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
                        lambda_window=(0, window),
                        corrector_score=(langevin or {}).get("score", "frozen"),
                        spread_match=spread_match, **(contrast or {}),
                        reward_guidance=reward_guidance,
                        expert_weights=expert_weights, refine=refine,
                        lora_adapter_name_late="late" if late_ckpt is not None else None,
                        adapter_switch_at=switch_at if late_ckpt is not None else None,
                    )
                    on = sum(1 for r in res.extras["per_step"] if r["adapter_on"])
                    if on != window:
                        raise SystemExit(f"{cell.pair_slug} seed {seed} {column}: adapter on {on} steps, not {window}")
                # mkdir is not idempotent on this NFS: jobs launched together race on the shared
                # pair folder and the losers raise, either FileExists or a stale FileNotFound.
                for _try in range(5):
                    try:
                        out.parent.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass
                    if out.parent.is_dir():
                        break
                    time.sleep(2)
                write_decoded_image(res.image, out)
                meta = out.with_suffix(".json")
                meta.write_text(json.dumps({
                    "pair": cell.pair_slug, "prompt_a": a, "prompt_b": b, "seed": seed,
                    "column": column, "noise_from": NOISE_PAIR,
                    "checkpoint": str(ckpt) if column != "mono" else None,
                    "adapter_steps": window, "tag": tag, "spread_match": spread_match, "contrast": contrast,
                    "late_checkpoint": str(late_ckpt) if late_ckpt else None, "switch_at": switch_at, "guidance": ctx.guidance_scale,
                    "steps": ctx.num_inference_steps, "size": [cell.width, cell.height],
                }, indent=1))
                del res
                gc.collect(); torch.cuda.empty_cache()
                print(f"{cell.pair_slug} seed {seed} {column}: wrote {out}", flush=True)


def sheet(pairs: list[tuple[str, str]], seeds: list[int]) -> None:
    """One sheet per pair: seeds as rows, the three columns across."""
    from poe_repair.experiments._eval_common import cell_for
    T, G, TOP, LEFT = 320, 8, 30, 70
    for a, b in pairs:
        slug = cell_for(a, b, seeds[0]).pair_slug
        W = LEFT + len(COLUMNS) * (T + G)
        H = TOP + len(seeds) * (T + G) + 24
        canvas = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(canvas)
        d.text((LEFT, 4), f"{a} and {b}", fill="black")
        for j, c in enumerate(COLUMNS):
            d.text((LEFT + j * (T + G), 16), LABELS[c], fill="#444444")
        for i, s in enumerate(seeds):
            y = TOP + i * (T + G)
            d.text((6, y + T // 2), f"seed {s}", fill="#444444")
            for j, c in enumerate(COLUMNS):
                p = cell_dir(slug, s) / f"{c}.png"
                x = LEFT + j * (T + G)
                if p.exists():
                    canvas.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                else:
                    d.rectangle((x, y, x + T, y + T), outline="#bbbbbb")
                    d.text((x + 10, y + T // 2), "not rendered", fill="#999999")
        out = OUT / "sheets" / f"{slug}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", choices=COLUMNS)
    ap.add_argument("--pairs", nargs="+", required=True, help='each as "a horse|an owl"')
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--window", type=int, nargs="+", help="ours only: adapter on for steps 0 to N, off after; several render in one process")
    ap.add_argument("--tag", default="", help="ours only: names the tile <column>_<tag>_wNN.png")
    ap.add_argument("--ema", action="store_true",
                    help="render the running average of the weights instead of the raw ones")
    ap.add_argument("--spread-match", type=float, default=0.0,
                    help="blend weight of the guidance rescale onto the product (0 off, 0.7 Lin et al.)")
    ap.add_argument("--late-checkpoint", type=Path, help="second adapter that draws from --switch-at on")
    ap.add_argument("--switch-at", type=int, help="step the late adapter takes over")
    ap.add_argument("--contrast-steps", type=int, default=0,
                    help="CO3's contrast corrector over the first N steps (0 off)")
    ap.add_argument("--contrast-iters", type=int, default=5)
    ap.add_argument("--contrast-beta", type=float, default=0.9)
    ap.add_argument("--contrast-w-multi", type=float, default=2.0)
    ap.add_argument("--langevin-k", type=int, default=0,
                    help="Langevin corrector steps inside the corrector window (0 is off)")
    ap.add_argument("--langevin-c", type=float, default=0.0,
                    help="corrector step size, multiplied by beta_t at each step")
    ap.add_argument("--corrector-window", type=int, nargs=2, metavar=("LO", "HI"),
                    help="the step range the corrector runs over, e.g. 20 50 for the tail")
    ap.add_argument("--corrector-score", default="frozen", choices=("frozen", "corrected"),
                    help="frozen: the chain settles into the base model's own distribution. "
                         "corrected: it settles into the adapted one.")
    ap.add_argument("--candidates", type=int, default=1,
                    help="best of N: render this many nearby starting noises, one tile each")
    ap.add_argument("--jitter", type=float, default=0.1,
                    help="how far each extra candidate sits from the cell's own starting noise")
    ap.add_argument("--guidance", type=float, default=None,
                    help="classifier-free guidance scale for every branch; 7.5 is what every run "
                         "in this project uses, and raising it tests whether the adapter's "
                         "correction is acting like a lower one")
    ap.add_argument("--expert-weights", type=float, nargs=2, metavar=("WA", "WB"),
                    help="weights on the two concept branches instead of --guidance, applied "
                         "explicitly rather than left for the adapter to learn")
    ap.add_argument("--refine-strength", type=float, default=0.0,
                    help="after the run, renoise to this fraction of the schedule and finish with "
                         "the frozen model (0 is off)")
    ap.add_argument("--refine-guidance", type=float, default=7.5)
    ap.add_argument("--reward-guidance", type=float, default=0.0,
                    help="weight of the plurality objective's gradient on the latent, as a "
                         "fraction of the latent's own norm per guided step (0 is off)")
    ap.add_argument("--guidance-window", type=int, nargs=2, default=(0, 12), metavar=("LO", "HI"),
                    help="the steps the guidance runs over; composition is decided early")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    pairs = [parse_pair(p) for p in a.pairs]
    if a.sheet:
        sheet(pairs, a.seeds)
    elif a.column:
        langevin = {"k": a.langevin_k, "c": a.langevin_c, "score": a.corrector_score,
                    "window": tuple(a.corrector_window) if a.corrector_window else None}
        guidance = None
        if a.reward_guidance > 0:
            import torch as _t
            from poe_repair.rewards.plurality import PluralityReward
            lo, hi = a.guidance_window
            guidance = {"reward": PluralityReward(_t.device("cuda")), "weight": a.reward_guidance,
                        "lo": int(lo), "hi": int(hi),
                        "prompt_a": pairs[0][0], "prompt_b": pairs[0][1]}
        render(a.column, pairs, a.seeds, a.checkpoint, a.rank, a.window or [None], a.tag,
               a.spread_match, a.late_checkpoint, a.ema, a.switch_at, langevin, a.candidates,
               a.jitter, guidance, a.guidance,
               tuple(a.expert_weights) if a.expert_weights else None,
               {"strength": a.refine_strength, "guidance": a.refine_guidance}
               if a.refine_strength > 0 else None,
               {"contrast_steps": a.contrast_steps, "contrast_iters": a.contrast_iters,
                "contrast_beta": a.contrast_beta, "contrast_w_multi": a.contrast_w_multi}
               if a.contrast_steps > 0 else None)
    else:
        raise SystemExit("pass --column <mono|poe|ours> or --sheet")
