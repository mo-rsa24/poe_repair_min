#!/usr/bin/env python
"""Separation steering on SDXL's plain product of experts, for two prompts (default "dog", "cat").

The loop is scripts/and_vector_sheet.py's render(): eps = eps_empty + w * sum_i (eps_i - eps_empty), w = 7.5,
50 DDIM steps, the starting noise of `a_cat__x__a_dog` per seed. At steps 0 to 9 only, before each PoE update,
the latent takes one gradient step on a loss read from the two experts' cross-attention:

    M_dog = expert "dog"'s attention on its "dog" token, M_cat = expert "cat"'s attention on its "cat" token,
            each averaged over heads and over every 32 x 32 attn2 layer, then divided to sum to 1
    L     = sum_p M_dog(p) M_cat(p)  +  (1 - max_p M_dog)  +  (1 - max_p M_cat)
    x_t  <- x_t - eta * dL/dx_t

The UNet is frozen; nothing is trained. One steering step is a forward and backward pass over the two
experts' prompts (batch 2), and the PoE update that follows is the usual forward over (dog, cat, empty).

    python scripts/separation_steering.py --calibrate --seed 1
    python scripts/separation_steering.py --render --eta small=0.5 medium=2 large=8 --seeds 1 2 3 4 5 6 7 8
    python scripts/separation_steering.py --sheet <out.png> --eta small=0.5 medium=2 large=8 --seeds 1 2 3 4 5 6 7 8

Tiles land as <OUT>/separation_steering/<a>__x__<b>/seedNN/eta_<label>.png with a JSON sidecar; an existing
tile is skipped. `--eta plain=0` renders the loop with no steering at all (the identity check against
and_vector/<a>__x__<b>/seedNN/poe.png).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from concept_pair_sheet import NOISE_PAIR, OUT, concept_slug, write_sidecar  # noqa: E402

STEER_STEPS = 10        # steps 0 to 9
LOSS_SCALE = 1024.0     # the backward pass runs in fp16; the loss is scaled up and the gradient scaled back
TOKEN = 1               # position of a one-word prompt's word: [BOS, word, EOS, pad...]


def tile(root: Path, slug: str, seed: int, label: str) -> Path:
    return root / "separation_steering" / slug / f"seed{seed:02d}" / f"eta_{label}.png"


def parse_etas(items: list[str]) -> list[tuple[str, float]]:
    return [(k, float(v)) for k, _, v in (s.partition("=") for s in items)]


class CrossAttnTap:
    """Attention processor for every attn2: at the 32 x 32 layers it computes the softmax explicitly and keeps
    the map of one token (per batch row, head-averaged); every other layer runs the stock processor."""

    def __init__(self, n_queries: int):
        from diffusers.models.attention_processor import AttnProcessor2_0
        self.fallback = AttnProcessor2_0()
        self.n_queries = n_queries
        self.maps: list | None = None      # a list while tapping, None otherwise

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if self.maps is None or hidden_states.shape[1] != self.n_queries:
            return self.fallback(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        b, n, _ = hidden_states.shape
        h = attn.heads
        q, k, v = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        d = q.shape[-1] // h
        q, k, v = (x.view(b, -1, h, d).transpose(1, 2) for x in (q, k, v))
        probs = ((q @ k.transpose(-1, -2)).float() * attn.scale).softmax(-1)      # b, heads, n, 77
        self.maps.append(probs[..., TOKEN].mean(1))                                # b, n
        out = (probs.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, h * d)
        return attn.to_out[1](attn.to_out[0](out))


def install_tap(unet, n_queries: int) -> CrossAttnTap:
    tap = CrossAttnTap(n_queries)
    procs = {name: (tap if name.endswith("attn2.processor") else proc)
             for name, proc in unet.attn_processors.items()}
    unet.set_attn_processor(procs)
    return tap


def loss_terms(maps: list):
    """maps: per layer a (2, n) tensor, row 0 = expert a (dog), row 1 = expert b (cat)."""
    import torch
    m = torch.stack(maps).mean(0)                 # 2, n   (layers averaged)
    m = m / m.sum(-1, keepdim=True)
    sep = (m[0] * m[1]).sum()
    pres_a, pres_b = 1 - m[0].max(), 1 - m[1].max()
    return sep, pres_a, pres_b, m


def run(a: str, b: str, seeds: list[int], etas: list[tuple[str, float]], mode: str) -> None:
    import numpy as np
    import torch

    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    ctx = make_ctx()
    slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
    unet, sched, w = ctx.models["unet"], ctx.scheduler, ctx.guidance_scale
    unet.requires_grad_(False)
    for name in ("tokenizer", "tokenizer_2"):
        for word in (a, b):
            ids = ctx.models[name](word).input_ids
            assert len(ids) == 3, f"{word!r} is not one token under {name}: {ids}"
    enc = lambda p: tuple(t.float() for t in encode_prompt_sdxl(p, models=ctx.models, device=ctx.device,
                                                                 dtype=ctx.dtype))
    ea, eb, e0 = enc(a), enc(b), enc("")
    dev, dt = ctx.device, ctx.dtype
    seq3 = torch.cat([ea[0], eb[0], e0[0]]).to(dt)
    pool3 = torch.cat([ea[1], eb[1], e0[1]]).to(dt)
    seq2, pool2 = seq3[:2], pool3[:2]

    first = cell_from_slug(NOISE_PAIR, seeds[0])
    n_q = (first.height // 32) * (first.width // 32)
    tap = install_tap(unet, n_q)
    cond3 = {"text_embeds": pool3, "time_ids": add_time_ids(height=first.height, width=first.width,
                                                            batch_size=3, device=dev, dtype=dt)}
    cond2 = {"text_embeds": pool2, "time_ids": add_time_ids(height=first.height, width=first.width,
                                                            batch_size=2, device=dev, dtype=dt)}

    def grad_pass(lat, t, parts: bool):
        """Loss terms at the current latent and the gradient of their sum (fp32), after one forward and
        backward over the two experts. With `parts`, also each term's own gradient norm."""
        tap.maps = []
        x = lat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            unet(sched.scale_model_input(x.repeat(2, 1, 1, 1), t), t, encoder_hidden_states=seq2,
                 added_cond_kwargs=cond2, timestep_cond=None)
            sep, pa, pb, m = loss_terms(tap.maps)
            total = sep + pa + pb
            if parts:
                norms = {}
                for key, term in (("sep", sep), ("presence_a", pa), ("presence_b", pb)):
                    g, = torch.autograd.grad(term * LOSS_SCALE, x, retain_graph=True)
                    norms[key] = (g.float() / LOSS_SCALE).norm().item()
            g, = torch.autograd.grad(total * LOSS_SCALE, x)
        tap.maps = None
        g = g.float() / LOSS_SCALE
        info = {"L": total.item(), "sep": sep.item(), "presence_a": pa.item(), "presence_b": pb.item(),
                "max_a": 1 - pa.item(), "max_b": 1 - pb.item(), "grad_norm": g.norm().item(),
                "finite": bool(torch.isfinite(g).all())}
        if parts:
            info["grad_norm_by_term"] = norms
        return g, info

    def trajectory(lat0, eta: float, calibrate: bool):
        lat = lat0.clone()
        log, grad_calls = [], 0
        for i, t in enumerate(sched.timesteps):
            if i < STEER_STEPS and (eta != 0 or calibrate):
                g, info = grad_pass(lat, t, parts=calibrate)
                grad_calls += 1
                info.update(step=i, latent_norm=lat.float().norm().item())
                if eta != 0 and info["finite"]:
                    lat = (lat.float() - eta * g).to(dt)
                    info["step_norm"] = (eta * g).norm().item()
                log.append(info)
            with torch.no_grad():
                eps = unet(sched.scale_model_input(lat.repeat(3, 1, 1, 1), t), t, encoder_hidden_states=seq3,
                           added_cond_kwargs=cond3, timestep_cond=None).sample
                e_u = eps[-1:]
                eg = e_u + w * ((eps[0:1] - e_u) + (eps[1:2] - e_u))
                abar = sched.alphas_cumprod[int(t.item())].to(device=dev, dtype=dt)
                x0 = tweedie_mean(lat, abar, eg)
                lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=eg)
        return lat, log, grad_calls

    for seed in seeds:
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        lat0 = (init / sigma).to(device=dev, dtype=dt)
        sched.set_timesteps(ctx.num_inference_steps)
        if mode == "calibrate":
            _, log, _ = trajectory(lat0, 0.0, calibrate=True)
            print(json.dumps(log, indent=1), flush=True)
            Path(os.environ.get("CALIBRATION_JSON", f"separation_calibration_seed{seed:02d}.json")
                 ).write_text(json.dumps(log, indent=1))
            continue
        for label, eta in etas:
            out = tile(OUT, slug, seed, label)
            if out.exists():
                print(f"{out}: already rendered", flush=True)
                continue
            sched.set_timesteps(ctx.num_inference_steps)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lat, log, grad_calls = trajectory(lat0, eta, calibrate=False)
            torch.cuda.synchronize()
            t_loop = time.perf_counter() - t0
            image = decode_latents(ctx.models, lat).cpu()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            out.parent.mkdir(parents=True, exist_ok=True)
            write_decoded_image(image, out)
            write_sidecar(out, {"label": label, "eta": eta, "a": a, "b": b, "seed": seed,
                                "noise_from": NOISE_PAIR, "steps": ctx.num_inference_steps, "guidance": w,
                                "size": [cell.width, cell.height], "steer_steps": STEER_STEPS,
                                "seconds_per_tile": seconds, "seconds_loop_only": t_loop,
                                "poe_forward_passes": ctx.num_inference_steps, "poe_rows_per_pass": 3,
                                "grad_passes": grad_calls, "grad_rows_per_pass": 2,
                                "steering_log": log, "device": torch.cuda.get_device_name(0)})
            gc.collect()
            torch.cuda.empty_cache()
            print(f"wrote {out}  {seconds:.1f} s  grad passes {grad_calls}", flush=True)


def sheet(out: Path, a: str, b: str, etas: list[tuple[str, float]], seeds: list[int]) -> None:
    from main_study_grids import grid

    slug = f"{concept_slug(a)}__x__{concept_slug(b)}"
    ref = lambda col, s: OUT / "and_vector" / slug / f"seed{s:02d}" / f"{col}.png"
    columns = [("poe", ["PoE, no steering", f'"{a}", "{b}"'])]
    columns += [(f"eta_{k}", [f"Steered, {k}", f"eta = {v:g}"]) for k, v in etas if v != 0]
    columns += [("joint", ["Joint prompt", f'"{a} and {b}"'])]
    where = lambda key, s: (ref("poe", s) if key == "poe" else ref("joint", s) if key == "joint"
                            else tile(OUT, slug, s, key[4:]))
    rows = [(f"seed {s}", lambda key, s=s: where(key, s)) for s in seeds]
    out.parent.mkdir(parents=True, exist_ok=True)
    record = grid(rows, columns, out, f'Separation steering on plain PoE: "{a}" and "{b}"', "",
                  rule_before=len(columns) - 1)
    seconds = {}
    for k, v in etas:
        vals = []
        for s in seeds:
            side = tile(OUT, slug, s, k).with_suffix(".json")
            if side.exists():
                vals.append(json.loads(side.read_text()).get("seconds_per_tile"))
        seconds[k] = vals
    record["seconds_per_tile"] = seconds
    out.with_suffix(".json").write_text(json.dumps(record, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sheet", type=Path, metavar="OUT_PNG")
    ap.add_argument("--a", default="dog")
    ap.add_argument("--b", default="cat")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--eta", nargs="+", default=[], metavar="LABEL=VALUE")
    args = ap.parse_args()
    etas = parse_etas(args.eta)
    if args.calibrate:
        run(args.a, args.b, [args.seed], [], "calibrate")
    if args.render:
        run(args.a, args.b, args.seeds, etas, "render")
    if args.sheet:
        sheet(args.sheet, args.a, args.b, etas, args.seeds)
