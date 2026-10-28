#!/usr/bin/env python
"""One learned "<and>" token as a third expert beside "dog" and "cat" in SDXL's plain PoE.

Adds the token "<and>" to both SDXL tokenizers and gives it one trainable input-embedding row in each
text encoder, initialised from the word "and". Nothing else is trained. The rows live in fp32; a forward
hook on each encoder's token embedding writes them into the fp16 activations, so gradient reaches them
through the frozen encoders and the frozen UNet.

    python scripts/learned_and_token.py --check-token      "<and>" must encode exactly as "and" does
    python scripts/learned_and_token.py --baseline         held-out share removed with the untrained token
    python scripts/learned_and_token.py --train --steps 2000 --eval-at 100 300 1000 2000 --name run1
"""
from __future__ import annotations

import argparse
import json
import random
import time
import os
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = Path(os.environ.get("POE_REPAIR_TRAINING_CACHE",
                            "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"))
POOL = REPO / "artifacts/_shared/cross_pair_pool_configs/cells_v58.json"
HELD_OUT = ("a_cat__x__a_dog", (9, 10, 11, 12, 13, 14, 15, 16))
EVAL_STEPS = (0, 10, 25, 40)
TOKEN = "<and>"
WORD = "and"
ENCODERS = (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2"))


class AndToken:
    """The two trainable rows and the hooks that splice them into each encoder's token embeddings."""

    def __init__(self, models: dict) -> None:
        self.models = models
        self.ids: list[int] = []
        self.rows = torch.nn.ParameterList()
        for tok_key, enc_key in ENCODERS:
            tok, enc = models[tok_key], models[enc_key]
            if tok.add_tokens([TOKEN]) != 1:
                raise RuntimeError(f"{tok_key}: {TOKEN} was already in the vocabulary")
            new_id = tok.convert_tokens_to_ids(TOKEN)
            enc.resize_token_embeddings(len(tok))
            word_ids = tok(WORD, add_special_tokens=False).input_ids
            if len(word_ids) != 1:
                raise RuntimeError(f"{tok_key}: {WORD!r} is {len(word_ids)} tokens, expected 1")
            table = enc.get_input_embeddings()
            self.ids.append(new_id)
            self.rows.append(torch.nn.Parameter(table.weight[word_ids[0]].detach().float().clone()))
            table.register_forward_hook(self._hook(len(self.ids) - 1))
            for p in enc.parameters():
                p.requires_grad_(False)
        self.rows.to(models["unet"].device)

    def _hook(self, k: int):
        def splice(_module, inputs, out):
            mask = inputs[0] == self.ids[k]
            if not mask.any():
                return out
            out = out.clone()
            out[mask] = self.rows[k].to(out.dtype)
            return out
        return splice

    def encode(self, prompt: str, *, device, dtype, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """The repo's SDXL prompt encoding, with the new token known and gradient optional."""
        m = self.models
        with torch.set_grad_enabled(grad):
            outs, all_ids = [], []
            for tok_key, enc_key in ENCODERS:
                tok = m[tok_key]
                ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(device)
                all_ids.append(ids)
                outs.append(m[enc_key](ids, output_hidden_states=True))
            embeds = torch.cat([outs[0].hidden_states[-2], outs[1].hidden_states[-2]], dim=-1).to(dtype)
            # HF pools at the highest token id, which is "<and>" (49408) rather than the end token
            # (49407), so pool at the first end-of-text position and project, as the library does.
            eos = (all_ids[1] == m["tokenizer_2"].eos_token_id).int().argmax(dim=-1)
            last = outs[1].last_hidden_state[torch.arange(len(eos), device=device), eos]
            pooled = m["text_encoder_2"].text_projection(last)
        return embeds, pooled.to(dtype)


def pool_cells() -> list[tuple[str, int]]:
    """The 182 (pair, seed) training cells, refusing cat x dog and any held-out seed."""
    cells = [(pair, seed) for pair, seeds in json.loads(POOL.read_text()).items() for seed in seeds]
    assert HELD_OUT[0] not in {p for p, _ in cells}, "cat x dog is in the pool"
    assert not {s for _, s in cells} & set(HELD_OUT[1]), "a held-out seed is in the pool"
    for pair, seed in cells:
        assert (CACHE / "train" / pair / f"seed_{seed}" / "residuals").is_dir(), f"{pair} seed {seed} not cached"
    return cells


def load_state(split: str, pair: str, seed: int, step: int, device) -> dict:
    """One cached state: x_t, its timestep and the four raw predictions, in fp32 on the device."""
    raw = torch.load(CACHE / split / pair / f"seed_{seed}" / "residuals" / f"step_{step:03d}.pt",
                     map_location="cpu", weights_only=True)
    return {k: (v.to(device) if k == "timestep" else v.to(device).float()) if torch.is_tensor(v) else v
            for k, v in raw.items()}


def and_target(st: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """What eps_<and> must equal, and the gap it has to close (guided joint minus PoE, over w)."""
    target = st["eps_j_raw"] + 2 * st["eps_uncond"] - st["eps_a_raw"] - st["eps_b_raw"]
    return target, target - st["eps_uncond"]


def eps_and(models: dict, tok: "AndToken", st: dict, device, dtype, grad: bool) -> torch.Tensor:
    from poe_repair.methods._sampling import add_time_ids

    seq, pool = tok.encode(TOKEN, device=device, dtype=dtype, grad=grad)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=1,
                                                          device=device, dtype=dtype)}
    t = torch.as_tensor(st["timestep"], device=device)
    with torch.set_grad_enabled(grad):
        return models["unet"](st["x_t"].to(dtype), t, encoder_hidden_states=seq, added_cond_kwargs=cond,
                              timestep_cond=None).sample.float()


def share_removed(models, tok, device, dtype) -> dict[int, list[float]]:
    """Per eval step, the share of the gap closed on each held-out cat x dog seed."""
    pair, seeds = HELD_OUT
    out: dict[int, list[float]] = {}
    for step in EVAL_STEPS:
        for seed in seeds:
            st = load_state("heldout", pair, seed, step, device)
            target, gap = and_target(st)
            e = eps_and(models, tok, st, device, dtype, grad=False)
            out.setdefault(step, []).append(1 - ((e - target).norm() / gap.norm()).item())
    return out


def baseline() -> None:
    from poe_repair.run import make_ctx

    cells = pool_cells()
    print(f"pool: {len(cells)} cells over {len({p for p, _ in cells})} pairs; no cat x dog, no seed 9 to 16",
          flush=True)
    ctx = make_ctx()
    tok = AndToken(ctx.models)
    res = share_removed(ctx.models, tok, ctx.device, ctx.dtype)
    for step, v in res.items():
        print(f"step {step:2d}: share removed per seed 9..16 = " + " ".join(f"{x:+.3f}" for x in v)
              + f"   mean {sum(v) / len(v):+.3f}", flush=True)


def report(tag: str, res: dict[int, list[float]]) -> bool:
    """One line per eval step; True when the 10% stop rule fires (every step's mean under 10%)."""
    means = {st: sum(v) / len(v) for st, v in res.items()}
    print(f"{tag} held-out share removed, mean over seeds 9..16: "
          + "  ".join(f"step {st}: {m:+.3f}" for st, m in means.items()), flush=True)
    return all(m < 0.10 for m in means.values())


def render_tiles(ctx, tok, a: str, b: str | None, seeds: list[int], third: bool, out_dir: Path) -> list[Path]:
    """PoE(a, b) or PoE(a, b, "<and>") per seed, the loop and noise of scripts/and_vector_sheet.py."""
    from concept_pair_sheet import NOISE_PAIR
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    m, device, dtype, sched, w = ctx.models, ctx.device, ctx.dtype, ctx.scheduler, ctx.guidance_scale
    enc = lambda q: encode_prompt_sdxl(q, models=m, device=device, dtype=dtype)
    embs = [enc(a)] + ([enc(b)] if b else []) + ([tok.encode(TOKEN, device=device, dtype=dtype, grad=False)] if third else [])
    embs.append(enc(""))
    seq = torch.cat([e for e, _ in embs]).to(dtype)
    pool = torch.cat([q for _, q in embs]).to(dtype)
    n = len(embs)
    unet = m["unet"]
    was_training = unet.training
    unet.eval()
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = out_dir / f"seed{seed:02d}.png"
        paths.append(path)
        cell = cell_from_slug(NOISE_PAIR, seed)
        init, sigma = init_latents_for_cell(cell, ctx)
        cond = {"text_embeds": pool, "time_ids": add_time_ids(height=cell.height, width=cell.width,
                                                              batch_size=n, device=device, dtype=dtype)}
        sched.set_timesteps(ctx.num_inference_steps)
        lat = (init / sigma).to(device=device, dtype=dtype)
        with torch.no_grad():
            for i, t in enumerate(sched.timesteps):
                eps = unet(sched.scale_model_input(lat.repeat(n, 1, 1, 1), t), t, encoder_hidden_states=seq,
                           added_cond_kwargs=cond, timestep_cond=None).sample
                e_u = eps[-1:]
                eg = e_u + w * sum(eps[k:k + 1] - e_u for k in range(n - 1))
                abar = sched.alphas_cumprod[int(t.item())].to(device=device, dtype=dtype)
                lat = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i,
                                            x0=tweedie_mean(lat, abar, eg), eps=eg)
            write_decoded_image(decode_latents(m, lat).cpu(), path)
        torch.cuda.empty_cache()
    unet.train(was_training)
    return paths


def contact_sheet(paths: list[Path], seeds: list[int], title: str, out: Path, size: int = 320):
    from PIL import Image, ImageDraw

    pad = 26
    sheet = Image.new("RGB", (len(paths) * size, size + pad), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill="black")
    for i, (pth, sd) in enumerate(zip(paths, seeds)):
        sheet.paste(Image.open(pth).convert("RGB").resize((size, size)), (i * size, pad))
        draw.text((i * size + 6, pad + 4), f"seed {sd}", fill="white")
    sheet.save(out)
    return sheet


def build_grid(run_dir: Path, iters: list[int], seeds: list[int]):
    """One row per seed: a cat alone, a dog alone, plain PoE, then PoE with <and> at each update in iters."""
    from PIL import Image, ImageDraw

    A, B = "a cat", "a dog"
    heads = [f"{A} alone", f"{B} alone", f"PoE({A}, {B})"] + [f"+ <and>, {it} updates" for it in iters]
    cols = [lambda sd: run_dir / "tiles" / "mono_a_cat" / f"seed{sd:02d}.png",
            lambda sd: run_dir / "tiles" / "mono_a_dog" / f"seed{sd:02d}.png",
            lambda sd: run_dir / "tiles" / "plain_poe" / f"seed{sd:02d}.png"] + [
        (lambda sd, it=it: run_dir / "tiles" / f"iter{it:05d}" / f"seed{sd:02d}.png") for it in iters]
    size, pad = 320, 24
    rows = []
    for sd in seeds:
        img = Image.new("RGB", (len(cols) * size, size + pad), "white")
        d = ImageDraw.Draw(img)
        for i, (h, c) in enumerate(zip(heads, cols)):
            d.text((i * size + 6, 6), f"seed {sd}: {h}", fill="black")
            img.paste(Image.open(c(sd)).convert("RGB").resize((size, size)), (i * size, pad))
        img.save(run_dir / f"grid_seed{sd:02d}.png")
        rows.append(img)
    full = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        full.paste(r, (0, y))
        y += r.height
    full.save(run_dir / "grid_all_seeds.png")
    return rows, full


def render_references(ctx, tok, run_dir: Path, seeds: list[int]) -> None:
    """The columns that do not change with training: each animal alone, and plain PoE of the two."""
    A, B = "a cat", "a dog"
    for name, a, b in (("mono_a_cat", A, None), ("mono_a_dog", B, None), ("plain_poe", A, B)):
        d = run_dir / "tiles" / name
        todo = [sd for sd in seeds if not (d / f"seed{sd:02d}.png").exists()]
        if todo:
            render_tiles(ctx, tok, a, b, todo, False, d)


def grid(run_dir: Path, iters: list[int], seeds: list[int], wandb_id: str | None) -> None:
    """Assemble the grid after the fact from a finished run's saved tiles."""
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    render_references(ctx, tok, run_dir, seeds)
    rows, full = build_grid(run_dir, iters, seeds)
    print(f"wrote {run_dir}/grid_all_seeds.png and grid_seedNN.png for {seeds}", flush=True)
    if wandb_id:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", id=wandb_id, resume="allow")
        table = wandb.Table(columns=["seed", "row"])
        for sd, r in zip(seeds, rows):
            table.add_data(sd, wandb.Image(r))
        run.log({"grid_per_seed": table, "grid_all_seeds": wandb.Image(full)})
        run.finish()


def train(steps: int, eval_at: list[int], lr: float, name: str, seed: int, use_wandb: bool,
          render_seeds: list[int]) -> None:
    from poe_repair.run import make_ctx

    out = Path("/datasets/mmolefe/poe_repair_min/outputs/learned_and_token") / name
    out.mkdir(parents=True, exist_ok=True)
    cells = pool_cells()
    ctx = make_ctx()
    models, device, dtype = ctx.models, ctx.device, ctx.dtype
    models["unet"].requires_grad_(False)
    models["unet"].enable_gradient_checkpointing()
    models["unet"].train()
    tok = AndToken(models)
    opt = torch.optim.Adam(tok.rows.parameters(), lr=lr)
    rng = random.Random(seed)
    print(f"train: {len(cells)} cells, {steps} steps, lr {lr}, evals at {eval_at}, out {out}", flush=True)
    run_wb = None
    if use_wandb:
        import wandb
        run_wb = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=f"learned-and-{name}",
                            group="learned-and-token", config={"steps": steps, "lr": lr, "pool": POOL.name,
                            "cells": len(cells), "held_out": HELD_OUT[0], "render_seeds": render_seeds})
        print(f"wandb run {run_wb.id} {run_wb.url}", flush=True)
    A, B = "a cat", "a dog"

    rendered: list[int] = []

    def evaluate(it: int) -> bool:
        res = share_removed(models, tok, device, dtype)
        fired = report(f"iter {it}", res)
        if run_wb:
            run_wb.log({f"held_out_share/step{st}": sum(v) / len(v) for st, v in res.items()}, step=it)
        if render_seeds:
            import wandb
            render_tiles(ctx, tok, A, B, render_seeds, True, out / "tiles" / f"iter{it:05d}")
            rendered.append(it)
            rows, full = build_grid(out, rendered, render_seeds)
            if run_wb:
                table = wandb.Table(columns=["seed", "row"])
                for sd, r in zip(render_seeds, rows):
                    table.add_data(sd, wandb.Image(r))
                run_wb.log({"grid_per_seed": table, "grid_all_seeds": wandb.Image(full)}, step=it)
        return fired

    if render_seeds:
        render_references(ctx, tok, out, render_seeds)
    evaluate(0)
    t0, run = time.time(), []
    for it in range(1, steps + 1):
        pair, cell_seed = rng.choice(cells)
        st = load_state("train", pair, cell_seed, rng.randrange(50), device)
        target, _ = and_target(st)
        e = eps_and(models, tok, st, device, dtype, grad=True)
        sq = (e - target) ** 2
        opt.zero_grad(set_to_none=True)
        sq.sum().backward()  # summed, so fp16 gradients through the UNet do not underflow; Adam is scale-free
        if all(torch.isfinite(r.grad).all() for r in tok.rows):
            opt.step()
        else:
            print(f"iter {it}: non-finite gradient, step skipped", flush=True)
        run.append(sq.mean().item())
        if run_wb:
            run_wb.log({"loss": run[-1], "row_norm_1": tok.rows[0].norm().item(),
                        "row_norm_2": tok.rows[1].norm().item()}, step=it)
        if it % 10 == 0:
            print(f"iter {it}: loss (mean squared error per element) {sum(run[-10:]) / 10:.5f}, "
                  f"rows' norms {[round(r.norm().item(), 4) for r in tok.rows]}, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if it in eval_at or it == steps:
            fired = evaluate(it)
            torch.save({"rows": [r.detach().cpu() for r in tok.rows], "ids": tok.ids, "iter": it},
                       out / f"rows_iter{it:05d}.pt")
            if it == steps:
                print(f"stop rule (every step's mean under 10% after {steps} steps): {fired}", flush=True)


def check_token() -> None:
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    kw = dict(device=ctx.device, dtype=ctx.dtype, grad=False)
    from poe_repair.runtime import encode_prompt_sdxl

    for p in ("and", "dog and cat"):
        er, pr = encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
        em, pm = tok.encode(p, **kw)
        print(f"repo encoder vs this one on {p!r}: embeds {(er - em).abs().max().item():.3g}, "
              f"pooled {(pr - pm).abs().max().item():.3g}", flush=True)
    for a, b in (("and", TOKEN), ("dog and cat", f"dog {TOKEN} cat")):
        ea, pa = tok.encode(a, **kw)
        eb, pb = tok.encode(b, **kw)
        print(f"{a!r} vs {b!r}: max |embeds| difference {(ea - eb).abs().max().item():.3g}, "
              f"max |pooled| difference {(pa - pb).abs().max().item():.3g}", flush=True)
    for (tok_key, _), new_id, row in zip(ENCODERS, tok.ids, tok.rows):
        print(f"{tok_key}: {TOKEN} is id {new_id}, row norm {row.norm().item():.4f}, "
              f"requires_grad {row.requires_grad}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-token", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-at", nargs="*", type=int, default=[100, 300, 1000, 2000])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--grid", type=Path, metavar="RUN_DIR")
    ap.add_argument("--iters", nargs="*", type=int, default=[0, 500, 1000, 1500, 2000])
    ap.add_argument("--wandb-id")
    ap.add_argument("--render-seeds", nargs="*", type=int, default=[])
    args = ap.parse_args()
    if args.grid:
        grid(args.grid, args.iters, args.render_seeds or list(HELD_OUT[1]), args.wandb_id)
    if args.train:
        train(args.steps, args.eval_at, args.lr, args.name, args.seed, args.wandb, args.render_seeds)
    if args.check_token:
        check_token()
    if args.baseline:
        baseline()
