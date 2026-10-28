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
        self.override: list[torch.Tensor] | None = None  # rows written by `hyper` for the bound pair
        self.hyper: "HyperRows | None" = None
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
            out[mask] = (self.rows if self.override is None else self.override)[k].to(out.dtype)
            return out
        return splice

    def bind(self, a: str, b: str, grad: bool = False) -> None:
        """With a hyper network, write the rows for this pair; otherwise the rows stay as they are."""
        if self.hyper is not None:
            with torch.set_grad_enabled(grad):
                self.override = self.hyper(self, a, b)

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


class HyperRows(torch.nn.Module):
    """Writes the two rows from the two animals: the "and" rows plus U(V(f)). f is built from each animal's pooled
    text vector, f = [pooled_A + pooled_B, pooled_A * pooled_B] rescaled to unit mean square, so A, B and B, A agree.
    U starts at zero, so the rows start as plain "and"."""

    def __init__(self, tok: "AndToken", rank: int) -> None:
        super().__init__()
        self.device = tok.rows[0].device
        self.base = [r.detach().clone() for r in tok.rows]
        self.dims = [r.numel() for r in tok.rows]
        self.V = torch.nn.Linear(2 * 1280, rank, bias=False)
        self.U = torch.nn.Linear(rank, sum(self.dims), bias=False)
        torch.nn.init.zeros_(self.U.weight)
        self.to(self.device)

    def forward(self, tok: "AndToken", a: str, b: str) -> list[torch.Tensor]:
        unet = tok.models["unet"]
        with torch.no_grad():
            pa, pb = (tok.encode(q, device=unet.device, dtype=unet.dtype, grad=False)[1].float()[0] for q in (a, b))
            f = torch.cat([pa + pb, pa * pb])
            f = f / f.pow(2).mean().sqrt()
        return [base + d for base, d in zip(self.base, self.U(self.V(f)).split(self.dims))]


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


def third_prompt(a: str, b: str, in_sentence: bool) -> str:
    """The third expert's text: the bare token, or the token between the two animals."""
    return f"{a} {TOKEN} {b}" if in_sentence else TOKEN


def eps_and(models: dict, tok: "AndToken", st: dict, device, dtype, grad: bool, prompt: str = TOKEN) -> torch.Tensor:
    from poe_repair.methods._sampling import add_time_ids

    seq, pool = tok.encode(prompt, device=device, dtype=dtype, grad=grad)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=1,
                                                          device=device, dtype=dtype)}
    t = torch.as_tensor(st["timestep"], device=device)
    with torch.set_grad_enabled(grad):
        return models["unet"](st["x_t"].to(dtype), t, encoder_hidden_states=seq, added_cond_kwargs=cond,
                              timestep_cond=None).sample.float()


def share_removed(models, tok, device, dtype, split="heldout", pair=None, seeds=None,
                  in_sentence: bool = False) -> dict[int, list[float]]:
    """Per eval step, the share of the gap closed on each seed (held-out cat x dog unless a pair is given)."""
    pair, seeds = (pair or HELD_OUT[0]), (seeds or HELD_OUT[1])
    prompt = third_prompt(*pair_prompts(pair)[:2], in_sentence)
    tok.bind(*pair_prompts(pair)[:2])
    out: dict[int, list[float]] = {}
    for step in EVAL_STEPS:
        for seed in seeds:
            st = load_state(split, pair, seed, step, device)
            target, gap = and_target(st)
            e = eps_and(models, tok, st, device, dtype, grad=False, prompt=prompt)
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


def render_tiles(ctx, tok, a: str, b: str | None, seeds: list[int], third: bool | str, out_dir: Path) -> list[Path]:
    """PoE(a, b) or PoE(a, b, third) per seed (third is True for the bare token, or the third expert's prompt), the loop and noise of scripts/and_vector_sheet.py."""
    from concept_pair_sheet import NOISE_PAIR
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.runtime import (ddim_prev_from_x0_eps, decode_latents, encode_prompt_sdxl,
                                    tweedie_mean)

    m, device, dtype, sched, w = ctx.models, ctx.device, ctx.dtype, ctx.scheduler, ctx.guidance_scale
    enc = lambda q: encode_prompt_sdxl(q, models=m, device=device, dtype=dtype)
    if third and b:
        tok.bind(a, b)
    embs = [enc(a)] + ([enc(b)] if b else []) + ([tok.encode(TOKEN if third is True else third, device=device, dtype=dtype, grad=False)] if third else [])
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


JOINT = "a cat and a dog"


def build_sheet(run_dir: Path, it: int, seeds: list[int], third_label: str = "<and>"):
    """Columns: dog alone, cat alone, the joint prompt, PoE(cat, dog), PoE(cat, dog, <and>) after `it` updates;
    one row per seed."""
    heads = ["a dog alone", "a cat alone", f'"{JOINT}"', "PoE(a cat, a dog)", f"PoE(a cat, a dog, {third_label}), {it} updates"]
    dirs = [run_dir / "tiles" / n for n in ("mono_a_dog", "mono_a_cat", "joint", "plain_poe", f"iter{it:05d}")]
    return _sheet(heads, dirs, seeds, run_dir / f"sheet_iter{it:05d}.png")


def _sheet(heads: list[str], dirs: list[Path], seeds: list[int], out: Path):
    from PIL import Image, ImageDraw

    size, head, label = 256, 34, 70
    sheet = Image.new("RGB", (label + len(dirs) * size, head + len(seeds) * size), "white")
    d = ImageDraw.Draw(sheet)
    for i, h in enumerate(heads):
        d.text((label + i * size + 6, 10), h, fill="black")
    for r, sd in enumerate(seeds):
        d.text((6, head + r * size + size // 2), f"seed {sd}", fill="black")
        for i, tdir in enumerate(dirs):
            sheet.paste(Image.open(tdir / f"seed{sd:02d}.png").convert("RGB").resize((size, size)),
                        (label + i * size, head + r * size))
    sheet.save(out)
    return sheet


def pair_prompts(pair: str) -> tuple[str, str, str]:
    """a_lion__x__a_meerkat -> 'a lion', 'a meerkat', 'a lion and a meerkat'."""
    a, b = (p.replace("_", " ") for p in pair.split("__x__"))
    return a, b, f"{a} and {b}"


def pairs_sheet(run_dir: Path, it: int, specs: list[str], wandb_id: str | None, in_sentence: bool = False,
                hyper_rank: int = 0) -> None:
    """The five-column sheet and the share removed for other pairs, with the rows saved at update `it`.
    Each spec is pair:split:seed,seed,... (split is train for a pool pair, heldout for a held-out one)."""
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    saved = torch.load(run_dir / f"rows_iter{it:05d}.pt", map_location="cpu")
    with torch.no_grad():
        for row, new in zip(tok.rows, saved["rows"]):
            row.copy_(new)
    if saved.get("hyper"):
        tok.hyper = HyperRows(tok, hyper_rank)
        tok.hyper.load_state_dict(saved["hyper"])
    sheets = {}
    for spec in specs:
        pair, split, seed_txt = spec.split(":")
        seeds = [int(s) for s in seed_txt.split(",")]
        a, b, joint = pair_prompts(pair)
        base = run_dir / "tiles_pairs" / pair
        for name, pa, pb, third in (("mono_a", a, None, False), ("mono_b", b, None, False), ("joint", joint, None, False),
                                    ("plain_poe", a, b, False),
                                    (f"iter{it:05d}", a, b, third_prompt(a, b, in_sentence))):
            todo = [sd for sd in seeds if not (base / name / f"seed{sd:02d}.png").exists()]
            if todo:
                render_tiles(ctx, tok, pa, pb, todo, third, base / name)
        heads = [f"{a} alone", f"{b} alone", f'"{joint}"', f"PoE({a}, {b})", f"PoE({a}, {b}, {third_prompt(a, b, in_sentence)}), {it} updates"]
        dirs = [base / n for n in ("mono_a", "mono_b", "joint", "plain_poe", f"iter{it:05d}")]
        sheets[pair] = _sheet(heads, dirs, seeds, run_dir / f"sheet_{pair}_iter{it:05d}.png")
        res = share_removed(ctx.models, tok, ctx.device, ctx.dtype, split, pair, seeds, in_sentence)
        print(f"{pair} ({split}) seeds {seeds}, {it} updates, share removed: "
              + "  ".join(f"step {st}: {sum(v) / len(v):+.3f}" for st, v in res.items()), flush=True)
    if wandb_id:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", id=wandb_id, resume="allow")
        run.log({f"sheet_{p}": wandb.Image(sh, caption=f"{p}, {it} updates") for p, sh in sheets.items()}, step=it)
        run.finish()


def render_references(ctx, tok, run_dir: Path, seeds: list[int]) -> None:
    """The columns that do not change with training: each animal alone, the joint prompt, and plain PoE."""
    for name, a, b in (("mono_a_dog", "a dog", None), ("mono_a_cat", "a cat", None), ("joint", JOINT, None),
                       ("plain_poe", "a cat", "a dog")):
        d = run_dir / "tiles" / name
        todo = [sd for sd in seeds if not (d / f"seed{sd:02d}.png").exists()]
        if todo:
            render_tiles(ctx, tok, a, b, todo, False, d)


def grid(run_dir: Path, iters: list[int], seeds: list[int], wandb_id: str | None) -> None:
    """Assemble the sheets after the fact from a finished run's saved tiles."""
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    render_references(ctx, tok, run_dir, seeds)
    sheets = {it: build_sheet(run_dir, it, seeds) for it in iters}
    print(f"wrote {run_dir}/sheet_iterNNNNN.png for {iters}", flush=True)
    if wandb_id:
        import wandb
        run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", id=wandb_id, resume="allow")
        for it, sh in sheets.items():
            run.log({"sheet": wandb.Image(sh, caption=f"{it} updates")}, step=it)
        run.finish()


def train(steps: int, eval_at: list[int], lr: float, name: str, seed: int, use_wandb: bool,
          render_seeds: list[int], in_sentence: bool = False, hyper_rank: int = 0) -> None:
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
    if hyper_rank:
        tok.hyper = HyperRows(tok, hyper_rank)
        for r in tok.rows:
            r.requires_grad_(False)
    trainable = list(tok.hyper.parameters()) if tok.hyper else list(tok.rows)
    opt = torch.optim.Adam(trainable, lr=lr)
    rng = random.Random(seed)
    print(f"train: {len(cells)} cells, {steps} steps, lr {lr}, evals at {eval_at}, out {out}", flush=True)
    run_wb = None
    if use_wandb:
        import wandb
        run_wb = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", name=f"learned-and-{name}",
                            group="learned-and-token", config={"steps": steps, "lr": lr, "pool": POOL.name,
                            "cells": len(cells), "held_out": HELD_OUT[0], "render_seeds": render_seeds,
                            "in_sentence": in_sentence, "hyper_rank": hyper_rank})
        print(f"wandb run {run_wb.id} {run_wb.url}", flush=True)
    A, B = "a cat", "a dog"

    def evaluate(it: int) -> bool:
        res = share_removed(models, tok, device, dtype, in_sentence=in_sentence)
        fired = report(f"iter {it}", res)
        if run_wb:
            run_wb.log({f"held_out_share/step{st}": sum(v) / len(v) for st, v in res.items()}, step=it)
        if render_seeds:
            import wandb
            render_tiles(ctx, tok, A, B, render_seeds, third_prompt(A, B, in_sentence) if in_sentence else True,
                         out / "tiles" / f"iter{it:05d}")
            sheet = build_sheet(out, it, render_seeds, third_prompt(A, B, in_sentence) if in_sentence else TOKEN)
            if run_wb:
                run_wb.log({"sheet": wandb.Image(sheet, caption=f"{it} updates")}, step=it)
        return fired

    if render_seeds:
        render_references(ctx, tok, out, render_seeds)
    evaluate(0)
    t0, run = time.time(), []
    for it in range(1, steps + 1):
        pair, cell_seed = rng.choice(cells)
        st = load_state("train", pair, cell_seed, rng.randrange(50), device)
        target, _ = and_target(st)
        tok.bind(*pair_prompts(pair)[:2], grad=True)
        e = eps_and(models, tok, st, device, dtype, grad=True,
                    prompt=third_prompt(*pair_prompts(pair)[:2], in_sentence))
        sq = (e - target) ** 2
        opt.zero_grad(set_to_none=True)
        sq.sum().backward()  # summed, so fp16 gradients through the UNet do not underflow; Adam is scale-free
        if all(torch.isfinite(r.grad).all() for r in trainable if r.grad is not None):
            opt.step()
        else:
            print(f"iter {it}: non-finite gradient, step skipped", flush=True)
        run.append(sq.mean().item())
        if run_wb:
            now = tok.override if tok.override is not None else tok.rows
            run_wb.log({"loss": run[-1], "row_norm_1": now[0].norm().item(),
                        "row_norm_2": now[1].norm().item()}, step=it)
        if it % 10 == 0:
            print(f"iter {it}: loss (mean squared error per element) {sum(run[-10:]) / 10:.5f}, "
                  f"rows' norms {[round(r.norm().item(), 4) for r in (tok.override if tok.override is not None else tok.rows)]}, "
                  f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if it in eval_at or it == steps:
            fired = evaluate(it)
            torch.save({"rows": [r.detach().cpu() for r in tok.rows], "ids": tok.ids, "iter": it,
                        "hyper": tok.hyper.state_dict() if tok.hyper else None},
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
    ap.add_argument("--in-sentence", action="store_true", help="third expert is 'A <and> B', not the bare token")
    ap.add_argument("--hyper-rank", type=int, default=0, help="rows written from the pair by a rank-r network")
    ap.add_argument("--pairs-sheet", type=Path, metavar="RUN_DIR")
    ap.add_argument("--iter", type=int, default=8000)
    ap.add_argument("--pairs", nargs="*", default=[], metavar="PAIR:SPLIT:SEEDS")
    args = ap.parse_args()
    if args.pairs_sheet:
        pairs_sheet(args.pairs_sheet, args.iter, args.pairs, args.wandb_id, args.in_sentence, args.hyper_rank)
    if args.grid:
        grid(args.grid, args.iters, args.render_seeds or list(HELD_OUT[1]), args.wandb_id)
    if args.train:
        train(args.steps, args.eval_at, args.lr, args.name, args.seed, args.wandb, args.render_seeds, args.in_sentence, args.hyper_rank)
    if args.check_token:
        check_token()
    if args.baseline:
        baseline()
