#!/usr/bin/env python
"""Figure 2 as a grid: three prompt pairs by four Stable Diffusion models, joint prompt beside
uncorrected product-of-experts composition in every cell.

Rows are the pairs, columns are SD 1.4, SD 2.1, SDXL and SD 3.5, and one cell is two images of one
pair on one model at one seed: the joint prompt "{a} and {b}" on the left, PoE on the right. The
plan is plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md.

The PoE rule is the repo's own (poe_repair/methods/_sampling.py, run_cfg_poe):
    eps = eps_u + w (eps_a - eps_u) + w (eps_b - eps_u)
with w the same guidance weight the joint-prompt cell beside it uses. SD 3.5 predicts a velocity,
which at a fixed step is affine in the noise prediction with the same coefficients for every
prompt, so the same line applied to velocities is the same rule.

SDXL goes through the repo's sampler and borrows the cached seed starting latent of cat x dog, as
every SDXL figure in the paper does. The other three draw their noise from a CUDA generator seeded
with the seed, at their own latent shape, so "seed 42" is a different tensor in each column and one
tensor within a column.

    python scripts/same_rule_four_models.py --dry-run
    python scripts/same_rule_four_models.py --check-load
    python scripts/same_rule_four_models.py --model sd14 --seeds 42 4 123
    python scripts/same_rule_four_models.py --sheet --seed 42 --out sheet-seed-42.pdf
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

OUT = Path(os.environ.get("SAME_RULE_OUT", "/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models"))
NOISE_PAIR = "a_cat__x__a_dog"          # whose cached starting latent every SDXL cell borrows

MODELS = {
    "sd14": dict(label="SD 1.4", ids=["CompVis/stable-diffusion-v1-4"],
                 size=512, steps=50, guidance=7.5, sampler="DDIM"),
    "sd21": dict(label="SD 2.1", ids=["Manojb/stable-diffusion-2-1-base"],
                 size=512, steps=50, guidance=7.5, sampler="DDIM"),
    "sdxl": dict(label="SDXL", ids=["stabilityai/stable-diffusion-xl-base-1.0"],
                 size=1024, steps=50, guidance=7.5, sampler="DDIM"),
    "sd35": dict(label="SD 3.5", ids=["stabilityai/stable-diffusion-3.5-medium"],
                 size=1024, steps=40, guidance=4.5, sampler="flow-matching Euler, rule on velocities"),
}
PAIRS = [("a butterfly", "a flower meadow"), ("a camel", "a forest"), ("a cat", "a dog")]
COLUMNS = ("joint", "poe")


def slug(a, b):
    return f"{a.replace(' ', '_')}__x__{b.replace(' ', '_')}"


def joint(a, b):
    return f"{a} and {b}"


def cell_path(model, a, b, seed, column):
    return OUT / model / slug(a, b) / f"seed_{seed}" / f"{column}.png"


def write(img, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[built] {path}", flush=True)


def todo(model, seeds):
    return [(a, b, s, c) for s in seeds for a, b in PAIRS for c in COLUMNS
            if not cell_path(model, a, b, s, c).exists()]


def meta_for(model, model_id, a, b, seed, column, noise):
    m = MODELS[model]
    return {"model": model, "model_id": model_id, "prompt_a": a, "prompt_b": b,
            "joint_prompt": joint(a, b), "seed": seed, "column": column,
            "size": m["size"], "steps": m["steps"], "guidance": m["guidance"],
            "sampler": m["sampler"], "noise": noise,
            "rule": "joint prompt with CFG" if column == "joint" else "eps_u + w(eps_a-eps_u) + w(eps_b-eps_u)"}


# ---- SD 1.4 and SD 2.1: UNet, loaded part by part (the SD 2.1 mirror has no model_index.json).
# stabilityai/stable-diffusion-2-1-base is no longer on the Hugging Face hub (404, 2026-09-22), so SD 2.1
# base comes from the mirror Manojb/stable-diffusion-2-1-base, and every meta.json records that id.

def load_unet_model(model):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    last = None
    for mid in MODELS[model]["ids"]:
        try:
            parts = dict(
                unet=UNet2DConditionModel.from_pretrained(mid, subfolder="unet", torch_dtype=torch.float16),
                vae=AutoencoderKL.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float16),
                text_encoder=CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=torch.float16),
                tokenizer=CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer"),
                scheduler=DDIMScheduler.from_pretrained(mid, subfolder="scheduler"),
            )
        except Exception as e:  # the next id is tried, and the one used is printed and recorded
            print(f"[load] {mid} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            last = e
            continue
        pred = parts["scheduler"].config.get("prediction_type", "epsilon")
        if pred != "epsilon":
            raise SystemExit(f"{mid} predicts {pred}, and this loop assumes epsilon")
        for k in ("unet", "vae", "text_encoder"):
            parts[k].to("cuda").eval()
        print(f"[load] {model} from {mid}", flush=True)
        return mid, parts
    raise SystemExit(f"no id for {model} loaded: {last}")


def render_unet(model, seeds):
    import torch
    m = MODELS[model]
    cells = todo(model, seeds)
    if not cells:
        print(f"{model}: everything already rendered", flush=True)
        return
    mid, p = load_unet_model(model)

    @torch.no_grad()
    def enc(text):
        ids = p["tokenizer"](text, padding="max_length", max_length=p["tokenizer"].model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to("cuda")
        return p["text_encoder"](ids)[0]

    e_u = enc("")
    h = m["size"] // 8
    w = m["guidance"]
    for a, b, seed, column in cells:
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((1, 4, h, h), generator=g, device="cuda", dtype=torch.float16)
        conds = [enc(joint(a, b))] if column == "joint" else [enc(a), enc(b)]
        emb = torch.cat(conds + [e_u])
        n = emb.shape[0]
        sch = p["scheduler"]
        sch.set_timesteps(m["steps"], device="cuda")
        x = x * sch.init_noise_sigma
        with torch.no_grad():
            for t in sch.timesteps:
                eps = p["unet"](sch.scale_model_input(x.repeat(n, 1, 1, 1), t), t,
                                encoder_hidden_states=emb).sample
                *e_c, e0 = eps.chunk(n)
                e = e0 + w * sum(c - e0 for c in e_c)
                x = sch.step(e, t, x).prev_sample
            img = p["vae"].decode(x / p["vae"].config.scaling_factor).sample
        img = to_pil(img)
        write(img, cell_path(model, a, b, seed, column),
              meta_for(model, mid, a, b, seed, column, f"torch.Generator(cuda).manual_seed({seed}), shape 1x4x{h}x{h}"))
    del p
    gc.collect(); torch.cuda.empty_cache()


def to_pil(img):
    from PIL import Image
    arr = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ---- SDXL: the repo's own sampler and cached starting latent

def render_sdxl(seeds):
    import torch
    from poe_repair.run import make_ctx
    from poe_repair._sdxl.runtime import encode_prompt_sdxl
    from poe_repair.composers._helpers import init_latents_for_cell
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import run_cfg, run_cfg_poe, write_decoded_image
    m = MODELS["sdxl"]
    cells = todo("sdxl", seeds)
    if not cells:
        print("sdxl: everything already rendered", flush=True)
        return
    ctx = make_ctx()
    if (ctx.guidance_scale, ctx.num_inference_steps) != (m["guidance"], m["steps"]):
        raise SystemExit(f"repo SDXL settings are {ctx.guidance_scale}/{ctx.num_inference_steps}, "
                         f"the plan says {m['guidance']}/{m['steps']}")
    enc = lambda text: encode_prompt_sdxl(text, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
    seq_e, pool_e = enc("")
    for a, b, seed, column in cells:
        init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
        common = dict(init_latents=init, models=ctx.models, scheduler=ctx.scheduler,
                      seq_e=seq_e, pool_e=pool_e, guidance_scale=ctx.guidance_scale,
                      num_inference_steps=ctx.num_inference_steps, height=m["size"], width=m["size"],
                      euler_init_noise_sigma=sigma, device=ctx.device, dtype=ctx.dtype)
        if column == "joint":
            s, pl = enc(joint(a, b))
            res = run_cfg(seq_cond=s, pool_cond=pl, **common)
        else:
            sa, pa = enc(a)
            sb, pb = enc(b)
            res = run_cfg_poe(seq_a=sa, pool_a=pa, seq_b=sb, pool_b=pb, **common)
        out = cell_path("sdxl", a, b, seed, column)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(res.image, out)
        out.with_suffix(".json").write_text(json.dumps(
            meta_for("sdxl", MODELS["sdxl"]["ids"][0], a, b, seed, column,
                     f"cached starting latent of {NOISE_PAIR} seed {seed}"), indent=1))
        print(f"[built] {out}", flush=True)
        del res
    del ctx
    gc.collect(); torch.cuda.empty_cache()


# ---- SD 3.5: the same rule on velocities

def render_sd35(seeds):
    import torch
    from diffusers import StableDiffusion3Pipeline
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
    m = MODELS["sd35"]
    cells = todo("sd35", seeds)
    if not cells:
        print("sd35: everything already rendered", flush=True)
        return
    mid = m["ids"][0]
    # Model CPU offload keeps one part on the GPU at a time (T5 alone is about 9.5 GB in bf16), so
    # this runs on the 12 GB cards of the batch partition as well as on a 24 GB one.
    pipe = StableDiffusion3Pipeline.from_pretrained(mid, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    if pipe.scheduler.config.get("use_dynamic_shifting"):
        raise SystemExit("the SD 3.5 scheduler uses dynamic shifting, which this loop does not pass")

    # Every prompt is encoded first, and the three text encoders are dropped before sampling.
    texts = {""} | {t for a, b, _, c in cells for t in ([joint(a, b)] if c == "joint" else [a, b])}
    emb = {}
    with torch.no_grad():
        for t in texts:
            pe, _, pooled, _ = pipe.encode_prompt(prompt=t, prompt_2=None, prompt_3=None, device="cuda",
                                                  do_classifier_free_guidance=False)
            emb[t] = (pe, pooled)
    # Sampling needs only the transformer and the VAE, about 5 GB together, so the offload hooks
    # come off and those two go to the GPU while the text encoders stay on the CPU.
    from accelerate.hooks import remove_hook_from_module
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"):
        remove_hook_from_module(getattr(pipe, name), recurse=True)
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipe, name).to("cpu")
    pipe.transformer.to("cuda"); pipe.vae.to("cuda")
    gc.collect(); torch.cuda.empty_cache()

    h = m["size"] // pipe.vae_scale_factor
    ch = pipe.transformer.config.in_channels
    w = m["guidance"]
    for a, b, seed, column in cells:
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((1, ch, h, h), generator=g, device="cuda", dtype=torch.bfloat16)
        keys = [joint(a, b)] if column == "joint" else [a, b]
        pe = torch.cat([emb[k][0] for k in keys] + [emb[""][0]])
        pooled = torch.cat([emb[k][1] for k in keys] + [emb[""][1]])
        n = pe.shape[0]
        timesteps, _ = retrieve_timesteps(pipe.scheduler, m["steps"], "cuda")
        with torch.no_grad():
            for t in timesteps:
                v = pipe.transformer(hidden_states=x.repeat(n, 1, 1, 1), timestep=t.expand(n),
                                     encoder_hidden_states=pe, pooled_projections=pooled,
                                     return_dict=False)[0]
                *v_c, v0 = v.chunk(n)
                vv = v0 + w * sum(c - v0 for c in v_c)
                x = pipe.scheduler.step(vv, t, x, return_dict=False)[0]
            lat = x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            img = pipe.vae.decode(lat, return_dict=False)[0]
        write(to_pil(img), cell_path("sd35", a, b, seed, column),
              meta_for("sd35", mid, a, b, seed, column, f"torch.Generator(cuda).manual_seed({seed}), shape 1x{ch}x{h}x{h}"))
    del pipe
    gc.collect(); torch.cuda.empty_cache()


# ---- the grid

def sheet(seed, out, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    order = list(MODELS)
    fig = plt.figure(figsize=(12, 4.9))
    gs = fig.add_gridspec(len(PAIRS), len(order), wspace=0.06, hspace=0.06,
                          left=0.105, right=0.995, top=0.86, bottom=0.01)
    missing = 0
    for r, (a, b) in enumerate(PAIRS):
        for c, model in enumerate(order):
            sub = gs[r, c].subgridspec(1, 2, wspace=0.02)
            for k, column in enumerate(COLUMNS):
                ax = fig.add_subplot(sub[0, k])
                ax.set_xticks([]); ax.set_yticks([])
                f = root / model / slug(a, b) / f"seed_{seed}" / f"{column}.png"
                if f.exists():
                    ax.imshow(Image.open(f).convert("RGB").resize((384, 384), Image.LANCZOS))
                else:
                    missing += 1
                    ax.set_facecolor("#fbe3e3")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#b00", fontsize=8,
                            transform=ax.transAxes)
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                if r == 0:
                    ax.set_title("joint prompt" if column == "joint" else "PoE", fontsize=8, pad=2)
                if r == 0 and k == 0:
                    ax.text(1.01, 1.16, MODELS[model]["label"], transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=11)
                if c == 0 and k == 0:
                    ax.set_ylabel(f"{a} ×\n{b}", fontsize=9.5, rotation=0, ha="right", va="center",
                                  labelpad=6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"[sheet] {out}  seed {seed}  missing {missing} of {len(PAIRS) * len(order) * 2}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--dry-run", action="store_true", help="print the cells and settings, render nothing")
    ap.add_argument("--check-load", action="store_true", help="load each model in turn and exit")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="--sheet only")
    ap.add_argument("--root", type=Path, default=OUT, help="--sheet only: where the PNGs are")
    ap.add_argument("--out", type=Path, help="--sheet only: the PDF to write")
    args = ap.parse_args()

    from disallowed_subjects import check as check_subject
    for a, b in PAIRS:
        check_subject(slug(a, b))

    if args.dry_run:
        for model, m in MODELS.items():
            print(f"{m['label']:7s} {m['ids'][0]}  {m['size']}px  {m['sampler']}  "
                  f"{m['steps']} steps  guidance {m['guidance']}")
        for s in args.seeds:
            for a, b in PAIRS:
                for model in MODELS:
                    for c in COLUMNS:
                        print(f"  seed {s:3d}  {slug(a, b):34s} {model}  {c:5s} -> {cell_path(model, a, b, s, c)}")
        print(f"{len(args.seeds) * len(PAIRS) * len(MODELS) * len(COLUMNS)} images, "
              f"{len(PAIRS) * len(MODELS) * len(COLUMNS)} per seed")
        return
    if args.sheet:
        sheet(args.seed, args.out or OUT / "sheets" / f"seed-{args.seed}.pdf", args.root)
        return
    if args.check_load:
        import torch
        for model in ("sd14", "sd21"):
            mid, p = load_unet_model(model)
            del p; gc.collect(); torch.cuda.empty_cache()
        from diffusers import StableDiffusion3Pipeline
        StableDiffusion3Pipeline.from_pretrained(MODELS["sd35"]["ids"][0], torch_dtype=torch.bfloat16)
        print("[load] sd35 ok", flush=True)
        from poe_repair.run import make_ctx
        ctx = make_ctx()
        print(f"[load] sdxl ok, guidance {ctx.guidance_scale}, steps {ctx.num_inference_steps}", flush=True)
        return
    if not args.model:
        raise SystemExit("--model, --dry-run, --check-load or --sheet")
    {"sd14": lambda s: render_unet("sd14", s), "sd21": lambda s: render_unet("sd21", s),
     "sdxl": render_sdxl, "sd35": render_sd35}[args.model](args.seeds)


if __name__ == "__main__":
    main()
