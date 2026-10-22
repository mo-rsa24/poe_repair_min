#!/usr/bin/env python
"""Joint, two-concept PoE and three-concept PoE on SDXL, on the laptop, from the same noise per seed.

The cluster's cached starting latents are not reachable from the laptop, so every column here starts
from fresh noise drawn from the seed, and the columns are comparable with each other, not with the
cluster's sheets. PoE is the repo's rule (poe_repair/_sdxl/metrics.py, guided_eps and poe_eps) written
for any number of conditions:

    eps = eps_empty + w * sum_i (eps_i - eps_empty)

    PY=/home/molef/micromamba/envs/sdxl-local/bin/python
    $PY scripts/local_sdxl_poe.py render --animals camel dog --seeds 1 2 3 4 5 6 7 8
    $PY scripts/local_sdxl_poe.py sheet
    $PY scripts/local_sdxl_poe.py tally          # after calls.json is written

50 DDIM steps, guidance 7.5, 1024 x 1024, fp16 UNet, fp32 VAE decode: the settings of every earlier sheet.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/results/does-an-effect-need-support/laptop-third-concept"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS, W, SIZE = 50, 7.5, 1024

# column name -> the list of conditions; one condition is an ordinary guided prompt, two or more is PoE
def columns(a: str) -> dict:
    return {
        "a {a}":                     [f"a {a}"],
        "joint: a {a} and rain":     [f"a {a} and rain"],
        "a {a} in the rain":         [f"a {a} in the rain"],
        "a wet {a}":                 [f"a wet {a}"],
        "PoE: a {a} x rain":         [f"a {a}", "rain"],
        "PoE: a {a} x wet":          [f"a {a}", "wet"],
        "PoE: a {a} x rain x wet":   [f"a {a}", "rain", "wet"],
        "PoE: a {a} x wet x wet":    [f"a {a}", "wet", "wet"],
    }


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").replace("__", "_")


def render(animals, seeds) -> None:
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True)
    dev = torch.device("cuda")
    # encode every prompt once, then free the text encoders so the UNet has the card to itself
    pipe.text_encoder.to(dev); pipe.text_encoder_2.to(dev)
    prompts = sorted({p for a in animals for ps in columns(a).values() for p in ps})
    emb = {}
    with torch.no_grad():
        for p in prompts:
            e, ne, pool, npool = pipe.encode_prompt(p, device=dev, num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True)
            emb[p] = (e, pool)
        emb[""] = (ne, npool)          # SDXL zeroes the empty prompt (force_zeros_for_empty_prompt)
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu"); torch.cuda.empty_cache()
    unet = pipe.unet.to(dev)
    sched = DDIMScheduler.from_config(pipe.scheduler.config)
    tid = torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]], device=dev, dtype=torch.float16)

    def eps(lat, t, p):
        e, pool = emb[p]
        return unet(sched.scale_model_input(lat, t), t, encoder_hidden_states=e,
                    added_cond_kwargs={"text_embeds": pool, "time_ids": tid}).sample

    for a in animals:
        for seed in seeds:
            for col, conds in columns(a).items():
                name = col.format(a=a)
                dst = OUT / a / f"seed{seed:02d}" / f"{slug(name)}.png"
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                g = torch.Generator(device=dev).manual_seed(seed)
                lat = torch.randn((1, 4, SIZE // 8, SIZE // 8), generator=g, device=dev, dtype=torch.float16)
                sched.set_timesteps(STEPS)
                lat = lat * sched.init_noise_sigma
                t0 = time.time()
                with torch.no_grad():
                    for t in sched.timesteps:
                        e0 = eps(lat, t, "")
                        e = e0 + W * sum(eps(lat, t, c) - e0 for c in conds)
                        lat = sched.step(e, t, lat).prev_sample
                    pipe.vae.to(dev, dtype=torch.float32)
                    img = pipe.vae.decode(lat.float() / pipe.vae.config.scaling_factor).sample
                    pipe.vae.to("cpu"); torch.cuda.empty_cache()
                img = ((img[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
                from PIL import Image
                Image.fromarray(img).save(dst)
                dst.with_suffix(".json").write_text(json.dumps(
                    {"column": name, "conditions": conds, "seed": seed, "steps": STEPS, "guidance": W,
                     "size": SIZE, "scheduler": "DDIM", "noise": "fresh, torch cuda generator from the seed",
                     "model": MODEL, "seconds": round(time.time() - t0, 1)}, indent=1))
                print(f"{a} seed {seed:2d} {name:28s} {time.time() - t0:5.0f} s", flush=True)


def sheet() -> None:
    from PIL import Image, ImageDraw
    for adir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        a = adir.name
        cols = [c.format(a=a) for c in columns(a)]
        seeds = sorted(adir.glob("seed*"))
        side, head, left = 256, 40, 90
        sh = Image.new("RGB", (left + side * len(cols), head + side * len(seeds)), "white")
        dr = ImageDraw.Draw(sh)
        for j, c in enumerate(cols):
            dr.text((left + j * side + 6, 6), c[:34], fill="black")
            dr.text((left + j * side + 6, 20), f"column {j + 1}", fill="gray")
        for i, sd in enumerate(seeds):
            dr.text((8, head + i * side + side // 2), sd.name, fill="black")
            for j, c in enumerate(cols):
                p = sd / f"{slug(c)}.png"
                if p.exists():
                    sh.paste(Image.open(p).resize((side - 4, side - 4)), (left + j * side + 2, head + i * side + 2))
        sh.save(OUT / f"{a}-sheet.jpg", quality=90)
        print(OUT / f"{a}-sheet.jpg", f"{len(seeds)} seeds x {len(cols)} columns")


def tally() -> None:
    """calls.json: {"camel": {"seed01": {"a wet camel": "e", ...}}}; e effect, n not, h half, x subject lost.
    A seed counts for the third-concept test when the joint prompt or the named effect shows the effect
    (e) and two-concept PoE with rain does not."""
    calls = json.loads((OUT / "calls.json").read_text())
    rows = []
    for a, by_seed in calls.items():
        c = {k.format(a=a): k for k in columns(a)}
        joint, named = f"joint: a {a} and rain", f"a wet {a}"
        poe2, poe3, poe_wet = f"PoE: a {a} x rain", f"PoE: a {a} x rain x wet", f"PoE: a {a} x wet"
        counted = [s for s, v in by_seed.items() if (v.get(joint) == "e" or v.get(named) == "e") and v.get(poe2) != "e"]
        back = [s for s in counted if by_seed[s].get(poe3) == "e"]
        # column 8 is judged on the pair seeds: "wet" alone did not bring the effect, "rain" with "wet" did
        pair = [s for s, v in by_seed.items() if v.get(poe_wet) != "e" and v.get(poe3) == "e"]
        twice = [s for s in pair if by_seed[s].get(f"PoE: a {a} x wet x wet") == "e"]
        per_col = {col: sum(v.get(col) == "e" for v in by_seed.values()) for col in c}
        rows.append({"animal": a, "seeds called": len(by_seed), "effect count per column": per_col,
                     "counted seeds": counted, "recovered by the third concept": back,
                     "PoE x wet effect count": per_col.get(poe_wet),
                     "pair seeds": pair, "matched by wet twice": twice})
    (OUT / "tally.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"\n{r['animal']}: {r['seeds called']} seeds")
        for col, n in r["effect count per column"].items():
            print(f"  {col:28s} effect on {n}")
        print(f"  counted seeds {len(r['counted seeds'])}, recovered by the third concept {len(r['recovered by the third concept'])}")
        print(f"  pair seeds {len(r['pair seeds'])}, matched by wet twice {len(r['matched by wet twice'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["render", "sheet", "tally"])
    ap.add_argument("--animals", nargs="+", default=["camel", "dog"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    a = ap.parse_args()
    {"render": lambda: render(a.animals, a.seeds), "sheet": sheet, "tally": tally}[a.stage]()
